// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The recording format without the SDK: what RecordingWriter writes,
// RecordingReader reads back exactly; a recording cut short reads up to its
// last full line; and the writer refuses what does not fit.

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/recorder/format.hpp>

#include "host_support.hpp"
#include "test_support.hpp"

namespace holoscan::recorder::test {
namespace {

std::filesystem::path g_work;

constexpr ElementType kU8{.code = 1, .bits = 8, .lanes = 1};
constexpr ElementType kU16{.code = 1, .bits = 16, .lanes = 1};
constexpr ElementType kF32{.code = 2, .bits = 32, .lanes = 1};

std::vector<std::byte> bytes_of(std::size_t count, std::uint8_t seed) {
  std::vector<std::byte> out(count);
  for (std::size_t i = 0; i < count; ++i) out[i] = static_cast<std::byte>(seed + i * 7U);
  return out;
}

ImageDescriptor descriptor(std::uint64_t i) {
  ImageDescriptor d;
  d.width = 4;
  d.height = 3;
  d.encoding = "NV12";
  d.color_space = "BT709";
  d.significant_bits = 8;
  d.roi_offset_x = -2;
  d.roi_offset_y = static_cast<std::int32_t>(i);
  d.plane_layouts = {{0, 4, 12}, {12, 4, 6}};
  d.exposure_time_ns = 33'000'000;
  d.gain = 1.0F / 3.0F;
  // No header on the second; a frame id with every character CSV must quote,
  // and an empty one.
  d.has_header = i != 1;
  if (d.has_header) {
    d.header_frame_id = i == 2 ? "" : "cam,\"left\"\nline";
    d.header_capture_timestamp_ns = -5;
    d.header_device_sequence = 18'000'000'000'000'000'000ULL;
  }
  return d;
}

Record image_record(std::uint64_t i) {
  Record record;
  record.stream = 0;
  record.sequence = i * 2;  // Gaps, as dropped samples leave.
  record.timestamp_ns = -1'000 + static_cast<std::int64_t>(i) * 1'000'000;
  record.frame_id = i + 1;
  record.dtype = kU8;
  record.shape = {6, 4};
  record.image = descriptor(i);
  return record;
}

Record tensor_record(std::uint64_t i, ElementType dtype, std::vector<std::int64_t> shape) {
  Record record;
  record.stream = 1;
  record.sequence = i;
  record.timestamp_ns = static_cast<std::int64_t>(i) * 500'000;
  record.dtype = dtype;
  record.shape = std::move(shape);
  return record;
}

void element_types() {
  for (const ElementType type : {kU8, kU16, kF32, ElementType{.code = 0, .bits = 16, .lanes = 4},
                                 ElementType{.code = 4, .bits = 16, .lanes = 1},
                                 ElementType{.code = 5, .bits = 64, .lanes = 1},
                                 ElementType{.code = 6, .bits = 8, .lanes = 1},
                                 ElementType{.code = 9, .bits = 8, .lanes = 1}}) {
    VERIFY_MSG(parse_element_type(to_string(type)) == type, to_string(type));
  }
  VERIFY(to_string(kF32) == "float32");
  VERIFY(to_string(ElementType{.code = 0, .bits = 16, .lanes = 4}) == "int16x4");
  VERIFY((ElementType{.code = 0, .bits = 16, .lanes = 4}.bytes() == 8U));
  VERIFY((ElementType{.code = 6, .bits = 1, .lanes = 1}.bytes() == 1U));
  for (const char* text : {"", "uint", "uint0", "float32x0", "u8", "uint8x", "int-8", "float300"}) {
    VERIFY_MSG(!error_message([&] { (void)parse_element_type(text); }).empty(), text);
  }
}

void round_trip() {
  const std::filesystem::path folder = fresh_folder(g_work, "round_trip");
  std::vector<std::vector<std::byte>> images;
  std::vector<std::vector<std::byte>> tensors;
  {
    RecordingWriter writer(
        folder, {{"camera", StreamKind::kImage}, {"imu.raw-1", StreamKind::kTensor}},
        WriterOptions{
            .calibration = {{"left", R"({"fx": 1.5, "d": [0, 1e-9]})"}, {"note", "\"text\""}},
            .metadata = {{"robot", "unit \"7\"\n"}}});
    for (std::uint64_t i = 0; i < 3; ++i) {
      images.push_back(bytes_of(24, static_cast<std::uint8_t>(i)));
      writer.append(image_record(i), images.back());
    }
    // A scalar, a vector, and a change of element type and rank.
    tensors.push_back(bytes_of(4, 9));
    writer.append(tensor_record(0, kF32, {}), tensors.back());
    tensors.push_back(bytes_of(12, 10));
    writer.append(tensor_record(1, kF32, {3}), tensors.back());
    tensors.push_back(bytes_of(16, 11));
    writer.append(tensor_record(2, kU16, {2, 4}), tensors.back());
    // An empty tensor.
    tensors.push_back({});
    writer.append(tensor_record(3, kU16, {0, 4}), tensors.back());
    VERIFY(writer.records() == 7U);
    VERIFY(writer.bytes() == 24U * 3U + 4U + 12U + 16U);
    const std::vector<std::uint64_t> dropped{5, 0};
    writer.finish(dropped);
    VERIFY(!error_message([&] { writer.append(image_record(9), images[0]); }).empty());
  }
  const RecordingReader reader(folder);
  VERIFY(reader.format() == RecordingFormat::kNative);
  VERIFY(reader.complete());
  VERIFY(reader.streams().size() == 2U);
  const StreamInfo& camera = reader.streams()[0];
  VERIFY(camera.name == "camera" && camera.kind == StreamKind::kImage);
  VERIFY(camera.records == 3U && camera.max_bytes == 24U && camera.dropped == 5U);
  VERIFY(camera.dtype == kU8 && camera.rank == 2U);
  const StreamInfo& imu = reader.streams()[1];
  VERIFY(imu.name == "imu.raw-1" && imu.kind == StreamKind::kTensor);
  VERIFY(imu.records == 4U && imu.max_bytes == 16U);
  // Its element type and rank change, so it promises neither.
  VERIFY(!imu.dtype && !imu.rank);
  VERIFY(reader.find_stream("imu.raw-1") == 1U);
  VERIFY(!reader.find_stream("none"));
  VERIFY(reader.calibration().at("left") == R"({"fx":1.5,"d":[0,1e-09]})");
  VERIFY(reader.calibration().at("note") == "\"text\"");
  VERIFY(reader.metadata().at("robot") == "unit \"7\"\n");

  VERIFY(reader.records().size() == 7U);
  std::size_t image = 0;
  std::size_t tensor = 0;
  for (const Record& record : reader.records()) {
    std::vector<std::byte> data(*record.bytes());
    reader.read(record, data);
    if (record.stream == 0) {
      const Record want = image_record(image);
      VERIFY(record.sequence == want.sequence && record.timestamp_ns == want.timestamp_ns);
      VERIFY(record.frame_id == want.frame_id && record.shape == want.shape);
      VERIFY_MSG(record.image == want.image, "image " + std::to_string(image) + ": descriptor");
      VERIFY(record.file == "camera.bin");
      VERIFY(data == images[image]);
      ++image;
    } else {
      VERIFY(!record.image);
      VERIFY(record.sequence == tensor);
      VERIFY(data == tensors[tensor]);
      ++tensor;
    }
  }
  VERIFY(image == 3U && tensor == 4U);
  VERIFY(reader.records()[3].shape.empty());
  VERIFY(!error_message([&] {
            std::vector<std::byte> wrong(3);
            reader.read(reader.records()[0], wrong);
          }).empty());
}

// The writer stops part way: data written, an index line cut short. The
// recording reads up to its last full line.
void cut_short() {
  const std::filesystem::path folder = fresh_folder(g_work, "cut_short");
  {
    RecordingWriter writer(folder, {{"a", StreamKind::kTensor}});
    for (std::uint64_t i = 0; i < 4; ++i) {
      Record record = tensor_record(i, kU8, {5});
      record.stream = 0;
      writer.append(record, bytes_of(5, static_cast<std::uint8_t>(i)));
    }
  }  // Not finished.
  {
    const RecordingReader reader(folder);
    VERIFY(!reader.complete());
    VERIFY(reader.records().size() == 4U);
  }
  std::string index = read_text(folder / "index.csv");
  index.resize(index.size() - 7);  // Into the last line.
  write_text(folder / "index.csv", index);
  {
    const RecordingReader reader(folder);
    VERIFY(reader.records().size() == 3U);
  }
  // A finished recording must not be cut short.
  std::string metadata = read_text(folder / "recording.json");
  metadata.replace(metadata.find("\"complete\": false"), 17, "\"complete\": true");
  write_text(folder / "recording.json", metadata);
  VERIFY(contains(error_message([&] { const RecordingReader reader(folder); }),
                  "the last line is cut short"));
}

void folders() {
  // Into a new folder, and its parents.
  const std::filesystem::path root = fresh_folder(g_work, "folders");
  const std::filesystem::path nested = root / "a" / "b";
  {
    RecordingWriter writer(nested, {{"x", StreamKind::kTensor}});
  }
  VERIFY(std::filesystem::exists(nested / "recording.json"));
  // Not over a recording, unless asked.
  VERIFY(
      contains(error_message([&] { RecordingWriter writer(nested, {{"x", StreamKind::kTensor}}); }),
               "already holds a recording"));
  {
    RecordingWriter writer(nested, {{"y", StreamKind::kTensor}}, WriterOptions{.overwrite = true});
    writer.finish();
  }
  VERIFY(!std::filesystem::exists(nested / "x.bin"));
  VERIFY(RecordingReader(nested).streams().front().name == "y");
  // Never into a folder of other files, nor over a file.
  write_text(nested / "mine.txt", "keep");
  VERIFY(contains(error_message([&] {
                    RecordingWriter writer(nested, {{"x", StreamKind::kTensor}},
                                           WriterOptions{.overwrite = true});
                  }),
                  "holds mine.txt, which is not part of a recording"));
  VERIFY(read_text(nested / "mine.txt") == "keep");
  VERIFY(contains(error_message([&] {
                    RecordingWriter writer(nested / "mine.txt", {{"x", StreamKind::kTensor}});
                  }),
                  "not a folder"));
}

void writer_refuses() {
  const std::filesystem::path folder = fresh_folder(g_work, "refuses");
  const auto refused = [&](std::vector<StreamDeclaration> streams, WriterOptions options,
                           const std::string& part) {
    std::filesystem::remove_all(folder);
    const std::string message = error_message(
        [&] { RecordingWriter writer(folder, std::move(streams), std::move(options)); });
    VERIFY_MSG(contains(message, part), "expected \"" + part + "\", got \"" + message + "\"");
  };
  refused({}, {}, "at least one stream");
  refused({{"", StreamKind::kImage}}, {}, "stream name \"\"");
  refused({{"a b", StreamKind::kImage}}, {}, "stream name \"a b\"");
  refused({{std::string(65, 'x'), StreamKind::kImage}}, {}, "is not 1 to 64");
  refused({{"a", StreamKind::kImage}, {"a", StreamKind::kTensor}}, {}, "two streams");
  refused({{"a", StreamKind::kImage}}, WriterOptions{.calibration = {{"c", "{"}}}, "not JSON");

  std::filesystem::remove_all(folder);
  RecordingWriter writer(folder, {{"img", StreamKind::kImage}, {"t", StreamKind::kTensor}});
  const auto append_refused = [&](Record record, std::size_t bytes, const std::string& part) {
    const std::string message = error_message([&] { writer.append(record, bytes_of(bytes, 0)); });
    VERIFY_MSG(contains(message, part), "expected \"" + part + "\", got \"" + message + "\"");
  };
  Record image = image_record(0);
  append_refused(image, 23, "23 bytes of data for a uint8 tensor of shape 6x4");
  Record tensor = tensor_record(0, kU8, {2});
  tensor.stream = 0;
  append_refused(tensor, 2, "has no image descriptor");
  image.stream = 1;
  append_refused(image, 24, "has an image descriptor");
  image.stream = 2;
  append_refused(image, 24, "record for stream 2 of 2");
  image = image_record(0);
  image.image->encoding = "nv12";
  append_refused(image, 24, "SDK enumerator names");
  writer.append(image_record(1), bytes_of(24, 0));
  append_refused(image_record(1), 24, "sequence 2 does not follow 2");
}

}  // namespace
}  // namespace holoscan::recorder::test

int main(int argc, char** argv) {
  namespace t = holoscan::recorder::test;
  t::g_work = argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::current_path();
  return t::run({
      t::TestCase{"element type names", [] { t::element_types(); }},
      t::TestCase{"what is written reads back exactly", [] { t::round_trip(); }},
      t::TestCase{"a recording cut short reads to its last full line", [] { t::cut_short(); }},
      t::TestCase{"folders written into, and refused", [] { t::folders(); }},
      t::TestCase{"the writer refuses what does not fit", [] { t::writer_refuses(); }},
  });
}
