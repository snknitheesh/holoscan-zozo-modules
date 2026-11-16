// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Malformed recordings, of either format, fail cleanly: RecordingReader throws a
// RecordingError that names the file, the line, and what is wrong, and never
// reads past what is there.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <holoscan/recorder/format.hpp>

#include "host_support.hpp"
#include "test_support.hpp"

namespace holoscan::recorder::test {
namespace {

namespace fs = std::filesystem;

fs::path g_work;

// A valid recording: an image stream and a tensor stream, two samples each.
fs::path native(const std::string& name) {
  const fs::path folder = g_work / "malformed" / name;
  fs::remove_all(folder);
  RecordingWriter writer(folder, {{"cam", StreamKind::kImage}, {"imu", StreamKind::kTensor}});
  for (std::uint64_t i = 0; i < 2; ++i) {
    Record image;
    image.stream = 0;
    image.sequence = i;
    image.timestamp_ns = static_cast<std::int64_t>(i) * 1000;
    image.dtype = ElementType{.code = 1, .bits = 8, .lanes = 1};
    image.shape = {2, 3};
    image.image = ImageDescriptor{};
    image.image->width = 3;
    image.image->height = 2;
    image.image->encoding = "MONO8";
    const std::vector<std::byte> pixels(6, std::byte{1});
    writer.append(image, pixels);
    Record tensor;
    tensor.stream = 1;
    tensor.sequence = i;
    tensor.timestamp_ns = static_cast<std::int64_t>(i) * 1000 + 1;
    tensor.dtype = ElementType{.code = 2, .bits = 32, .lanes = 1};
    tensor.shape = {2};
    const std::vector<std::byte> values(8, std::byte{2});
    writer.append(tensor, values);
  }
  writer.finish();
  return folder;
}

// A valid stereo camera recording: two frames of 4x2 eyes, RGB and depth.
fs::path stereo(const std::string& name) {
  const fs::path folder = g_work / "malformed" / name;
  fs::remove_all(folder);
  for (const char* sub : {"left", "right", "rgb", "depth"}) fs::create_directories(folder / sub);
  write_text(folder / "calibration.json", R"({"version": 1, "rectified": true})");
  std::string index = "frame,timestamp_ns,device_sequence,left,right,rgb,depth\n";
  for (int f = 0; f < 2; ++f) {
    const std::string n = "00000" + std::to_string(f);
    write_text(folder / "left" / (n + ".pgm"), "P5\n4 2\n255\n" + std::string(8, 'L'));
    write_text(folder / "right" / (n + ".pgm"),
               "P5\n# a comment\n4 2\n255\n" + std::string(8, 'R'));
    write_text(folder / "rgb" / (n + ".ppm"), "P6 4 2 255\n" + std::string(24, 'C'));
    std::string depth = "P5\n4 2\n65535\n";
    for (int i = 0; i < 8; ++i) depth += std::string{'\x01', static_cast<char>(i)};
    write_text(folder / "depth" / (n + ".pgm"), depth);
    index += std::to_string(f) + "," + std::to_string(1000 + 33 * f) + "," +
             std::to_string(31 + f) + ",left/" + n + ".pgm,right/" + n + ".pgm,rgb/" + n +
             ".ppm,depth/" + n + ".pgm\n";
  }
  write_text(folder / "index.csv", index);
  return folder;
}

void replace(const fs::path& file, const std::string& from, const std::string& to) {
  std::string text = read_text(file);
  const std::size_t at = text.find(from);
  if (at == std::string::npos) throw std::runtime_error("no \"" + from + "\" in " + file.string());
  text.replace(at, from.size(), to);
  write_text(file, text);
}

// Opening the recording fails with a message holding every one of `parts`.
void fails(const fs::path& folder, const std::vector<std::string>& parts) {
  const std::string message = error_message([&] { const RecordingReader reader(folder); });
  if (message.empty()) {
    VERIFY_MSG(false, folder.filename().string() + ": opened, but should have failed");
    return;
  }
  for (const std::string& part : parts) {
    VERIFY_MSG(contains(message, part),
               folder.filename().string() + ": expected \"" + part + "\" in \"" + message + "\"");
  }
}

void valid_ones_open() {
  const RecordingReader a(native("valid_native"));
  VERIFY(a.records().size() == 4U);
  const RecordingReader b(stereo("valid_stereo"));
  VERIFY(b.format() == RecordingFormat::kStereoCamera);
  VERIFY(b.streams().size() == 4U && b.records().size() == 8U);
  VERIFY(b.calibration().at("stereo_camera") == R"({"version":1,"rectified":true})");
  const Record& depth = b.records()[3];
  VERIFY(b.streams()[depth.stream].name == "depth");
  VERIFY(depth.image->encoding == "MONO16" && depth.dtype.bits == 16U);
  VERIFY((depth.shape == std::vector<std::int64_t>{2, 4}));
  VERIFY(depth.frame_id == 1U && depth.timestamp_ns == 1000);
  VERIFY(depth.image->header_device_sequence == 31U && depth.image->header_frame_id == "depth");
  std::vector<std::byte> data(16);
  b.read(depth, data);
  // Big-endian 0x01NN comes out as host-order 0x01NN.
  for (int i = 0; i < 8; ++i) {
    std::uint16_t value = 0;
    std::memcpy(&value, data.data() + 2 * i, 2);
    VERIFY(value == 0x0100U + static_cast<std::uint16_t>(i));
  }
  std::vector<std::byte> rgb(24);
  b.read(b.records()[2], rgb);
  VERIFY(rgb == std::vector<std::byte>(24, std::byte{'C'}));
  std::vector<std::byte> right(8);
  b.read(b.records()[1], right);
  VERIFY(right == std::vector<std::byte>(8, std::byte{'R'}));
}

void not_recordings() {
  fails(g_work / "malformed" / "no_such_folder", {"no such folder"});
  const fs::path empty = fresh_folder(g_work / "malformed", "empty");
  fails(empty, {"not a recording"});
}

void bad_metadata() {
  struct Case {
    std::string name;
    std::string from;
    std::string to;
    std::vector<std::string> parts;
  };
  const std::vector<Case> cases{
      {"json_syntax", "\"version\": 1,", "\"version\": 1", {"recording.json", "line 4"}},
      {"format", "\"holoscan-recording\"", "\"other\"", {"not a holoscan-recording"}},
      {"version", "\"version\": 1", "\"version\": 2", {"format version 2"}},
      {"complete_type", "\"complete\": true", "\"complete\": 1", {"\"complete\" must be"}},
      {"kind", "\"kind\": \"tensor\"", "\"kind\": \"audio\"", {"kind \"audio\""}},
      {"name", "{\"name\": \"cam\"", "{\"name\": \"c/m\"", {"stream name \"c/m\""}},
      {"duplicate", "{\"name\": \"imu\"", "{\"name\": \"cam\"", {"two streams are named \"cam\""}},
      {"escaping_file", "\"file\": \"cam.bin\"", "\"file\": \"../cam.bin\"", {"not a file of"}},
      {"missing_file", "\"file\": \"cam.bin\"", "\"file\": \"gone.bin\"", {"gone.bin", "open"}},
      {"metadata_type", "\"metadata\": {}", "\"metadata\": {\"a\": 1}", {"must be a string"}},
      {"no_streams", "\"streams\"", "\"rivers\"", {"needs \"streams\""}},
  };
  for (const Case& c : cases) {
    const fs::path folder = native("metadata_" + c.name);
    replace(folder / "recording.json", c.from, c.to);
    fails(folder, c.parts);
  }
}

void bad_index() {
  struct Case {
    std::string name;
    std::function<void(const fs::path&)> corrupt;
    std::vector<std::string> parts;
  };
  const auto edit = [](std::string from, std::string to) {
    return [from, to](const fs::path& folder) { replace(folder / "index.csv", from, to); };
  };
  const std::vector<Case> cases{
      {"header", edit("stream,sequence", "stream,seq"), {"index.csv", "the first line must be"}},
      {"fields", edit("imu,0,1,imu.bin", "imu,0,imu.bin"), {"line 3", "21 fields; expected 22"}},
      {"stream", edit("imu,0,1,imu.bin", "gyro,0,1,imu.bin"), {"stream \"gyro\" is not in"}},
      {"file", edit("imu,0,1,imu.bin", "imu,0,1,cam.bin"), {"file \"cam.bin\" is not"}},
      {"number", edit("imu,0,1,imu.bin", "imu,zero,1,imu.bin"), {"line 3", "\"zero\""}},
      {"negative", edit("imu,0,1,imu.bin", "imu,-1,1,imu.bin"), {"unsigned", "\"-1\""}},
      {"dtype", edit(",float32,", ",real32,"), {"unknown element type \"real32\""}},
      {"shape", edit(",2x3,", ",2x-3,"), {"negative extent"}},
      {"bytes", edit(",8,0,float32,", ",9,0,float32,"), {"bytes 9 do not match"}},
      {"past_end", edit("imu,1,1001,imu.bin,8,", "imu,1,1001,imu.bin,80,"), {"runs past"}},
      {"sequence", edit("imu,1,1001", "imu,0,1001"), {"sequence 0 of stream \"imu\" does not"}},
      {"tensor_image_columns", edit(",float32,2,", ",float32,2,MONO8"), {"holds tensors"}},
      {"encoding", edit(",MONO8,", ",mono8,"), {"enumerator names"}},
      {"planes", edit(",0,0,,0,", ",0,0,1:2,0,"), {"planes must be"}},
      {"quote", edit("cam,1,", "\"cam,1,"), {"quoted field is not closed"}},
      {"cut_short",
       [](const fs::path& folder) {
         std::string text = read_text(folder / "index.csv");
         write_text(folder / "index.csv", text.substr(0, text.size() - 3));
       },
       {"the last line is cut short"}},
      {"data_shrunk",
       [](const fs::path& folder) { fs::resize_file(folder / "cam.bin", 7); },
       {"runs past the 7 bytes of cam.bin"}},
  };
  for (const Case& c : cases) {
    const fs::path folder = native("index_" + c.name);
    c.corrupt(folder);
    fails(folder, c.parts);
  }
}

void bad_stereo() {
  struct Case {
    std::string name;
    std::function<void(const fs::path&)> corrupt;
    std::vector<std::string> parts;
  };
  const auto edit = [](std::string file, std::string from, std::string to) {
    return [file, from, to](const fs::path& folder) { replace(folder / file, from, to); };
  };
  const std::vector<Case> cases{
      {"calibration", edit("calibration.json", "}", ""), {"calibration.json"}},
      {"calibration_array",
       [](const fs::path& folder) { write_text(folder / "calibration.json", "[1]"); },
       {"must hold an object"}},
      {"header", edit("index.csv", "frame,timestamp_ns", "frame,time"), {"the first line must"}},
      {"column", edit("index.csv", ",rgb,depth", ",rgb,ir"), {"unknown column \"ir\""}},
      {"twice", edit("index.csv", ",rgb,depth", ",rgb,rgb"), {"column \"rgb\" appears twice"}},
      {"escaping",
       edit("index.csv", "left/000001.pgm", "../left/000001.pgm"),
       {"line 3", "not a path inside"}},
      {"absolute", edit("index.csv", "left/000001.pgm", "/etc/passwd"), {"not a path inside"}},
      {"missing", edit("index.csv", "left/000001.pgm", "left/000009.pgm"), {"is missing"}},
      {"frame", edit("index.csv", "\n1,1033", "\n0,1033"), {"frame 0 does not follow 0"}},
      {"time", edit("index.csv", "\n1,1033", "\n1,999"), {"goes back in time"}},
      {"fields", edit("index.csv", ",depth/000001.pgm", ""), {"6 fields; expected 7"}},
      {"not_netpbm",
       [](const fs::path& folder) { write_text(folder / "left" / "000000.pgm", "GIF89a"); },
       {"000000.pgm", "not a binary PGM"}},
      {"bad_size",
       [](const fs::path& folder) { write_text(folder / "rgb" / "000000.ppm", "P6\n0 2\n255\n"); },
       {"out of range"}},
  };
  for (const Case& c : cases) {
    const fs::path folder = stereo("stereo_" + c.name);
    c.corrupt(folder);
    fails(folder, c.parts);
  }
}

// What opening cannot see, reading does: an image that changed size, or ends
// early, fails that read, cleanly.
void bad_stereo_reads() {
  const fs::path folder = stereo("stereo_reads");
  const RecordingReader reader(folder);
  write_text(folder / "left" / "000001.pgm", "P5\n2 4\n255\n" + std::string(8, 'L'));
  std::vector<std::byte> data(8);
  VERIFY(contains(error_message([&] { reader.read(reader.records()[4], data); }),
                  "a 2x4 P5 8-bit image in a stream of 4x2 P5 8-bit images"));
  write_text(folder / "right" / "000001.pgm", "P5\n4 2\n255\nRR");
  VERIFY(contains(error_message([&] { reader.read(reader.records()[5], data); }),
                  "the file ends before"));
}

}  // namespace
}  // namespace holoscan::recorder::test

int main(int argc, char** argv) {
  namespace t = holoscan::recorder::test;
  t::g_work = argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::current_path();
  return t::run({
      t::TestCase{"valid recordings of both formats open", [] { t::valid_ones_open(); }},
      t::TestCase{"folders that are not recordings", [] { t::not_recordings(); }},
      t::TestCase{"malformed recording.json", [] { t::bad_metadata(); }},
      t::TestCase{"malformed index.csv and data", [] { t::bad_index(); }},
      t::TestCase{"malformed stereo camera recordings", [] { t::bad_stereo(); }},
      t::TestCase{"stereo camera images that change", [] { t::bad_stereo_reads(); }},
  });
}
