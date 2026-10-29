// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Every pixel format of the SDK, and tensors of other element types and ranks,
// recorded by RecorderOp from host and from device memory, and played back by
// ReplayerOp into host and into device memory: every byte, capture time, frame
// id and image descriptor comes back exactly as it was sent.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/recorder/format.hpp>
#include <holoscan/recorder/recorder_op.hpp>
#include <holoscan/recorder/replayer_op.hpp>

#include "graph_support.hpp"
#include "test_support.hpp"

namespace holoscan::recorder::test {
namespace {

constexpr std::uint64_t kFrames = 5;

const holoscan::ConnectionOptions kLossless{.queue_policy = holoscan::QueuePolicy::kBlocking};

// Records kFrames frames of `streams`; returns the recording's folder.
std::filesystem::path record(const std::vector<Stream>& streams, const std::string& name) {
  const std::filesystem::path folder = fresh_folder(name);
  holoscan::Graph graph{"record"};
  auto sent = std::make_shared<std::atomic<std::uint64_t>>(0);
  auto source =
      graph.op<PatternSourceOp>("source", SourceConfig{.streams = streams,
                                                       .frames = kFrames,
                                                       .period = std::chrono::milliseconds{5},
                                                       .calls = kFrames * 40U,
                                                       .sent = sent});
  std::vector<RecorderStream> inputs;
  for (const Stream& stream : streams) inputs.push_back({stream.name, stream.kind});
  auto stats = std::make_shared<RecorderStats>();
  auto recorder = graph.op<RecorderOp>(
      "recorder", RecorderConfig{.directory = folder.string(),
                                 .streams = inputs,
                                 .calibration = {{"camera", R"({"fx": 336.98760986328125})"}},
                                 .metadata = {{"test", "round trip"}},
                                 .stats = stats});
  holoscan::CompileOptions options;
  for (const Stream& stream : streams) {
    if (stream.kind == StreamKind::kImage) {
      graph.add_flow(source->image(stream.name), recorder->image(stream.name), kLossless);
    } else {
      graph.add_flow(source->tensor(stream.name), recorder->tensor(stream.name), kLossless);
    }
    if (stream.device) {
      options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
          .operator_path = "source", .output_port = stream.name, .device = holoscan::DeviceId{0}});
    }
  }
  const RunOutcome outcome = run_graph(graph, std::move(options));
  VERIFY_MSG(outcome.compiled && outcome.completed, outcome.diagnostics);
  VERIFY(*sent == kFrames);
  const std::uint64_t samples = kFrames * streams.size();
  VERIFY(stats->received == samples);
  VERIFY(stats->recorded == samples);
  VERIFY(stats->dropped == 0U);
  return folder;
}

void check_image(const schema::ImageT& got, const schema::ImageT& want, const std::string& what) {
  VERIFY_MSG(got.width == want.width && got.height == want.height, what + ": size");
  VERIFY_MSG(got.encoding == want.encoding, what + ": encoding");
  VERIFY_MSG(got.color_space == want.color_space, what + ": color space");
  VERIFY_MSG(got.significant_bits == want.significant_bits, what + ": significant bits");
  VERIFY_MSG(got.roi_offset_x == want.roi_offset_x && got.roi_offset_y == want.roi_offset_y,
             what + ": roi");
  VERIFY_MSG(got.exposure_time_ns == want.exposure_time_ns, what + ": exposure");
  VERIFY_MSG(got.gain == want.gain, what + ": gain");
  VERIFY_MSG(got.plane_layouts.size() == want.plane_layouts.size(), what + ": planes");
  for (std::size_t p = 0; p < got.plane_layouts.size() && p < want.plane_layouts.size(); ++p) {
    VERIFY_MSG(
        got.plane_layouts[p].offset_bytes() == want.plane_layouts[p].offset_bytes() &&
            got.plane_layouts[p].row_stride_bytes() == want.plane_layouts[p].row_stride_bytes() &&
            got.plane_layouts[p].size_bytes() == want.plane_layouts[p].size_bytes(),
        what + ": plane " + std::to_string(p));
  }
  // The SDK gives a received image a header, an empty one when its producer
  // sent none; the recorder records it like any other.
  if (!want.header) {
    VERIFY_MSG(
        !got.header || (got.header->frame_id.empty() && got.header->capture_timestamp_ns == 0 &&
                        got.header->device_sequence == 0U),
        what + ": empty header");
  }
  if (want.header) {
    VERIFY_MSG(static_cast<bool>(got.header), what + ": header");
  }
  if (got.header && want.header) {
    VERIFY_MSG(got.header->frame_id == want.header->frame_id, what + ": header frame id");
    VERIFY_MSG(got.header->capture_timestamp_ns == want.header->capture_timestamp_ns,
               what + ": header capture time");
    VERIFY_MSG(got.header->device_sequence == want.header->device_sequence,
               what + ": header device sequence");
  }
}

// The recording holds exactly what the source sent.
void check_recording(const std::filesystem::path& folder, const std::vector<Stream>& streams) {
  const RecordingReader reader(folder);
  VERIFY(reader.format() == RecordingFormat::kNative);
  VERIFY(reader.complete());
  VERIFY(reader.streams().size() == streams.size());
  VERIFY(reader.records().size() == kFrames * streams.size());
  VERIFY(reader.calibration().at("camera") == R"({"fx":336.98760986328125})");
  VERIFY(reader.metadata().at("test") == "round trip");
  std::vector<std::uint64_t> seen(streams.size(), 0);
  for (const Record& record : reader.records()) {
    const std::size_t s = record.stream;
    const Stream& stream = streams[s];
    const std::uint64_t frame = seen[s]++;
    const std::string what = stream.name + " frame " + std::to_string(frame);
    VERIFY_MSG(reader.streams()[s].name == stream.name, what + ": name");
    VERIFY_MSG(record.sequence == frame, what + ": sequence");
    VERIFY_MSG(record.timestamp_ns == capture_ns(frame), what + ": timestamp");
    VERIFY_MSG(record.frame_id == frame_id_of(frame), what + ": frame id");
    VERIFY_MSG(record.shape == stream.shape, what + ": shape");
    VERIFY_MSG(record.dtype.code == stream.dtype.code && record.dtype.bits == stream.dtype.bits &&
                   record.dtype.lanes == stream.dtype.lanes,
               what + ": dtype");
    std::vector<std::byte> data(stream.bytes());
    reader.read(record, data);
    VERIFY_MSG(
        std::memcmp(data.data(), pattern_bytes(s, frame, data.size()).data(), data.size()) == 0,
        what + ": data");
    VERIFY_MSG(record.image.has_value() == (stream.kind == StreamKind::kImage), what + ": kind");
  }
}

// Plays the recording back and checks every sample.
void replay(const std::filesystem::path& folder, const std::vector<Stream>& streams, bool device) {
  holoscan::Graph graph{"replay"};
  auto stats = std::make_shared<ReplayerStats>();
  auto replayer = graph.op<ReplayerOp>(
      "replayer", ReplayerConfig{.directory = folder.string(),
                                 .memory = device ? ReplayMemory::kDevice : ReplayMemory::kHost,
                                 .stats = stats});
  auto collected = std::make_shared<Collected>();
  auto sink = graph.op<CollectSinkOp>(
      "sink",
      SinkConfig{.streams = streams, .expected = kFrames * streams.size(), .collected = collected});
  for (const Stream& stream : streams) {
    if (stream.kind == StreamKind::kImage) {
      graph.add_flow(replayer->image(stream.name), sink->image(stream.name), kLossless);
    } else {
      graph.add_flow(replayer->tensor(stream.name), sink->tensor(stream.name), kLossless);
    }
  }
  holoscan::CompileOptions options;
  replayer->bind_device(options, "replayer");
  const RunOutcome outcome = run_graph(graph, std::move(options));
  VERIFY_MSG(outcome.compiled && outcome.completed, outcome.diagnostics);
  VERIFY(stats->emitted == kFrames * streams.size());
  for (std::size_t s = 0; s < streams.size(); ++s) {
    const Stream& stream = streams[s];
    const std::vector<Received>& samples = collected->streams[stream.name];
    VERIFY_MSG(samples.size() == kFrames, stream.name + ": samples");
    for (std::uint64_t frame = 0; frame < samples.size(); ++frame) {
      const Received& got = samples[frame];
      const std::string what =
          stream.name + " frame " + std::to_string(frame) + (device ? " (device)" : " (host)");
      VERIFY_MSG(got.capture_valid && got.capture_ns == capture_ns(frame), what + ": capture");
      VERIFY_MSG(got.frame_id == frame_id_of(frame), what + ": frame id");
      VERIFY_MSG(got.shape == stream.shape, what + ": shape");
      VERIFY_MSG(got.dtype.code == stream.dtype.code && got.dtype.bits == stream.dtype.bits &&
                     got.dtype.lanes == stream.dtype.lanes,
                 what + ": dtype");
      VERIFY_MSG((got.device == kDLCUDA) == device, what + ": memory");
      VERIFY_MSG(got.bytes == pattern_bytes(s, frame, stream.bytes()), what + ": data");
      VERIFY_MSG(got.image.has_value() == (stream.kind == StreamKind::kImage), what + ": kind");
      if (got.image) {
        check_image(*got.image, descriptor_of(stream, s, frame), what);
      }
    }
  }
}

void round_trip(bool record_device) {
  const std::vector<Stream> streams = every_format(64, 48, record_device);
  const std::filesystem::path folder =
      record(streams, record_device ? "every_format_device" : "every_format_host");
  check_recording(folder, streams);
  replay(folder, streams, false);
  replay(folder, streams, true);
}

// A large frame, as a camera sends: 1280x720 RGB8 and MONO16 from the GPU.
void large_frames() {
  const std::vector<Stream> streams{
      Stream{.name = "rgb",
             .encoding = schema::ImageEncoding_RGB8,
             .dtype = kU8,
             .shape = {720, 1280, 3},
             .device = true},
      Stream{.name = "depth",
             .encoding = schema::ImageEncoding_MONO16,
             .dtype = kU16,
             .shape = {720, 1280},
             .device = false},
  };
  const std::filesystem::path folder = record(streams, "large_frames");
  check_recording(folder, streams);
  replay(folder, streams, true);
}

}  // namespace
}  // namespace holoscan::recorder::test

int main() {
  using holoscan::recorder::test::TestCase;
  namespace t = holoscan::recorder::test;
  return t::run({
      TestCase{"every format, recorded from host memory, replayed to host and device",
               [] { t::round_trip(false); }},
      TestCase{"every format, recorded from device memory, replayed to host and device",
               [] { t::round_trip(true); }},
      TestCase{"1280x720 RGB8 and MONO16 frames", [] { t::large_frames(); }},
  });
}
