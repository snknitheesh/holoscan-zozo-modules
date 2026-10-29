// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// StereoCameraOp playing recordings of a real camera, when HOLOSCAN_TEST_DATA
// points at a folder with a stereo-camera/ folder of them; skipped (exit 77)
// otherwise. Each recording plays once through the operator, into device memory,
// with every stream it has: every frame arrives, with the recording's images,
// the eyes the same size, the calibration the recording's, and the source's
// timestamps and the capture times increasing.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/stereo_camera/stereo_camera_op.hpp>

#include "graph_support.hpp"
#include "test_support.hpp"

namespace holoscan::stereo_camera {
namespace {

constexpr int kSkip = 77;

void check_recording(const std::filesystem::path& folder) {
  const RecordingReader reader(folder);
  const bool rgb = reader.has_rgb();
  const bool depth = reader.has_depth();
  const auto frames = static_cast<std::uint32_t>(reader.size());
  auto received = std::make_shared<test::Received>();
  holoscan::Graph graph{"real_data"};
  StereoCameraConfig config;
  config.source = SourceConfig{.backend = Backend::kPlayback,
                               .rgb = rgb,
                               .depth = depth,
                               .playback = PlaybackOptions{.directory = folder}};
  config.period = std::chrono::milliseconds{2};
  config.memory = OutputMemory::kDevice;
  auto camera = graph.op<StereoCameraOp>("camera", config);
  auto sink = graph.op<test::FrameSinkOp>(
      "sink",
      test::FrameSinkConfig{.rgb = rgb, .depth = depth, .expected = frames, .received = received});
  test::connect(graph, camera, sink, rgb, depth,
                holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kBlocking});
  const auto outcome = test::run_graph(graph, test::device_outputs(rgb, depth));
  VERIFY_MSG(outcome.completed, folder.string() + ": " + outcome.diagnostics);
  VERIFY_MSG(received->frames.size() == frames, folder.string() + ": " +
                                                    std::to_string(received->frames.size()) +
                                                    " of " + std::to_string(frames) + " frames");
  const std::string calibration = format_calibration(reader.calibration());
  Image left;
  Image right;
  Image rgb_image;
  Image depth_image;
  std::size_t wrong = 0;
  std::optional<std::int64_t> last_timestamp;
  std::optional<std::int64_t> last_capture;
  for (std::size_t i = 0; i < received->frames.size(); ++i) {
    const test::ReceivedFrame& frame = received->frames[i];
    reader.read(i, left, right, rgb ? &rgb_image : nullptr, depth ? &depth_image : nullptr);
    wrong += !frame.consistent;
    wrong += frame.left.width != frame.right.width || frame.left.height != frame.right.height;
    wrong += !same_pixels(frame.left.view(left.format), left.view());
    wrong += !same_pixels(frame.right.view(right.format), right.view());
    if (rgb) wrong += !same_pixels(frame.rgb->view(PixelFormat::kRgb8), rgb_image.view());
    if (depth) wrong += !same_pixels(frame.depth->view(PixelFormat::kMono16), depth_image.view());
    wrong += frame.calibration.frame != i ||
             frame.calibration.timestamp_ns != reader.frames()[i].timestamp_ns;
    // Copy the camera's text fields back from the recording: the message carries no settings.
    StereoCalibration from = from_message(frame.calibration);
    from.camera.settings = reader.calibration().camera.settings;
    wrong += format_calibration(from) != calibration;
    wrong += last_timestamp && frame.calibration.timestamp_ns <= *last_timestamp;
    wrong += last_capture && frame.capture_ns <= *last_capture;
    last_timestamp = frame.calibration.timestamp_ns;
    last_capture = frame.capture_ns;
  }
  VERIFY_MSG(wrong == 0, folder.string() + ": " + std::to_string(wrong) + " mismatches");
  std::cout << folder.filename().string() << ": " << received->frames.size() << " frames through "
            << "StereoCameraOp" << (rgb ? " with rgb" : "") << (depth ? " and depth" : "") << '\n';
}

}  // namespace
}  // namespace holoscan::stereo_camera

int main() {
  namespace sc = holoscan::stereo_camera;
  const char* root = std::getenv("HOLOSCAN_TEST_DATA");
  if (root == nullptr || *root == '\0') {
    std::cout << "skipped: HOLOSCAN_TEST_DATA is not set\n";
    return sc::kSkip;
  }
  const std::filesystem::path data = std::filesystem::path(root) / "stereo-camera";
  std::vector<std::filesystem::path> recordings;
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator(data, error)) {
    if (entry.is_directory() && std::filesystem::exists(entry.path() / sc::kCalibrationFile)) {
      recordings.push_back(entry.path());
    }
  }
  if (recordings.empty()) {
    std::cout << "skipped: no recordings in " << data.string() << '\n';
    return sc::kSkip;
  }
  std::sort(recordings.begin(), recordings.end());
  std::vector<sc::test::TestCase> cases;
  for (const auto& folder : recordings) {
    cases.push_back({folder.filename().string(), [folder] { sc::check_recording(folder); }});
  }
  return sc::test::run(cases);
}
