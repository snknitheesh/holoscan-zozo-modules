// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Recordings of a real camera, when HOLOSCAN_TEST_DATA points at a folder with
// a stereo-camera/ folder of them; skipped (exit 77) otherwise. For every
// recording there: the calibration reads and is a usable stereo calibration,
// every frame decodes through the playback backend, the eyes have the same size
// as each other and as the calibration, the RGB and depth images have theirs,
// and the timestamps and the camera's frame numbers increase.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/stereo_camera/source.hpp>

#include "test_support.hpp"

namespace holoscan::stereo_camera {
namespace {

constexpr int kSkip = 77;

bool matches(const ImageView& view, const CameraModel& camera) {
  return view.width == camera.width && view.height == camera.height;
}

void check_recording(const std::filesystem::path& folder) {
  const RecordingReader reader(folder);
  const StereoCalibration& calibration = reader.calibration();
  VERIFY(calibration.baseline() > 0.0 && calibration.left.fx > 0.0);
  VERIFY(reader.size() > 0);
  auto source = create_source(SourceConfig{.backend = Backend::kPlayback,
                                           .rgb = reader.has_rgb(),
                                           .depth = reader.has_depth(),
                                           .playback = PlaybackOptions{.directory = folder}});
  source->open();
  std::size_t frames = 0;
  std::size_t wrong_sizes = 0;
  std::size_t nonincreasing = 0;
  std::uint64_t valid_depth = 0;
  std::uint64_t depth_pixels = 0;
  std::optional<std::int64_t> last_time;
  std::optional<std::uint64_t> last_sequence;
  while (const auto frame = source->grab()) {
    ++frames;
    wrong_sizes +=
        !(matches(frame->left, calibration.left) && matches(frame->right, calibration.right) &&
          frame->left.width == frame->right.width && frame->left.height == frame->right.height &&
          frame->left.format == frame->right.format);
    if (reader.has_rgb()) wrong_sizes += !matches(*frame->rgb, calibration.rgb->model);
    if (reader.has_depth()) {
      wrong_sizes += !matches(*frame->depth, calibration.depth->model);
      const auto* depth = reinterpret_cast<const std::uint16_t*>(frame->depth->data);
      for (std::size_t i = 0; i < frame->depth->packed_bytes() / 2; ++i) {
        valid_depth += depth[i] != 0U;
      }
      depth_pixels += frame->depth->packed_bytes() / 2;
    }
    nonincreasing += last_time && frame->timestamp_ns <= *last_time;
    nonincreasing += last_sequence && frame->device_sequence <= *last_sequence;
    last_time = frame->timestamp_ns;
    last_sequence = frame->device_sequence;
  }
  VERIFY_MSG(frames == reader.size(), folder.string() + ": " + std::to_string(frames) + " of " +
                                          std::to_string(reader.size()) + " frames decoded");
  VERIFY_MSG(wrong_sizes == 0, folder.string() + ": " + std::to_string(wrong_sizes) +
                                   " frames with images of the wrong size");
  VERIFY_MSG(nonincreasing == 0, folder.string() + ": timestamps or frame numbers do not increase");
  std::cout << folder.filename().string() << ": " << calibration.camera.name << " "
            << calibration.camera.serial << ", " << frames << " frames, eyes "
            << calibration.left.width << "x" << calibration.left.height << ", baseline "
            << 1000.0 * calibration.baseline() << " mm, fx " << calibration.left.fx;
  if (depth_pixels != 0) {
    std::cout << ", valid depth " << 100.0 * valid_depth / depth_pixels << " %";
  }
  std::cout << '\n';
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
