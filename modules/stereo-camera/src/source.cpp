// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/stereo_camera/source.hpp"

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sources.hpp"

namespace holoscan::stereo_camera {
namespace {

constexpr std::array<std::pair<Backend, std::string_view>, 3> kBackendNames{{
    {Backend::kRealSense, "realsense"},
    {Backend::kV4l2SideBySide, "v4l2-sbs"},
    {Backend::kPlayback, "playback"},
}};

std::string no_realsense() {
  return "backend realsense: this build has no librealsense; install librealsense2-dev and rebuild";
}

// Largest image side and frame rate a live camera may be asked for.
constexpr int kMaxSide = 8192;
constexpr int kMaxFps = 1000;

void check_size(int width, int height, int fps, const std::string& what) {
  if (width < 1 || height < 1 || width > kMaxSide || height > kMaxSide) {
    throw std::invalid_argument(what + ": width and height must be 1.." + std::to_string(kMaxSide));
  }
  if (fps < 1 || fps > kMaxFps) {
    throw std::invalid_argument(what + ": fps must be 1.." + std::to_string(kMaxFps));
  }
}

}  // namespace

std::string_view backend_name(Backend backend) noexcept {
  for (const auto& [kind, name] : kBackendNames) {
    if (kind == backend) return name;
  }
  return "unknown";
}

std::optional<Backend> parse_backend(std::string_view name) noexcept {
  for (const auto& [kind, backend] : kBackendNames) {
    if (backend == name) return kind;
  }
  return std::nullopt;
}

bool realsense_available() noexcept { return HOLOSCAN_STEREO_CAMERA_WITH_REALSENSE != 0; }

std::vector<Backend> available_backends() {
  std::vector<Backend> backends;
  if (realsense_available()) backends.push_back(Backend::kRealSense);
  backends.push_back(Backend::kV4l2SideBySide);
  backends.push_back(Backend::kPlayback);
  return backends;
}

void validate(const SourceConfig& config) {
  switch (config.backend) {
    case Backend::kRealSense: {
      if (!realsense_available()) {
        throw std::invalid_argument(no_realsense());
      }
      const RealSenseOptions& options = config.realsense;
      check_size(options.width, options.height, options.fps, "realsense");
      if (config.rgb) {
        check_size(options.rgb_width, options.rgb_height, options.fps, "realsense rgb");
      }
      return;
    }
    case Backend::kV4l2SideBySide: {
      const V4l2SideBySideOptions& options = config.v4l2;
      if (config.rgb || config.depth) {
        throw std::invalid_argument("v4l2-sbs: a side-by-side camera gives no rgb or depth image");
      }
      check_size(options.width, options.height, options.fps, "v4l2-sbs");
      if (options.width % 2 != 0 ||
          (options.format == SideBySideFormat::kYuyv && options.width % 4 != 0)) {
        throw std::invalid_argument(
            "v4l2-sbs: the frame width must split into two eyes (a multiple of 4 for YUYV)");
      }
      if (options.output != PixelFormat::kMono8 && options.output != PixelFormat::kRgb8) {
        throw std::invalid_argument("v4l2-sbs: the eyes are mono8 or rgb8");
      }
      if (options.device.empty()) {
        throw std::invalid_argument("v4l2-sbs: needs a device");
      }
      if (options.calibration_file.empty()) {
        throw std::invalid_argument("v4l2-sbs: needs a calibration file");
      }
      const StereoCalibration calibration = read_calibration(options.calibration_file);
      if (calibration.left.width != options.width / 2 ||
          calibration.left.height != options.height) {
        throw std::invalid_argument(options.calibration_file.string() + ": its eyes are " +
                                    std::to_string(calibration.left.width) + "x" +
                                    std::to_string(calibration.left.height) + ", the camera's " +
                                    std::to_string(options.width / 2) + "x" +
                                    std::to_string(options.height));
      }
      return;
    }
    case Backend::kPlayback: {
      if (config.playback.directory.empty()) {
        throw std::invalid_argument("playback: needs a recording directory");
      }
      const RecordingReader reader(config.playback.directory);
      if (config.rgb && !reader.has_rgb()) {
        throw std::invalid_argument(config.playback.directory.string() +
                                    ": the recording has no rgb images");
      }
      if (config.depth && !reader.has_depth()) {
        throw std::invalid_argument(config.playback.directory.string() +
                                    ": the recording has no depth images");
      }
      return;
    }
  }
  throw std::invalid_argument("unknown backend");
}

std::unique_ptr<StereoSource> create_source(const SourceConfig& config) {
  validate(config);
  switch (config.backend) {
    case Backend::kRealSense:
      return detail::create_realsense_source(config);
    case Backend::kV4l2SideBySide:
      return detail::create_v4l2_source(config);
    case Backend::kPlayback:
      return detail::create_playback_source(config);
  }
  throw std::invalid_argument("unknown backend");
}

namespace detail {

SourceLayout realsense_layout(const SourceConfig& config) {
  const RealSenseOptions& options = config.realsense;
  SourceLayout layout;
  layout.left = StreamLayout{options.width, options.height, PixelFormat::kMono8};
  layout.right = layout.left;
  if (config.rgb) {
    layout.rgb = StreamLayout{options.rgb_width, options.rgb_height, PixelFormat::kRgb8};
  }
  if (config.depth) {
    layout.depth = StreamLayout{options.width, options.height, PixelFormat::kMono16};
  }
  layout.live = true;
  return layout;
}

#if !HOLOSCAN_STEREO_CAMERA_WITH_REALSENSE
std::unique_ptr<StereoSource> create_realsense_source(const SourceConfig&) {
  throw std::invalid_argument(no_realsense());
}
#endif

}  // namespace detail
}  // namespace holoscan::stereo_camera
