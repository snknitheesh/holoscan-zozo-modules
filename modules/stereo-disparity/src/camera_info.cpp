// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/stereo_disparity/camera_info.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "holoscan/stereo_disparity/disparity_options.hpp"

namespace holoscan::stereo_disparity {
namespace {

std::string size_text(std::uint32_t width, std::uint32_t height) {
  return std::to_string(width) + "x" + std::to_string(height);
}

}  // namespace

std::optional<std::string> camera_info_problem(const messages::StereoCameraInfo& info, int width,
                                               int height) {
  if (info.left.state != messages::CameraState::kRectified ||
      info.right.state != messages::CameraState::kRectified) {
    return "the stereo camera info describes " +
           std::string(messages::camera_state_name(info.left.state)) +
           " cameras; the images must be a rectified pair (rectify them first), and the camera "
           "info rectified to match";
  }
  const auto image_width = static_cast<std::uint32_t>(width);
  const auto image_height = static_cast<std::uint32_t>(height);
  if (width < 1 || height < 1 || info.left.width != image_width ||
      info.left.height != image_height || info.right.width != image_width ||
      info.right.height != image_height) {
    return "the stereo camera info is for " + size_text(info.left.width, info.left.height) +
           " (left) and " + size_text(info.right.width, info.right.height) +
           " (right) images, and the images are " + size_text(image_width, image_height);
  }
  return std::nullopt;
}

messages::CameraInfo depth_camera_info(const messages::StereoCameraInfo& info, int downscale) {
  if (downscale != 1 && downscale != 2 && downscale != 4 && downscale != 8) {
    throw std::invalid_argument("downscale must be 1, 2, 4 or 8");
  }
  const messages::CameraInfo& left = info.left;
  if (auto problem =
          camera_info_problem(info, static_cast<int>(left.width), static_cast<int>(left.height))) {
    throw std::invalid_argument(*problem);
  }
  messages::CameraInfo depth = left;
  depth.header.capture_timestamp_ns = 0;
  depth.header.device_sequence = 0;
  if (depth.header.frame_id.empty()) depth.header.frame_id = info.header.frame_id;
  const auto factor = static_cast<std::uint32_t>(downscale);
  depth.width = static_cast<std::uint32_t>(output_side(static_cast<int>(left.width), downscale));
  depth.height = static_cast<std::uint32_t>(output_side(static_cast<int>(left.height), downscale));
  const double scale = 1.0 / downscale;
  const double shift = (downscale - 1) / 2.0;
  depth.fx = left.fx * scale;
  depth.fy = left.fy * scale;
  depth.cx = (left.cx - shift) * scale;
  depth.cy = (left.cy - shift) * scale;
  depth.binning_x = left.binning_x * factor;
  depth.binning_y = left.binning_y * factor;
  messages::validate(depth, "depth_camera_info");
  return depth;
}

}  // namespace holoscan::stereo_disparity
