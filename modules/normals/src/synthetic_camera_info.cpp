// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <utility>

#include <holoscan/normals/synthetic_scene.hpp>

namespace holoscan::normals {

holoscan::messages::CameraInfo synthetic_camera_info(const SyntheticScene& scene,
                                                     std::string frame_id) {
  holoscan::messages::CameraInfo info;
  info.header.frame_id = std::move(frame_id);
  info.width = static_cast<std::uint32_t>(scene.width);
  info.height = static_cast<std::uint32_t>(scene.height);
  info.state = holoscan::messages::CameraState::kUndistorted;
  info.fx = scene.fx;
  info.fy = scene.fy;
  info.cx = scene.cx;
  info.cy = scene.cy;
  info.distortion_model = holoscan::messages::DistortionModel::kNone;
  return info;
}

}  // namespace holoscan::normals
