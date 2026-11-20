// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <utility>

#include <holoscan/height_map/synthetic_terrain.hpp>

namespace holoscan::height_map {

holoscan::messages::CameraInfo camera_info_of(const TerrainCamera& camera, std::string frame_id) {
  holoscan::messages::CameraInfo info;
  info.header.frame_id = std::move(frame_id);
  info.width = static_cast<std::uint32_t>(camera.width);
  info.height = static_cast<std::uint32_t>(camera.height);
  info.state = holoscan::messages::CameraState::kUndistorted;
  info.fx = camera.fx;
  info.fy = camera.fy;
  info.cx = camera.cx;
  info.cy = camera.cy;
  info.distortion_model = holoscan::messages::DistortionModel::kNone;
  return info;
}

}  // namespace holoscan::height_map
