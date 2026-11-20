// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <utility>

#include <holoscan/tsdf/synthetic_scene.hpp>

namespace holoscan::tsdf {

holoscan::messages::CameraInfo synthetic_camera_info(int width, int height, std::string frame_id) {
  const CameraIntrinsics k = synthetic_intrinsics(width, height);
  holoscan::messages::CameraInfo info;
  info.header.frame_id = std::move(frame_id);
  info.width = static_cast<std::uint32_t>(width);
  info.height = static_cast<std::uint32_t>(height);
  info.state = holoscan::messages::CameraState::kUndistorted;
  info.fx = k.fx;
  info.fy = k.fy;
  info.cx = k.cx;
  info.cy = k.cy;
  holoscan::messages::validate(info);
  return info;
}

}  // namespace holoscan::tsdf
