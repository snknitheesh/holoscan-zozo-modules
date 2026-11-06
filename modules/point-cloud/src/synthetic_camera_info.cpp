// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <utility>

#include <holoscan/point_cloud/synthetic_depth.hpp>

namespace holoscan::point_cloud {

holoscan::messages::CameraInfo synthetic_camera_info(const SyntheticFrame& f,
                                                     std::string frame_id) {
  holoscan::messages::CameraInfo info;
  info.header.frame_id = std::move(frame_id);
  info.width = static_cast<std::uint32_t>(f.width);
  info.height = static_cast<std::uint32_t>(f.height);
  info.state = holoscan::messages::CameraState::kUndistorted;
  info.fx = f.intrinsics.fx;
  info.fy = f.intrinsics.fy;
  info.cx = f.intrinsics.cx;
  info.cy = f.intrinsics.cy;
  info.distortion_model = holoscan::messages::DistortionModel::kNone;
  return info;
}

}  // namespace holoscan::point_cloud
