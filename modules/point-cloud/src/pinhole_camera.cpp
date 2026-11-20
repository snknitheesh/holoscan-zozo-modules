// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "pinhole_camera.hpp"

#include <string>

namespace holoscan::point_cloud::detail {

holoscan::expected<CameraIntrinsics, holoscan::Error> pinhole_intrinsics(
    const holoscan::messages::CameraInfo& info, std::string_view op) {
  const auto refuse = [&](const std::string& why) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                        std::string(op) + ": camera info '" + info.header.frame_id + "' " + why});
  };
  if (info.state == holoscan::messages::CameraState::kRaw) {
    return refuse("describes raw images, with " +
                  std::string(holoscan::messages::distortion_model_name(info.distortion_model)) +
                  " distortion coefficients that are not zero; the operator models undistorted "
                  "or rectified images only: undistort or rectify the images first");
  }
  if (info.skew != 0.0) {
    return refuse("has a skew of " + std::to_string(info.skew) +
                  "; the operator models cameras without skew only");
  }
  return CameraIntrinsics{static_cast<float>(info.fx), static_cast<float>(info.fy),
                          static_cast<float>(info.cx), static_cast<float>(info.cy)};
}

}  // namespace holoscan::point_cloud::detail
