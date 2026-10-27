// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Pinhole camera intrinsics, as the module's operators and kernels take them.

#pragma once

namespace holoscan::point_cloud {

// Pinhole intrinsics in pixels; pixel centers are at integer coordinates, as in
// OpenCV and ROS CameraInfo.
struct CameraIntrinsics {
  float fx{};
  float fy{};
  float cx{};
  float cy{};
};

}  // namespace holoscan::point_cloud
