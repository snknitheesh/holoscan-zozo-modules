// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The pinhole intrinsics the operators take from a CameraInfo: the volume's
// kernels model undistorted images without skew only.

#pragma once

#include <string_view>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/messages/camera_info.hpp>
#include <holoscan/tsdf/tsdf_volume.hpp>

namespace holoscan::tsdf::detail {

// The intrinsics of `info` (already validated), or kInvalidArgument naming
// `op` and the camera for one the kernels cannot model: a raw camera, whose
// images still have their lens distortion, or one with skew.
[[nodiscard]] holoscan::expected<CameraIntrinsics, holoscan::Error> pinhole_intrinsics(
    const holoscan::messages::CameraInfo& info, std::string_view op);

}  // namespace holoscan::tsdf::detail
