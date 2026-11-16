// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Calibrations with known values, for tests and demos that need ground truth
// without a real camera.

#pragma once

#include <cstdint>

#include "holoscan/messages/camera_info.hpp"

namespace holoscan::messages {

// The rotation by `angle_rad` about `axis` (any length > 0), right-handed.
[[nodiscard]] Matrix3 rotation_matrix(const Vector3& axis, double angle_rad) noexcept;

// A stereo pair as a stereo calibration gives it: two cameras of `width` x
// `height` (> 0) with `model`'s distortion (kRaw; kNone gives kUndistorted
// cameras), slightly different intrinsics, the right camera 0.12 m to the
// right of the left one and turned by about a degree, and the rectifications
// that turn both into one frame in which the right camera lies along +x of
// the left one. Frame ids "left" and "right"; the pair's is "left".
[[nodiscard]] StereoCameraInfo synthetic_stereo_camera_info(DistortionModel model,
                                                            std::uint32_t width = 1280,
                                                            std::uint32_t height = 720);

}  // namespace holoscan::messages
