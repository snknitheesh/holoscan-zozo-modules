// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// How a camera's lens bends the pinhole image, for the cameras of
// holoscan-messages' CameraInfo: the checks rectification needs beyond the
// message's own, and distortion and its inverse. No SDK, no CUDA.

#pragma once

#include <cstddef>
#include <optional>
#include <string_view>

#include <holoscan/messages/camera_info.hpp>

namespace holoscan::rectify {

// Largest image side a camera may have.
inline constexpr int kMaxImageSide = 16384;

// Checks that `camera` is valid (messages::validate()) and that this module
// can remap its images: each side at most kMaxImageSide, no skew, the whole
// sensor (no region of interest) and no binning, since its intrinsics then
// describe the images as they are.
// @throws std::invalid_argument naming the first unusable field, prefixed by `what`.
void check_camera(const messages::CameraInfo& camera, std::string_view what = "camera");

// The distortion coefficient `index` of `camera` in its model's order, 0 when
// the model has fewer.
[[nodiscard]] double coefficient(const messages::CameraInfo& camera, std::size_t index) noexcept;

// A point in normalized image coordinates (x = X/Z, y = Y/Z).
struct Point2 {
  double x{};
  double y{};
};

// Where the lens moves the normalized point `ideal`: its distorted normalized
// coordinates. Pixel coordinates are then (fx * x + cx, fy * y + cy).
[[nodiscard]] Point2 distort(const messages::CameraInfo& camera, Point2 ideal) noexcept;
// The inverse of distort(): the ideal normalized point the lens moved to
// `distorted`, solved by Newton's method to 1e-12. nullopt when there is none
// (a fisheye ray at or beyond 90 degrees from the axis, or no convergence).
[[nodiscard]] std::optional<Point2> undistort(const messages::CameraInfo& camera,
                                              Point2 distorted) noexcept;

}  // namespace holoscan::rectify
