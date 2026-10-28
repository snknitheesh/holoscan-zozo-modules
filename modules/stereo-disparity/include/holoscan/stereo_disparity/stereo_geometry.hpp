// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The geometry of a rectified stereo pair: what turns a disparity into a
// depth. No SDK, CUDA or VPI.

#pragma once

#include <optional>

// Lets the kernels share the host's depth formula.
#ifdef __CUDACC__
#define HOLOSCAN_STEREO_DISPARITY_HD __host__ __device__
#else
#define HOLOSCAN_STEREO_DISPARITY_HD
#endif

namespace holoscan::stereo_disparity {

// A rectified stereo pair: both eyes share the focal length and the rows, and
// the right eye sits `baseline` meters to the left eye's right. Pixel centers
// are at integer coordinates, as in OpenCV and ROS CameraInfo.
//
// A point at depth z (meters, along the optical axis) seen at column x_left in
// the left eye is at x_right = x_left - fx * baseline / z + doffs in the right
// one, so its disparity d = x_left - x_right gives
//
//   z = fx * baseline / (d + doffs)
//
// doffs is the right eye's principal point minus the left eye's (cx_right -
// cx_left, Middlebury's convention): zero for a pair rectified to one
// principal point, as RealSense and OpenCV's stereoRectify with
// CALIB_ZERO_DISPARITY give.
struct StereoGeometry {
  double fx{};        // Focal length in pixels of the rectified images.
  double baseline{};  // Meters between the two eyes' optical centers.
  int width{};        // Image size the geometry is for; images must match it.
  int height{};
  // The left eye's principal point; carried for consumers (deprojection), not
  // used for depth.
  std::optional<double> cx;
  std::optional<double> cy;
  double doffs{};  // Pixels; see above.
};

// @throws std::invalid_argument naming the first unusable field: fx and
//         baseline must be finite and positive, width and height 1..16384,
//         cx, cy and doffs finite.
void validate_geometry(const StereoGeometry& geometry);

// fx * baseline / (disparity + doffs) in meters, or 0 when that is not a
// positive, finite depth (a disparity of 0 or less is invalid).
[[nodiscard]] HOLOSCAN_STEREO_DISPARITY_HD inline constexpr float depth_from_disparity(
    float disparity, float fx_baseline, float doffs) noexcept {
  if (!(disparity > 0.0F)) {
    return 0.0F;
  }
  const float shifted = disparity + doffs;
  return shifted > 0.0F ? fx_baseline / shifted : 0.0F;
}

}  // namespace holoscan::stereo_disparity
