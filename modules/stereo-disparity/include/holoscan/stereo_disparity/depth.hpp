// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Depth from the disparity of a rectified stereo pair, as the operator's
// kernels write it. No SDK, CUDA or VPI.

#pragma once

// Lets the kernels share the host's depth formula.
#ifdef __CUDACC__
#define HOLOSCAN_STEREO_DISPARITY_HD __host__ __device__
#else
#define HOLOSCAN_STEREO_DISPARITY_HD
#endif

namespace holoscan::stereo_disparity {

// A point at depth z (meters, along the optical axis) seen at column x_left in
// the left image of a rectified pair is at x_right = x_left - fx * baseline / z
// + doffs in the right one, so its disparity d = x_left - x_right gives
//
//   z = fx * baseline / (d + doffs)
//
// with doffs = cx_right - cx_left, as messages::StereoCameraInfo holds it.
// Returns that in meters, or 0 when it is not a positive, finite depth (a
// disparity of 0 or less is invalid), where messages::depth_from_disparity()
// returns NaN: the operator's depth image marks invalid pixels with 0.
[[nodiscard]] HOLOSCAN_STEREO_DISPARITY_HD inline constexpr float depth_from_disparity(
    float disparity, float fx_baseline, float doffs) noexcept {
  if (!(disparity > 0.0F)) {
    return 0.0F;
  }
  const float shifted = disparity + doffs;
  return shifted > 0.0F ? fx_baseline / shifted : 0.0F;
}

}  // namespace holoscan::stereo_disparity
