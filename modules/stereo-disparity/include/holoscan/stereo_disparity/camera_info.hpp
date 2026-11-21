// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What StereoDisparityOp needs of the stereo camera info it takes, and the
// camera info it sends with its depth. No SDK, CUDA or VPI.

#pragma once

#include <optional>
#include <string>

#include <holoscan/messages/camera_info.hpp>

namespace holoscan::stereo_disparity {

// Why a pair `info` (already valid, messages::validate()) cannot be matched
// in `width` x `height` images, or nullopt: both cameras must be
// messages::CameraState::kRectified, which gives one fx, the baseline and
// doffs, and of the images' size.
[[nodiscard]] std::optional<std::string> camera_info_problem(const messages::StereoCameraInfo& info,
                                                             int width, int height);

// The camera of a depth or disparity image that the rectified pair `info`
// gives, downscaled by `downscale` (1, 2, 4 or 8): the rectified left camera
// (kRectified, no distortion) of output_side(width, downscale) x
// output_side(height, downscale) pixels, its frame id, and its intrinsics in
// those pixels. Output pixel u covers input pixels u * downscale ..
// u * downscale + downscale - 1, centered on u * downscale + (downscale - 1) / 2,
// so fx and fy are divided by the factor and cx' = (cx - (downscale - 1) / 2)
// / downscale, and likewise cy. The binning is multiplied by the factor (sensor
// pixels per image pixel); the region of interest stays. The header's capture
// time is left 0, for the sender to stamp.
// @throws std::invalid_argument if camera_info_problem() finds one for the
// pair's own size, or for another factor.
[[nodiscard]] messages::CameraInfo depth_camera_info(const messages::StereoCameraInfo& info,
                                                     int downscale);

}  // namespace holoscan::stereo_disparity
