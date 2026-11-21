// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What FoundationStereoOp needs of the stereo camera info it takes, and the
// camera info it sends with its depth. No SDK or CUDA.

#pragma once

#include <optional>
#include <string>

#include <holoscan/messages/camera_info.hpp>

namespace holoscan::foundation_stereo {

// Why a pair `info` (already valid, messages::validate()) cannot be used for
// `width` x `height` images, or nullopt: both cameras must be
// messages::CameraState::kRectified, which gives one fx, the baseline and
// doffs, and of the images' size.
[[nodiscard]] std::optional<std::string> camera_info_problem(const messages::StereoCameraInfo& info,
                                                             int width, int height);

// The camera of the depth and disparity images that the rectified pair `info`
// gives: the rectified left camera (kRectified, no distortion) at the images'
// size, which the operator's outputs keep, with the pair's frame id when the
// left camera has none. The header's capture time is left 0, for the sender
// to stamp.
// @throws std::invalid_argument if camera_info_problem() finds one for the
// pair's own size.
[[nodiscard]] messages::CameraInfo depth_camera_info(const messages::StereoCameraInfo& info);

}  // namespace holoscan::foundation_stereo
