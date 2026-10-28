// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Pose from a detection's corners. Every backend reports corners only, and
// poses are estimated here, so all backends report poses the same way.

#pragma once

#include <array>
#include <optional>

#include <holoscan/apriltag/detection.hpp>

namespace holoscan::apriltag {

using Point2 = std::array<double, 2>;

// The pose of a tag `tag_size` meters across its black border, from its corners
// and the camera's intrinsics; nullopt if the corners are degenerate.
[[nodiscard]] std::optional<TagPose> estimate_pose(const Detection& detection,
                                                   const CameraIntrinsics& intrinsics,
                                                   double tag_size);

}  // namespace holoscan::apriltag
