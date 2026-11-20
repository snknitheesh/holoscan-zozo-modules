// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Camera projection and the pose of a planar target from its points, for markers and boards
// alike. Poses follow holoscan-apriltag's convention: the target's frame in the camera frame,
// camera x right, y down, z forward; target x right, y down as printed, z into the target.

#pragma once

#include <array>
#include <optional>
#include <span>

#include <holoscan/fiducials/types.hpp>
#include <holoscan/messages/camera_info.hpp>

namespace holoscan::fiducials {

// Cameras are holoscan-messages' CameraInfo, valid (messages::validate()), with any of its lens
// models: none, plumb_bob, rational_polynomial or equidistant, and skew.

// Checks that `camera` describes images of `width` x `height` pixels: intrinsics are in the
// pixels of one image size, so a calibration of another size would give wrong poses.
// @throws std::invalid_argument naming both sizes.
void check_camera_size(const messages::CameraInfo& camera, int width, int height);

// Where a point in the camera frame (z > 0) appears, distortion applied.
[[nodiscard]] Point2 project(const messages::CameraInfo& camera, const Point3& camera_point);
// Where a point of the target appears, for the target at `pose`.
[[nodiscard]] Point2 project(const messages::CameraInfo& camera, const Pose& pose,
                             const Point3& target_point);
// A pixel's undistorted normalized coordinates: (x / z, y / z) of the points it sees.
[[nodiscard]] Point2 undistort(const messages::CameraInfo& camera, const Point2& pixel);

// The pose of a planar target from at least four of its points (on its z = 0 plane, not all on
// one line) and where they appear. Both poses a plane seen from afar can take are refined, and
// the one that reprojects better is returned; nullopt if no pose fits.
[[nodiscard]] std::optional<Pose> solve_planar_pose(std::span<const Point2> target_xy,
                                                    std::span<const Point2> image,
                                                    const messages::CameraInfo& camera);

// The corners of a marker `length` meters across its black border, in its own frame (centered
// on it), in Marker::corners' order: top-left (-l/2, -l/2, 0), top-right (l/2, -l/2, 0),
// bottom-right (l/2, l/2, 0), bottom-left (-l/2, l/2, 0).
[[nodiscard]] std::array<Point3, 4> marker_object_corners(double length);

// The pose of `marker`'s frame, from its corners; nullopt if none fits.
[[nodiscard]] std::optional<Pose> marker_pose(const Marker& marker,
                                              const messages::CameraInfo& camera, double length);

}  // namespace holoscan::fiducials
