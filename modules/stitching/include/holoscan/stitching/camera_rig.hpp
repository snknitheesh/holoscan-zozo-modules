// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A camera rig: several cameras, each with its calibration as a ROS
// camera_info file gives it (holoscan-messages' CameraInfo: image size,
// intrinsics and lens model) and its pose in the rig's frame, which
// camera_info does not carry. No SDK, no CUDA.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/messages/camera_info.hpp>

namespace holoscan::stitching {

// The transform from frame A to frame B: p_B = rotation * p_A + translation,
// row-major, metres (holoscan-messages' RigidTransform).
using RigidTransform = messages::RigidTransform;

// The transform from B to A.
[[nodiscard]] RigidTransform inverse(const RigidTransform& b_from_a) noexcept;
// a_from_c = a_from_b * b_from_c.
[[nodiscard]] RigidTransform compose(const RigidTransform& a_from_b,
                                     const RigidTransform& b_from_c) noexcept;

// The pose of a camera mounted on a rig, as camera_from_rig. The rig frame is
// x forward, y left, z up (ROS REP 103); a camera frame is optical: x right,
// y down, z forward. With all angles 0 the camera looks along the rig's +x,
// upright. `yaw_deg` turns it to the left (about +z), then `pitch_deg` tilts
// it up (negative looks down), then `roll_deg` turns it clockwise about its
// own axis, as seen from behind. `position` is its optical centre in metres.
[[nodiscard]] RigidTransform mount(const std::array<double, 3>& position, double yaw_deg,
                                   double pitch_deg, double roll_deg = 0.0) noexcept;

// One camera of the rig. `info` describes its images as they are: their size,
// the intrinsics in their pixels (pixel centres at integer coordinates) and
// the lens model, plumb_bob, rational_polynomial, equidistant (a fisheye,
// which sees beyond 90 degrees) or none. Its state and rectification are not
// used: `camera_from_rig` is the pose of the frame the intrinsics describe.
struct Camera {
  messages::CameraInfo info;
  RigidTransform camera_from_rig;
};

// Up to kMaxCameras cameras, in the order of the stitcher's inputs.
struct CameraRig {
  std::vector<Camera> cameras;
};

inline constexpr int kMaxImageSide = 16384;
inline constexpr std::size_t kMaxCameras = 8;

// Checks a camera's calibration: valid (messages::validate()), each side at
// most kMaxImageSide, no skew, and neither a region of interest nor binning
// (the intrinsics then describe the images as they are), and a lens whose
// projection moves outwards from its axis.
// @throws std::invalid_argument naming the first unusable field, prefixed by `what`.
void check_camera_info(const messages::CameraInfo& info, std::string_view what = "camera");
// Checks the calibration as above, and that the pose's rotation is a rotation
// (orthonormal with determinant 1, to 1e-6) and its translation finite.
// @throws std::invalid_argument naming the first unusable field, prefixed by `what`.
void validate(const Camera& camera, std::string_view what = "camera");
// Checks every camera, and that there are 1..kMaxCameras of them.
// @throws std::invalid_argument naming the first unusable field.
void validate(const CameraRig& rig);
// Checks a pose as validate(const Camera&) does.
// @throws std::invalid_argument naming the problem, prefixed by `what`.
void validate(const RigidTransform& camera_from_rig, std::string_view what = "camera_from_rig");

// "camera <index>", and its frame id in parentheses when it has one: how
// errors and logs name a camera.
[[nodiscard]] std::string camera_label(const Camera& camera, std::size_t index);

// The coefficient `index` of `camera`'s lens model, 0 when the model has fewer.
[[nodiscard]] double coefficient(const messages::CameraInfo& camera, std::size_t index) noexcept;

// A point in normalized image coordinates (x = X/Z, y = Y/Z), or in pixels.
struct Point2 {
  double x{};
  double y{};
};

// Where the lens moves the normalized point `ideal`: its distorted normalized
// coordinates. Pixel coordinates are then (fx * x + cx, fy * y + cy). For
// equidistant, `ideal` is the tangent-plane point of a ray in front of the
// camera.
[[nodiscard]] Point2 distort(const messages::CameraInfo& camera, Point2 ideal) noexcept;

// The pixel a point in the camera's frame (metres, or a direction) lands on,
// before checking that it is inside the image; nullopt when the lens cannot
// see it: behind a pinhole camera, beyond a fisheye's field of view, or where
// the distortion polynomial folds back (see max_valid_angle()).
[[nodiscard]] std::optional<Point2> project_to_pixel(const messages::CameraInfo& camera,
                                                     const std::array<double, 3>& point) noexcept;
// The ray of a pixel, a unit vector in the camera's frame: the inverse of
// project_to_pixel(), solved by Newton's method. nullopt when no ray in the
// lens' valid range lands there.
[[nodiscard]] std::optional<std::array<double, 3>> pixel_ray(const messages::CameraInfo& camera,
                                                             Point2 pixel) noexcept;

// The largest angle from the optical axis, in radians, up to which the lens'
// projection keeps moving outwards: beyond it the distortion polynomial folds
// back and two rays would land on one pixel. Below pi/2 for the pinhole
// models, up to pi for equidistant. The stitcher ignores rays beyond it.
[[nodiscard]] double max_valid_angle(const messages::CameraInfo& camera) noexcept;

}  // namespace holoscan::stitching
