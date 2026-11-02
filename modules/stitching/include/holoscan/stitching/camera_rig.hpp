// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A camera rig: several calibrated cameras, each with its image size, pinhole
// intrinsics, lens distortion and pose in the rig's frame. Read from a rig
// file (calibration.json style), or from a recording's calibration.json.
// No SDK, no CUDA.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace holoscan::stitching {

// How a lens bends the pinhole image. With normalized coordinates x = X/Z,
// y = Y/Z of a point in the camera frame:
//   kNone                 no distortion, no coefficients
//   kBrownConrady         k1, k2, p1, p2, k3: OpenCV's and ROS's "plumb_bob"
//   kRationalPolynomial   k1, k2, p1, p2, k3, k4, k5, k6: OpenCV's rational model
//   kKannalaBrandt4       k1, k2, k3, k4: equidistant fisheye, OpenCV's fisheye
//                         model and ROS's "equidistant"; sees beyond 90 degrees
enum class DistortionModel : std::uint8_t {
  kNone,
  kBrownConrady,
  kRationalPolynomial,
  kKannalaBrandt4,
};

// "none", "brown_conrady", "rational_polynomial" or "kannala_brandt4".
[[nodiscard]] std::string_view distortion_model_name(DistortionModel model) noexcept;
// One of those names, or ROS's "plumb_bob", "equidistant" and "fisheye";
// nullopt for anything else.
[[nodiscard]] std::optional<DistortionModel> parse_distortion_model(std::string_view name) noexcept;
// The number of coefficients `model` takes.
[[nodiscard]] std::size_t distortion_coefficient_count(DistortionModel model) noexcept;

// A rigid transform from frame A to frame B: p_B = rotation * p_A + translation.
// Row-major rotation; translation in metres.
struct RigidTransform {
  std::array<double, 9> rotation{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  std::array<double, 3> translation{};
};

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

// One camera of the rig: image size, pinhole intrinsics in pixels with pixel
// centres at integer coordinates (OpenCV, ROS CameraInfo), distortion, and
// where it sits on the rig. No skew.
struct Camera {
  std::string name;
  int width{};
  int height{};
  double fx{};
  double fy{};
  double cx{};
  double cy{};
  DistortionModel distortion_model{DistortionModel::kNone};
  // distortion_coefficient_count(distortion_model) values. Empty means all zero.
  std::vector<double> distortion;
  RigidTransform camera_from_rig;
};

// Up to kMaxCameras cameras, in the order of the stitcher's inputs.
struct CameraRig {
  std::vector<Camera> cameras;
};

inline constexpr int kMaxImageSide = 16384;
inline constexpr std::size_t kMaxCameras = 8;

// Checks a camera: size 1..kMaxImageSide, finite positive fx and fy, finite cx
// and cy, the model's coefficient count (or none), finite coefficients, and a
// rotation that is a rotation (orthonormal with determinant 1, to 1e-6) and a
// finite translation.
// @throws std::invalid_argument naming the first unusable field, prefixed by `what`.
void validate(const Camera& camera, std::string_view what = "camera");
// Checks every camera, and that there are 1..kMaxCameras of them with
// distinct, non-empty names.
// @throws std::invalid_argument naming the first unusable field.
void validate(const CameraRig& rig);

// The coefficient `index` of `camera`, 0 when it has none.
[[nodiscard]] double coefficient(const Camera& camera, std::size_t index) noexcept;

// A point in normalized image coordinates (x = X/Z, y = Y/Z), or in pixels.
struct Point2 {
  double x{};
  double y{};
};

// Where the lens moves the normalized point `ideal`: its distorted normalized
// coordinates. Pixel coordinates are then (fx * x + cx, fy * y + cy). For
// kKannalaBrandt4, `ideal` is the tangent-plane point of a ray in front of the
// camera.
[[nodiscard]] Point2 distort(const Camera& camera, Point2 ideal) noexcept;

// The pixel a point in the camera's frame (metres, or a direction) lands on,
// before checking that it is inside the image; nullopt when the lens cannot
// see it: behind a pinhole camera, beyond a fisheye's field of view, or where
// the distortion polynomial folds back (see max_valid_angle()).
[[nodiscard]] std::optional<Point2> project_to_pixel(const Camera& camera,
                                                     const std::array<double, 3>& point) noexcept;
// The ray of a pixel, a unit vector in the camera's frame: the inverse of
// project_to_pixel(), solved by Newton's method. nullopt when no ray in the
// lens' valid range lands there.
[[nodiscard]] std::optional<std::array<double, 3>> pixel_ray(const Camera& camera,
                                                             Point2 pixel) noexcept;

// The largest angle from the optical axis, in radians, up to which the lens'
// projection keeps moving outwards: beyond it the distortion polynomial folds
// back and two rays would land on one pixel. Below pi/2 for the pinhole
// models, up to pi for kKannalaBrandt4. The stitcher ignores rays beyond it.
[[nodiscard]] double max_valid_angle(const Camera& camera) noexcept;

// --- Files ---------------------------------------------------------------

// Parses a rig file:
//
//   {
//     "version": 1,
//     "cameras": [
//       {
//         "name": "front", "width": 1280, "height": 720,
//         "fx": 330.0, "fy": 330.0, "cx": 639.5, "cy": 359.5,
//         "distortion": {"model": "kannala_brandt4", "coefficients": [0.0, 0.0, 0.0, 0.0]},
//         "camera_from_rig": {"rotation": [9 numbers, row-major], "translation": [3 numbers]}
//       }
//     ]
//   }
//
// Each camera takes `camera_from_rig` or its inverse `rig_from_camera`. The
// camera objects, distortion models and transforms are those of a recording's
// calibration.json, which this also reads: its `left`, `right` and, when
// present, `rgb` cameras, in the left camera's frame (`right_from_left` and
// `rgb.from_left` give the others' poses). Other keys are ignored. The result
// is validated.
// @throws std::invalid_argument naming the problem.
[[nodiscard]] CameraRig parse_rig(std::string_view json);
// parse_rig() of a file.
// @throws std::invalid_argument naming the file and the problem.
[[nodiscard]] CameraRig read_rig(const std::filesystem::path& path);
// A rig file parse_rig() reads back to the same rig.
[[nodiscard]] std::string to_json(const CameraRig& rig);

}  // namespace holoscan::stitching
