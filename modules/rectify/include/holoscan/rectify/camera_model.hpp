// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A camera's calibration: image size, pinhole intrinsics and lens distortion,
// and a stereo pair's two cameras with the pose of the right one relative to
// the left one. Read from a recording's calibration.json or from ROS
// camera_info YAML files. No SDK, no CUDA.

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace holoscan::rectify {

// How a lens bends the pinhole image. With normalized coordinates x = X/Z,
// y = Y/Z of a point in the camera frame:
//   kNone                 no distortion, no coefficients
//   kBrownConrady         k1, k2, p1, p2, k3: OpenCV's and ROS's "plumb_bob"
//   kRationalPolynomial   k1, k2, p1, p2, k3, k4, k5, k6: OpenCV's rational model
//   kKannalaBrandt4       k1, k2, k3, k4: equidistant fisheye, OpenCV's fisheye
//                         model and ROS's "equidistant"
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

// One camera: image size, pinhole intrinsics in pixels with pixel centers at
// integer coordinates (OpenCV, ROS CameraInfo), and distortion. No skew.
struct CameraModel {
  int width{};
  int height{};
  double fx{};
  double fy{};
  double cx{};
  double cy{};
  DistortionModel distortion_model{DistortionModel::kNone};
  // distortion_coefficient_count(distortion_model) values. Empty means all zero.
  std::vector<double> distortion;
};

// A rigid transform from frame A to frame B: p_B = rotation * p_A + translation.
// Row-major rotation; translation in metres. Camera frames are optical: x right,
// y down, z forward.
struct RigidTransform {
  std::array<double, 9> rotation{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  std::array<double, 3> translation{};
};

// A stereo pair. The fields match holoscan-stereo-camera's StereoCalibration.
struct StereoCalibration {
  CameraModel left;
  CameraModel right;
  // From the left camera's frame to the right camera's: for a horizontal pair
  // with the right camera `b` metres to the right, the translation is (-b, 0, 0).
  RigidTransform right_from_left;
};

// Largest image side a camera may have.
inline constexpr int kMaxImageSide = 16384;

// Checks a camera: size 1..kMaxImageSide, finite positive fx and fy, finite cx
// and cy, the model's coefficient count (or none), finite coefficients.
// @throws std::invalid_argument naming the first unusable field, prefixed by `what`.
void validate(const CameraModel& camera, std::string_view what = "camera");
// Checks both cameras, that the rotation is a rotation (orthonormal with
// determinant 1, to 1e-6) and that the baseline is finite and nonzero.
// @throws std::invalid_argument naming the first unusable field.
void validate(const StereoCalibration& calibration);

// The coefficient `index` of `camera`, 0 when it has none.
[[nodiscard]] double coefficient(const CameraModel& camera, std::size_t index) noexcept;

// A point in normalized image coordinates (x = X/Z, y = Y/Z).
struct Point2 {
  double x{};
  double y{};
};

// Where the lens moves the normalized point `ideal`: its distorted normalized
// coordinates. Pixel coordinates are then (fx * x + cx, fy * y + cy).
[[nodiscard]] Point2 distort(const CameraModel& camera, Point2 ideal) noexcept;
// The inverse of distort(): the ideal normalized point the lens moved to
// `distorted`, solved by Newton's method to 1e-12. nullopt when there is none
// (a fisheye ray at or beyond 90 degrees from the axis, or no convergence).
[[nodiscard]] std::optional<Point2> undistort(const CameraModel& camera, Point2 distorted) noexcept;

// --- Files ---------------------------------------------------------------

// Parses the JSON of a recording's calibration.json: `left` and `right`, each
// with `width`, `height`, `fx`, `fy`, `cx`, `cy` and `distortion` (`model`,
// `coefficients`), and `right_from_left` (`rotation`, 9 numbers row-major, and
// `translation`, 3, in metres). Other keys are ignored. Models: "none",
// "brown_conrady" (4 or 5 coefficients), "rational_polynomial" (8) and
// "kannala_brandt4" (4); others, such as librealsense's
// "modified_brown_conrady", are refused by name. The result is validated.
// @throws std::invalid_argument naming the problem.
[[nodiscard]] StereoCalibration parse_stereo_calibration(std::string_view json);
// parse_stereo_calibration() of a file.
// @throws std::invalid_argument naming the file and the problem.
[[nodiscard]] StereoCalibration read_stereo_calibration(const std::filesystem::path& path);

// What a ROS camera_info YAML file holds: the camera, and the rectification
// rotation and projection matrix a ROS stereo calibration writes.
struct CameraInfo {
  std::string camera_name;
  CameraModel camera;
  // R: from the camera's frame to the rectified frame, row-major. Identity
  // when the file has none.
  std::array<double, 9> rectification{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  // P: the rectified camera's 3x4 projection, row-major; all zero when the
  // file has none. For a ROS stereo pair's right camera, P[3] = -fx' * baseline.
  std::array<double, 12> projection{};
};

// Parses a ROS camera_info YAML file (as camera_calibration and
// camera_info_manager write it): image_width, image_height, camera_name,
// camera_matrix, distortion_model ("plumb_bob", "rational_polynomial",
// "equidistant"), distortion_coefficients, rectification_matrix and
// projection_matrix. A nonzero skew is refused. The camera is validated.
// @throws std::invalid_argument naming the problem.
[[nodiscard]] CameraInfo parse_camera_info(std::string_view yaml);
// parse_camera_info() of a file.
// @throws std::invalid_argument naming the file and the problem.
[[nodiscard]] CameraInfo read_camera_info(const std::filesystem::path& path);

// The stereo calibration of a pair of ROS camera_info files written by a
// stereo calibration: right_from_left from their rectification rotations and
// the right camera's projection (R = R_right^T * R_left,
// t = R_right^T * (P_right[3] / P_right[0], P_right[7] / P_right[5], 0)).
// @throws std::invalid_argument if the right projection carries no baseline.
[[nodiscard]] StereoCalibration stereo_from_camera_info(const CameraInfo& left,
                                                        const CameraInfo& right);

}  // namespace holoscan::rectify
