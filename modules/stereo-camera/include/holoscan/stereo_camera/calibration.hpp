// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The calibration of a stereo camera: each eye's intrinsics and distortion, the
// right eye's pose relative to the left one, and optionally an RGB camera and a
// depth image, each with its own intrinsics and pose. Read from and written to
// the JSON file a recording keeps beside its frames. No SDK.

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace holoscan::stereo_camera {

// How an image's pixels are distorted from the pinhole model. The coefficients'
// meaning and count follow the model:
//   kNone                    none
//   kBrownConrady            k1, k2, p1, p2, k3 (OpenCV's "plumb bob")
//   kModifiedBrownConrady    k1, k2, p1, p2, k3, as librealsense applies them
//   kInverseBrownConrady     k1, k2, p1, p2, k3, from distorted to undistorted
//   kKannalaBrandt4          k1, k2, k3, k4 (fisheye)
//   kFTheta                  k1 (librealsense's F-theta fisheye)
//   kRationalPolynomial      k1, k2, p1, p2, k3, k4, k5, k6 (OpenCV)
enum class DistortionModel : std::uint8_t {
  kNone,
  kBrownConrady,
  kModifiedBrownConrady,
  kInverseBrownConrady,
  kKannalaBrandt4,
  kFTheta,
  kRationalPolynomial,
};

// "none", "brown_conrady", "modified_brown_conrady", "inverse_brown_conrady",
// "kannala_brandt4", "ftheta" or "rational_polynomial".
[[nodiscard]] std::string_view distortion_model_name(DistortionModel model) noexcept;
// The model of one of those names; nullopt for anything else.
[[nodiscard]] std::optional<DistortionModel> parse_distortion_model(std::string_view name) noexcept;
// The number of coefficients `model` takes.
[[nodiscard]] std::size_t distortion_coefficient_count(DistortionModel model) noexcept;

// One camera's image size, pinhole intrinsics in pixels and distortion. Pixel
// centers are at integer coordinates, as in OpenCV and ROS CameraInfo. The
// fields fx, fy, cx and cy are those of holoscan::point_cloud::CameraIntrinsics.
struct CameraModel {
  int width{};
  int height{};
  double fx{};
  double fy{};
  double cx{};
  double cy{};
  DistortionModel distortion_model{DistortionModel::kNone};
  // distortion_coefficient_count(distortion_model) values; all zero, or empty,
  // means no distortion.
  std::vector<double> distortion;

  // Whether every distortion coefficient is zero.
  [[nodiscard]] bool undistorted() const noexcept;
};

// A rigid transform from frame A to frame B: p_B = rotation * p_A + translation.
// Row-major rotation; translation in metres. Camera frames are optical: x right,
// y down, z forward.
struct RigidTransform {
  std::array<double, 9> rotation{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  std::array<double, 3> translation{};
};

// A camera beside the stereo pair, registered to the left eye.
struct AuxiliaryCamera {
  CameraModel model;
  // From the left eye's frame to this camera's frame.
  RigidTransform from_left;
};

// The depth image, when the camera computes one, and what its values mean.
struct DepthCamera {
  CameraModel model;
  // From the left eye's frame to the depth image's frame. A RealSense computes
  // depth in its left infrared camera's frame, so this is the identity there.
  RigidTransform from_left;
  // Metres per unit of the uint16 depth values; 0 is no depth.
  double scale{0.001};
};

// What the calibration belongs to. `settings` holds what the camera ran with,
// such as "emitter_enabled" or "fps", as text; nothing reads them back.
struct CameraInfo {
  std::string name;
  std::string serial;
  std::string firmware;
  std::map<std::string, std::string> settings;
};

struct StereoCalibration {
  CameraModel left;
  CameraModel right;
  // From the left eye's frame to the right eye's: for a horizontal pair with the
  // right eye `b` metres to the right, the translation is (-b, 0, 0).
  RigidTransform right_from_left;
  // The images are already rectified: undistorted, with rows aligned between the
  // eyes, so that a point's disparity is along its row.
  bool rectified{};
  std::optional<AuxiliaryCamera> rgb;
  std::optional<DepthCamera> depth;
  CameraInfo camera;

  // The distance between the eyes in metres: the length of the translation.
  [[nodiscard]] double baseline() const noexcept;
  // fx of the left eye times the baseline: depth in metres = this / disparity in
  // pixels, for a rectified pair.
  [[nodiscard]] double depth_from_disparity() const noexcept;
};

// Checks what every consumer relies on: positive image sizes; finite, positive
// focal lengths; finite principal points, coefficients and transforms; the
// coefficient count of each model; rotations that are rotations (orthonormal
// with determinant 1, to 1e-6); a nonzero baseline; equal eye sizes; for a
// rectified pair, no distortion; and a positive depth scale.
// @throws std::invalid_argument naming the first unusable field.
void validate(const StereoCalibration& calibration);

// The version of the calibration file format this library reads and writes.
inline constexpr int kCalibrationFormatVersion = 1;

// Parses the JSON text of a calibration file and validates it (see the README
// for its fields). Unknown keys are ignored, so that later versions can add some.
// @throws std::invalid_argument naming the first unusable field, or where the
//         JSON is malformed.
[[nodiscard]] StereoCalibration parse_calibration(std::string_view json);
// The JSON text of a calibration file for `calibration`, which parse_calibration()
// reads back exactly: numbers are written with every digit they need.
[[nodiscard]] std::string format_calibration(const StereoCalibration& calibration);

// parse_calibration() of a file's contents.
// @throws std::invalid_argument naming the file and the problem.
[[nodiscard]] StereoCalibration read_calibration(const std::filesystem::path& path);
// Writes format_calibration() to `path`, replacing it.
// @throws std::runtime_error if the file cannot be written.
void write_calibration(const std::filesystem::path& path, const StereoCalibration& calibration);

}  // namespace holoscan::stereo_camera
