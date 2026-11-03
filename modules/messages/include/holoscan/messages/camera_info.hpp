// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// CameraInfo and StereoCameraInfo as plain C++ values: what the messages of
// the same names carry, usable and checkable without the Holoscan SDK.
// conversion.hpp turns them into the messages and back.
//
// Conventions, the same as the messages':
// - Pixel centers are at integer coordinates (as in ROS and OpenCV).
// - Matrices are row-major std::array<double, 9>.
// - A camera's frame has x right, y down and z forward; distances are metres.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace holoscan::messages {

using Matrix3 = std::array<double, 9>;
using Vector3 = std::array<double, 3>;

inline constexpr Matrix3 kIdentity3{1, 0, 0, 0, 1, 0, 0, 0, 1};

// The longest frame id a message carries, in bytes: the messages have a fixed
// largest size.
inline constexpr std::size_t kMaxFrameIdBytes = 256;

// The SDK's Header: when the data is for and the frame it is in.
struct Header {
  std::string frame_id;
  // In the producer's clock domain; 0 when not known.
  std::int64_t capture_timestamp_ns{};
  // The device's own sample counter, or 0.
  std::uint64_t device_sequence{};

  friend bool operator==(const Header&, const Header&) = default;
};

// What has been done to the images a CameraInfo describes.
enum class CameraState : std::uint8_t {
  // As the sensor saw them: some distortion coefficient is not zero.
  kRaw = 0,
  // No distortion. A rectification may still rotate them into a stereo
  // pair's rectified frame.
  kUndistorted = 1,
  // Undistorted and in the rectified frame of a stereo pair.
  kRectified = 2,
};

// The lens models, with ROS's names and coefficient orders.
enum class DistortionModel : std::uint8_t {
  kNone = 0,                // No coefficients.
  kPlumbBob = 1,            // "plumb_bob": k1 k2 p1 p2 k3.
  kRationalPolynomial = 2,  // "rational_polynomial": k1 k2 p1 p2 k3 k4 k5 k6.
  kEquidistant = 3,         // "equidistant": k1 k2 k3 k4 (OpenCV's fisheye).
};

// Full-resolution sensor pixels: columns x .. x + width - 1, rows
// y .. y + height - 1.
struct RegionOfInterest {
  std::uint32_t x{};
  std::uint32_t y{};
  std::uint32_t width{};
  std::uint32_t height{};

  friend bool operator==(const RegionOfInterest&, const RegionOfInterest&) = default;
};

// One camera's calibration for the images it goes with.
struct CameraInfo {
  Header header;
  // Pixels, both > 0.
  std::uint32_t width{};
  std::uint32_t height{};
  CameraState state{CameraState::kRaw};
  // Pixels. A point (x, y, z) projects to u = fx * x / z + skew * y / z + cx,
  // v = fy * y / z + cy, before distortion.
  double fx{};
  double fy{};
  double cx{};
  double cy{};
  double skew{};
  DistortionModel distortion_model{DistortionModel::kNone};
  // Exactly coefficient_count(distortion_model) values.
  std::vector<double> coefficients;
  // Rotation from this camera's frame to its stereo pair's rectified frame;
  // empty when there is none to apply (always for kRectified).
  std::optional<Matrix3> rectification;
  // Empty: the images hold the whole sensor.
  std::optional<RegionOfInterest> roi;
  // Sensor pixels per image pixel, at least 1.
  std::uint32_t binning_x{1};
  std::uint32_t binning_y{1};

  friend bool operator==(const CameraInfo&, const CameraInfo&) = default;
};

// p_right = rotation * p_left + translation, translation in metres.
struct RigidTransform {
  Matrix3 rotation{kIdentity3};
  Vector3 translation{};

  friend bool operator==(const RigidTransform&, const RigidTransform&) = default;
};

// A stereo pair's calibration, both cameras and their extrinsics together.
struct StereoCameraInfo {
  // frame_id: the left camera's frame.
  Header header;
  CameraInfo left;
  CameraInfo right;
  RigidTransform right_from_left;
  // Set when both cameras are kRectified, else 0: the right camera's distance
  // along +x of the rectified left frame in metres (> 0), and
  // cx_right - cx_left in pixels.
  double baseline{};
  double doffs{};

  friend bool operator==(const StereoCameraInfo&, const StereoCameraInfo&) = default;
};

// 0, 5, 8 or 4.
[[nodiscard]] std::size_t coefficient_count(DistortionModel model) noexcept;
// "none", "plumb_bob", "rational_polynomial" or "equidistant".
[[nodiscard]] std::string_view distortion_model_name(DistortionModel model) noexcept;
// ROS's names, and "none" or "" for kNone.
[[nodiscard]] std::optional<DistortionModel> parse_distortion_model(std::string_view name) noexcept;
// "raw", "undistorted" or "rectified".
[[nodiscard]] std::string_view camera_state_name(CameraState state) noexcept;

// Checks every field, and throws std::invalid_argument naming the first one
// that is wrong ("camera_info.fx must be finite and > 0"); `name` begins the
// field's name.
// A CameraInfo is valid when: the size is > 0; fx, fy > 0 and cx, cy, skew are
// finite; the coefficients match the model's count and are finite; kRaw has a
// coefficient that is not zero and the other states have none; a
// rectification is a rotation (orthonormal, determinant +1) and kRectified
// has none; the ROI lies in the sensor and is not empty; binning is >= 1; and
// the frame id has at most kMaxFrameIdBytes bytes.
void validate(const CameraInfo& info, std::string_view name = "camera_info");

// Checks both cameras and the pair, as above. Further, the rotation is a
// rotation and the translation finite and not zero; both cameras or neither
// are kRectified; a rectified pair has equal fx, fy and cy, no skew, the
// rotation the identity, the translation (-baseline, 0, 0) with baseline > 0
// and doffs = cx_right - cx_left; any other pair has baseline = doffs = 0;
// and when a camera has a rectification (an absent one is the identity),
// they rotate the pair into one frame with the right camera on its -x axis.
void validate(const StereoCameraInfo& info, std::string_view name = "stereo_camera_info");

// fx * baseline, in pixel metres: depth = fx_baseline / (disparity + doffs).
// 0 unless both cameras are rectified.
[[nodiscard]] double fx_baseline(const StereoCameraInfo& info) noexcept;
// The depth in metres of a disparity d = x_left - x_right in pixels of a
// rectified pair: fx * baseline / (d + doffs). NaN when d + doffs <= 0 or
// the pair is not rectified.
[[nodiscard]] double depth_from_disparity(const StereoCameraInfo& info, double disparity) noexcept;

// Whether two calibrations are the same apart from their headers: what a
// consumer compares to know whether to recompute what it derives from them.
[[nodiscard]] bool same_calibration(const CameraInfo& a, const CameraInfo& b) noexcept;
[[nodiscard]] bool same_calibration(const StereoCameraInfo& a, const StereoCameraInfo& b) noexcept;

}  // namespace holoscan::messages
