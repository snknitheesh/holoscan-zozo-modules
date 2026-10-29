// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The geometry of undistortion and stereo rectification, computed once per
// calibration on the CPU: the output cameras (OpenCV's
// getOptimalNewCameraMatrix and Bouguet's stereoRectify), and the maps that
// send every output pixel to the input pixel it samples. No SDK, no CUDA.

#pragma once

#include <array>
#include <optional>
#include <vector>

#include <holoscan/rectify/camera_model.hpp>

namespace holoscan::rectify {

using Matrix3 = std::array<double, 9>;  // Row-major.

// Pinhole intrinsics in pixels, pixel centers at integer coordinates.
struct Intrinsics {
  double fx{};
  double fy{};
  double cx{};
  double cy{};
};

// Whole pixels: columns x .. x + width - 1, rows y .. y + height - 1.
struct PixelRect {
  int x{};
  int y{};
  int width{};
  int height{};
};

// How UndistortOp chooses its output camera.
struct UndistortOptions {
  // 0 keeps only valid pixels: every output pixel sees the input, and some of
  // the input is cut off. 1 keeps every input pixel, with black around them.
  // Values between blend the two. Ignored when `intrinsics` is given.
  double alpha{0.0};
  // The output size; 0 for the input's.
  int width{0};
  int height{0};
  // The output camera, explicitly.
  std::optional<Intrinsics> intrinsics;
};

// One camera's undistortion.
struct Undistortion {
  CameraModel input;
  // The undistorted camera: the output size and intrinsics, no distortion.
  CameraModel output;
  // The output pixels that see the input, as a rectangle (approximately: the
  // largest one inside the input's outline, from its border sampled densely).
  PixelRect valid;
};

// @throws std::invalid_argument for an unusable camera or options.
[[nodiscard]] Undistortion compute_undistortion(const CameraModel& camera,
                                                const UndistortOptions& options = {});

// How StereoRectifyOp chooses its rectified cameras.
struct StereoRectifyOptions {
  // As UndistortOptions::alpha, for both cameras together (OpenCV's alpha):
  // 0 crops to pixels both cameras see, 1 keeps every pixel of both.
  double alpha{0.0};
  // The output size; 0 for the input's.
  int width{0};
  int height{0};
  // Give both rectified cameras the same principal point, so that a point at
  // infinity has zero disparity (OpenCV's CALIB_ZERO_DISPARITY). When false,
  // each keeps its own cx, which crops less for close scenes; doffs then says
  // how they differ.
  bool zero_disparity{true};
};

// A rectified stereo pair: both cameras share fx = fy, cy and the image size,
// and the right one sits `baseline` metres along +x of the left one, so a
// scene point lies on the same row in both images.
//
// A disparity d = x_left - x_right (pixels) is at depth
//   Z = fx * baseline / (d + doffs)  metres,
// and at X = (x_left - cx_left) * Z / fx, Y = (y - cy) * Z / fx in the
// rectified left camera's frame. With zero_disparity, doffs is 0.
struct StereoRectification {
  int width{};
  int height{};
  double fx{};  // = fy
  double cx_left{};
  double cx_right{};
  double cy{};
  double baseline{};  // Metres, positive.
  double doffs{};     // cx_right - cx_left, pixels.

  // From each input camera's frame to its rectified frame (OpenCV's R1, R2).
  Matrix3 rect_from_left{};
  Matrix3 rect_from_right{};
  // OpenCV's P1, P2 (3x4) and Q (4x4), row-major. Q maps (x, y, d, 1) to a
  // homogeneous point in the rectified left frame.
  std::array<double, 12> projection_left{};
  std::array<double, 12> projection_right{};
  std::array<double, 16> disparity_to_depth{};

  // The output pixels that see the input, per camera (approximately).
  PixelRect valid_left;
  PixelRect valid_right;

  // fx * baseline: depth Z = this / (d + doffs).
  [[nodiscard]] double depth_from_disparity() const noexcept { return fx * baseline; }
  // Each rectified camera, with no distortion.
  [[nodiscard]] CameraModel left_camera() const;
  [[nodiscard]] CameraModel right_camera() const;
};

// Bouguet's method, as OpenCV's stereoRectify: rotates each camera half of the
// way to the other's orientation, then both so that the baseline lies along x;
// picks the smaller vertical focal length (adjusted for barrel distortion) and
// principal points that center the images, then scales by `alpha`.
// Horizontal pairs only, with the right camera to the right of the left one.
// @throws std::invalid_argument for an unusable calibration or options, a
//         vertical pair or swapped cameras.
[[nodiscard]] StereoRectification compute_stereo_rectification(
    const StereoCalibration& calibration, const StereoRectifyOptions& options = {});

// The mapping of one output image into its input: for an output pixel (u, v),
// the input pixel it samples. Pixel centers are at integer coordinates.
class RemapModel {
 public:
  // `rect_from_camera` rotates the input camera's frame into the output's
  // (identity for undistortion).
  RemapModel(CameraModel input, const Matrix3& rect_from_camera, const Intrinsics& output);
  // The input pixel output pixel (u, v) samples; nullopt where it sees nothing
  // (behind the camera).
  [[nodiscard]] std::optional<Point2> source(double u, double v) const noexcept;
  // source() as the backends take it: within half a pixel outside the input,
  // moved onto its edge (still the edge pixel's area); further out, clamped to
  // [-2, input size + 1], a small number that samples nothing; and (-2, -2)
  // where source() has none.
  [[nodiscard]] Point2 clamped_source(double u, double v) const noexcept;

  [[nodiscard]] const CameraModel& input() const noexcept { return input_; }

 private:
  CameraModel input_;
  Matrix3 camera_from_rect_{};
  Intrinsics output_;
};

// The model for an undistortion, or for one camera of a rectification.
[[nodiscard]] RemapModel undistortion_remap(const Undistortion& undistortion);
[[nodiscard]] RemapModel left_remap(const StereoCalibration& calibration,
                                    const StereoRectification& rectification);
[[nodiscard]] RemapModel right_remap(const StereoCalibration& calibration,
                                     const StereoRectification& rectification);

// A dense map: clamped_source() of every output pixel, (x, y) interleaved, row
// by row.
struct RemapTable {
  int width{};
  int height{};
  std::vector<float> xy;
};
[[nodiscard]] RemapTable build_remap_table(const RemapModel& model, int width, int height);

// Rotation helpers.
[[nodiscard]] Matrix3 multiply(const Matrix3& a, const Matrix3& b) noexcept;
[[nodiscard]] Matrix3 transpose(const Matrix3& m) noexcept;
[[nodiscard]] std::array<double, 3> rotate(const Matrix3& m,
                                           const std::array<double, 3>& v) noexcept;
// The rotation by `angle` radians about `axis` (Rodrigues); identity for a zero axis.
[[nodiscard]] Matrix3 axis_angle(const std::array<double, 3>& axis, double angle) noexcept;

}  // namespace holoscan::rectify
