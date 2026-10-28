// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A synthetic scene with known geometry, for tests and the demo: a plane of
// Gaussian dots over a faint checkerboard, seen through distorted cameras, and
// the checks that measure each dot where an undistorted or rectified image
// shows it. No SDK, no CUDA.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <holoscan/rectify/camera_model.hpp>
#include <holoscan/rectify/rectification.hpp>

namespace holoscan::rectify {

// The plane z = depth in the left camera's frame, with a dot every `spacing`
// metres; each dot is a Gaussian of `sigma` metres. Frame f shifts the dots.
struct DotPlane {
  double depth{2.0};
  double spacing{0.2};
  double sigma{0.008};
};

// A distorted stereo pair looking at a DotPlane: each camera with its own
// intrinsics and distortion, the right one about 12 cm to the right and
// rotated by a degree or two, so rectification has real work to do.
struct SyntheticStereo {
  StereoCalibration calibration;
  DotPlane plane;
};

// A pair of width x height cameras (at least 160 x 120) with `model`
// distortion: kBrownConrady (barrel, with tangential terms),
// kRationalPolynomial or kKannalaBrandt4 (a wide fisheye).
// @throws std::invalid_argument for a smaller size or kNone.
[[nodiscard]] SyntheticStereo synthetic_stereo(
    int width, int height, DistortionModel model = DistortionModel::kBrownConrady);

// One camera's view of the plane, rendered exactly: every pixel is undistorted
// and its ray met with the plane, so an undistorted or rectified image shows
// each dot as a Gaussian centered where the dot projects.
class SyntheticView {
 public:
  // `camera_from_plane`: from the plane's (left camera's) frame to this camera's.
  SyntheticView(CameraModel camera, const RigidTransform& camera_from_plane, DotPlane plane);
  // Frame `frame`: MONO8, packed rows.
  [[nodiscard]] std::vector<std::uint8_t> render(std::uint32_t frame) const;
  [[nodiscard]] const CameraModel& camera() const noexcept { return camera_; }

 private:
  CameraModel camera_;
  DotPlane plane_;
  // Where each pixel's ray meets the plane (x, y), NaN where it does not.
  std::vector<float> plane_xy_;
};

// The left and right views of `scene`.
[[nodiscard]] SyntheticView left_view(const SyntheticStereo& scene);
[[nodiscard]] SyntheticView right_view(const SyntheticStereo& scene);

// The centers of frame `frame`'s dots in the plane's frame, within
// `half_extent` metres of the optical axis.
[[nodiscard]] std::vector<std::array<double, 3>> dot_centers(const DotPlane& plane,
                                                             std::uint32_t frame,
                                                             double half_extent);

// The intensity-weighted centroid of the dot near `predicted` in a MONO8
// image, from the pixels within 4 * sigma_px that rise above the checkerboard;
// nullopt when that window leaves `region` or holds no dot.
[[nodiscard]] std::optional<Point2> measure_dot(const std::uint8_t* image, std::size_t pitch,
                                                const PixelRect& region, Point2 predicted,
                                                double sigma_px);

// How well images show the dots where the geometry says they are.
struct AlignmentReport {
  int dots{};                   // Dots measured.
  int missing{};                // Dots predicted in the region but not found.
  double max_row_error{};       // Stereo: largest |y_left - y_right|, pixels.
  double rms_row_error{};       // Stereo: its root mean square.
  double max_position_error{};  // Largest distance from the predicted position, pixels.
  double rms_position_error{};
};

// Measures frame `frame`'s dots in a rectified pair (MONO8, `pitch` bytes per
// row) inside both valid rectangles: the rows they lie on in each image, and
// their distance from where `rectification` projects them. Dots that a raw
// image shows narrower than 1.2 pixels (a fisheye's rim) or cut by its border
// are left out: they measure the rendering, not the remap.
[[nodiscard]] AlignmentReport check_rectified(const SyntheticStereo& scene,
                                              const StereoRectification& rectification,
                                              const std::uint8_t* left, const std::uint8_t* right,
                                              std::size_t pitch, std::uint32_t frame);

// Measures frame `frame`'s dots in an undistorted image of the camera at the
// plane's frame: their distance from where `undistortion.output` projects them.
[[nodiscard]] AlignmentReport check_undistorted(const DotPlane& plane,
                                                const Undistortion& undistortion,
                                                const std::uint8_t* image, std::size_t pitch,
                                                std::uint32_t frame);

}  // namespace holoscan::rectify
