// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What a stitched image shows: a panorama (cylindrical or spherical) seen from
// a point of the rig, or a top-down view of the ground. Each output pixel is a
// 3D point in the rig's frame (x forward, y left, z up), which every camera
// that sees it contributes to. No SDK, no CUDA.

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

#include <holoscan/stitching/camera_rig.hpp>

namespace holoscan::stitching {

enum class Projection : std::uint8_t {
  // Columns are azimuth, rows the height on a vertical cylinder around
  // `PanoramaParams::center` (tan of the elevation): vertical lines stay
  // straight.
  kCylindrical,
  // Columns are azimuth, rows elevation (equirectangular).
  kSpherical,
  // The ground plane z = 0 of the rig frame, seen from above: up is +x
  // (forward), right is -y (the rig's right).
  kBirdsEye,
};

// "cylindrical", "spherical" or "birds_eye".
[[nodiscard]] std::string_view projection_name(Projection projection) noexcept;
[[nodiscard]] std::optional<Projection> parse_projection(std::string_view name) noexcept;

struct PanoramaParams {
  // The azimuth of the middle column, in degrees: 0 looks along +x, 90 along
  // +y (left). Columns run left to right, so azimuth falls across the image.
  double azimuth_center_deg{0.0};
  // The azimuth the width spans, 1..360 degrees. At 360 the image wraps around.
  double horizontal_fov_deg{360.0};
  // The elevations of the bottom and top edges, in degrees, within (-90, 90).
  double elevation_min_deg{-30.0};
  double elevation_max_deg{30.0};
  // The point the panorama is seen from, in the rig frame, in metres.
  std::array<double, 3> center{0.0, 0.0, 0.0};
  // How far the scene is assumed to be. 0: at infinity, so each camera's
  // offset from `center` is ignored and only its rotation counts; > 0: on the
  // cylinder (kCylindrical) or sphere (kSpherical) of this radius in metres
  // around `center`, so cameras away from it see that surface without parallax.
  double radius{0.0};
};

struct BirdsEyeParams {
  // The ground area the image covers, in the rig frame, in metres: x from
  // x_min (bottom edge) to x_max (top edge), y from y_max (left edge) to y_min
  // (right edge).
  double x_min{-10.0};
  double x_max{10.0};
  double y_min{-10.0};
  double y_max{10.0};
};

struct OutputGeometry {
  Projection projection{Projection::kBirdsEye};
  int width{1024};
  int height{1024};
  PanoramaParams panorama;   // For kCylindrical and kSpherical.
  BirdsEyeParams birds_eye;  // For kBirdsEye.
};

// Checks the size (1..kMaxImageSide) and the parameters of the projection.
// @throws std::invalid_argument naming the first unusable field.
void validate(const OutputGeometry& geometry);

// A bird's-eye view of `area` at `metres_per_pixel`: the size is the area
// over the resolution, rounded up, and the area grows to match it.
// @throws std::invalid_argument if the area or resolution is unusable.
[[nodiscard]] OutputGeometry birds_eye_view(const BirdsEyeParams& area, double metres_per_pixel);

// Whether the image wraps around: a 360-degree panorama.
[[nodiscard]] bool wraps(const OutputGeometry& geometry) noexcept;

// The 3D point an output pixel shows, at pixel coordinates (x, y) with pixel
// centres at integers. `at_infinity` marks a direction (a panorama with
// radius 0) rather than a point.
struct OutputPoint {
  std::array<double, 3> point{};
  bool at_infinity{false};
};
[[nodiscard]] OutputPoint output_point(const OutputGeometry& geometry, double x, double y) noexcept;

// The inverse: where a point of the rig frame appears in the output, in pixel
// coordinates; for a panorama, along its direction from the centre. nullopt
// when it is not in the projection's domain (a panorama point straight above
// or below the centre, or at the centre).
[[nodiscard]] std::optional<Point2> output_pixel(const OutputGeometry& geometry,
                                                 const std::array<double, 3>& point) noexcept;

}  // namespace holoscan::stitching
