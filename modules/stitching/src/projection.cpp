// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/stitching/projection.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

#include "stitch_math.hpp"

namespace holoscan::stitching {
namespace {

constexpr double kDegree = std::numbers::pi / 180.0;

bool finite(const std::array<double, 3>& v) {
  return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

}  // namespace

std::string_view projection_name(Projection projection) noexcept {
  switch (projection) {
    case Projection::kCylindrical:
      return "cylindrical";
    case Projection::kSpherical:
      return "spherical";
    case Projection::kBirdsEye:
      return "birds_eye";
  }
  return "unknown";
}

std::optional<Projection> parse_projection(std::string_view name) noexcept {
  if (name == "cylindrical") return Projection::kCylindrical;
  if (name == "spherical") return Projection::kSpherical;
  if (name == "birds_eye") return Projection::kBirdsEye;
  return std::nullopt;
}

void validate(const OutputGeometry& geometry) {
  if (geometry.width < 1 || geometry.height < 1 || geometry.width > kMaxImageSide ||
      geometry.height > kMaxImageSide) {
    throw std::invalid_argument("output width and height must be 1.." +
                                std::to_string(kMaxImageSide));
  }
  if (geometry.projection == Projection::kBirdsEye) {
    const BirdsEyeParams& b = geometry.birds_eye;
    if (!(std::isfinite(b.x_min) && std::isfinite(b.x_max) && b.x_min < b.x_max &&
          std::isfinite(b.y_min) && std::isfinite(b.y_max) && b.y_min < b.y_max)) {
      throw std::invalid_argument("birds_eye: need finite x_min < x_max and y_min < y_max");
    }
    return;
  }
  if (geometry.projection != Projection::kCylindrical &&
      geometry.projection != Projection::kSpherical) {
    throw std::invalid_argument("unknown projection");
  }
  const PanoramaParams& p = geometry.panorama;
  if (!(std::isfinite(p.azimuth_center_deg))) {
    throw std::invalid_argument("panorama: azimuth_center_deg must be finite");
  }
  if (!(p.horizontal_fov_deg >= 1.0 && p.horizontal_fov_deg <= 360.0)) {
    throw std::invalid_argument("panorama: horizontal_fov_deg must be 1..360");
  }
  if (!(p.elevation_min_deg > -90.0 && p.elevation_min_deg < p.elevation_max_deg &&
        p.elevation_max_deg < 90.0)) {
    throw std::invalid_argument("panorama: need -90 < elevation_min_deg < elevation_max_deg < 90");
  }
  if (!finite(p.center)) {
    throw std::invalid_argument("panorama: center must be finite");
  }
  if (!(std::isfinite(p.radius) && p.radius >= 0.0)) {
    throw std::invalid_argument("panorama: radius must be 0 (infinity) or positive");
  }
}

OutputGeometry birds_eye_view(const BirdsEyeParams& area, double metres_per_pixel) {
  if (!(std::isfinite(metres_per_pixel) && metres_per_pixel > 0.0)) {
    throw std::invalid_argument("birds_eye: metres_per_pixel must be finite and positive");
  }
  OutputGeometry geometry;
  geometry.projection = Projection::kBirdsEye;
  geometry.birds_eye = area;
  validate(geometry);
  const double columns = std::ceil((area.y_max - area.y_min) / metres_per_pixel - 1e-9);
  const double rows = std::ceil((area.x_max - area.x_min) / metres_per_pixel - 1e-9);
  if (!(columns >= 1 && rows >= 1 && columns <= kMaxImageSide && rows <= kMaxImageSide)) {
    throw std::invalid_argument("birds_eye: the area at this resolution is larger than " +
                                std::to_string(kMaxImageSide) + " pixels");
  }
  geometry.width = static_cast<int>(columns);
  geometry.height = static_cast<int>(rows);
  const double cx = 0.5 * (area.x_min + area.x_max);
  const double cy = 0.5 * (area.y_min + area.y_max);
  geometry.birds_eye.x_min = cx - 0.5 * rows * metres_per_pixel;
  geometry.birds_eye.x_max = cx + 0.5 * rows * metres_per_pixel;
  geometry.birds_eye.y_min = cy - 0.5 * columns * metres_per_pixel;
  geometry.birds_eye.y_max = cy + 0.5 * columns * metres_per_pixel;
  return geometry;
}

bool wraps(const OutputGeometry& geometry) noexcept {
  return geometry.projection != Projection::kBirdsEye &&
         geometry.panorama.horizontal_fov_deg == 360.0;
}

OutputPoint output_point(const OutputGeometry& geometry, double x, double y) noexcept {
  const detail::GeometryParams g = detail::geometry_params(geometry);
  OutputPoint result;
  double p[3];
  result.at_infinity = detail::output_point(g, x, y, p);
  result.point = {p[0], p[1], p[2]};
  return result;
}

std::optional<Point2> output_pixel(const OutputGeometry& geometry,
                                   const std::array<double, 3>& point) noexcept {
  const detail::GeometryParams g = detail::geometry_params(geometry);
  if (geometry.projection == Projection::kBirdsEye) {
    return Point2{(g.y_left - point[1]) / g.y_step - 0.5, (g.x_top - point[0]) / g.x_step - 0.5};
  }
  const double dx = point[0] - g.center[0];
  const double dy = point[1] - g.center[1];
  const double dz = point[2] - g.center[2];
  const double horizontal = std::hypot(dx, dy);
  if (!(horizontal > 1e-12)) {
    return std::nullopt;
  }
  // The azimuth's offset from the left edge, in [0, 2 pi).
  double delta = std::fmod(g.azimuth_left - std::atan2(dy, dx), 2.0 * std::numbers::pi);
  if (delta < 0.0) delta += 2.0 * std::numbers::pi;
  const double row =
      geometry.projection == Projection::kSpherical ? std::atan2(dz, horizontal) : dz / horizontal;
  return Point2{delta / g.azimuth_step - 0.5, (g.row_top - row) / g.row_step - 0.5};
}

namespace detail {

GeometryParams geometry_params(const OutputGeometry& geometry) {
  GeometryParams g{};
  g.width = geometry.width;
  g.height = geometry.height;
  const PanoramaParams& p = geometry.panorama;
  const BirdsEyeParams& b = geometry.birds_eye;
  switch (geometry.projection) {
    case Projection::kCylindrical:
      g.projection = kCylindrical;
      g.row_top = std::tan(p.elevation_max_deg * kDegree);
      g.row_step = (g.row_top - std::tan(p.elevation_min_deg * kDegree)) / geometry.height;
      break;
    case Projection::kSpherical:
      g.projection = kSpherical;
      g.row_top = p.elevation_max_deg * kDegree;
      g.row_step = (p.elevation_max_deg - p.elevation_min_deg) * kDegree / geometry.height;
      break;
    case Projection::kBirdsEye:
      g.projection = kBirdsEye;
      break;
  }
  const double fov = p.horizontal_fov_deg * kDegree;
  g.azimuth_left = p.azimuth_center_deg * kDegree + 0.5 * fov;
  g.azimuth_step = fov / geometry.width;
  for (int i = 0; i < 3; ++i) g.center[i] = p.center[i];
  g.radius = p.radius;
  g.x_top = b.x_max;
  g.x_step = (b.x_max - b.x_min) / geometry.height;
  g.y_left = b.y_max;
  g.y_step = (b.y_max - b.y_min) / geometry.width;
  return g;
}

}  // namespace detail
}  // namespace holoscan::stitching
