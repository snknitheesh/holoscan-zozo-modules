// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The synthetic world's host half: rigs, poses, geometries, the expected
// checkerboard corners and the comparison with the truth. The renders are in
// synthetic_render.cu.

#include "holoscan/stitching/synthetic_world.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace holoscan::stitching {
namespace {

constexpr double kDegree = std::numbers::pi / 180.0;

// A point of the rig frame from one of the world frame.
std::array<double, 3> rig_from_world(const RigPose& pose, const std::array<double, 3>& p) {
  const double yaw = pose.yaw_deg * kDegree;
  const double dx = p[0] - pose.x;
  const double dy = p[1] - pose.y;
  return {std::cos(yaw) * dx + std::sin(yaw) * dy, -std::sin(yaw) * dx + std::cos(yaw) * dy, p[2]};
}

std::array<double, 3> world_from_rig(const RigPose& pose, const std::array<double, 3>& p) {
  const double yaw = pose.yaw_deg * kDegree;
  return {std::cos(yaw) * p[0] - std::sin(yaw) * p[1] + pose.x,
          std::sin(yaw) * p[0] + std::cos(yaw) * p[1] + pose.y, p[2]};
}

bool inside(const OutputGeometry& g, const Point2& p, double margin) {
  return p.x >= margin && p.y >= margin && p.x <= g.width - 1 - margin &&
         p.y <= g.height - 1 - margin;
}

}  // namespace

RigPose moving_pose(std::uint32_t frame, std::uint32_t frames_per_lap) {
  const double t =
      2.0 * std::numbers::pi * static_cast<double>(frame) / std::max(1U, frames_per_lap);
  constexpr double kRadius = 3.0;
  return RigPose{kRadius * std::sin(t), kRadius * (1.0 - std::cos(t)), t / kDegree};
}

CameraRig surround_rig(int width, int height) {
  CameraRig rig;
  const double f = 330.0 * width / 1280.0;
  const auto fisheye = [&](const char* name, std::array<double, 3> position, double yaw,
                           double pitch) {
    Camera c;
    c.name = name;
    c.width = width;
    c.height = height;
    c.fx = f;
    c.fy = f;
    c.cx = (width - 1) / 2.0;
    c.cy = (height - 1) / 2.0;
    c.distortion_model = DistortionModel::kKannalaBrandt4;
    c.distortion = {0.02, -0.005, 0.001, -0.0001};
    c.camera_from_rig = mount(position, yaw, pitch);
    return c;
  };
  rig.cameras.push_back(fisheye("front", {2.0, 0.0, 0.8}, 0.0, -30.0));
  rig.cameras.push_back(fisheye("left", {0.0, 1.0, 1.0}, 90.0, -40.0));
  rig.cameras.push_back(fisheye("rear", {-2.0, 0.0, 0.8}, 180.0, -30.0));
  rig.cameras.push_back(fisheye("right", {0.0, -1.0, 1.0}, -90.0, -40.0));
  return rig;
}

CameraRig ring_rig(int count, int width, int height, double horizontal_fov_deg, double radius) {
  if (count < 1 || count > static_cast<int>(kMaxCameras)) {
    throw std::invalid_argument("ring_rig: 1..8 cameras");
  }
  if (!(horizontal_fov_deg > 0.0 && horizontal_fov_deg < 170.0)) {
    throw std::invalid_argument("ring_rig: the field of view must be in (0, 170) degrees");
  }
  CameraRig rig;
  const double f = 0.5 * width / std::tan(0.5 * horizontal_fov_deg * kDegree);
  for (int i = 0; i < count; ++i) {
    const double yaw = 360.0 * i / count;
    Camera c;
    c.name = "camera_" + std::to_string(i);
    c.width = width;
    c.height = height;
    c.fx = f;
    c.fy = f;
    c.cx = (width - 1) / 2.0;
    c.cy = (height - 1) / 2.0;
    c.distortion_model = DistortionModel::kBrownConrady;
    c.distortion = {-0.08, 0.02, 0.0005, -0.0003, 0.0};
    c.camera_from_rig =
        mount({radius * std::cos(yaw * kDegree), radius * std::sin(yaw * kDegree), 1.0}, yaw, 0.0);
    rig.cameras.push_back(std::move(c));
  }
  return rig;
}

OutputGeometry wall_panorama(const SyntheticWorld& world, Projection projection, int width,
                             int height, double center_height) {
  if (projection == Projection::kBirdsEye) {
    throw std::invalid_argument("wall_panorama: a panorama is cylindrical or spherical");
  }
  OutputGeometry g;
  g.projection = projection;
  g.width = width;
  g.height = height;
  g.panorama.horizontal_fov_deg = 360.0;
  // Square pixels around the middle of the wall: a row spans what a column
  // does, in elevation (spherical) or its tangent (cylindrical).
  const double bottom = -center_height / world.wall_radius;
  const double top = (world.wall_height - center_height) / world.wall_radius;
  const double span = 2.0 * std::numbers::pi * height / width;
  if (projection == Projection::kCylindrical) {
    const double middle = 0.5 * (bottom + top);
    g.panorama.elevation_min_deg = std::atan(middle - 0.5 * span) / kDegree;
    g.panorama.elevation_max_deg = std::atan(middle + 0.5 * span) / kDegree;
  } else {
    const double middle = 0.5 * (std::atan(bottom) + std::atan(top));
    g.panorama.elevation_min_deg = std::max(-89.0, (middle - 0.5 * span) / kDegree);
    g.panorama.elevation_max_deg = std::min(89.0, (middle + 0.5 * span) / kDegree);
  }
  g.panorama.center = {0.0, 0.0, center_height};
  g.panorama.radius = world.wall_radius;
  validate(g);
  return g;
}

OutputGeometry ground_view(int width, int height, double half_extent) {
  OutputGeometry g;
  g.projection = Projection::kBirdsEye;
  g.width = width;
  g.height = height;
  g.birds_eye = BirdsEyeParams{-half_extent, half_extent, -half_extent, half_extent};
  validate(g);
  return g;
}

std::vector<Point2> checker_corners(const SyntheticWorld& world, const OutputGeometry& geometry,
                                    const RigPose& pose, double margin) {
  std::vector<Point2> corners;
  if (geometry.projection == Projection::kBirdsEye) {
    const BirdsEyeParams& b = geometry.birds_eye;
    // The ground squares around the area, in the world frame.
    double x_lo = 1e300;
    double x_hi = -1e300;
    double y_lo = 1e300;
    double y_hi = -1e300;
    for (const double x : {b.x_min, b.x_max}) {
      for (const double y : {b.y_min, b.y_max}) {
        const auto w = world_from_rig(pose, {x, y, 0.0});
        x_lo = std::min(x_lo, w[0]);
        x_hi = std::max(x_hi, w[0]);
        y_lo = std::min(y_lo, w[1]);
        y_hi = std::max(y_hi, w[1]);
      }
    }
    const double s = world.ground_square;
    for (double i = std::ceil(x_lo / s); i * s <= x_hi; i += 1.0) {
      for (double j = std::ceil(y_lo / s); j * s <= y_hi; j += 1.0) {
        const auto p = rig_from_world(pose, {i * s, j * s, 0.0});
        if (std::hypot(p[0], p[1]) > world.wall_radius - s) continue;
        const auto pixel = output_pixel(geometry, p);
        if (pixel && inside(geometry, *pixel, margin)) corners.push_back(*pixel);
      }
    }
    return corners;
  }
  // The wall's corners: its column edges at each inner row edge.
  const double yaw = pose.yaw_deg * kDegree;
  const int rows = static_cast<int>(std::floor(world.wall_height / world.wall_row_height + 1e-9));
  for (int k = 0; k < world.wall_columns; ++k) {
    const double azimuth = 2.0 * std::numbers::pi * k / world.wall_columns - yaw;
    for (int m = 1; m < rows; ++m) {
      const std::array<double, 3> p{world.wall_radius * std::cos(azimuth),
                                    world.wall_radius * std::sin(azimuth),
                                    m * world.wall_row_height};
      const auto pixel = output_pixel(geometry, p);
      if (pixel && inside(geometry, *pixel, margin)) corners.push_back(*pixel);
    }
  }
  return corners;
}

std::optional<Point2> refine_corner(std::span<const float> gray, int width, int height,
                                    Point2 guess, int radius) {
  const auto at = [&](int x, int y) { return gray[static_cast<std::size_t>(y) * width + x]; };
  Point2 q = guess;
  for (int iteration = 0; iteration < 20; ++iteration) {
    const int cx = static_cast<int>(std::lround(q.x));
    const int cy = static_cast<int>(std::lround(q.y));
    if (cx - radius - 1 < 0 || cy - radius - 1 < 0 || cx + radius + 1 >= width ||
        cy + radius + 1 >= height) {
      return std::nullopt;
    }
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double bx = 0.0;
    double by = 0.0;
    for (int y = cy - radius; y <= cy + radius; ++y) {
      for (int x = cx - radius; x <= cx + radius; ++x) {
        const double gx = 0.5 * (at(x + 1, y) - at(x - 1, y));
        const double gy = 0.5 * (at(x, y + 1) - at(x, y - 1));
        a += gx * gx;
        b += gx * gy;
        c += gy * gy;
        bx += gx * gx * x + gx * gy * y;
        by += gx * gy * x + gy * gy * y;
      }
    }
    const double det = a * c - b * b;
    if (!(det > 1e-6 * (a + c) * (a + c)) || a + c < 1e3) {
      return std::nullopt;
    }
    const Point2 next{(c * bx - b * by) / det, (a * by - b * bx) / det};
    const double step = std::hypot(next.x - q.x, next.y - q.y);
    q = next;
    if (step < 1e-3) break;
  }
  if (std::hypot(q.x - guess.x, q.y - guess.y) > 0.5 * radius) {
    return std::nullopt;
  }
  return q;
}

Comparison compare_with_truth(const std::uint8_t* stitched, std::size_t stitched_pitch,
                              const std::uint8_t* truth, std::size_t truth_pitch, int width,
                              int height, std::span<const Point2> corners, int border,
                              std::span<const float> resolution, float min_resolution) {
  const auto pixel = [](const std::uint8_t* image, std::size_t pitch, int x, int y) {
    return image + static_cast<std::size_t>(y) * pitch + static_cast<std::size_t>(x) * 4U;
  };
  const auto usable = [&](int x, int y) {
    return x >= border && y >= border && x < width - border && y < height - border &&
           pixel(truth, truth_pitch, x, y)[3] == 255 &&
           pixel(stitched, stitched_pitch, x, y)[3] == 255;
  };
  Comparison result;
  std::size_t comparable = 0;
  double st = 0.0;
  double tt = 0.0;
  for (int y = border; y < height - border; ++y) {
    for (int x = border; x < width - border; ++x) {
      if (pixel(truth, truth_pitch, x, y)[3] != 255) continue;
      ++comparable;
      if (!usable(x, y)) continue;
      ++result.compared_pixels;
      const std::uint8_t* s = pixel(stitched, stitched_pitch, x, y);
      const std::uint8_t* t = pixel(truth, truth_pitch, x, y);
      for (int ch = 0; ch < 3; ++ch) {
        st += s[ch];
        tt += t[ch];
      }
    }
  }
  result.coverage = comparable == 0 ? 0.0
                                    : static_cast<double>(result.compared_pixels) /
                                          static_cast<double>(comparable);
  result.gain = tt > 0.0 ? st / tt : 0.0;
  double squared = 0.0;
  for (int y = border; y < height - border; ++y) {
    for (int x = border; x < width - border; ++x) {
      if (!usable(x, y)) continue;
      const std::uint8_t* s = pixel(stitched, stitched_pitch, x, y);
      const std::uint8_t* t = pixel(truth, truth_pitch, x, y);
      for (int ch = 0; ch < 3; ++ch) {
        const double d = static_cast<double>(s[ch]) - result.gain * t[ch];
        squared += d * d;
      }
    }
  }
  const double mse = result.compared_pixels == 0
                         ? 0.0
                         : squared / (3.0 * static_cast<double>(result.compared_pixels));
  result.psnr_db = mse > 0.0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : 99.0;

  // The corners, where their whole window is comparable.
  std::vector<float> gray_s(static_cast<std::size_t>(width) * height);
  std::vector<float> gray_t(gray_s.size());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::uint8_t* s = pixel(stitched, stitched_pitch, x, y);
      const std::uint8_t* t = pixel(truth, truth_pitch, x, y);
      const std::size_t i = static_cast<std::size_t>(y) * width + x;
      gray_s[i] = 0.299F * s[0] + 0.587F * s[1] + 0.114F * s[2];
      gray_t[i] = 0.299F * t[0] + 0.587F * t[1] + 0.114F * t[2];
    }
  }
  constexpr int kRadius = 8;
  double error_sum = 0.0;
  for (const Point2& corner : corners) {
    const int cx = static_cast<int>(std::lround(corner.x));
    const int cy = static_cast<int>(std::lround(corner.y));
    if (!resolution.empty() && cx >= 0 && cy >= 0 && cx < width && cy < height &&
        resolution[static_cast<std::size_t>(cy) * width + cx] < min_resolution) {
      continue;
    }
    bool window = true;
    for (int y = cy - kRadius - 3; y <= cy + kRadius + 3 && window; ++y) {
      for (int x = cx - kRadius - 3; x <= cx + kRadius + 3 && window; ++x) {
        window = usable(x, y);
      }
    }
    if (!window) continue;
    const auto in_truth = refine_corner(gray_t, width, height, corner, kRadius);
    if (!in_truth) continue;
    const auto in_stitch = refine_corner(gray_s, width, height, corner, kRadius);
    // A corner the truth has and the stitch lost is as wrong as can be.
    const double error = in_stitch
                             ? std::hypot(in_stitch->x - in_truth->x, in_stitch->y - in_truth->y)
                             : static_cast<double>(kRadius);
    ++result.corners;
    error_sum += error;
    result.max_corner_error = std::max(result.max_corner_error, error);
  }
  result.mean_corner_error = result.corners == 0 ? 0.0 : error_sum / result.corners;
  return result;
}

}  // namespace holoscan::stitching
