// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/rectify/synthetic_stereo.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace holoscan::rectify {
namespace {

// The dots and the checkerboard, in gray levels.
constexpr double kDark = 40.0;
constexpr double kLight = 70.0;
constexpr double kPeak = 255.0;
// measure_dot() counts only what rises above the checkerboard.
constexpr double kDotThreshold = 80.0;
// Dot spacing and size in pixels of the left camera, at the plane.
constexpr double kSpacingPixels = 56.0;
constexpr double kSigmaPixels = 3.0;
// Dots that a raw image shows smaller than this (in pixels, along their most
// compressed direction), or cut by its border, are not measured: their
// centroids would carry the rendering's aliasing or the cut rather than the
// remap's error. A fisheye compresses dots so near its rim.
constexpr double kMinRawSigmaPixels = 1.2;

double degrees(double d) { return d * std::numbers::pi / 180.0; }

CameraModel camera(int width, int height, DistortionModel model, bool right) {
  CameraModel c;
  c.width = width;
  c.height = height;
  c.distortion_model = model;
  const double w = width;
  const double h = height;
  const double sign = right ? -1.0 : 1.0;
  c.cx = (w - 1) / 2 + sign * 0.0098 * w;
  c.cy = (h - 1) / 2 - sign * 0.0085 * h;
  switch (model) {
    case DistortionModel::kBrownConrady:
      c.fx = (right ? 0.615 : 0.62) * w;
      c.fy = (right ? 0.618 : 0.625) * w;
      c.distortion = right ? std::vector<double>{-0.26, 0.07, -0.0009, 0.0011, -0.008}
                           : std::vector<double>{-0.28, 0.08, 0.0012, -0.0008, -0.01};
      break;
    case DistortionModel::kRationalPolynomial:
      c.fx = (right ? 0.6 : 0.605) * w;
      c.fy = (right ? 0.602 : 0.607) * w;
      c.distortion = right ? std::vector<double>{0.04, 0.0, -0.0007, 0.0005, 0.0, 0.33, 0.0, 0.0}
                           : std::vector<double>{0.05, 0.0, 0.0008, -0.0006, 0.0, 0.35, 0.0, 0.0};
      break;
    case DistortionModel::kKannalaBrandt4:
      c.fx = (right ? 0.448 : 0.45) * w;
      c.fy = (right ? 0.449 : 0.451) * w;
      c.distortion = right ? std::vector<double>{0.045, -0.009, 0.0018, -0.0004}
                           : std::vector<double>{0.05, -0.01, 0.002, -0.0005};
      break;
    case DistortionModel::kNone:
      break;
  }
  return c;
}

Point2 project(const CameraModel& c, const std::array<double, 3>& p) {
  return {c.fx * p[0] / p[2] + c.cx, c.fy * p[1] / p[2] + c.cy};
}

bool window_inside(const PixelRect& region, Point2 center, double radius) {
  return center.x - radius >= region.x && center.y - radius >= region.y &&
         center.x + radius <= region.x + region.width - 1 &&
         center.y + radius <= region.y + region.height - 1;
}

// Whether a raw image of `camera` shows the dot of `sigma` metres at `point`
// (the plane's frame) whole and large enough to measure the remap by: at
// least kMinRawSigmaPixels along its most compressed direction, and 4 sigma
// inside the image.
bool raw_dot_measurable(const CameraModel& camera, const RigidTransform& camera_from_plane,
                        const std::array<double, 3>& point, double sigma) {
  const auto pixel = [&](double dx, double dy) {
    auto p = rotate(camera_from_plane.rotation, {point[0] + dx, point[1] + dy, point[2]});
    for (int k = 0; k < 3; ++k) p[k] += camera_from_plane.translation[k];
    const Point2 d = distort(camera, {p[0] / p[2], p[1] / p[2]});
    return Point2{camera.fx * d.x + camera.cx, camera.fy * d.y + camera.cy};
  };
  const Point2 center = pixel(0.0, 0.0);
  double smallest = std::numeric_limits<double>::infinity();
  double largest = 0.0;
  for (int i = 0; i < 8; ++i) {
    const double angle = i * std::numbers::pi / 8;
    const Point2 moved = pixel(sigma * std::cos(angle), sigma * std::sin(angle));
    const double extent = std::hypot(moved.x - center.x, moved.y - center.y);
    smallest = std::min(smallest, extent);
    largest = std::max(largest, extent);
  }
  const double margin = 4 * largest + 1;
  return smallest >= kMinRawSigmaPixels && center.x >= margin && center.y >= margin &&
         center.x <= camera.width - 1 - margin && center.y <= camera.height - 1 - margin;
}

void accumulate(AlignmentReport& report, double position_error, double row_error, double& sum_pos,
                double& sum_row) {
  ++report.dots;
  report.max_position_error = std::max(report.max_position_error, position_error);
  report.max_row_error = std::max(report.max_row_error, row_error);
  sum_pos += position_error * position_error;
  sum_row += row_error * row_error;
}

void finish(AlignmentReport& report, double sum_pos, double sum_row) {
  if (report.dots > 0) {
    report.rms_position_error = std::sqrt(sum_pos / report.dots);
    report.rms_row_error = std::sqrt(sum_row / report.dots);
  }
}

}  // namespace

SyntheticStereo synthetic_stereo(int width, int height, DistortionModel model) {
  if (width < 160 || height < 120 || width > kMaxImageSide || height > kMaxImageSide) {
    throw std::invalid_argument("synthetic stereo images must be 160x120 to 16384x16384");
  }
  if (model == DistortionModel::kNone) {
    throw std::invalid_argument("the synthetic pair needs a distortion model");
  }
  SyntheticStereo scene;
  scene.calibration.left = camera(width, height, model, false);
  scene.calibration.right = camera(width, height, model, true);
  // A rotation of a degree or so about each axis, and a 12 cm baseline that
  // is not quite horizontal.
  const Matrix3 yaw = axis_angle({0.0, 1.0, 0.0}, degrees(1.2));
  const Matrix3 pitch = axis_angle({1.0, 0.0, 0.0}, degrees(-0.8));
  const Matrix3 roll = axis_angle({0.0, 0.0, 1.0}, degrees(0.6));
  scene.calibration.right_from_left.rotation = multiply(roll, multiply(pitch, yaw));
  scene.calibration.right_from_left.translation = {-0.12, 0.0035, -0.0021};
  const double f = scene.calibration.left.fx;
  scene.plane.depth = 2.0;
  scene.plane.spacing = kSpacingPixels * scene.plane.depth / f;
  scene.plane.sigma = kSigmaPixels * scene.plane.depth / f;
  return scene;
}

SyntheticView::SyntheticView(CameraModel camera, const RigidTransform& camera_from_plane,
                             DotPlane plane)
    : camera_(std::move(camera)), plane_(plane) {
  validate(camera_);
  const Matrix3 plane_from_camera = transpose(camera_from_plane.rotation);
  const auto t_plane = rotate(plane_from_camera, camera_from_plane.translation);
  plane_xy_.assign(static_cast<std::size_t>(camera_.width) * camera_.height * 2U,
                   std::numeric_limits<float>::quiet_NaN());
  std::size_t i = 0;
  for (int v = 0; v < camera_.height; ++v) {
    for (int u = 0; u < camera_.width; ++u, i += 2) {
      const auto ideal =
          undistort(camera_, {(u - camera_.cx) / camera_.fx, (v - camera_.cy) / camera_.fy});
      if (!ideal) continue;
      // The ray s * (x, y, 1) in the plane's frame: plane_from_camera * (ray - t).
      const auto direction = rotate(plane_from_camera, {ideal->x, ideal->y, 1.0});
      if (!(direction[2] > 1e-9)) continue;
      const double s = (plane_.depth + t_plane[2]) / direction[2];
      if (!(s > 0)) continue;
      plane_xy_[i] = static_cast<float>(s * direction[0] - t_plane[0]);
      plane_xy_[i + 1] = static_cast<float>(s * direction[1] - t_plane[1]);
    }
  }
}

std::vector<std::uint8_t> SyntheticView::render(std::uint32_t frame) const {
  const double spacing = plane_.spacing;
  const double ox = std::fmod(0.37 * frame, 1.0) * spacing;
  const double oy = std::fmod(0.23 * frame, 1.0) * spacing;
  const double inv_two_sigma2 = 1.0 / (2 * plane_.sigma * plane_.sigma);
  std::vector<std::uint8_t> image(static_cast<std::size_t>(camera_.width) * camera_.height);
  for (std::size_t p = 0; p < image.size(); ++p) {
    const double x = plane_xy_[2 * p];
    const double y = plane_xy_[2 * p + 1];
    if (std::isnan(x)) {
      image[p] = 0;
      continue;
    }
    const double dx = x - ox;
    const double dy = y - oy;
    const double rx = dx - std::round(dx / spacing) * spacing;
    const double ry = dy - std::round(dy / spacing) * spacing;
    const auto cell = static_cast<std::int64_t>(std::floor(x / (2 * spacing))) +
                      static_cast<std::int64_t>(std::floor(y / (2 * spacing)));
    const double background = (cell & 1) != 0 ? kLight : kDark;
    const double dot = std::exp(-(rx * rx + ry * ry) * inv_two_sigma2);
    // The larger of the two, so that a dot over a checkerboard edge stays
    // symmetric above the checkerboard, where measure_dot() looks.
    image[p] = static_cast<std::uint8_t>(
        std::clamp(std::round(std::max(background, kPeak * dot)), 0.0, 255.0));
  }
  return image;
}

SyntheticView left_view(const SyntheticStereo& scene) {
  return SyntheticView(scene.calibration.left, RigidTransform{}, scene.plane);
}

SyntheticView right_view(const SyntheticStereo& scene) {
  return SyntheticView(scene.calibration.right, scene.calibration.right_from_left, scene.plane);
}

std::vector<std::array<double, 3>> dot_centers(const DotPlane& plane, std::uint32_t frame,
                                               double half_extent) {
  const double ox = std::fmod(0.37 * frame, 1.0) * plane.spacing;
  const double oy = std::fmod(0.23 * frame, 1.0) * plane.spacing;
  const int n = static_cast<int>(std::ceil(half_extent / plane.spacing)) + 1;
  std::vector<std::array<double, 3>> centers;
  for (int j = -n; j <= n; ++j) {
    for (int i = -n; i <= n; ++i) {
      const double x = ox + i * plane.spacing;
      const double y = oy + j * plane.spacing;
      if (std::abs(x) <= half_extent && std::abs(y) <= half_extent) {
        centers.push_back({x, y, plane.depth});
      }
    }
  }
  return centers;
}

std::optional<Point2> measure_dot(const std::uint8_t* image, std::size_t pitch,
                                  const PixelRect& region, Point2 predicted, double sigma_px) {
  const double radius = std::ceil(4 * sigma_px);
  if (!window_inside(region, predicted, radius)) return std::nullopt;
  const int x0 = static_cast<int>(std::lround(predicted.x - radius));
  const int x1 = static_cast<int>(std::lround(predicted.x + radius));
  const int y0 = static_cast<int>(std::lround(predicted.y - radius));
  const int y1 = static_cast<int>(std::lround(predicted.y + radius));
  double sum = 0.0;
  double sx = 0.0;
  double sy = 0.0;
  for (int y = y0; y <= y1; ++y) {
    for (int x = x0; x <= x1; ++x) {
      const double w = std::max(
          0.0, static_cast<double>(image[static_cast<std::size_t>(y) * pitch + x]) - kDotThreshold);
      sum += w;
      sx += w * x;
      sy += w * y;
    }
  }
  // A dot of a fraction of its full height, or none, is not measured.
  if (sum < (kPeak - kDotThreshold) * 2.0) return std::nullopt;
  return Point2{sx / sum, sy / sum};
}

AlignmentReport check_rectified(const SyntheticStereo& scene,
                                const StereoRectification& rectification, const std::uint8_t* left,
                                const std::uint8_t* right, std::size_t pitch, std::uint32_t frame) {
  const CameraModel left_camera = rectification.left_camera();
  const CameraModel right_camera = rectification.right_camera();
  const RigidTransform& pose = scene.calibration.right_from_left;
  AlignmentReport report;
  double sum_pos = 0.0;
  double sum_row = 0.0;
  for (const auto& point : dot_centers(scene.plane, frame, 3 * scene.plane.depth)) {
    const auto in_left = rotate(rectification.rect_from_left, point);
    auto in_right = rotate(pose.rotation, point);
    for (int k = 0; k < 3; ++k) in_right[k] += pose.translation[k];
    in_right = rotate(rectification.rect_from_right, in_right);
    if (!(in_left[2] > 0 && in_right[2] > 0)) continue;
    const Point2 predicted_left = project(left_camera, in_left);
    const Point2 predicted_right = project(right_camera, in_right);
    const double sigma_left = scene.plane.sigma * rectification.fx / in_left[2];
    const double sigma_right = scene.plane.sigma * rectification.fx / in_right[2];
    const double radius = std::ceil(4 * std::max(sigma_left, sigma_right));
    if (!window_inside(rectification.valid_left, predicted_left, radius) ||
        !window_inside(rectification.valid_right, predicted_right, radius) ||
        !raw_dot_measurable(scene.calibration.left, RigidTransform{}, point, scene.plane.sigma) ||
        !raw_dot_measurable(scene.calibration.right, pose, point, scene.plane.sigma)) {
      continue;
    }
    const auto measured_left =
        measure_dot(left, pitch, rectification.valid_left, predicted_left, sigma_left);
    const auto measured_right =
        measure_dot(right, pitch, rectification.valid_right, predicted_right, sigma_right);
    if (!measured_left || !measured_right) {
      ++report.missing;
      continue;
    }
    const double position_error = std::max(
        std::hypot(measured_left->x - predicted_left.x, measured_left->y - predicted_left.y),
        std::hypot(measured_right->x - predicted_right.x, measured_right->y - predicted_right.y));
    accumulate(report, position_error, std::abs(measured_left->y - measured_right->y), sum_pos,
               sum_row);
  }
  finish(report, sum_pos, sum_row);
  return report;
}

AlignmentReport check_undistorted(const DotPlane& plane, const Undistortion& undistortion,
                                  const std::uint8_t* image, std::size_t pitch,
                                  std::uint32_t frame) {
  const CameraModel& out = undistortion.output;
  AlignmentReport report;
  double sum_pos = 0.0;
  double sum_row = 0.0;
  for (const auto& point : dot_centers(plane, frame, 3 * plane.depth)) {
    const Point2 predicted = project(out, point);
    const double sigma = plane.sigma * std::min(out.fx, out.fy) / point[2];
    if (!window_inside(undistortion.valid, predicted, std::ceil(4 * sigma)) ||
        !raw_dot_measurable(undistortion.input, RigidTransform{}, point, plane.sigma)) {
      continue;
    }
    const auto measured = measure_dot(image, pitch, undistortion.valid, predicted, sigma);
    if (!measured) {
      ++report.missing;
      continue;
    }
    accumulate(report, std::hypot(measured->x - predicted.x, measured->y - predicted.y), 0.0,
               sum_pos, sum_row);
  }
  finish(report, sum_pos, sum_row);
  return report;
}

}  // namespace holoscan::rectify
