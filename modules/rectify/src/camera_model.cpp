// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/rectify/camera_model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace holoscan::rectify {
namespace {

constexpr std::array<std::pair<DistortionModel, std::string_view>, 4> kModelNames{{
    {DistortionModel::kNone, "none"},
    {DistortionModel::kBrownConrady, "brown_conrady"},
    {DistortionModel::kRationalPolynomial, "rational_polynomial"},
    {DistortionModel::kKannalaBrandt4, "kannala_brandt4"},
}};

// Other names for them: ROS camera_info's, and OpenCV's fisheye.
constexpr std::array<std::pair<DistortionModel, std::string_view>, 3> kAliases{{
    {DistortionModel::kBrownConrady, "plumb_bob"},
    {DistortionModel::kKannalaBrandt4, "equidistant"},
    {DistortionModel::kKannalaBrandt4, "fisheye"},
}};

constexpr int kNewtonIterations = 50;
constexpr double kNewtonTolerance = 1e-12;

// The fisheye angle of a distorted radius: solves
// theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8) = theta_d.
std::optional<double> fisheye_theta(const CameraModel& camera, double theta_d) noexcept {
  const double k1 = coefficient(camera, 0);
  const double k2 = coefficient(camera, 1);
  const double k3 = coefficient(camera, 2);
  const double k4 = coefficient(camera, 3);
  double theta = theta_d;
  for (int i = 0; i < kNewtonIterations; ++i) {
    const double t2 = theta * theta;
    const double f = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d;
    const double df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)));
    if (!(std::abs(df) > 1e-12)) return std::nullopt;
    const double step = f / df;
    theta -= step;
    if (std::abs(step) < kNewtonTolerance) {
      return std::isfinite(theta) && theta >= 0 ? std::optional<double>(theta) : std::nullopt;
    }
  }
  return std::nullopt;
}

}  // namespace

std::string_view distortion_model_name(DistortionModel model) noexcept {
  for (const auto& [kind, name] : kModelNames) {
    if (kind == model) return name;
  }
  return "unknown";
}

std::optional<DistortionModel> parse_distortion_model(std::string_view name) noexcept {
  for (const auto& [kind, model_name] : kModelNames) {
    if (model_name == name) return kind;
  }
  for (const auto& [kind, alias] : kAliases) {
    if (alias == name) return kind;
  }
  return std::nullopt;
}

std::size_t distortion_coefficient_count(DistortionModel model) noexcept {
  switch (model) {
    case DistortionModel::kNone:
      return 0;
    case DistortionModel::kBrownConrady:
      return 5;
    case DistortionModel::kRationalPolynomial:
      return 8;
    case DistortionModel::kKannalaBrandt4:
      return 4;
  }
  return 0;
}

double coefficient(const CameraModel& camera, std::size_t index) noexcept {
  return index < camera.distortion.size() ? camera.distortion[index] : 0.0;
}

void validate(const CameraModel& camera, std::string_view what) {
  const std::string prefix = std::string(what) + ": ";
  if (camera.width < 1 || camera.height < 1 || camera.width > kMaxImageSide ||
      camera.height > kMaxImageSide) {
    throw std::invalid_argument(prefix + "width and height must be 1.." +
                                std::to_string(kMaxImageSide));
  }
  if (!(std::isfinite(camera.fx) && camera.fx > 0 && std::isfinite(camera.fy) && camera.fy > 0)) {
    throw std::invalid_argument(prefix + "fx and fy must be finite and positive");
  }
  if (!(std::isfinite(camera.cx) && std::isfinite(camera.cy))) {
    throw std::invalid_argument(prefix + "cx and cy must be finite");
  }
  if (distortion_model_name(camera.distortion_model) == "unknown") {
    throw std::invalid_argument(prefix + "unknown distortion model");
  }
  const std::size_t count = distortion_coefficient_count(camera.distortion_model);
  if (!camera.distortion.empty() && camera.distortion.size() != count) {
    throw std::invalid_argument(
        prefix + std::string(distortion_model_name(camera.distortion_model)) + " takes " +
        std::to_string(count) + " coefficients, not " + std::to_string(camera.distortion.size()));
  }
  if (!std::all_of(camera.distortion.begin(), camera.distortion.end(),
                   [](double k) { return std::isfinite(k); })) {
    throw std::invalid_argument(prefix + "distortion coefficients must be finite");
  }
}

void validate(const StereoCalibration& calibration) {
  validate(calibration.left, "left");
  validate(calibration.right, "right");
  const auto& r = calibration.right_from_left.rotation;
  const auto& t = calibration.right_from_left.translation;
  const auto finite = [](double v) { return std::isfinite(v); };
  if (!std::all_of(r.begin(), r.end(), finite) || !std::all_of(t.begin(), t.end(), finite)) {
    throw std::invalid_argument("right_from_left must be finite");
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double dot = 0.0;
      for (int k = 0; k < 3; ++k) dot += r[3 * k + i] * r[3 * k + j];
      if (std::abs(dot - (i == j ? 1.0 : 0.0)) > 1e-6) {
        throw std::invalid_argument("right_from_left.rotation is not orthonormal");
      }
    }
  }
  const double det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) +
                     r[2] * (r[3] * r[7] - r[4] * r[6]);
  if (std::abs(det - 1.0) > 1e-6) {
    throw std::invalid_argument("right_from_left.rotation has determinant " + std::to_string(det) +
                                ", not 1");
  }
  if (!(std::hypot(t[0], t[1], t[2]) > 1e-9)) {
    throw std::invalid_argument("right_from_left.translation is zero: there is no baseline");
  }
}

Point2 distort(const CameraModel& camera, Point2 ideal) noexcept {
  const double x = ideal.x;
  const double y = ideal.y;
  switch (camera.distortion_model) {
    case DistortionModel::kNone:
      return ideal;
    case DistortionModel::kBrownConrady:
    case DistortionModel::kRationalPolynomial: {
      const double k1 = coefficient(camera, 0);
      const double k2 = coefficient(camera, 1);
      const double p1 = coefficient(camera, 2);
      const double p2 = coefficient(camera, 3);
      const double k3 = coefficient(camera, 4);
      const double k4 = coefficient(camera, 5);
      const double k5 = coefficient(camera, 6);
      const double k6 = coefficient(camera, 7);
      const double r2 = x * x + y * y;
      const double radial =
          (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)));
      return {x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
              y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y};
    }
    case DistortionModel::kKannalaBrandt4: {
      const double r = std::hypot(x, y);
      if (r < 1e-15) return ideal;
      const double theta = std::atan(r);
      const double t2 = theta * theta;
      const double theta_d =
          theta * (1 + t2 * (coefficient(camera, 0) +
                             t2 * (coefficient(camera, 1) +
                                   t2 * (coefficient(camera, 2) + t2 * coefficient(camera, 3)))));
      const double scale = theta_d / r;
      return {x * scale, y * scale};
    }
  }
  return ideal;
}

std::optional<Point2> undistort(const CameraModel& camera, Point2 distorted) noexcept {
  switch (camera.distortion_model) {
    case DistortionModel::kNone:
      return distorted;
    case DistortionModel::kKannalaBrandt4: {
      const double theta_d = std::hypot(distorted.x, distorted.y);
      if (theta_d < 1e-15) return distorted;
      const auto theta = fisheye_theta(camera, theta_d);
      // At 90 degrees or more the ray never meets the image plane.
      if (!theta || *theta >= std::numbers::pi / 2 - 1e-9) return std::nullopt;
      const double scale = std::tan(*theta) / theta_d;
      return Point2{distorted.x * scale, distorted.y * scale};
    }
    case DistortionModel::kBrownConrady:
    case DistortionModel::kRationalPolynomial:
      break;
  }
  // Newton's method on distort(p) = distorted, from p = distorted, with a
  // central-difference Jacobian.
  Point2 p = distorted;
  constexpr double h = 1e-7;
  for (int i = 0; i < kNewtonIterations; ++i) {
    const Point2 f = distort(camera, p);
    const double ex = f.x - distorted.x;
    const double ey = f.y - distorted.y;
    if (std::hypot(ex, ey) < kNewtonTolerance) {
      return std::isfinite(p.x) && std::isfinite(p.y) ? std::optional<Point2>(p) : std::nullopt;
    }
    const Point2 xp = distort(camera, {p.x + h, p.y});
    const Point2 xm = distort(camera, {p.x - h, p.y});
    const Point2 yp = distort(camera, {p.x, p.y + h});
    const Point2 ym = distort(camera, {p.x, p.y - h});
    const double a = (xp.x - xm.x) / (2 * h);
    const double b = (yp.x - ym.x) / (2 * h);
    const double c = (xp.y - xm.y) / (2 * h);
    const double d = (yp.y - ym.y) / (2 * h);
    const double det = a * d - b * c;
    if (!(std::abs(det) > 1e-12) || !std::isfinite(det)) return std::nullopt;
    p.x -= (d * ex - b * ey) / det;
    p.y -= (a * ey - c * ex) / det;
  }
  return std::nullopt;
}

}  // namespace holoscan::rectify
