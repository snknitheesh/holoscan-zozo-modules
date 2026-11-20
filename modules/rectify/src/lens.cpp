// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/rectify/lens.hpp"

#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <string>

namespace holoscan::rectify {
namespace {

constexpr int kNewtonIterations = 50;
constexpr double kNewtonTolerance = 1e-12;

// The fisheye angle of a distorted radius: solves
// theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8) = theta_d.
std::optional<double> fisheye_theta(const messages::CameraInfo& camera, double theta_d) noexcept {
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

double coefficient(const messages::CameraInfo& camera, std::size_t index) noexcept {
  return index < camera.coefficients.size() ? camera.coefficients[index] : 0.0;
}

void check_camera(const messages::CameraInfo& camera, std::string_view what) {
  messages::validate(camera, what);
  const std::string prefix = std::string(what) + ": ";
  if (camera.width > static_cast<std::uint32_t>(kMaxImageSide) ||
      camera.height > static_cast<std::uint32_t>(kMaxImageSide)) {
    throw std::invalid_argument(prefix + "width and height must be at most " +
                                std::to_string(kMaxImageSide));
  }
  if (camera.skew != 0.0) {
    throw std::invalid_argument(prefix + "a camera with skew is not supported");
  }
  // ROS gives the intrinsics of the full-resolution sensor and leaves it to
  // the consumer to apply a region of interest and binning; refuse them rather
  // than remap with intrinsics that do not describe the images.
  if (camera.roi || camera.binning_x != 1 || camera.binning_y != 1) {
    throw std::invalid_argument(prefix +
                                "a region of interest or binning is not supported; give the "
                                "calibration of the images as they are");
  }
}

Point2 distort(const messages::CameraInfo& camera, Point2 ideal) noexcept {
  const double x = ideal.x;
  const double y = ideal.y;
  switch (camera.distortion_model) {
    case messages::DistortionModel::kNone:
      return ideal;
    case messages::DistortionModel::kPlumbBob:
    case messages::DistortionModel::kRationalPolynomial: {
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
    case messages::DistortionModel::kEquidistant: {
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

std::optional<Point2> undistort(const messages::CameraInfo& camera, Point2 distorted) noexcept {
  switch (camera.distortion_model) {
    case messages::DistortionModel::kNone:
      return distorted;
    case messages::DistortionModel::kEquidistant: {
      const double theta_d = std::hypot(distorted.x, distorted.y);
      if (theta_d < 1e-15) return distorted;
      const auto theta = fisheye_theta(camera, theta_d);
      // At 90 degrees or more the ray never meets the image plane.
      if (!theta || *theta >= std::numbers::pi / 2 - 1e-9) return std::nullopt;
      const double scale = std::tan(*theta) / theta_d;
      return Point2{distorted.x * scale, distorted.y * scale};
    }
    case messages::DistortionModel::kPlumbBob:
    case messages::DistortionModel::kRationalPolynomial:
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
