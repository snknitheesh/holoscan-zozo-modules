// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/stitching/camera_rig.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "stitch_math.hpp"

namespace holoscan::stitching {

static_assert(detail::kMaxCameras == static_cast<int>(kMaxCameras),
              "the kernels and the operator take the same number of cameras");

namespace {

using Matrix3 = std::array<double, 9>;

Matrix3 multiply(const Matrix3& a, const Matrix3& b) noexcept {
  Matrix3 c{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) s += a[3 * i + k] * b[3 * k + j];
      c[3 * i + j] = s;
    }
  }
  return c;
}

Matrix3 transpose(const Matrix3& a) noexcept {
  return {a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]};
}

std::array<double, 3> rotate(const Matrix3& r, const std::array<double, 3>& v) noexcept {
  return {r[0] * v[0] + r[1] * v[1] + r[2] * v[2], r[3] * v[0] + r[4] * v[1] + r[5] * v[2],
          r[6] * v[0] + r[7] * v[1] + r[8] * v[2]};
}

Matrix3 rotation_z(double angle) noexcept {
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  return {c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0};
}

Matrix3 rotation_y(double angle) noexcept {
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  return {c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c};
}

Matrix3 rotation_x(double angle) noexcept {
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  return {1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c};
}

constexpr double kDegree = std::numbers::pi / 180.0;

template <typename Values>
bool all_finite(const Values& values) {
  return std::all_of(std::begin(values), std::end(values),
                     [](double v) { return std::isfinite(v); });
}
// Steps of the search for where a lens' projection stops moving outwards.
constexpr double kAngleStep = 1e-3;

// The distorted radius (normalized) of a ray at `theta`, ignoring tangential
// distortion, for the search below.
double distorted_radius(const messages::CameraInfo& camera, double theta) noexcept {
  const auto k = [&camera](std::size_t i) { return coefficient(camera, i); };
  if (camera.distortion_model == messages::DistortionModel::kEquidistant) {
    const double t2 = theta * theta;
    return theta * (1.0 + t2 * (k(0) + t2 * (k(1) + t2 * (k(2) + t2 * k(3)))));
  }
  const double r = std::tan(theta);
  const double r2 = r * r;
  const double a = 1.0 + r2 * k(0) + r2 * r2 * k(1) + r2 * r2 * r2 * k(4);
  double b = 1.0;
  if (camera.distortion_model == messages::DistortionModel::kRationalPolynomial) {
    b += r2 * k(5) + r2 * r2 * k(6) + r2 * r2 * r2 * k(7);
  }
  return r * a / b;
}

}  // namespace

RigidTransform inverse(const RigidTransform& b_from_a) noexcept {
  RigidTransform a_from_b;
  a_from_b.rotation = transpose(b_from_a.rotation);
  const std::array<double, 3> t = rotate(a_from_b.rotation, b_from_a.translation);
  a_from_b.translation = {-t[0], -t[1], -t[2]};
  return a_from_b;
}

RigidTransform compose(const RigidTransform& a_from_b, const RigidTransform& b_from_c) noexcept {
  RigidTransform a_from_c;
  a_from_c.rotation = multiply(a_from_b.rotation, b_from_c.rotation);
  const std::array<double, 3> t = rotate(a_from_b.rotation, b_from_c.translation);
  for (int i = 0; i < 3; ++i) a_from_c.translation[i] = t[i] + a_from_b.translation[i];
  return a_from_c;
}

RigidTransform mount(const std::array<double, 3>& position, double yaw_deg, double pitch_deg,
                     double roll_deg) noexcept {
  // The optical axes of an upright camera looking along +x: x right (-y),
  // y down (-z), z forward (+x), as the columns of rig_from_camera.
  constexpr Matrix3 kOptical{0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0};
  const Matrix3 body =
      multiply(multiply(rotation_z(yaw_deg * kDegree), rotation_y(-pitch_deg * kDegree)),
               rotation_x(roll_deg * kDegree));
  RigidTransform rig_from_camera;
  rig_from_camera.rotation = multiply(body, kOptical);
  rig_from_camera.translation = position;
  return inverse(rig_from_camera);
}

void validate(const RigidTransform& camera_from_rig, std::string_view what) {
  const std::string where(what);
  const Matrix3& r = camera_from_rig.rotation;
  if (!all_finite(r) || !all_finite(camera_from_rig.translation)) {
    throw std::invalid_argument(where + " must be finite");
  }
  const Matrix3 should_be_identity = multiply(r, transpose(r));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (std::abs(should_be_identity[3 * i + j] - (i == j ? 1.0 : 0.0)) > 1e-6) {
        throw std::invalid_argument(where + "'s rotation is not orthonormal");
      }
    }
  }
  const double det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) +
                     r[2] * (r[3] * r[7] - r[4] * r[6]);
  if (std::abs(det - 1.0) > 1e-6) {
    throw std::invalid_argument(where + "'s rotation is a reflection");
  }
}

void check_camera_info(const messages::CameraInfo& info, std::string_view what) {
  messages::validate(info, what);
  const std::string prefix = std::string(what) + ": ";
  if (info.width > static_cast<std::uint32_t>(kMaxImageSide) ||
      info.height > static_cast<std::uint32_t>(kMaxImageSide)) {
    throw std::invalid_argument(prefix + "width and height must be at most " +
                                std::to_string(kMaxImageSide));
  }
  if (info.skew != 0.0) {
    throw std::invalid_argument(prefix + "a camera with skew is not supported");
  }
  // ROS gives the intrinsics of the full-resolution sensor and leaves it to
  // the consumer to apply a region of interest and binning; refuse them rather
  // than stitch with intrinsics that may not describe the images.
  if (info.roi || info.binning_x != 1 || info.binning_y != 1) {
    throw std::invalid_argument(prefix +
                                "a region of interest or binning is not supported; give the "
                                "calibration of the images as they are");
  }
  if (!(max_valid_angle(info) > 0.01)) {
    throw std::invalid_argument(prefix + "the distortion folds back at the optical axis");
  }
}

void validate(const Camera& camera, std::string_view what) {
  check_camera_info(camera.info, what);
  validate(camera.camera_from_rig, std::string(what) + ": camera_from_rig");
}

void validate(const CameraRig& rig) {
  if (rig.cameras.empty() || rig.cameras.size() > kMaxCameras) {
    throw std::invalid_argument("a rig has 1.." + std::to_string(kMaxCameras) + " cameras, not " +
                                std::to_string(rig.cameras.size()));
  }
  for (std::size_t i = 0; i < rig.cameras.size(); ++i) {
    validate(rig.cameras[i], camera_label(rig.cameras[i], i));
  }
}

std::string camera_label(const Camera& camera, std::size_t index) {
  std::string label = "camera " + std::to_string(index);
  if (!camera.info.header.frame_id.empty()) {
    label += " (" + camera.info.header.frame_id + ")";
  }
  return label;
}

double coefficient(const messages::CameraInfo& camera, std::size_t index) noexcept {
  return index < camera.coefficients.size() ? camera.coefficients[index] : 0.0;
}

double max_valid_angle(const messages::CameraInfo& camera) noexcept {
  const bool fisheye = camera.distortion_model == messages::DistortionModel::kEquidistant;
  // A pinhole sees less than 90 degrees from its axis; a fisheye less than 180.
  const double limit = (fisheye ? std::numbers::pi : std::numbers::pi / 2.0) - 1e-3;
  double previous = 0.0;
  for (double theta = kAngleStep; theta <= limit; theta += kAngleStep) {
    const double radius = distorted_radius(camera, theta);
    if (!(radius > previous)) {
      return theta - kAngleStep;
    }
    previous = radius;
  }
  return limit;
}

Point2 distort(const messages::CameraInfo& camera, Point2 ideal) noexcept {
  const detail::CameraParams params = detail::camera_params(camera, RigidTransform{});
  double xd = 0.0;
  double yd = 0.0;
  if (camera.distortion_model == messages::DistortionModel::kEquidistant) {
    const double r = std::hypot(ideal.x, ideal.y);
    const double theta = std::atan(r);
    detail::distort_ray(params, theta, r > 0 ? ideal.x / r : 0.0, r > 0 ? ideal.y / r : 0.0, 0.0,
                        0.0, xd, yd);
  } else {
    detail::distort_ray(params, 0.0, 0.0, 0.0, ideal.x, ideal.y, xd, yd);
  }
  return {xd, yd};
}

std::optional<Point2> project_to_pixel(const messages::CameraInfo& camera,
                                       const std::array<double, 3>& point) noexcept {
  // Through the camera's own frame: an identity pose.
  detail::CameraParams params = detail::camera_params(camera, RigidTransform{});
  // Only the lens counts here, not the image's border.
  params.width = kMaxImageSide * 4;
  params.height = kMaxImageSide * 4;
  params.cx = camera.cx + kMaxImageSide * 2;
  params.cy = camera.cy + kMaxImageSide * 2;
  const double p[3] = {point[0], point[1], point[2]};
  double u = 0.0;
  double v = 0.0;
  double margin = 0.0;
  if (!detail::project(params, p, false, u, v, margin)) {
    return std::nullopt;
  }
  return Point2{u - kMaxImageSide * 2, v - kMaxImageSide * 2};
}

std::optional<std::array<double, 3>> pixel_ray(const messages::CameraInfo& camera,
                                               Point2 pixel) noexcept {
  const detail::CameraParams params = detail::camera_params(camera, RigidTransform{});
  double ray[3];
  if (!detail::pixel_ray(params, pixel.x, pixel.y, ray)) {
    return std::nullopt;
  }
  return std::array<double, 3>{ray[0], ray[1], ray[2]};
}

namespace detail {

CameraParams camera_params(const messages::CameraInfo& camera,
                           const RigidTransform& camera_from_rig) {
  CameraParams p{};
  p.width = static_cast<int>(camera.width);
  p.height = static_cast<int>(camera.height);
  p.fx = camera.fx;
  p.fy = camera.fy;
  p.cx = camera.cx;
  p.cy = camera.cy;
  switch (camera.distortion_model) {
    case messages::DistortionModel::kNone:
      p.model = kModelNone;
      break;
    case messages::DistortionModel::kPlumbBob:
      p.model = kModelPlumbBob;
      break;
    case messages::DistortionModel::kRationalPolynomial:
      p.model = kModelRational;
      break;
    case messages::DistortionModel::kEquidistant:
      p.model = kModelEquidistant;
      break;
  }
  for (std::size_t i = 0; i < 8; ++i) p.k[i] = coefficient(camera, i);
  std::copy(camera_from_rig.rotation.begin(), camera_from_rig.rotation.end(), p.rotation);
  std::copy(camera_from_rig.translation.begin(), camera_from_rig.translation.end(), p.translation);
  p.max_angle = max_valid_angle(camera);
  return p;
}

CameraParams camera_params(const Camera& camera) {
  return camera_params(camera.info, camera.camera_from_rig);
}

}  // namespace detail
}  // namespace holoscan::stitching
