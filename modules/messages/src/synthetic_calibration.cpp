// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/messages/synthetic_calibration.hpp"

#include <cmath>
#include <vector>

#include "geometry.hpp"

namespace holoscan::messages {
namespace {

std::vector<double> coefficients(DistortionModel model, double scale) {
  switch (model) {
    case DistortionModel::kPlumbBob:
      return {-0.12 * scale, 0.031 * scale, 0.0004, -0.0002, -0.004 * scale};
    case DistortionModel::kRationalPolynomial:
      return {0.41 * scale, -0.07, 0.0003, -0.0001, 0.002, 0.77 * scale, -0.02, 0.006};
    case DistortionModel::kEquidistant:
      return {0.052 * scale, -0.011, 0.0021, -0.0004};
    case DistortionModel::kNone:
      break;
  }
  return {};
}

CameraInfo camera(DistortionModel model, std::uint32_t width, std::uint32_t height,
                  const char* frame_id, double focal_scale) {
  CameraInfo info;
  info.header.frame_id = frame_id;
  info.width = width;
  info.height = height;
  info.state = model == DistortionModel::kNone ? CameraState::kUndistorted : CameraState::kRaw;
  info.fx = 0.71 * width * focal_scale;
  info.fy = 0.7105 * width * focal_scale;
  info.cx = 0.5 * (width - 1) + 3.25 * focal_scale;
  info.cy = 0.5 * (height - 1) - 2.5 * focal_scale;
  info.distortion_model = model;
  info.coefficients = coefficients(model, focal_scale);
  return info;
}

}  // namespace

Matrix3 rotation_matrix(const Vector3& axis, double angle_rad) noexcept {
  const double length = detail::norm(axis);
  const double x = axis[0] / length;
  const double y = axis[1] / length;
  const double z = axis[2] / length;
  const double c = std::cos(angle_rad);
  const double s = std::sin(angle_rad);
  const double t = 1.0 - c;
  return {t * x * x + c,     t * x * y - s * z, t * x * z + s * y, t * x * y + s * z, t * y * y + c,
          t * y * z - s * x, t * x * z - s * y, t * y * z + s * x, t * z * z + c};
}

StereoCameraInfo synthetic_stereo_camera_info(DistortionModel model, std::uint32_t width,
                                              std::uint32_t height) {
  StereoCameraInfo info;
  info.header.frame_id = "left";
  info.left = camera(model, width, height, "left", 1.0);
  info.right = camera(model, width, height, "right", 1.004);
  info.right_from_left.rotation = rotation_matrix({0.2, 1.0, 0.1}, 0.0175);
  info.right_from_left.translation = {-0.12, 0.0011, -0.0007};
  // The right camera's center c in the left frame; rect_left turns c onto +x,
  // and rect_right = rect_left * R^T, so that R = rect_right^T * rect_left.
  const Matrix3 r_t = detail::transpose(info.right_from_left.rotation);
  const Vector3 t = detail::rotate(r_t, info.right_from_left.translation);
  const Vector3 center{-t[0], -t[1], -t[2]};
  const double length = detail::norm(center);
  const Vector3 axis{0.0, center[2], -center[1]};  // center x (1, 0, 0)
  const Matrix3 rect_left = rotation_matrix(axis, std::acos(center[0] / length));
  info.left.rectification = rect_left;
  info.right.rectification = detail::multiply(rect_left, r_t);
  validate(info);
  return info;
}

}  // namespace holoscan::messages
