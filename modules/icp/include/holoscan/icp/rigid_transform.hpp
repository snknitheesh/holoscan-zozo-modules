// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Rigid transforms as 4x4 row-major double matrices, on the host: the
// transforms IcpRegistration estimates and the poses IcpOdometry accumulates.
// Header-only; needs neither the Holoscan SDK nor CUDA.

#pragma once

#include <array>
#include <cmath>

namespace holoscan::icp {

// A rigid transform [R | t; 0 0 0 1], row-major: p_out = R * p_in + t, with
// R at [0..2], [4..6], [8..10] and t at [3], [7], [11].
using Matrix4 = std::array<double, 16>;

[[nodiscard]] constexpr Matrix4 identity_transform() {
  return {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
}

// a * b: first b, then a.
[[nodiscard]] constexpr Matrix4 compose(const Matrix4& a, const Matrix4& b) {
  Matrix4 out{};
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      double sum = 0.0;
      for (int k = 0; k < 4; ++k) {
        sum += a[4 * row + k] * b[4 * k + col];
      }
      out[4 * row + col] = sum;
    }
  }
  return out;
}

// The inverse of a rigid transform: [R^T | -R^T t].
[[nodiscard]] constexpr Matrix4 rigid_inverse(const Matrix4& m) {
  Matrix4 out = identity_transform();
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      out[4 * row + col] = m[4 * col + row];
    }
  }
  for (int row = 0; row < 3; ++row) {
    out[4 * row + 3] = -(out[4 * row] * m[3] + out[4 * row + 1] * m[7] + out[4 * row + 2] * m[11]);
  }
  return out;
}

// The rotation by angle |axis_angle| about axis_angle's direction (Rodrigues),
// followed by the translation `translation`.
[[nodiscard]] inline Matrix4 transform_from(const std::array<double, 3>& axis_angle,
                                            const std::array<double, 3>& translation) {
  const double theta = std::sqrt(axis_angle[0] * axis_angle[0] + axis_angle[1] * axis_angle[1] +
                                 axis_angle[2] * axis_angle[2]);
  // sin(theta) / theta and (1 - cos(theta)) / theta^2, by series near zero.
  double a = 1.0;
  double b = 0.5;
  if (theta > 1e-8) {
    a = std::sin(theta) / theta;
    b = (1.0 - std::cos(theta)) / (theta * theta);
  } else {
    a = 1.0 - theta * theta / 6.0;
    b = 0.5 - theta * theta / 24.0;
  }
  const double x = axis_angle[0];
  const double y = axis_angle[1];
  const double z = axis_angle[2];
  // R = I + a [w]x + b [w]x^2.
  Matrix4 m = identity_transform();
  m[0] = 1.0 - b * (y * y + z * z);
  m[1] = -a * z + b * x * y;
  m[2] = a * y + b * x * z;
  m[4] = a * z + b * x * y;
  m[5] = 1.0 - b * (x * x + z * z);
  m[6] = -a * x + b * y * z;
  m[8] = -a * y + b * x * z;
  m[9] = a * x + b * y * z;
  m[10] = 1.0 - b * (x * x + y * y);
  m[3] = translation[0];
  m[7] = translation[1];
  m[11] = translation[2];
  return m;
}

// The angle of the rotation part, in radians, 0..pi; from its sine and cosine,
// so that small angles keep their precision.
[[nodiscard]] inline double rotation_angle(const Matrix4& m) {
  const double c = (m[0] + m[5] + m[10] - 1.0) * 0.5;
  const double x = m[9] - m[6];
  const double y = m[2] - m[8];
  const double z = m[4] - m[1];
  const double s = 0.5 * std::sqrt(x * x + y * y + z * z);
  return std::atan2(s, c);
}

// The length of the translation part.
[[nodiscard]] inline double translation_norm(const Matrix4& m) {
  return std::sqrt(m[3] * m[3] + m[7] * m[7] + m[11] * m[11]);
}

// The difference of `estimate` from `truth`, as the rotation angle (radians)
// and translation length of truth^-1 * estimate.
struct TransformError {
  double rotation_rad{};
  double translation{};
};

[[nodiscard]] inline TransformError transform_error(const Matrix4& estimate, const Matrix4& truth) {
  const Matrix4 delta = compose(rigid_inverse(truth), estimate);
  return {rotation_angle(delta), translation_norm(delta)};
}

// Whether every entry is finite and the last row is 0 0 0 1.
[[nodiscard]] inline bool is_rigid_shaped(const Matrix4& m) {
  for (const double value : m) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return m[12] == 0.0 && m[13] == 0.0 && m[14] == 0.0 && m[15] == 1.0;
}

}  // namespace holoscan::icp
