// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The few row-major 3x3 operations the calibration checks and the extrinsics
// recovery need.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "holoscan/messages/camera_info.hpp"

namespace holoscan::messages::detail {

// Rotation matrices are checked to this, entry by entry: far above the
// rounding of a rotation computed in double precision and written with 17
// digits, far below any real error.
inline constexpr double kRotationTolerance = 1e-9;

[[nodiscard]] inline Matrix3 multiply(const Matrix3& a, const Matrix3& b) noexcept {
  Matrix3 c{};
  for (std::size_t r = 0; r < 3; ++r) {
    for (std::size_t k = 0; k < 3; ++k) {
      c[3 * r + k] = a[3 * r] * b[k] + a[3 * r + 1] * b[3 + k] + a[3 * r + 2] * b[6 + k];
    }
  }
  return c;
}

[[nodiscard]] inline Matrix3 transpose(const Matrix3& m) noexcept {
  return {m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]};
}

[[nodiscard]] inline Vector3 rotate(const Matrix3& m, const Vector3& v) noexcept {
  return {m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
          m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
}

[[nodiscard]] inline double max_abs_difference(const Matrix3& a, const Matrix3& b) noexcept {
  double worst = 0.0;
  for (std::size_t i = 0; i < 9; ++i) worst = std::max(worst, std::abs(a[i] - b[i]));
  return worst;
}

[[nodiscard]] inline bool all_finite(const Matrix3& m) noexcept {
  return std::all_of(m.begin(), m.end(), [](double v) { return std::isfinite(v); });
}

[[nodiscard]] inline double determinant(const Matrix3& m) noexcept {
  return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
         m[2] * (m[3] * m[7] - m[4] * m[6]);
}

// A rotation read from a file is taken to this: ROS's stereo calibration
// writes R with 8 decimals, whose rounding leaves R R^T up to ~1e-8 from I.
inline constexpr double kRotationFileTolerance = 1e-6;

// Orthonormal with determinant +1, to `tolerance`.
[[nodiscard]] inline bool is_rotation(const Matrix3& m,
                                      double tolerance = kRotationTolerance) noexcept {
  return all_finite(m) && max_abs_difference(multiply(m, transpose(m)), kIdentity3) <= tolerance &&
         std::abs(determinant(m) - 1.0) <= tolerance;
}

// The rotation nearest to `m`, a rotation to kRotationFileTolerance: the
// iteration m <- m (3 I - m^T m) / 2, which converges quadratically to it
// (each step squares the distance from orthonormal).
[[nodiscard]] inline Matrix3 nearest_rotation(Matrix3 m) noexcept {
  for (int step = 0; step < 4; ++step) {
    Matrix3 correction = multiply(transpose(m), m);
    for (std::size_t i = 0; i < 9; ++i)
      correction[i] = 0.5 * ((i % 4 == 0 ? 3.0 : 0.0) - correction[i]);
    m = multiply(m, correction);
  }
  return m;
}

[[nodiscard]] inline double norm(const Vector3& v) noexcept {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// |a - b| <= tolerance * max(1, |a|, |b|).
[[nodiscard]] inline bool nearly_equal(double a, double b, double tolerance) noexcept {
  return std::abs(a - b) <= tolerance * std::max({1.0, std::abs(a), std::abs(b)});
}

// Pair checks: equal intrinsics of a rectified pair, and a translation along
// -x, relative to the values compared.
inline constexpr double kPairTolerance = 1e-9;

// Whether the cameras' rectifications (an absent one is the identity) turn
// the pair into one frame in which the right camera lies along +x of the left
// one (right_from_left's translation, rotated, along -x), as the ROS format
// and a rectifier need them to.
[[nodiscard]] inline bool rectifications_agree(const StereoCameraInfo& info) noexcept {
  const Matrix3 rect_left = info.left.rectification.value_or(kIdentity3);
  const Matrix3 rect_right = info.right.rectification.value_or(kIdentity3);
  if (max_abs_difference(multiply(transpose(rect_right), rect_left),
                         info.right_from_left.rotation) > kRotationTolerance) {
    return false;
  }
  const Vector3 offset = rotate(rect_right, info.right_from_left.translation);
  const double length = norm(offset);
  return offset[0] < 0 && std::abs(offset[1]) <= kPairTolerance * length &&
         std::abs(offset[2]) <= kPairTolerance * length;
}

}  // namespace holoscan::messages::detail
