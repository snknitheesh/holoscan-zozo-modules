// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The header-only parts, without the SDK, CUDA or a GPU: rigid transforms
// (composition, inverse, axis-angle, errors) and the solve of the 6x6 normal
// equations, which must give the exact step for a linear problem and refuse
// one with no solution.

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

#include <holoscan/icp/rigid_transform.hpp>

#include "normal_equations.hpp"

namespace {

using holoscan::icp::compose;
using holoscan::icp::identity_transform;
using holoscan::icp::is_rigid_shaped;
using holoscan::icp::Matrix4;
using holoscan::icp::rigid_inverse;
using holoscan::icp::rotation_angle;
using holoscan::icp::transform_error;
using holoscan::icp::transform_from;
using holoscan::icp::translation_norm;
namespace detail = holoscan::icp::detail;

int failures = 0;

void check(bool condition, const std::string& what) {
  if (!condition) {
    ++failures;
    std::cerr << "check failed: " << what << '\n';
  }
}

bool near(const Matrix4& a, const Matrix4& b, double tolerance) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::fabs(a[i] - b[i]) > tolerance) return false;
  }
  return true;
}

void transforms() {
  const Matrix4 a = transform_from({0.3, -0.2, 0.1}, {1.0, -2.0, 0.5});
  const Matrix4 b = transform_from({-0.1, 0.4, 0.25}, {0.2, 0.3, -0.7});
  check(near(compose(a, rigid_inverse(a)), identity_transform(), 1e-12), "a * a^-1 is I");
  check(near(compose(rigid_inverse(compose(a, b)), compose(a, b)), identity_transform(), 1e-12),
        "the inverse of a product");
  check(std::fabs(rotation_angle(transform_from({0.0, 0.0, 0.7}, {0.0, 0.0, 0.0})) - 0.7) < 1e-12,
        "the angle of an axis-angle rotation");
  check(
      std::fabs(rotation_angle(transform_from({1e-10, 0.0, 0.0}, {0.0, 0.0, 0.0})) - 1e-10) < 1e-15,
      "a tiny rotation by its series");
  check(std::fabs(translation_norm(a) - std::sqrt(1.0 + 4.0 + 0.25)) < 1e-12, "translation norm");
  const auto error =
      transform_error(compose(a, transform_from({0.0, 0.01, 0.0}, {0.0, 0.0, 0.0})), a);
  check(std::fabs(error.rotation_rad - 0.01) < 1e-12 && error.translation < 1e-12,
        "the error of a transform turned by 0.01 rad");
  // Rotation matrices are orthonormal with determinant 1.
  const double det = a[0] * (a[5] * a[10] - a[6] * a[9]) - a[1] * (a[4] * a[10] - a[6] * a[8]) +
                     a[2] * (a[4] * a[9] - a[5] * a[8]);
  check(std::fabs(det - 1.0) < 1e-12, "a proper rotation");
  check(is_rigid_shaped(a), "a rigid transform's shape");
  Matrix4 bad = a;
  bad[14] = 1.0;
  check(!is_rigid_shaped(bad), "a last row other than 0 0 0 1");
  bad = a;
  bad[3] = std::nan("");
  check(!is_rigid_shaped(bad), "a non-finite entry");
}

// The sums of J^T J and J^T r for residuals r = J x0 + c, whose least-squares
// step from 0 is -(J^T J)^-1 J^T r: x0's components must come back.
void solve() {
  const std::array<double, 6> truth{0.01, -0.02, 0.03, 0.1, -0.2, 0.05};
  std::array<double, detail::kSums> sums{};
  // Twelve well spread rows.
  for (int i = 0; i < 12; ++i) {
    std::array<double, 6> j{};
    for (int k = 0; k < 6; ++k) {
      j[k] = std::sin(1.0 + 0.7 * i + 1.3 * k) + (k == i % 6 ? 2.0 : 0.0);
    }
    double r = 0.0;
    for (int k = 0; k < 6; ++k) r -= j[k] * truth[k];  // r(0) = -J x0, minimum at x0.
    for (int row = 0; row < 6; ++row) {
      for (int col = row; col < 6; ++col) {
        sums[detail::kJtJ + detail::upper_index(row, col)] += j[row] * j[col];
      }
      sums[detail::kJtr + row] += j[row] * r;
    }
  }
  const std::optional<std::array<double, 6>> step = detail::solve_normal_equations(sums);
  check(step.has_value(), "a well-posed system is solved");
  if (step) {
    for (int k = 0; k < 6; ++k) {
      check(std::fabs((*step)[k] - truth[k]) < 1e-7, "component " + std::to_string(k));
    }
  }
  check(detail::upper_index(0, 0) == 0 && detail::upper_index(0, 5) == 5 &&
            detail::upper_index(1, 1) == 6 && detail::upper_index(5, 5) == 20,
        "the upper triangle is indexed row by row");
  // No equations at all, and non-finite sums, have no solution.
  check(!detail::solve_normal_equations(std::array<double, detail::kSums>{}).has_value(),
        "an empty system is refused");
  std::array<double, detail::kSums> nan_sums = sums;
  nan_sums[detail::kJtr] = std::nan("");
  check(!detail::solve_normal_equations(nan_sums).has_value(), "a non-finite system is refused");
}

}  // namespace

int main() {
  transforms();
  solve();
  std::cout << (failures == 0 ? "icp_host_test: all passed" : "icp_host_test: FAILED") << '\n';
  return failures == 0 ? 0 : 1;
}
