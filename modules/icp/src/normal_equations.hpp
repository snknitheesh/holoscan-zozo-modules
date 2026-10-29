// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The 6x6 normal equations of one Gauss-Newton step, as the GPU sums them,
// and their solution on the host. Header-only and free of CUDA, so the host
// tests check it on its own.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>

namespace holoscan::icp::detail {

// The sums one association pass produces, in this order: the upper triangle
// of J^T W J row by row (21), J^T W r (6), the sum of squared point distances
// of the correspondences (1) and their count (1), padded to 32.
inline constexpr int kJtJ = 0;
inline constexpr int kJtr = 21;
inline constexpr int kSquaredDistance = 27;
inline constexpr int kCount = 28;
inline constexpr int kSums = 32;

// The index into the upper triangle of row `row` and column `col` >= `row`.
constexpr int upper_index(int row, int col) { return row * 6 - row * (row - 1) / 2 + (col - row); }

// The update x = [w, v] (rotation as axis-angle w, then translation v) that
// minimizes the linearized cost: (J^T W J + damping) x = -J^T W r. A small
// damping proportional to the largest diagonal keeps a degenerate geometry
// (a single plane, for point-to-plane) solvable without moving along its
// unconstrained directions. Returns nothing if the matrix is not positive
// definite even so.
inline std::optional<std::array<double, 6>> solve_normal_equations(
    const std::array<double, kSums>& sums) {
  double a[6][6];
  for (int row = 0; row < 6; ++row) {
    for (int col = row; col < 6; ++col) {
      a[row][col] = sums[kJtJ + upper_index(row, col)];
      a[col][row] = a[row][col];
    }
  }
  double largest = 0.0;
  for (int i = 0; i < 6; ++i) {
    largest = std::fmax(largest, a[i][i]);
  }
  if (!(largest > 0.0) || !std::isfinite(largest)) {
    return std::nullopt;
  }
  for (int i = 0; i < 6; ++i) {
    a[i][i] += 1e-9 * largest;
  }
  // Cholesky: a = L L^T, L in the lower triangle.
  for (int j = 0; j < 6; ++j) {
    double diagonal = a[j][j];
    for (int k = 0; k < j; ++k) {
      diagonal -= a[j][k] * a[j][k];
    }
    if (!(diagonal > 1e-15 * largest)) {
      return std::nullopt;
    }
    a[j][j] = std::sqrt(diagonal);
    for (int i = j + 1; i < 6; ++i) {
      double value = a[i][j];
      for (int k = 0; k < j; ++k) {
        value -= a[i][k] * a[j][k];
      }
      a[i][j] = value / a[j][j];
    }
  }
  std::array<double, 6> x{};
  // L y = -b, then L^T x = y.
  for (int i = 0; i < 6; ++i) {
    double value = -sums[kJtr + i];
    for (int k = 0; k < i; ++k) {
      value -= a[i][k] * x[k];
    }
    x[i] = value / a[i][i];
  }
  for (int i = 5; i >= 0; --i) {
    double value = x[i];
    for (int k = i + 1; k < 6; ++k) {
      value -= a[k][i] * x[k];
    }
    x[i] = value / a[i][i];
  }
  for (const double value : x) {
    if (!std::isfinite(value)) {
      return std::nullopt;
    }
  }
  return x;
}

}  // namespace holoscan::icp::detail
