// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The 6x6 normal equations of one Gauss-Newton step, as the GPU sums them,
// their solution, and the update of the transform by the step. Plain C++ on
// C arrays, compiled for the GPU, where the registration runs them, and for
// the host, where the tests check them; free of CUDA headers.

#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define HOLOSCAN_ICP_HOST_DEVICE __host__ __device__
#else
#define HOLOSCAN_ICP_HOST_DEVICE
#endif

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
HOLOSCAN_ICP_HOST_DEVICE constexpr int upper_index(int row, int col) {
  return row * 6 - row * (row - 1) / 2 + (col - row);
}

// Writes into `x` the update [w, v] (rotation as axis-angle w, then
// translation v) that minimizes the linearized cost:
// (J^T W J + damping) x = -J^T W r. A small damping proportional to the
// largest diagonal keeps a degenerate geometry (a single plane, for
// point-to-plane) solvable without moving along its unconstrained
// directions. Returns false if the matrix is not positive definite even so.
HOLOSCAN_ICP_HOST_DEVICE inline bool solve_normal_equations(const double* sums, double x[6]) {
  using std::fmax;
  using std::isfinite;
  using std::sqrt;
  double a[6][6];
  for (int row = 0; row < 6; ++row) {
    for (int col = row; col < 6; ++col) {
      a[row][col] = sums[kJtJ + upper_index(row, col)];
      a[col][row] = a[row][col];
    }
  }
  double largest = 0.0;
  for (int i = 0; i < 6; ++i) {
    largest = fmax(largest, a[i][i]);
  }
  if (!(largest > 0.0) || !isfinite(largest)) {
    return false;
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
      return false;
    }
    a[j][j] = sqrt(diagonal);
    for (int i = j + 1; i < 6; ++i) {
      double value = a[i][j];
      for (int k = 0; k < j; ++k) {
        value -= a[i][k] * a[j][k];
      }
      a[i][j] = value / a[j][j];
    }
  }
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
  for (int i = 0; i < 6; ++i) {
    if (!isfinite(x[i])) {
      return false;
    }
  }
  return true;
}

// transform <- exp(step) * transform, for a row-major 4x4 rigid transform:
// the rotation by axis-angle step[0..2] (Rodrigues), then the translation
// step[3..5], after the transform.
HOLOSCAN_ICP_HOST_DEVICE inline void apply_step(const double step[6], double transform[16]) {
  using std::cos;
  using std::sin;
  using std::sqrt;
  const double x = step[0];
  const double y = step[1];
  const double z = step[2];
  const double theta = sqrt(x * x + y * y + z * z);
  // sin(theta) / theta and (1 - cos(theta)) / theta^2, by series near zero.
  double a = 1.0 - theta * theta / 6.0;
  double b = 0.5 - theta * theta / 24.0;
  if (theta > 1e-8) {
    a = sin(theta) / theta;
    b = (1.0 - cos(theta)) / (theta * theta);
  }
  // R = I + a [w]x + b [w]x^2.
  const double r[9] = {
      1.0 - b * (y * y + z * z), -a * z + b * x * y,        a * y + b * x * z,
      a * z + b * x * y,         1.0 - b * (x * x + z * z), -a * x + b * y * z,
      -a * y + b * x * z,        a * x + b * y * z,         1.0 - b * (x * x + y * y)};
  double out[12];
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 4; ++col) {
      double sum = 0.0;
      for (int k = 0; k < 3; ++k) {
        sum += r[3 * row + k] * transform[4 * k + col];
      }
      out[4 * row + col] = sum;
    }
    out[4 * row + 3] += step[3 + row];
  }
  for (int i = 0; i < 12; ++i) {
    transform[i] = out[i];
  }
}

}  // namespace holoscan::icp::detail
