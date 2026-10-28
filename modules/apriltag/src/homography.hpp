// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Homography and linear-solve helpers shared by the host pose and the GPU decoder.

#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define TAG_DETECTION_HOST_DEVICE __host__ __device__
#else
#define TAG_DETECTION_HOST_DEVICE
#endif

namespace holoscan::apriltag {

// The ideal tag corners (-1, 1), (1, 1), (1, -1), (-1, -1), in Detection::corners' order.
TAG_DETECTION_HOST_DEVICE constexpr double ideal_x(int corner) {
  return corner == 0 || corner == 3 ? -1.0 : 1.0;
}
TAG_DETECTION_HOST_DEVICE constexpr double ideal_y(int corner) { return corner < 2 ? 1.0 : -1.0; }

// Solves the augmented N x (N + 1) system `m` by Gauss-Jordan elimination with
// partial pivoting into x[0..N-1]. False if a pivot is below `tiny`.
template <int N>
TAG_DETECTION_HOST_DEVICE inline bool solve(double (&m)[N][N + 1], double tiny, double* x) {
  using std::fabs;
  for (int col = 0; col < N; ++col) {
    int pivot = col;
    for (int row = col + 1; row < N; ++row) {
      if (fabs(m[row][col]) > fabs(m[pivot][col])) pivot = row;
    }
    if (fabs(m[pivot][col]) < tiny) {
      return false;
    }
    for (int k = 0; k <= N; ++k) {
      const double swap = m[col][k];
      m[col][k] = m[pivot][k];
      m[pivot][k] = swap;
    }
    for (int row = 0; row < N; ++row) {
      if (row == col) continue;
      const double factor = m[row][col] / m[col][col];
      for (int k = col; k <= N; ++k) m[row][k] -= factor * m[col][k];
    }
  }
  for (int i = 0; i < N; ++i) x[i] = m[i][N] / m[i][i];
  return true;
}

// The homography (row-major 3x3, h[8] = 1) taking the ideal tag corners to
// `corners`, by the direct linear transform. False if the corners are degenerate.
TAG_DETECTION_HOST_DEVICE inline bool homography_from_corners(const double (&corners)[4][2],
                                                              double (&h)[9]) {
  using std::isfinite;
  double a[8][9];
  for (int i = 0; i < 4; ++i) {
    const double x = ideal_x(i);
    const double y = ideal_y(i);
    const double u = corners[i][0];
    const double v = corners[i][1];
    const double row0[9] = {x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u, u};
    const double row1[9] = {0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v, v};
    for (int k = 0; k < 9; ++k) {
      a[2 * i][k] = row0[k];
      a[2 * i + 1][k] = row1[k];
    }
  }
  if (!solve(a, 1e-12, h)) {
    return false;
  }
  for (int i = 0; i < 8; ++i) {
    if (!isfinite(h[i])) {
      return false;
    }
  }
  h[8] = 1.0;
  return true;
}

TAG_DETECTION_HOST_DEVICE inline void project(const double (&h)[9], double x, double y,
                                              double (&out)[2]) {
  const double z = h[6] * x + h[7] * y + h[8];
  out[0] = (h[0] * x + h[1] * y + h[2]) / z;
  out[1] = (h[3] * x + h[4] * y + h[5]) / z;
}

}  // namespace holoscan::apriltag
