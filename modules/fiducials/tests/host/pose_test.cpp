// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Projection, undistortion and planar poses: the convention (z into the target), exact poses
// from exact points with and without distortion, markers seen obliquely and from afar (the
// planar ambiguity), noisy corners, and too few or collinear points.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <holoscan/fiducials/pose.hpp>
#include <holoscan/fiducials/synthetic_scene.hpp>

#include "test_support.hpp"

namespace holoscan::fiducials {
namespace {

using Matrix3 = std::array<double, 9>;

Matrix3 rotation(double yaw, double pitch, double roll) {
  const double cy = std::cos(yaw), sy = std::sin(yaw);
  const double cp = std::cos(pitch), sp = std::sin(pitch);
  const double cr = std::cos(roll), sr = std::sin(roll);
  const Matrix3 y{cy, 0, sy, 0, 1, 0, -sy, 0, cy};
  const Matrix3 x{1, 0, 0, 0, cp, -sp, 0, sp, cp};
  const Matrix3 z{cr, -sr, 0, sr, cr, 0, 0, 0, 1};
  const auto mul = [](const Matrix3& a, const Matrix3& b) {
    Matrix3 c{};
    for (int r = 0; r < 3; ++r) {
      for (int k = 0; k < 3; ++k) {
        for (int i = 0; i < 3; ++i) c[r * 3 + k] += a[r * 3 + i] * b[i * 3 + k];
      }
    }
    return c;
  };
  return mul(y, mul(x, z));
}

const CameraIntrinsics kCamera{900.0, 905.0, 639.5, 359.5, {}};
// A camera with strong barrel distortion.
CameraIntrinsics distorted() {
  return {900.0, 905.0, 641.2, 357.9, {-0.28, 0.09, 0.0007, -0.0004, -0.01}};
}

double translation_error(const Pose& a, const Pose& b) {
  return std::hypot(a.translation[0] - b.translation[0], a.translation[1] - b.translation[1],
                    a.translation[2] - b.translation[2]);
}

// Target x right and y down as printed, z into it: a marker facing the camera (identity
// rotation) shows its top-left corner up and to the left.
void check_convention() {
  const Pose facing{.rotation = {1, 0, 0, 0, 1, 0, 0, 0, 1}, .translation = {0, 0, 1}};
  const auto corners = marker_object_corners(0.1);
  const Point2 top_left = project(kCamera, facing, corners[0]);
  const Point2 bottom_right = project(kCamera, facing, corners[2]);
  VERIFY(top_left[0] < kCamera.cx && top_left[1] < kCamera.cy);
  VERIFY(bottom_right[0] > kCamera.cx && bottom_right[1] > kCamera.cy);
  // The target's z points away from the camera.
  VERIFY(facing.rotation[8] > 0.0);
}

void check_undistort() {
  double worst = 0.0;
  for (double x = -0.6; x <= 0.6; x += 0.1) {
    for (double y = -0.35; y <= 0.35; y += 0.07) {
      const Point2 pixel = project(distorted(), Point3{x, y, 1.0});
      const Point2 back = undistort(distorted(), pixel);
      worst = std::max(worst, std::hypot(back[0] - x, back[1] - y));
    }
  }
  VERIFY_MSG(worst < 1e-6, "undistort inverts project: " + std::to_string(worst));
}

void check_exact_board_poses() {
  std::mt19937 random(7);
  std::uniform_real_distribution<double> angle(-0.6, 0.6);
  std::vector<Point2> board;
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 8; ++x) board.push_back({0.04 * x, 0.04 * y});
  }
  double worst_t = 0.0;
  double worst_r = 0.0;
  for (const CameraIntrinsics& camera : {kCamera, distorted()}) {
    for (int trial = 0; trial < 30; ++trial) {
      const Pose truth{.rotation = rotation(angle(random), angle(random), 3.0 * angle(random)),
                       .translation = {-0.14 + 0.1 * angle(random), -0.1 + 0.1 * angle(random),
                                       0.5 + std::abs(angle(random))}};
      std::vector<Point2> image;
      for (const Point2& p : board) image.push_back(project(camera, truth, {p[0], p[1], 0.0}));
      const auto pose = solve_planar_pose(board, image, camera);
      VERIFY(pose.has_value());
      if (!pose) continue;
      worst_t = std::max(worst_t, translation_error(*pose, truth));
      worst_r = std::max(worst_r, rotation_error_deg(pose->rotation, truth.rotation));
      VERIFY(pose->reprojection_error < 1e-4);
    }
  }
  std::cout << "  exact boards: translation " << worst_t << " m, rotation " << worst_r << " deg\n";
  VERIFY(worst_t < 1e-6 && worst_r < 1e-4);
}

// Four corners of a small, far, tilted marker: the pose must be the true one, not its mirror.
void check_marker_ambiguity() {
  int right = 0;
  int trials = 0;
  for (double tilt = -0.9; tilt <= 0.9; tilt += 0.15) {
    for (const double distance : {0.5, 1.5, 3.0}) {
      const Pose truth{.rotation = rotation(tilt, 0.4 * tilt, 0.3),
                       .translation = {0.1, -0.05, distance}};
      Marker marker;
      const auto corners = marker_object_corners(0.08);
      for (int i = 0; i < 4; ++i) marker.corners[i] = project(kCamera, truth, corners[i]);
      const auto pose = marker_pose(marker, kCamera, 0.08);
      ++trials;
      right += pose && rotation_error_deg(pose->rotation, truth.rotation) < 1e-3 &&
               translation_error(*pose, truth) < 1e-6;
    }
  }
  VERIFY_MSG(right == trials, std::to_string(right) + " of " + std::to_string(trials) +
                                  " exact marker poses recovered");
}

// Corners off by up to a tenth of a pixel: a 100-pixel marker's pose stays close.
void check_noisy_marker() {
  std::mt19937 random(3);
  std::uniform_real_distribution<double> noise(-0.1, 0.1);
  double worst_t = 0.0;
  double worst_r = 0.0;
  for (int trial = 0; trial < 50; ++trial) {
    const double tilt = -0.6 + 1.2 * trial / 50.0;
    const Pose truth{.rotation = rotation(tilt, -0.5 * tilt, 0.1 * trial),
                     .translation = {0.05, 0.02, 0.9}};
    Marker marker;
    const auto corners = marker_object_corners(0.1);
    for (int i = 0; i < 4; ++i) {
      const Point2 p = project(kCamera, truth, corners[i]);
      marker.corners[i] = {p[0] + noise(random), p[1] + noise(random)};
    }
    const auto pose = marker_pose(marker, kCamera, 0.1);
    VERIFY(pose.has_value());
    if (!pose) continue;
    worst_t = std::max(worst_t, translation_error(*pose, truth) / 0.9);
    worst_r = std::max(worst_r, rotation_error_deg(pose->rotation, truth.rotation));
  }
  std::cout << "  noisy markers: translation " << 100 * worst_t << "%, rotation " << worst_r
            << " deg\n";
  VERIFY(worst_t < 0.005 && worst_r < 2.0);
}

void check_degenerate() {
  const std::vector<Point2> three{{0, 0}, {1, 0}, {0, 1}};
  VERIFY(!solve_planar_pose(three, three, kCamera));
  const std::vector<Point2> line{{0, 0}, {1, 0}, {2, 0}, {3, 0}};
  const std::vector<Point2> image{{100, 100}, {200, 100}, {300, 100}, {400, 100}};
  VERIFY(!solve_planar_pose(line, image, kCamera));
  const std::vector<Point2> four{{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  // A square seen as a line: whatever comes back, nothing breaks.
  (void)solve_planar_pose(four, image, kCamera);
  VERIFY(test::throws<std::invalid_argument>(
      [] { validate_intrinsics(CameraIntrinsics{0.0, 900, 320, 240, {}}); }));
  VERIFY(test::throws<std::invalid_argument>(
      [] { validate_intrinsics(CameraIntrinsics{900, 900, 320, 240, {NAN, 0, 0, 0, 0}}); }));
}

}  // namespace
}  // namespace holoscan::fiducials

int main() {
  namespace f = holoscan::fiducials;
  return f::test::run({
      {"pose convention", f::check_convention},
      {"undistort inverts project", f::check_undistort},
      {"exact board poses, with distortion", f::check_exact_board_poses},
      {"marker poses: the planar ambiguity", f::check_marker_ambiguity},
      {"noisy marker corners", f::check_noisy_marker},
      {"degenerate input", f::check_degenerate},
  });
}
