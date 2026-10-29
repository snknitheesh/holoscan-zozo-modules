// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The geometry, without a GPU: distortion round trips for every model; the
// undistorted camera for alpha 0 and 1; Bouguet rectification of pairs with
// known R and t, whose rectified projections share rows exactly and whose
// maps send each rectified pixel to the raw pixel that sees the same point;
// the identity a rectified camera's calibration gives; synthetic images
// undistorted and rectified on the CPU, measured against the truth; and how
// far VPI's sparse warp grid strays from the dense map.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/rectify/backend.hpp>
#include <holoscan/rectify/camera_model.hpp>
#include <holoscan/rectify/rectification.hpp>
#include <holoscan/rectify/synthetic_stereo.hpp>

#include "reference_remap.hpp"
#include "test_support.hpp"

namespace holoscan::rectify {
namespace {

using enum DistortionModel;
constexpr DistortionModel kModels[] = {kBrownConrady, kRationalPolynomial, kKannalaBrandt4};

std::string model_name(DistortionModel model) { return std::string(distortion_model_name(model)); }

// Distort then undistort, and undistort then distort, over the image.
void test_distortion_round_trip() {
  for (const DistortionModel model : kModels) {
    const CameraModel camera = synthetic_stereo(640, 480, model).calibration.left;
    double worst_ideal = 0.0;
    double worst_pixel = 0.0;
    int unsolved = 0;
    for (int v = 0; v < camera.height; v += 7) {
      for (int u = 0; u < camera.width; u += 7) {
        // A pixel of the distorted image, undistorted and distorted back.
        const Point2 d{(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy};
        const auto ideal = undistort(camera, d);
        if (!ideal) {
          ++unsolved;
          continue;
        }
        const Point2 again = distort(camera, *ideal);
        worst_pixel = std::max(
            worst_pixel, std::hypot((again.x - d.x) * camera.fx, (again.y - d.y) * camera.fy));
        // An ideal point distorted and undistorted back.
        const auto back = undistort(camera, distort(camera, *ideal));
        if (back) {
          worst_ideal = std::max(worst_ideal, std::hypot(back->x - ideal->x, back->y - ideal->y));
        } else {
          ++unsolved;
        }
      }
    }
    std::cout << "  " << model_name(model) << ": pixel round trip " << worst_pixel
              << " px, normalized " << worst_ideal << ", unsolved " << unsolved << '\n';
    VERIFY_MSG(unsolved == 0, model_name(model) + ": points without an undistortion");
    VERIFY_MSG(worst_pixel < 1e-8, model_name(model) + ": pixel round trip too large");
    VERIFY_MSG(worst_ideal < 1e-9, model_name(model) + ": normalized round trip too large");
  }
}

// Alpha 0: every output pixel sees the input. Alpha 1: every input pixel lands in the output.
void test_undistortion_alpha() {
  for (const DistortionModel model : kModels) {
    const CameraModel camera = synthetic_stereo(640, 480, model).calibration.left;
    const Undistortion cropped = compute_undistortion(camera, {.alpha = 0.0});
    const RemapModel map = undistortion_remap(cropped);
    int outside = 0;
    for (int v = 0; v < camera.height; ++v) {
      for (int u = 0; u < camera.width; ++u) {
        const auto s = map.source(u, v);
        outside += !s || s->x < -0.5 || s->y < -0.5 || s->x > camera.width - 0.5 ||
                   s->y > camera.height - 0.5;
      }
    }
    VERIFY_MSG(outside == 0, model_name(model) + ": alpha 0 leaves " + std::to_string(outside) +
                                 " pixels outside the input");
    VERIFY(cropped.valid.width == camera.width && cropped.valid.height == camera.height);
    if (model == DistortionModel::kKannalaBrandt4) continue;  // Its full view is huge.
    const Undistortion full = compute_undistortion(camera, {.alpha = 1.0});
    const CameraModel& out = full.output;
    int lost = 0;
    for (int i = 0; i <= 100; ++i) {
      const double t = i / 100.0;
      for (const auto& [u, v] : {std::pair{t * (camera.width - 1), 0.0},
                                 std::pair{t * (camera.width - 1), camera.height - 1.0},
                                 std::pair{0.0, t * (camera.height - 1)},
                                 std::pair{camera.width - 1.0, t * (camera.height - 1)}}) {
        const auto ideal =
            undistort(camera, {(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy});
        const double x = out.fx * ideal->x + out.cx;
        const double y = out.fy * ideal->y + out.cy;
        lost += x < -0.5 || y < -0.5 || x > out.width - 0.5 || y > out.height - 0.5;
      }
    }
    VERIFY_MSG(lost == 0,
               model_name(model) + ": alpha 1 loses " + std::to_string(lost) + " border points");
    // An explicit camera is taken as given.
    const Undistortion fixed = compute_undistortion(
        camera, {.width = 320, .height = 240, .intrinsics = Intrinsics{200, 201, 160, 120}});
    VERIFY(fixed.output.width == 320 && fixed.output.fx == 200 && fixed.output.cy == 120);
  }
  VERIFY(test::throws<std::invalid_argument>([] {
    (void)compute_undistortion(synthetic_stereo(640, 480).calibration.left, {.alpha = 2});
  }));
}

// Random points in front of both cameras: the rectified projections share a
// row, the maps send each rectified pixel to the raw pixel of the same point,
// and Q turns the disparity back into the point's depth.
void check_pair(const StereoCalibration& calibration, const StereoRectifyOptions& options,
                const std::string& what) {
  const StereoRectification r = compute_stereo_rectification(calibration, options);
  const RemapModel left = left_remap(calibration, r);
  const RemapModel right = right_remap(calibration, r);
  const auto& pose = calibration.right_from_left;
  std::mt19937 random(7);
  std::uniform_real_distribution<double> lateral(-1.0, 1.0);
  std::uniform_real_distribution<double> depth(0.6, 8.0);
  double worst_row = 0.0;
  double worst_map = 0.0;
  double worst_depth = 0.0;
  int used = 0;
  for (int i = 0; i < 2000; ++i) {
    const double z = depth(random);
    const std::array<double, 3> p{lateral(random) * 0.5 * z, lateral(random) * 0.4 * z, z};
    auto q = rotate(pose.rotation, p);
    for (int k = 0; k < 3; ++k) q[k] += pose.translation[k];
    const auto pl = rotate(r.rect_from_left, p);
    const auto pr = rotate(r.rect_from_right, q);
    const double xl = r.fx * pl[0] / pl[2] + r.cx_left;
    const double yl = r.fx * pl[1] / pl[2] + r.cy;
    const double xr = r.fx * pr[0] / pr[2] + r.cx_right;
    const double yr = r.fx * pr[1] / pr[2] + r.cy;
    // Only points both raw cameras see.
    const auto raw = [](const CameraModel& c, const std::array<double, 3>& x) {
      const Point2 d = distort(c, {x[0] / x[2], x[1] / x[2]});
      return Point2{c.fx * d.x + c.cx, c.fy * d.y + c.cy};
    };
    const Point2 raw_left = raw(calibration.left, p);
    const Point2 raw_right = raw(calibration.right, q);
    const auto inside = [](const CameraModel& c, Point2 x) {
      return x.x >= 0 && x.y >= 0 && x.x <= c.width - 1 && x.y <= c.height - 1;
    };
    if (!inside(calibration.left, raw_left) || !inside(calibration.right, raw_right)) continue;
    ++used;
    worst_row = std::max(worst_row, std::abs(yl - yr));
    const auto sl = left.source(xl, yl);
    const auto sr = right.source(xr, yr);
    VERIFY(sl && sr);
    if (!sl || !sr) continue;
    worst_map = std::max({worst_map, std::hypot(sl->x - raw_left.x, sl->y - raw_left.y),
                          std::hypot(sr->x - raw_right.x, sr->y - raw_right.y)});
    const double d = xl - xr;
    worst_depth = std::max(worst_depth, std::abs(r.depth_from_disparity() / (d + r.doffs) - pl[2]));
    // Q: (x, y, d, 1) -> (X, Y, Z, W).
    const auto& Q = r.disparity_to_depth;
    const double w = Q[14] * d + Q[15];
    const double qz = Q[11] / w;
    worst_depth = std::max(worst_depth, std::abs(qz - pl[2]));
  }
  std::cout << "  " << what << ": " << used << " points, row error " << worst_row
            << " px, map error " << worst_map << " px, depth error " << worst_depth << " m; fx "
            << r.fx << " cx " << r.cx_left << '/' << r.cx_right << " cy " << r.cy << " baseline "
            << r.baseline << '\n';
  VERIFY_MSG(used > 200, what + ": too few points seen by both cameras");
  VERIFY_MSG(worst_row < 1e-9, what + ": rectified rows differ");
  VERIFY_MSG(worst_map < 1e-6, what + ": the map misses the raw pixel");
  VERIFY_MSG(worst_depth < 1e-9, what + ": depth from disparity is wrong");
  // The rectifying rotations are rotations.
  for (const Matrix3* m : {&r.rect_from_left, &r.rect_from_right}) {
    const Matrix3 id = multiply(*m, transpose(*m));
    for (int k = 0; k < 9; ++k) VERIFY(std::abs(id[k] - (k % 4 == 0 ? 1.0 : 0.0)) < 1e-12);
  }
  VERIFY(r.baseline > 0);
  VERIFY(options.zero_disparity ? r.doffs == 0.0 : true);
}

void test_rectification() {
  for (const DistortionModel model : kModels) {
    const SyntheticStereo scene = synthetic_stereo(640, 480, model);
    for (const double alpha : {0.0, 0.5, 1.0}) {
      if (model == DistortionModel::kKannalaBrandt4 && alpha > 0) continue;
      check_pair(scene.calibration, {.alpha = alpha},
                 model_name(model) + " alpha " + std::to_string(alpha).substr(0, 3));
    }
  }
  const SyntheticStereo scene = synthetic_stereo(640, 480);
  check_pair(scene.calibration, {.zero_disparity = false}, "separate principal points");
  check_pair(scene.calibration, {.width = 800, .height = 400}, "output 800x400");
  // A larger rotation and a baseline with a vertical part.
  StereoCalibration tilted = scene.calibration;
  tilted.right_from_left.rotation = axis_angle({0.3, 1.0, 0.2}, 0.12);
  tilted.right_from_left.translation = {-0.3, 0.05, 0.04};
  check_pair(tilted, {.alpha = 0.5}, "7 degrees, tilted baseline");

  // Swapped cameras and vertical pairs are refused.
  StereoCalibration swapped = scene.calibration;
  swapped.right_from_left.translation[0] = 0.12;
  VERIFY(test::throws<std::invalid_argument>([&] { (void)compute_stereo_rectification(swapped); }));
  StereoCalibration vertical = scene.calibration;
  vertical.right_from_left.translation = {-0.01, 0.12, 0.0};
  VERIFY(
      test::throws<std::invalid_argument>([&] { (void)compute_stereo_rectification(vertical); }));
}

// A pair that is already rectified (identity rotation, no distortion, equal
// intrinsics), as the RealSense recordings are: the maps are the identity.
void test_rectified_input_is_identity() {
  StereoCalibration c;
  c.left = CameraModel{.width = 1280,
                       .height = 720,
                       .fx = 336.98760986328125,
                       .fy = 336.98760986328125,
                       .cx = 674.053955078125,
                       .cy = 355.81134033203125,
                       .distortion_model = DistortionModel::kBrownConrady,
                       .distortion = {0, 0, 0, 0, 0}};
  c.right = c.left;
  c.right_from_left.translation = {-0.094843290746212, 0.0, 0.0};
  const StereoRectification r = compute_stereo_rectification(c);
  VERIFY(std::abs(r.fx - c.left.fx) < 1e-9);
  VERIFY(std::abs(r.cx_left - c.left.cx) < 1e-9 && std::abs(r.cy - c.left.cy) < 1e-9);
  VERIFY(std::abs(r.baseline - 0.094843290746212) < 1e-15);
  const RemapTable table = build_remap_table(left_remap(c, r), r.width, r.height);
  double worst = 0.0;
  for (int v = 0; v < r.height; ++v) {
    for (int u = 0; u < r.width; ++u) {
      const std::size_t i = (static_cast<std::size_t>(v) * r.width + u) * 2;
      worst = std::max({worst, static_cast<double>(std::abs(table.xy[i] - static_cast<float>(u))),
                        static_cast<double>(std::abs(table.xy[i + 1] - static_cast<float>(v)))});
    }
  }
  std::cout << "  identity map: largest offset " << worst << " px\n";
  VERIFY_MSG(worst < 1e-3, "the map of a rectified pair is not the identity");
  VERIFY(r.valid_left.width == 1280 && r.valid_left.height == 720);
}

// Synthetic images, undistorted and rectified on the CPU: the dots appear
// where the geometry puts them.
void test_synthetic_images() {
  for (const DistortionModel model : kModels) {
    const SyntheticStereo scene = synthetic_stereo(640, 480, model);
    const StereoRectification r = compute_stereo_rectification(scene.calibration);
    const auto left_image = left_view(scene).render(3);
    const auto right_image = right_view(scene).render(3);
    const auto left_out = test::reference_remap(
        left_image, 640, 480, 1,
        build_remap_table(left_remap(scene.calibration, r), r.width, r.height), true);
    const auto right_out = test::reference_remap(
        right_image, 640, 480, 1,
        build_remap_table(right_remap(scene.calibration, r), r.width, r.height), true);
    const AlignmentReport pair =
        check_rectified(scene, r, left_out.data(), right_out.data(), r.width, 3);
    std::cout << "  " << model_name(model) << " rectified: " << pair.dots << " dots, "
              << pair.missing << " missing, row error max " << pair.max_row_error << " rms "
              << pair.rms_row_error << " px, position error max " << pair.max_position_error
              << " px\n";
    VERIFY(pair.dots > 50 && pair.missing == 0);
    VERIFY(pair.max_row_error < 0.1);
    VERIFY(pair.max_position_error < 0.15);

    const Undistortion u = compute_undistortion(scene.calibration.left);
    const auto undistorted = test::reference_remap(
        left_image, 640, 480, 1,
        build_remap_table(undistortion_remap(u), u.output.width, u.output.height), true);
    const AlignmentReport single =
        check_undistorted(scene.plane, u, undistorted.data(), u.output.width, 3);
    std::cout << "  " << model_name(model) << " undistorted: " << single.dots
              << " dots, position error max " << single.max_position_error << " rms "
              << single.rms_position_error << " px\n";
    VERIFY(single.dots > 50 && single.missing == 0);
    VERIFY(single.max_position_error < 0.15);
    // The raw images are visibly distorted: the dots are far from the rectified positions.
    const AlignmentReport raw =
        check_rectified(scene, r, left_image.data(), right_image.data(), 640, 3);
    VERIFY(raw.dots == 0 || raw.max_position_error > 1.0);
  }
}

// VPI's VIC and PVA sample the map every 4 pixels and interpolate between.
void test_sparse_grid_error() {
  const SyntheticStereo scene = synthetic_stereo(1280, 720);
  const StereoRectification r = compute_stereo_rectification(scene.calibration);
  const RemapModel model = left_remap(scene.calibration, r);
  constexpr int kInterval = 4;
  double worst = 0.0;
  for (int v = 0; v + kInterval < r.height; v += 3) {
    for (int u = 0; u + kInterval < r.width; u += 3) {
      const int u0 = u / kInterval * kInterval;
      const int v0 = v / kInterval * kInterval;
      const double a = static_cast<double>(u - u0) / kInterval;
      const double b = static_cast<double>(v - v0) / kInterval;
      const Point2 p00 = model.clamped_source(u0, v0);
      const Point2 p10 = model.clamped_source(u0 + kInterval, v0);
      const Point2 p01 = model.clamped_source(u0, v0 + kInterval);
      const Point2 p11 = model.clamped_source(u0 + kInterval, v0 + kInterval);
      const double x =
          (1 - a) * (1 - b) * p00.x + a * (1 - b) * p10.x + (1 - a) * b * p01.x + a * b * p11.x;
      const double y =
          (1 - a) * (1 - b) * p00.y + a * (1 - b) * p10.y + (1 - a) * b * p01.y + a * b * p11.y;
      const Point2 exact = model.clamped_source(u, v);
      worst = std::max(worst, std::hypot(x - exact.x, y - exact.y));
    }
  }
  std::cout << "  4-pixel warp grid at 1280x720: largest error " << worst << " px\n";
  VERIFY(worst < 0.02);
}

}  // namespace
}  // namespace holoscan::rectify

int main() {
  namespace rc = holoscan::rectify;
  return rc::test::run({
      {"distortion round trip", rc::test_distortion_round_trip},
      {"undistortion alpha", rc::test_undistortion_alpha},
      {"Bouguet rectification", rc::test_rectification},
      {"a rectified pair maps to itself", rc::test_rectified_input_is_identity},
      {"synthetic images on the CPU", rc::test_synthetic_images},
      {"sparse warp grid", rc::test_sparse_grid_error},
  });
}
