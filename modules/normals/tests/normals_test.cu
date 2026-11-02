// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// NormalEstimator without a graph, on clouds of the synthetic scene and
// random clouds made at test time: the scene's GPU render against its host
// version; the normals of each search against the scene's analytic normals
// (median and 95th percentile of the angle error, orientation) on organized
// and shuffled unorganized clouds; the curvature on planes and on the sphere;
// the GPU against the CPU reference on small clouds; determinism; empty,
// all-invalid, single-point and degenerate clouds; and rejected settings.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <holoscan/normals/normals.hpp>
#include <holoscan/normals/synthetic_scene.hpp>

#include "reference.hpp"
#include "scene_support.hpp"
#include "test_support.hpp"

namespace holoscan::normals {
namespace {

using test::AngleReport;
using test::Device;
using test::SceneCloud;

void print(const char* what, const AngleReport& report) {
  std::printf("  %-34s n=%zu median=%.3f deg p95=%.3f deg flipped=%zu missing=%zu\n", what,
              report.angles.count, report.angles.median, report.angles.p95, report.flipped,
              report.missing);
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

void check_scene_render() {
  const SyntheticScene scene = synthetic_scene(320, 240, 7);
  const SceneCloud cloud = test::scene_cloud(scene);
  Device<float> xyz(cloud.xyz.size());
  Device<float> depth(cloud.size());
  Device<std::uint8_t> rgb(cloud.rgb.size());
  test::cuda_check(render_synthetic_scene(scene, depth.get(), rgb.get(), xyz.get(), nullptr),
                   "render");
  VERIFY_MSG(same_bits(xyz.host(), cloud.xyz), "the GPU render is the host cloud, bit for bit");
  VERIFY(rgb.host() == cloud.rgb);
  // Every surface the tests rely on is in view.
  std::vector<std::size_t> pixels(256, 0);
  for (const std::uint8_t surface : cloud.surfaces) {
    ++pixels[surface];
  }
  for (const int surface : {int{kSurfaceFloor}, int{kSurfaceBackWall}, int{kSurfaceSideWall},
                            int{kSurfaceSphere}, int{kSurfaceTableTop}}) {
    VERIFY_MSG(pixels[surface] > 1000U, "surface " + std::to_string(surface) + " is in view");
  }
  // The true normals of flat surfaces are their planes' normals, and every
  // flat surface's points lie on its plane.
  std::size_t off_plane = 0;
  SyntheticScene clean = scene;
  clean.noise_sigma = 0.0F;
  const SceneCloud exact = test::scene_cloud(clean);
  for (std::size_t i = 0; i < exact.size(); ++i) {
    float plane[4];
    if (!synthetic_surface_plane(clean, exact.surfaces[i], plane)) {
      continue;
    }
    const float* p = exact.xyz.data() + 3 * i;
    const float* n = exact.normals.data() + 3 * i;
    off_plane +=
        std::fabs(plane[0] * p[0] + plane[1] * p[1] + plane[2] * p[2] + plane[3]) > 1e-4F ||
                test::angle_deg(plane, n) > 0.01
            ? 1U
            : 0U;
  }
  VERIFY_MSG(off_plane == 0U, std::to_string(off_plane) + " points off their true plane");
}

// The window search on 640x480: the interior of every surface within 2
// degrees (median) and 8 (95th percentile) of the truth, every normal toward
// the camera.
void check_window_accuracy() {
  const SceneCloud cloud = test::scene_cloud(640, 480);
  NormalParams params;
  const test::Estimate result = test::estimate(cloud.xyz, 640, 480, params);
  const AngleReport interior = test::angle_report(result.normals, cloud.normals,
                                                  test::interior_mask(cloud, params.window_radius));
  print("window r3, interior", interior);
  VERIFY(interior.angles.count > 200000U);
  VERIFY(interior.angles.median < 1.5);
  VERIFY(interior.angles.p95 < 5.0);
  VERIFY(interior.flipped == 0U);
  VERIFY(interior.missing == 0U);
  // Every normal faces the camera, and nearly every valid point has one.
  std::size_t valid = 0;
  std::size_t with_normal = 0;
  std::size_t away = 0;
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    const float* p = cloud.xyz.data() + 3 * i;
    const float* n = result.normals.data() + 3 * i;
    valid += test::finite3(p) ? 1U : 0U;
    if (std::isfinite(n[0])) {
      ++with_normal;
      away += n[0] * p[0] + n[1] * p[1] + n[2] * p[2] > 0.0F ? 1U : 0U;
      VERIFY(std::fabs(n[0] * n[0] + n[1] * n[1] + n[2] * n[2] - 1.0F) < 1e-5F);
    } else {
      VERIFY(std::isnan(result.curvature[i]));
    }
  }
  std::printf("  window r3: %zu of %zu valid points have a normal\n", with_normal, valid);
  VERIFY(away == 0U);
  VERIFY(with_normal >= valid * 99 / 100);
  // A larger window is smoother.
  params.window_radius = 5;
  const test::Estimate wide = test::estimate(cloud.xyz, 640, 480, params);
  const AngleReport wide_interior =
      test::angle_report(wide.normals, cloud.normals, test::interior_mask(cloud, 5));
  print("window r5, interior", wide_interior);
  VERIFY(wide_interior.angles.median < interior.angles.median);
  VERIFY(wide_interior.angles.p95 < 3.0);
}

// The k-nearest and radius searches on the scene's points in a random order,
// as an unorganized [N, 3] cloud with its invalid points left in.
void check_unorganized_accuracy() {
  const SceneCloud cloud = test::scene_cloud(640, 480);
  const std::size_t n = cloud.size();
  std::vector<std::uint32_t> order(n);
  std::iota(order.begin(), order.end(), 0U);
  std::shuffle(order.begin(), order.end(), std::mt19937(17U));
  std::vector<float> shuffled(3 * n);
  std::vector<float> truth(3 * n);
  for (std::size_t i = 0; i < n; ++i) {
    std::copy_n(cloud.xyz.begin() + 3 * order[i], 3, shuffled.begin() + 3 * i);
    std::copy_n(cloud.normals.begin() + 3 * order[i], 3, truth.begin() + 3 * i);
  }
  const auto unshuffle = [&](const std::vector<std::uint8_t>& mask) {
    std::vector<std::uint8_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = mask[order[i]];
    }
    return out;
  };
  // A pixel's neighborhood of `radius` meters spans radius * fx / z pixels.
  const auto pixels_of = [&](float radius) {
    return [radius, fx = cloud.scene.fx](float z) {
      return static_cast<int>(std::ceil(radius * fx / z)) + 1;
    };
  };

  // Where rows of points lie far apart on the surface (seen at a grazing
  // angle, far away), a neighborhood can be one row, a line, whose normal is
  // undefined. Judge the points whose neighborhood spans three rows: `spacing`
  // pixels' spacing on the surface within the radius.
  const auto judged = [&](float radius, float spacing) {
    std::vector<std::uint8_t> mask = test::interior_mask(cloud, pixels_of(radius));
    for (std::size_t i = 0; i < n; ++i) {
      const float* p = cloud.xyz.data() + 3 * i;
      const float* t = cloud.normals.data() + 3 * i;
      const float range = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
      const float facing = -(p[0] * t[0] + p[1] * t[1] + p[2] * t[2]) / range;
      if (mask[i] != 0U && !(spacing * p[2] / (cloud.scene.fx * facing) <= radius)) {
        mask[i] = 0U;
      }
    }
    return unshuffle(mask);
  };
  NormalParams knn;
  knn.search = NeighborSearch::kKnn;
  knn.k = 24;
  knn.radius = 0.1F;
  const test::Estimate by_knn = test::estimate(shuffled, static_cast<int>(n), 1, knn);
  const AngleReport knn_report = test::angle_report(by_knn.normals, truth, judged(0.02F, 2.0F));
  print("knn k24, interior", knn_report);
  VERIFY(knn_report.angles.count > 40000U);
  VERIFY(knn_report.angles.median < 3.0);
  VERIFY(knn_report.angles.p95 < 9.0);
  VERIFY(knn_report.flipped == 0U);
  VERIFY(knn_report.missing == 0U);

  NormalParams radius;
  radius.search = NeighborSearch::kRadius;
  radius.radius = 0.04F;
  const test::Estimate by_radius = test::estimate(shuffled, static_cast<int>(n), 1, radius);
  const AngleReport radius_report =
      test::angle_report(by_radius.normals, truth, judged(0.04F, 3.0F));
  print("radius 4 cm, interior", radius_report);
  VERIFY(radius_report.angles.count > 30000U);
  VERIFY(radius_report.angles.median < 1.5);
  VERIFY(radius_report.angles.p95 < 5.0);
  VERIFY(radius_report.flipped == 0U);
  VERIFY(radius_report.missing == 0U);

  // kAuto on an unorganized cloud is the k-nearest search.
  NormalParams automatic = knn;
  automatic.search = NeighborSearch::kAuto;
  VERIFY(same_bits(test::estimate(shuffled, static_cast<int>(n), 1, automatic).normals,
                   by_knn.normals));
}

// The surface variation: near zero on the flat surfaces, higher on the sphere,
// exactly so without noise.
void check_curvature() {
  for (const float noise : {0.0F, 0.001F}) {
    SyntheticScene scene = synthetic_scene(640, 480, 0);
    scene.noise_sigma = noise;
    const SceneCloud cloud = test::scene_cloud(scene);
    NormalParams params;
    params.window_radius = 5;
    const test::Estimate result = test::estimate(cloud.xyz, 640, 480, params);
    const std::vector<std::uint8_t> interior = test::interior_mask(cloud, params.window_radius);
    std::vector<double> flat;
    std::vector<double> sphere;
    for (std::size_t i = 0; i < cloud.size(); ++i) {
      if (interior[i] == 0U || !std::isfinite(result.curvature[i])) {
        continue;
      }
      float plane[4];
      if (synthetic_surface_plane(scene, cloud.surfaces[i], plane)) {
        flat.push_back(result.curvature[i]);
      } else if (cloud.surfaces[i] == kSurfaceSphere) {
        sphere.push_back(result.curvature[i]);
      }
    }
    const test::Stats f = test::stats(flat);
    const test::Stats s = test::stats(sphere);
    std::printf(
        "  curvature, noise %.1f mm: flat median %.2e p95 %.2e (n=%zu); sphere median %.2e "
        "(n=%zu)\n",
        1000.0 * noise, f.median, f.p95, f.count, s.median, s.count);
    VERIFY(f.count > 100000U && s.count > 2000U);
    if (noise == 0.0F) {
      VERIFY(f.p95 < 1e-5);
      VERIFY(s.median > 5e-4);
      VERIFY(s.median > 100.0 * f.p95);
    } else {
      VERIFY(f.median < 2e-3);
      VERIFY(s.median > 2.0 * f.median);
    }
  }
}

// Compares GPU normals with the CPU reference: the same points have normals,
// within `tolerance_deg` and the same orientation, and the same curvature
// within 1e-4.
void compare(const std::string& what, const std::vector<float>& points, const float view[3],
             const test::Estimate& gpu, const test::NormalResult& cpu, double tolerance_deg) {
  std::size_t mismatched_validity = 0;
  std::size_t compared = 0;
  std::size_t flipped = 0;
  double worst = 0.0;
  double worst_curvature = 0.0;
  for (std::size_t i = 0; i < cpu.curvature.size(); ++i) {
    const bool g = std::isfinite(gpu.normals[3 * i]);
    const bool c = std::isfinite(cpu.normals[3 * i]);
    if (g != c) {
      ++mismatched_validity;
      continue;
    }
    if (!g) {
      VERIFY(std::isnan(gpu.curvature[i]));
      continue;
    }
    ++compared;
    const double angle = test::angle_deg(gpu.normals.data() + 3 * i, cpu.normals.data() + 3 * i);
    // The orientation of a normal nearly across the line of sight is a toss-up
    // between float and double.
    const float* p = points.data() + 3 * i;
    const float* r = cpu.normals.data() + 3 * i;
    const double along =
        std::fabs((view[0] - p[0]) * r[0] + (view[1] - p[1]) * r[1] + (view[2] - p[2]) * r[2]) /
        std::sqrt((view[0] - p[0]) * (view[0] - p[0]) + (view[1] - p[1]) * (view[1] - p[1]) +
                  (view[2] - p[2]) * (view[2] - p[2]));
    flipped += angle > 90.0 && along > 1e-3 ? 1U : 0U;
    worst = std::max(worst, std::min(angle, 180.0 - angle));
    worst_curvature = std::max(worst_curvature,
                               std::fabs(static_cast<double>(gpu.curvature[i]) - cpu.curvature[i]));
  }
  std::printf("  %-34s compared=%zu worst=%.2e deg curvature=%.2e flipped=%zu\n", what.c_str(),
              compared, worst, worst_curvature, flipped);
  VERIFY_MSG(mismatched_validity == 0U, what + ": the same points have normals");
  VERIFY_MSG(compared > cpu.curvature.size() / 2, what + ": most points compared");
  VERIFY_MSG(worst < tolerance_deg, what + ": normals match the reference");
  VERIFY_MSG(flipped == 0U, what + ": orientations match the reference");
  VERIFY_MSG(worst_curvature < 1e-4, what + ": curvature matches the reference");
}

void check_cpu_reference() {
  // Organized: the scene at 96x72, two window sizes.
  const SceneCloud cloud = test::scene_cloud(96, 72, 3);
  for (const int radius : {1, 3}) {
    NormalParams params;
    params.window_radius = radius;
    compare("window r" + std::to_string(radius) + " vs CPU", cloud.xyz, params.viewpoint,
            test::estimate(cloud.xyz, 96, 72, params),
            test::window_reference(cloud.xyz, 96, 72, params), 0.05);
  }
  // Unorganized: the same points, and a random noisy blob with a viewpoint.
  std::vector<float> blob;
  std::mt19937 random(5U);
  std::normal_distribution<float> gauss(0.0F, 1.0F);
  for (int i = 0; i < 3000; ++i) {
    const float x = gauss(random);
    const float y = gauss(random);
    blob.insert(blob.end(), {x, y, 0.3F * std::sin(x) + 0.02F * gauss(random)});
  }
  blob.insert(blob.end(), {NAN, 0.0F, 0.0F});
  for (const bool knn : {true, false}) {
    NormalParams params;
    params.search = knn ? NeighborSearch::kKnn : NeighborSearch::kRadius;
    params.k = 16;
    params.radius = knn ? 0.6F : 0.3F;
    params.viewpoint[2] = 10.0F;
    const auto n = static_cast<int>(blob.size() / 3);
    compare(std::string(knn ? "knn" : "radius") + " blob vs CPU", blob, params.viewpoint,
            test::estimate(blob, n, 1, params), test::grid_reference(blob, params, knn), 0.05);
    params.radius = knn ? 0.1F : 0.15F;
    params.viewpoint[2] = 0.0F;
    const auto m = static_cast<int>(cloud.size());
    compare(std::string(knn ? "knn" : "radius") + " scene vs CPU", cloud.xyz, params.viewpoint,
            test::estimate(cloud.xyz, m, 1, params), test::grid_reference(cloud.xyz, params, knn),
            0.05);
  }
}

void check_determinism() {
  const SceneCloud cloud = test::scene_cloud(640, 480, 2);
  const auto n = static_cast<int>(cloud.size());
  for (const NeighborSearch search :
       {NeighborSearch::kWindow, NeighborSearch::kKnn, NeighborSearch::kRadius}) {
    NormalParams params;
    params.search = search;
    const int width = search == NeighborSearch::kWindow ? 640 : n;
    const int height = search == NeighborSearch::kWindow ? 480 : 1;
    NormalEstimator estimator;
    const test::Estimate first = test::estimate(estimator, cloud.xyz, width, height, params);
    const test::Estimate again = test::estimate(estimator, cloud.xyz, width, height, params);
    const test::Estimate fresh = test::estimate(cloud.xyz, width, height, params);
    VERIFY(same_bits(first.normals, again.normals) && same_bits(first.curvature, again.curvature));
    VERIFY(same_bits(first.normals, fresh.normals) && same_bits(first.curvature, fresh.curvature));
  }
}

void check_edge_cases() {
  NormalParams window;
  NormalParams knn;
  knn.search = NeighborSearch::kKnn;
  NormalEstimator estimator;
  // Empty.
  VERIFY(estimator.estimate(nullptr, 0, 0, 0, window, nullptr, nullptr, nullptr) == cudaSuccess);
  VERIFY(estimator.estimate(nullptr, 0, 0, 1, knn, nullptr, nullptr, nullptr) == cudaSuccess);
  // All invalid.
  const std::vector<float> invalid(3 * 64 * 48, std::numeric_limits<float>::quiet_NaN());
  for (const NormalParams& params : {window, knn}) {
    const test::Estimate result = test::estimate(invalid, 64, 48, params);
    VERIFY(std::all_of(result.normals.begin(), result.normals.end(),
                       [](float v) { return std::isnan(v); }));
    VERIFY(std::all_of(result.curvature.begin(), result.curvature.end(),
                       [](float v) { return std::isnan(v); }));
  }
  // A single point, and points on a line, have no normal.
  const std::vector<float> single{0.1F, 0.2F, 1.0F};
  VERIFY(std::isnan(test::estimate(single, 1, 1, window).normals[0]));
  VERIFY(std::isnan(test::estimate(single, 1, 1, knn).normals[0]));
  std::vector<float> line;
  for (int i = 0; i < 20; ++i) {
    line.insert(line.end(), {0.01F * static_cast<float>(i), 0.0F, 1.0F});
  }
  const test::Estimate on_line = test::estimate(line, 20, 1, knn);
  VERIFY(std::all_of(on_line.normals.begin(), on_line.normals.end(),
                     [](float v) { return std::isnan(v); }));
  // A flat 3x3 patch: every point has the plane's normal, toward the viewpoint.
  std::vector<float> patch;
  for (int v = 0; v < 3; ++v) {
    for (int u = 0; u < 3; ++u) {
      patch.insert(patch.end(),
                   {0.01F * static_cast<float>(u), 0.01F * static_cast<float>(v), 2.0F});
    }
  }
  NormalParams small = window;
  small.window_radius = 2;
  const test::Estimate flat = test::estimate(patch, 3, 3, small);
  for (std::size_t i = 0; i < 9; ++i) {
    VERIFY(std::fabs(flat.normals[3 * i + 2] + 1.0F) < 1e-6F);
    VERIFY(std::fabs(flat.curvature[i]) < 1e-6F);
  }
  // A window over a grid that is not the cloud's size, and unusable settings.
  Device<float> xyz(30);
  Device<float> normals(30);
  Device<float> curvature(10);
  VERIFY(estimator.estimate(xyz.get(), 10, 3, 3, window, normals.get(), curvature.get(), nullptr) ==
         cudaErrorInvalidValue);
  const auto rejects = [&](NormalParams params, const std::string& what) {
    const char* reason = nullptr;
    VERIFY_MSG(!valid_params(params, &reason) && reason != nullptr &&
                   std::string(reason).find(what) != std::string::npos,
               "rejects " + what);
    VERIFY(estimator.estimate(xyz.get(), 10, 10, 1, params, normals.get(), curvature.get(),
                              nullptr) == cudaErrorInvalidValue);
  };
  NormalParams bad;
  bad.window_radius = 0;
  rejects(bad, "window_radius");
  bad = NormalParams{};
  bad.window_radius = kMaxWindowRadius + 1;
  rejects(bad, "window_radius");
  bad = NormalParams{};
  bad.k = 2;
  rejects(bad, "k must be");
  bad = NormalParams{};
  bad.k = kMaxNeighbors + 1U;
  rejects(bad, "k must be");
  bad = NormalParams{};
  bad.radius = 0.0F;
  rejects(bad, "radius");
  bad = NormalParams{};
  bad.depth_change_factor = std::numeric_limits<float>::infinity();
  rejects(bad, "depth_change_factor");
  bad = NormalParams{};
  bad.min_neighbors = 2;
  rejects(bad, "min_neighbors");
  bad = NormalParams{};
  bad.viewpoint[1] = NAN;
  rejects(bad, "viewpoint");
}

}  // namespace
}  // namespace holoscan::normals

int main() {
  using holoscan::normals::test::TestCase;
  namespace hn = holoscan::normals;
  return holoscan::normals::test::run({
      TestCase{"synthetic scene", hn::check_scene_render},
      TestCase{"window normals against the truth", hn::check_window_accuracy},
      TestCase{"unorganized normals against the truth", hn::check_unorganized_accuracy},
      TestCase{"curvature", hn::check_curvature},
      TestCase{"CPU reference", hn::check_cpu_reference},
      TestCase{"determinism", hn::check_determinism},
      TestCase{"edge cases and settings", hn::check_edge_cases},
  });
}
