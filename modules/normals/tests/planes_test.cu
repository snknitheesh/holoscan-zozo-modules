// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// PlaneSegmenter without a graph, on clouds made at test time: the floor,
// the walls and the table top of the synthetic scene recovered (angle,
// offset, and how many of their points are labeled with them, and how few
// others) from organized and unorganized clouds; the GPU against the CPU
// reference, bit for bit; coplanar planes merged; determinism; empty,
// all-invalid, single-plane and plane-free clouds; and rejected settings.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <holoscan/normals/normals.hpp>
#include <holoscan/normals/planes.hpp>
#include <holoscan/normals/synthetic_scene.hpp>

#include "reference.hpp"
#include "scene_support.hpp"
#include "test_support.hpp"

namespace holoscan::normals {
namespace {

using test::Device;
using test::PlaneResult;
using test::SceneCloud;

PlaneResult segment(PlaneSegmenter& segmenter, const std::vector<float>& xyz,
                    const std::vector<float>& normals, const PlaneParams& params) {
  const std::size_t n = xyz.size() / 3;
  Device<float> points(xyz);
  Device<float> point_normals(normals);
  Device<std::uint8_t> labels(n);
  Device<float> coefficients(4U * params.max_planes);
  Device<float> centroids(3U * params.max_planes);
  Device<std::uint32_t> inliers(params.max_planes);
  Device<std::uint32_t> count(1);
  test::cuda_check(
      segmenter.segment(points.get(), point_normals.get(), static_cast<std::uint32_t>(n), params,
                        PlaneOutputs{labels.get(), coefficients.get(), centroids.get(),
                                     inliers.get(), count.get()},
                        nullptr),
      "segment");
  PlaneResult result;
  result.labels = labels.host();
  result.coefficients = coefficients.host();
  result.centroids = centroids.host();
  result.inliers = inliers.host();
  result.count = count.host()[0];
  return result;
}

PlaneResult segment(const std::vector<float>& xyz, const std::vector<float>& normals,
                    const PlaneParams& params) {
  PlaneSegmenter segmenter;
  return segment(segmenter, xyz, normals, params);
}

bool same_bits(const PlaneResult& a, const PlaneResult& b) {
  const auto same = [](const auto& x, const auto& y) {
    return x.size() == y.size() &&
           (x.empty() || std::memcmp(x.data(), y.data(), x.size() * sizeof(x[0])) == 0);
  };
  return a.count == b.count && a.labels == b.labels && a.inliers == b.inliers &&
         same(a.coefficients, b.coefficients) && same(a.centroids, b.centroids);
}

// How well the planes found in a cloud of the scene match its large planes.
struct Recovery {
  double max_angle_deg{};
  double max_offset_m{};
  double min_recall{1.0};
  double min_precision{1.0};
  bool all_found{true};
};

// `surfaces` is each labeled point's true surface.
Recovery recovery(const SyntheticScene& scene, const PlaneResult& result,
                  const std::vector<std::uint8_t>& surfaces, const char* what) {
  Recovery r;
  for (const int surface :
       {int{kSurfaceFloor}, int{kSurfaceBackWall}, int{kSurfaceSideWall}, int{kSurfaceTableTop}}) {
    float truth[4];
    synthetic_surface_plane(scene, static_cast<std::uint8_t>(surface), truth);
    // The plane found closest to the truth: within 5 degrees, the nearest.
    int best = -1;
    double best_angle = 180.0;
    double best_offset = 1e30;
    for (std::uint32_t k = 0; k < result.count; ++k) {
      const double angle = test::angle_deg(result.coefficients.data() + 4 * k, truth);
      const double offset = std::fabs(result.coefficients[4 * k + 3] - truth[3]);
      if (angle < 5.0 && offset < best_offset) {
        best_angle = angle;
        best_offset = offset;
        best = static_cast<int>(k);
      }
    }
    if (best < 0) {
      r.all_found = false;
      continue;
    }
    const double offset = std::fabs(result.coefficients[4 * best + 3] - truth[3]);
    std::size_t on_surface = 0;
    std::size_t labeled = 0;
    std::size_t both = 0;
    const auto label = static_cast<std::uint8_t>(best + 1);
    for (std::size_t i = 0; i < surfaces.size(); ++i) {
      on_surface += surfaces[i] == surface ? 1U : 0U;
      labeled += result.labels[i] == label ? 1U : 0U;
      both += surfaces[i] == surface && result.labels[i] == label ? 1U : 0U;
    }
    const double recall = static_cast<double>(both) / static_cast<double>(on_surface);
    const double precision = labeled == 0 ? 0.0 : static_cast<double>(both) / labeled;
    std::printf(
        "  %-12s surface %2d: plane %d angle=%.4f deg offset=%.2f mm inliers=%u "
        "recall=%.4f precision=%.4f\n",
        what, surface, best + 1, best_angle, 1000.0 * offset, result.inliers[best], recall,
        precision);
    r.all_found = r.all_found && best_angle < 10.0;
    r.max_angle_deg = std::max(r.max_angle_deg, best_angle);
    r.max_offset_m = std::max(r.max_offset_m, offset);
    r.min_recall = std::min(r.min_recall, recall);
    r.min_precision = std::min(r.min_precision, precision);
  }
  return r;
}

void verify_recovery(const Recovery& r) {
  VERIFY(r.all_found);
  VERIFY(r.max_angle_deg < 1.0);
  VERIFY(r.max_offset_m < 0.015);
  VERIFY(r.min_recall > 0.90);
  // Points next to an edge whose normal leans toward the plane are labeled
  // with it: the table top's rim.
  VERIFY(r.min_precision > 0.96);
}

// The floor, both walls and the table top, from organized normals.
void check_scene_organized() {
  for (const std::uint32_t frame : {0U, 40U}) {
    const SceneCloud cloud = test::scene_cloud(640, 480, frame);
    const test::Estimate normals = test::estimate(cloud.xyz, 640, 480, NormalParams{});
    const PlaneResult result = segment(cloud.xyz, normals.normals, PlaneParams{});
    std::printf("  frame %u: %u planes\n", frame, result.count);
    VERIFY(result.count >= 4U && result.count <= 8U);
    verify_recovery(recovery(cloud.scene, result, cloud.surfaces, "organized"));
    // Every label is a plane found, or no plane, or invalid where there is no
    // normal; the inlier counts are the labels'.
    std::vector<std::uint32_t> counted(256, 0);
    for (std::size_t i = 0; i < cloud.size(); ++i) {
      const std::uint8_t label = result.labels[i];
      ++counted[label];
      VERIFY((label == kLabelInvalid) == !std::isfinite(normals.normals[3 * i]));
    }
    for (std::uint32_t k = 0; k < 8; ++k) {
      VERIFY(counted[k + 1] == result.inliers[k]);
      VERIFY((k < result.count) == std::isfinite(result.coefficients[4 * k]));
      // A plane's centroid lies on it.
      if (k < result.count) {
        const float* c = result.centroids.data() + 3 * k;
        const float* p = result.coefficients.data() + 4 * k;
        VERIFY(std::fabs(p[0] * c[0] + p[1] * c[1] + p[2] * c[2] + p[3]) < 2e-6F);
        VERIFY(p[3] > 0.0F);  // Toward the camera at the origin.
      }
    }
  }
}

// The same from an unorganized cloud of the valid points, with k-nearest
// normals.
void check_scene_unorganized() {
  const SceneCloud cloud = test::scene_cloud(640, 480, 0);
  std::vector<float> xyz;
  std::vector<std::uint8_t> surfaces;
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    if (test::finite3(cloud.xyz.data() + 3 * i)) {
      xyz.insert(xyz.end(), cloud.xyz.begin() + 3 * i, cloud.xyz.begin() + 3 * i + 3);
      surfaces.push_back(cloud.surfaces[i]);
    }
  }
  NormalParams knn;
  knn.search = NeighborSearch::kKnn;
  const auto n = static_cast<int>(surfaces.size());
  const test::Estimate normals = test::estimate(xyz, n, 1, knn);
  const PlaneResult result = segment(xyz, normals.normals, PlaneParams{});
  std::printf("  unorganized: %u planes\n", result.count);
  verify_recovery(recovery(cloud.scene, result, surfaces, "unorganized"));
}

void check_cpu_reference() {
  for (const bool merge : {false, true}) {
    for (const std::uint32_t frame : {1U, 30U}) {
      const SceneCloud cloud = test::scene_cloud(160, 120, frame);
      const test::Estimate normals = test::estimate(cloud.xyz, 160, 120, NormalParams{});
      PlaneParams params;
      params.min_inliers = 60;
      params.max_planes = 10;
      params.samples = 1500;
      params.merge_coplanar = merge;
      params.merge_angle_deg = 10.0F;
      params.merge_distance = 0.3F;
      const PlaneResult gpu = segment(cloud.xyz, normals.normals, params);
      const PlaneResult cpu = test::plane_reference(cloud.xyz, normals.normals, params);
      std::printf("  reference %s frame %u: GPU %u planes, CPU %u planes\n",
                  merge ? "merged" : "unmerged", frame, gpu.count, cpu.count);
      VERIFY(gpu.count >= 3U);
      VERIFY_MSG(same_bits(gpu, cpu), "the GPU's planes and labels are the reference's");
    }
  }
}

void check_determinism() {
  const SceneCloud cloud = test::scene_cloud(640, 480, 9);
  const test::Estimate normals = test::estimate(cloud.xyz, 640, 480, NormalParams{});
  PlaneParams params;
  params.merge_coplanar = true;
  PlaneSegmenter segmenter;
  const PlaneResult first = segment(segmenter, cloud.xyz, normals.normals, params);
  const PlaneResult again = segment(segmenter, cloud.xyz, normals.normals, params);
  const PlaneResult fresh = segment(cloud.xyz, normals.normals, params);
  VERIFY(same_bits(first, again));
  VERIFY(same_bits(first, fresh));
  // Another seed finds the same planes, to within the noise.
  params.seed = 12345U;
  const PlaneResult other = segment(cloud.xyz, normals.normals, params);
  VERIFY(other.count >= 4U);
  verify_recovery(recovery(cloud.scene, other, cloud.surfaces, "seed 12345"));
}

// A noisy tilted plane with its exact normals, and random points around it.
struct Patch {
  std::vector<float> xyz;
  std::vector<float> normals;
};

Patch tilted_plane(int side, float offset, float noise, std::uint32_t seed) {
  Patch patch;
  std::mt19937 random(seed);
  std::normal_distribution<float> gauss(0.0F, noise);
  // z = 0.1 x + 2 + offset, seen from the origin.
  const float length = std::sqrt(1.0F + 0.01F);
  for (int v = 0; v < side; ++v) {
    for (int u = 0; u < side; ++u) {
      const float x = -1.0F + 2.0F * static_cast<float>(u) / static_cast<float>(side);
      const float y = -1.0F + 2.0F * static_cast<float>(v) / static_cast<float>(side);
      patch.xyz.insert(patch.xyz.end(), {x, y, 0.1F * x + 2.0F + offset + gauss(random)});
      patch.normals.insert(patch.normals.end(), {0.1F / length, 0.0F, -1.0F / length});
    }
  }
  return patch;
}

void check_edge_cases() {
  PlaneParams params;
  // Empty: no planes.
  {
    const PlaneResult result = segment({}, {}, params);
    VERIFY(result.count == 0U && result.labels.empty());
    VERIFY(std::all_of(result.coefficients.begin(), result.coefficients.end(),
                       [](float v) { return std::isnan(v); }));
    VERIFY(std::all_of(result.inliers.begin(), result.inliers.end(),
                       [](std::uint32_t v) { return v == 0U; }));
  }
  // All invalid, and valid points without normals: invalid labels, no planes.
  {
    const Patch patch = tilted_plane(40, 0.0F, 0.001F, 1U);
    std::vector<float> nan_points(patch.xyz.size(), std::numeric_limits<float>::quiet_NaN());
    std::vector<float> nan_normals(patch.normals.size(), std::numeric_limits<float>::quiet_NaN());
    for (const auto& [xyz, normals] :
         {std::pair{nan_points, patch.normals}, std::pair{patch.xyz, nan_normals}}) {
      const PlaneResult result = segment(xyz, normals, params);
      VERIFY(result.count == 0U);
      VERIFY(std::all_of(result.labels.begin(), result.labels.end(),
                         [](std::uint8_t v) { return v == kLabelInvalid; }));
    }
  }
  // One plane: found, every point on it, and the search stops there.
  {
    const Patch patch = tilted_plane(100, 0.0F, 0.002F, 2U);
    const PlaneResult result = segment(patch.xyz, patch.normals, params);
    VERIFY(result.count == 1U);
    VERIFY(std::count(result.labels.begin(), result.labels.end(), 1U) >= 9990);
    const float truth[4] = {0.1F / std::sqrt(1.01F), 0.0F, -1.0F / std::sqrt(1.01F),
                            2.0F / std::sqrt(1.01F)};
    VERIFY(test::angle_deg(result.coefficients.data(), truth) < 0.1);
    VERIFY(std::fabs(result.coefficients[3] - truth[3]) < 0.001F);
  }
  // No plane: random points with random normals.
  {
    std::mt19937 random(3U);
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    std::vector<float> xyz;
    std::vector<float> normals;
    for (int i = 0; i < 20000; ++i) {
      xyz.insert(xyz.end(), {uniform(random), uniform(random), 3.0F + uniform(random)});
      float n[3] = {uniform(random), uniform(random), uniform(random)};
      const float length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      normals.insert(normals.end(), {n[0] / length, n[1] / length, n[2] / length});
    }
    params.min_inliers = 200;
    const PlaneResult result = segment(xyz, normals, params);
    VERIFY(result.count == 0U);
    VERIFY(std::all_of(result.labels.begin(), result.labels.end(),
                       [](std::uint8_t v) { return v == kLabelNoPlane; }));
  }
}

// Two parallel planes 1.5 cm apart: two planes, or one when merged.
void check_merge() {
  Patch near = tilted_plane(80, 0.0F, 0.001F, 4U);
  const Patch far = tilted_plane(80, 0.015F, 0.001F, 5U);
  near.xyz.insert(near.xyz.end(), far.xyz.begin(), far.xyz.end());
  near.normals.insert(near.normals.end(), far.normals.begin(), far.normals.end());
  PlaneParams params;
  params.distance_threshold = 0.005F;
  const PlaneResult apart = segment(near.xyz, near.normals, params);
  VERIFY(apart.count == 2U);
  VERIFY(apart.inliers[0] + apart.inliers[1] >= 12700U);
  params.merge_coplanar = true;
  params.merge_distance = 0.03F;
  const PlaneResult merged = segment(near.xyz, near.normals, params);
  VERIFY(merged.count == 1U);
  VERIFY(merged.inliers[0] == apart.inliers[0] + apart.inliers[1]);
  VERIFY(std::count(merged.labels.begin(), merged.labels.end(), 1U) ==
         static_cast<std::ptrdiff_t>(merged.inliers[0]));
  VERIFY(std::isnan(merged.coefficients[4]));
  // Between the two.
  VERIFY(merged.coefficients[3] > apart.coefficients[3] - 0.016F &&
         merged.coefficients[3] < apart.coefficients[3] + 0.016F);
  // Too far apart to merge.
  params.merge_distance = 0.01F;
  VERIFY(segment(near.xyz, near.normals, params).count == 2U);
}

void check_settings() {
  PlaneSegmenter segmenter;
  Device<float> xyz(30);
  Device<std::uint8_t> labels(10);
  Device<float> coefficients(4 * kMaxPlanes);
  Device<float> centroids(3 * kMaxPlanes);
  Device<std::uint32_t> inliers(kMaxPlanes);
  const PlaneOutputs outputs{labels.get(), coefficients.get(), centroids.get(), inliers.get()};
  const auto rejects = [&](const PlaneParams& params, const std::string& what) {
    const char* reason = nullptr;
    VERIFY_MSG(!valid_params(params, &reason) && reason != nullptr &&
                   std::string(reason).find(what) != std::string::npos,
               "rejects " + what);
    VERIFY(segmenter.segment(xyz.get(), xyz.get(), 10, params, outputs, nullptr) ==
           cudaErrorInvalidValue);
  };
  PlaneParams bad;
  bad.max_planes = 0;
  rejects(bad, "max_planes");
  bad = PlaneParams{};
  bad.max_planes = kMaxPlanes + 1U;
  rejects(bad, "max_planes");
  bad = PlaneParams{};
  bad.distance_threshold = -1.0F;
  rejects(bad, "distance_threshold");
  bad = PlaneParams{};
  bad.max_normal_angle_deg = 91.0F;
  rejects(bad, "max_normal_angle_deg");
  bad = PlaneParams{};
  bad.iterations = 0;
  rejects(bad, "iterations");
  bad = PlaneParams{};
  bad.samples = kMaxSamples + 1U;
  rejects(bad, "samples");
  bad = PlaneParams{};
  bad.min_inliers = 2;
  rejects(bad, "min_inliers");
  bad = PlaneParams{};
  bad.refine_iterations = 9;
  rejects(bad, "refine_iterations");
  bad = PlaneParams{};
  bad.viewpoint[0] = INFINITY;
  rejects(bad, "viewpoint");
  bad = PlaneParams{};
  bad.merge_coplanar = true;
  bad.merge_distance = -1.0F;
  rejects(bad, "merge_distance");
  // A missing output.
  PlaneOutputs missing = outputs;
  missing.inliers = nullptr;
  VERIFY(segmenter.segment(xyz.get(), xyz.get(), 10, PlaneParams{}, missing, nullptr) ==
         cudaErrorInvalidValue);
}

}  // namespace
}  // namespace holoscan::normals

int main() {
  using holoscan::normals::test::TestCase;
  namespace hn = holoscan::normals;
  return holoscan::normals::test::run({
      TestCase{"scene planes, organized", hn::check_scene_organized},
      TestCase{"scene planes, unorganized", hn::check_scene_unorganized},
      TestCase{"CPU reference", hn::check_cpu_reference},
      TestCase{"determinism", hn::check_determinism},
      TestCase{"empty, invalid, one plane, no plane", hn::check_edge_cases},
      TestCase{"coplanar merge", hn::check_merge},
      TestCase{"rejected settings", hn::check_settings},
  });
}
