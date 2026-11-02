// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The clusterer against the truth and the CPU references, on clouds made at
// test time: boxes and spheres on a removed ground, touching and nearly
// touching, of a range of sizes, with jitter and isolated noise; random
// clouds and lattices whose distances fall on the tolerance; organized clouds
// in both connectivities and the synthetic depth camera; summaries and
// oriented boxes; the size limits; determinism; and empty, all-invalid,
// all-noise, single-point and very large clouds.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/clustering/clustering.hpp>
#include <holoscan/clustering/render.hpp>
#include <holoscan/clustering/synthetic_scene.hpp>

#include "reference.hpp"
#include "test_support.hpp"

namespace holoscan::clustering {
namespace {

constexpr float kTol = 0.02F;

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

// A device copy of a host vector, freed with it.
template <typename T>
class Device {
 public:
  explicit Device(std::size_t count) : count_(count) {
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&data_), std::max<std::size_t>(count, 1U) * sizeof(T)),
        "cudaMalloc");
  }
  explicit Device(const std::vector<T>& host) : Device(host.size()) {
    check_cuda(cudaMemcpy(data_, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice),
               "cudaMemcpy");
  }
  ~Device() { cudaFree(data_); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  [[nodiscard]] T* get() const { return data_; }
  [[nodiscard]] std::vector<T> host(std::size_t count) const {
    std::vector<T> out(count);
    check_cuda(cudaMemcpy(out.data(), data_, count * sizeof(T), cudaMemcpyDeviceToHost),
               "cudaMemcpy");
    return out;
  }

 private:
  T* data_{};
  std::size_t count_{};
};

// Everything the clusterer gives for one cloud.
struct Result {
  cudaError_t status{cudaSuccess};
  std::uint32_t clusters{};
  std::uint32_t clustered{};
  std::vector<std::int32_t> labels;
  std::vector<ClusterSummary> summaries;
  std::vector<float> grouped_xyz;
  std::vector<std::int32_t> grouped_labels;
  std::vector<float> kept_xyz;
};

Result run(EuclideanClusterer& clusterer, const std::vector<float>& xyz, std::uint32_t width,
           const ClusterParams& params, bool oriented = false, cudaStream_t stream = nullptr) {
  Result r;
  const auto n = static_cast<std::uint32_t>(xyz.size() / 3);
  Device<float> points(xyz);
  r.status = clusterer.cluster(points.get(), n, width, params, r.clusters, stream);
  if (r.status != cudaSuccess) {
    return r;
  }
  check_cuda(clusterer.clustered_count(r.clustered, stream), "clustered_count");
  Device<std::int32_t> labels(n);
  Device<ClusterSummary> summaries(r.clusters);
  Device<float> grouped(std::size_t{r.clustered} * 3U);
  Device<std::int32_t> grouped_labels(r.clustered);
  Device<float> kept(std::size_t{n} * 3U);
  check_cuda(clusterer.write_labels(labels.get(), stream), "write_labels");
  check_cuda(clusterer.write_summaries(points.get(), oriented,
                                       reinterpret_cast<float*>(summaries.get()), stream),
             "write_summaries");
  check_cuda(clusterer.write_grouped(points.get(), nullptr, grouped.get(), nullptr,
                                     grouped_labels.get(), stream),
             "write_grouped");
  check_cuda(clusterer.write_kept(points.get(), nullptr, kept.get(), nullptr, stream),
             "write_kept");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  r.labels = labels.host(n);
  r.summaries = summaries.host(r.clusters);
  r.grouped_xyz = grouped.host(std::size_t{r.clustered} * 3U);
  r.grouped_labels = grouped_labels.host(r.clustered);
  r.kept_xyz = kept.host(std::size_t{n} * 3U);
  return r;
}

Result run(const std::vector<float>& xyz, std::uint32_t width, const ClusterParams& params,
           bool oriented = false) {
  EuclideanClusterer clusterer;
  return run(clusterer, xyz, width, params, oriented);
}

// The truth with the objects of fewer than `min_size` points as noise.
std::vector<std::int32_t> truth_labels(const std::vector<std::int32_t>& truth,
                                       std::uint32_t min_size) {
  std::map<std::int32_t, std::uint32_t> count;
  for (const std::int32_t t : truth) {
    ++count[t];
  }
  std::vector<std::int32_t> labels(truth);
  for (std::int32_t& label : labels) {
    if (label >= 0 && count[label] < min_size) {
      label = -1;
    }
  }
  return labels;
}

bool same_bits(const void* a, const void* b, std::size_t bytes) {
  return bytes == 0 || std::memcmp(a, b, bytes) == 0;
}

// Checks the grouped, kept and summary outputs against the labels: every
// output agrees with the labels, and each summary with the CPU's.
void check_outputs(const std::string& what, const std::vector<float>& xyz, const Result& r) {
  const std::vector<test::ReferenceCluster> ref = test::reference_clusters(xyz, r.labels);
  VERIFY_MSG(ref.size() == r.clusters, what + ": cluster count of the labels");
  std::uint64_t clustered = 0;
  for (std::size_t k = 0; k < ref.size() && k < r.summaries.size(); ++k) {
    const ClusterSummary& s = r.summaries[k];
    const std::string at = what + " cluster " + std::to_string(k);
    VERIFY_MSG(s.id == static_cast<float>(k), at + ": id");
    VERIFY_MSG(s.point_count == static_cast<float>(ref[k].count), at + ": point count");
    VERIFY_MSG(k == 0 || r.summaries[k - 1].point_count >= s.point_count, at + ": by size");
    for (int axis = 0; axis < 3; ++axis) {
      VERIFY_MSG(std::abs(s.centroid[axis] - ref[k].centroid[axis]) <=
                     2e-6 + 1e-6 * std::abs(ref[k].centroid[axis]),
                 at + ": centroid");
      VERIFY_MSG(s.min[axis] == ref[k].lo[axis] && s.max[axis] == ref[k].hi[axis], at + ": box");
    }
    clustered += ref[k].count;
  }
  VERIFY_MSG(r.clustered == clustered, what + ": clustered count");
  // Grouped: cluster by cluster, input order within one.
  std::vector<float> want_xyz;
  std::vector<std::int32_t> want_labels;
  for (std::int32_t k = 0; k < static_cast<std::int32_t>(r.clusters); ++k) {
    for (std::size_t i = 0; i < r.labels.size(); ++i) {
      if (r.labels[i] == k) {
        want_xyz.insert(want_xyz.end(), xyz.begin() + 3 * i, xyz.begin() + 3 * i + 3);
        want_labels.push_back(k);
      }
    }
  }
  VERIFY_MSG(want_labels == r.grouped_labels, what + ": grouped labels");
  VERIFY_MSG(want_xyz.size() == r.grouped_xyz.size() &&
                 same_bits(want_xyz.data(), r.grouped_xyz.data(), want_xyz.size() * 4),
             what + ": grouped points");
  bool kept = r.kept_xyz.size() == xyz.size();
  for (std::size_t i = 0; kept && i < r.labels.size(); ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      const float got = r.kept_xyz[3 * i + axis];
      kept = kept && (r.labels[i] >= 0 ? same_bits(&got, &xyz[3 * i + axis], 4) : std::isnan(got));
    }
  }
  VERIFY_MSG(kept, what + ": kept points");
}

// ---- Known scenes against the truth -------------------------------------------------------------

void check_scene_exact() {
  const test::TruthCloud scene = test::test_scene(kTol, 0.05F * kTol, 200, 7U);
  ClusterParams params{.tolerance = kTol, .min_cluster_size = 10};
  const Result r = run(scene.xyz, 0, params);
  VERIFY(r.status == cudaSuccess);
  // Every object but the five-point clump; noise points are alone.
  VERIFY_MSG(r.clusters == test::kSceneObjects - 1,
             "6 clusters, got " + std::to_string(r.clusters));
  VERIFY_MSG(test::same_partition(r.labels, truth_labels(scene.truth, 10)),
             "the labels are the true objects exactly");
  // Touching boxes are one cluster; nearly touching spheres two.
  std::int32_t box = -2;
  std::int32_t top = -2;
  std::int32_t big_sphere = -2;
  std::int32_t small_sphere = -2;
  for (std::size_t i = 0; i < scene.size(); ++i) {
    if (scene.truth[i] == 1) {
      // Object 1's first point is on the large box, its last on the small one.
      box = box == -2 ? r.labels[i] : box;
      top = r.labels[i];
    }
    big_sphere = scene.truth[i] == 2 && big_sphere == -2 ? r.labels[i] : big_sphere;
    small_sphere = scene.truth[i] == 3 && small_sphere == -2 ? r.labels[i] : small_sphere;
  }
  VERIFY_MSG(box >= 0 && box == top, "the touching boxes are one cluster");
  VERIFY_MSG(big_sphere >= 0 && small_sphere >= 0 && big_sphere != small_sphere,
             "the nearly touching spheres are two clusters");
  // The CPU reference gives the very same labels, ids included.
  const std::vector<std::int32_t> ref =
      test::reference_labels(test::reference_components(scene.xyz, kTol), params);
  VERIFY_MSG(ref == r.labels, "labels equal the CPU reference");
  check_outputs("scene", scene.xyz, r);
}

void check_scene_noisy() {
  // More jitter, and a thousand isolated points.
  const test::TruthCloud scene = test::test_scene(kTol, 0.15F * kTol, 1000, 11U);
  ClusterParams params{.tolerance = kTol, .min_cluster_size = 10};
  const Result r = run(scene.xyz, 0, params);
  VERIFY(r.status == cudaSuccess);
  VERIFY_MSG(r.clusters == test::kSceneObjects - 1,
             "6 clusters, got " + std::to_string(r.clusters));
  const double ari = test::adjusted_rand_index(r.labels, truth_labels(scene.truth, 10));
  VERIFY_MSG(ari >= 0.999, "adjusted Rand index " + std::to_string(ari));
  const std::vector<std::int32_t> ref =
      test::reference_labels(test::reference_components(scene.xyz, kTol), params);
  VERIFY_MSG(ref == r.labels, "labels equal the CPU reference");
  check_outputs("noisy scene", scene.xyz, r);
}

void check_scene_far_away() {
  // Far from the origin, where floats are coarse: still exactly the reference.
  const test::TruthCloud scene =
      test::test_scene(kTol, 0.05F * kTol, 100, 13U, {120.0F, -75.0F, 30.0F});
  const ClusterParams params{.tolerance = kTol, .min_cluster_size = 10};
  const Result r = run(scene.xyz, 0, params);
  VERIFY(r.status == cudaSuccess);
  VERIFY(r.clusters == test::kSceneObjects - 1);
  const std::vector<std::int32_t> ref =
      test::reference_labels(test::reference_components(scene.xyz, kTol), params);
  VERIFY_MSG(ref == r.labels, "labels equal the CPU reference");
  check_outputs("far scene", scene.xyz, r);
}

void check_scene_summaries() {
  // No jitter: the objects' true centers and boxes are known.
  const test::TruthCloud scene = test::test_scene(kTol, 0.0F, 0, 17U);
  const Result r =
      run(scene.xyz, 0, ClusterParams{.tolerance = kTol, .min_cluster_size = 10}, true);
  VERIFY(r.status == cudaSuccess && r.clusters == test::kSceneObjects - 1);
  check_outputs("summaries", scene.xyz, r);
  // The rotated box, object 0: its oriented box is the box.
  std::int32_t k = -1;
  for (std::size_t i = 0; i < scene.size(); ++i) {
    if (scene.truth[i] == 0) {
      k = r.labels[i];
      break;
    }
  }
  VERIFY(k >= 0);
  if (k < 0) return;
  const ClusterSummary& s = r.summaries[static_cast<std::size_t>(k)];
  const std::array<float, 9> rot = test::rotation(30.0, 10.0, 5.0);
  const float half[3] = {0.2F, 0.15F, 0.1F};
  for (int axis = 0; axis < 3; ++axis) {
    // Its axes are the box's, by decreasing extent, up to sign.
    const double dot =
        std::abs(s.obb_axes[axis][0] * rot[axis] + s.obb_axes[axis][1] * rot[3 + axis] +
                 s.obb_axes[axis][2] * rot[6 + axis]);
    VERIFY_MSG(dot >= std::cos(0.2 * std::numbers::pi / 180.0),
               "oriented axis " + std::to_string(axis) + " within 0.2 degrees");
    VERIFY_MSG(std::abs(s.obb_half_extents[axis] - half[axis]) <= 1e-3F,
               "half extent " + std::to_string(axis) + " within 1 mm");
  }
  const float center[3] = {0.0F, 0.0F, 0.5F};
  for (int axis = 0; axis < 3; ++axis) {
    VERIFY_MSG(std::abs(s.obb_center[axis] - center[axis]) <= 1e-3F, "oriented center within 1 mm");
  }
  // The touching boxes: the axis-aligned box spans both.
  for (std::size_t i = 0; i < scene.size(); ++i) {
    if (scene.truth[i] == 1) {
      const ClusterSummary& b = r.summaries[static_cast<std::size_t>(r.labels[i])];
      VERIFY(std::abs(b.min[0] - 1.25F) < 1e-5F && std::abs(b.max[0] - 1.75F) < 1e-5F);
      VERIFY(std::abs(b.min[2] - 0.0F) < 1e-5F && std::abs(b.max[2] - 0.7F) < 1e-5F);
      break;
    }
  }
  // Every cluster's oriented box: orthonormal, right-handed, holding its points.
  for (std::size_t c = 0; c < r.summaries.size(); ++c) {
    const ClusterSummary& b = r.summaries[c];
    const auto& a = b.obb_axes;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        const float dot = a[i][0] * a[j][0] + a[i][1] * a[j][1] + a[i][2] * a[j][2];
        VERIFY_MSG(std::abs(dot - (i == j ? 1.0F : 0.0F)) < 1e-5F, "orthonormal axes");
      }
    }
    const float det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
                      a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
                      a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    VERIFY_MSG(std::abs(det - 1.0F) < 1e-5F, "right-handed axes");
    for (std::size_t i = 0; i < scene.size(); ++i) {
      if (r.labels[i] != static_cast<std::int32_t>(c)) continue;
      for (int axis = 0; axis < 3; ++axis) {
        float projection = 0.0F;
        for (int k2 = 0; k2 < 3; ++k2) {
          projection += (scene.xyz[3 * i + k2] - b.obb_center[k2]) * a[axis][k2];
        }
        VERIFY_MSG(std::abs(projection) <= b.obb_half_extents[axis] + 1e-5F,
                   "the oriented box holds cluster " + std::to_string(c));
        if (std::abs(projection) > b.obb_half_extents[axis] + 1e-5F) return;
      }
    }
  }
}

void check_size_limits() {
  const test::TruthCloud scene = test::test_scene(kTol, 0.05F * kTol, 50, 19U);
  const Result all = run(scene.xyz, 0, ClusterParams{.tolerance = kTol});
  VERIFY(all.status == cudaSuccess);
  // Every object and every noise point is a cluster of its own.
  VERIFY_MSG(all.clusters == test::kSceneObjects + 50,
             "57 clusters, got " + std::to_string(all.clusters));
  std::vector<std::uint32_t> sizes;
  for (const ClusterSummary& s : all.summaries) {
    sizes.push_back(static_cast<std::uint32_t>(s.point_count));
  }
  for (const auto& [min_size, max_size, max_clusters] :
       std::vector<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>>{
           {10, kMaxPointsLimit, 1024},
           {5, 5, 1024},
           {300, 20000, 1024},
           {1, kMaxPointsLimit, 3},
           {2, 4, 1024},
           {1, 1, 10}}) {
    const ClusterParams params{.tolerance = kTol,
                               .min_cluster_size = min_size,
                               .max_cluster_size = max_size,
                               .max_clusters = max_clusters};
    const Result r = run(scene.xyz, 0, params);
    std::uint32_t want = 0;
    for (const std::uint32_t size : sizes) {
      want += size >= min_size && size <= max_size ? 1U : 0U;
    }
    want = std::min(want, max_clusters);
    const std::string what = "sizes " + std::to_string(min_size) + ".." + std::to_string(max_size) +
                             " max " + std::to_string(max_clusters);
    VERIFY_MSG(r.clusters == want, what + ": cluster count");
    for (const ClusterSummary& s : r.summaries) {
      VERIFY_MSG(s.point_count >= min_size && s.point_count <= max_size, what + ": sizes");
    }
    const std::vector<std::int32_t> ref =
        test::reference_labels(test::reference_components(scene.xyz, kTol), params);
    VERIFY_MSG(ref == r.labels, what + ": labels equal the CPU reference");
    check_outputs(what, scene.xyz, r);
  }
}

// ---- Against the CPU reference ------------------------------------------------------------------

void check_random_clouds() {
  // Uniform points near the percolation density: clusters of every size, and
  // many distances close to the tolerance.
  for (std::uint32_t seed = 1; seed <= 6; ++seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(-0.5F, 0.5F);
    const int n = 2000 + 1000 * static_cast<int>(seed);
    std::vector<float> xyz(3 * static_cast<std::size_t>(n));
    for (float& value : xyz) {
      value = unit(rng);
    }
    // A few invalid points.
    xyz[3 * 5] = std::numeric_limits<float>::quiet_NaN();
    xyz[3 * 17 + 2] = std::numeric_limits<float>::infinity();
    const float tol = 0.86F / std::cbrt(static_cast<float>(n));
    const ClusterParams params{.tolerance = tol, .min_cluster_size = 1 + seed % 3};
    const Result r = run(xyz, 0, params);
    const std::vector<std::int32_t> ref =
        test::reference_labels(test::reference_components(xyz, tol), params);
    const std::string what = "random cloud " + std::to_string(seed);
    VERIFY_MSG(r.status == cudaSuccess && ref == r.labels,
               what + ": labels equal the CPU reference");
    check_outputs(what, xyz, r);
  }
}

void check_lattice() {
  // Points on a lattice of exactly the tolerance: every neighbor distance is
  // on the limit, and only exact arithmetic decides.
  for (const float spacing : {0.02F, 0.05F, 0.1F}) {
    std::vector<float> xyz;
    for (int i = 0; i < 20; ++i) {
      for (int j = 0; j < 20; ++j) {
        for (int k = 0; k < 4; ++k) {
          xyz.insert(xyz.end(),
                     {static_cast<float>(i) * spacing + 0.3F,
                      static_cast<float>(j) * spacing - 0.7F, static_cast<float>(k) * spacing});
        }
      }
    }
    const ClusterParams params{.tolerance = spacing};
    const Result r = run(xyz, 0, params);
    const std::vector<std::int32_t> ref =
        test::reference_labels(test::reference_components(xyz, spacing), params);
    VERIFY_MSG(r.status == cudaSuccess && ref == r.labels,
               "lattice " + std::to_string(spacing) + ": labels equal the CPU reference");
  }
}

void check_image_connectivity() {
  // Organized clouds of the synthetic camera, against the window reference.
  for (const int window : {1, 2, 4}) {
    const SyntheticScene scene = synthetic_scene(160, 120, 3);
    std::vector<float> xyz;
    std::vector<std::int32_t> groups;
    synthetic_scene_cloud(scene, xyz, groups);
    const ClusterParams params{.tolerance = 0.03F,
                               .min_cluster_size = 5,
                               .connectivity = Connectivity::kImage,
                               .image_window = window};
    const Result r = run(xyz, 160, params);
    const std::vector<std::int32_t> ref = test::reference_labels(
        test::reference_image_components(xyz, 160, params.tolerance, window), params);
    const std::string what = "image window " + std::to_string(window);
    VERIFY_MSG(r.status == cudaSuccess && ref == r.labels,
               what + ": labels equal the CPU reference");
    check_outputs(what, xyz, r);
  }
  // The Euclidean mode on an organized cloud: the same as on its points.
  const SyntheticScene scene = synthetic_scene(160, 120, 5);
  std::vector<float> xyz;
  std::vector<std::int32_t> groups;
  synthetic_scene_cloud(scene, xyz, groups);
  const ClusterParams params{.tolerance = 0.03F, .min_cluster_size = 5};
  const Result organized = run(xyz, 160, params);
  const Result points = run(xyz, 0, params);
  VERIFY(organized.labels == points.labels);
  const std::vector<std::int32_t> ref =
      test::reference_labels(test::reference_components(xyz, params.tolerance), params);
  VERIFY_MSG(ref == organized.labels, "organized Euclidean: labels equal the CPU reference");
}

void check_depth_scene() {
  // The synthetic depth camera against its truth, in both connectivities.
  for (const auto& [width, height] : {std::pair{640, 480}, std::pair{1280, 720}}) {
    for (std::uint32_t frame : {0U, 40U}) {
      const SyntheticScene scene = synthetic_scene(width, height, frame);
      std::vector<float> xyz;
      std::vector<std::int32_t> groups;
      synthetic_scene_cloud(scene, xyz, groups);
      const std::uint32_t min_size = 300;
      std::vector<std::int32_t> truth(groups.size(), -1);
      std::vector<std::uint32_t> count(kSceneGroups, 0U);
      for (const std::int32_t g : groups) {
        if (g >= 0) ++count[static_cast<std::size_t>(g)];
      }
      std::uint32_t expected = 0;
      for (std::size_t g = 0; g < count.size(); ++g) {
        expected += count[g] >= min_size ? 1U : 0U;
      }
      VERIFY_MSG(count[kSceneTinyGroup] < min_size && expected == kSceneGroups - 1,
                 "every object but the tiny box is large enough");
      for (std::size_t i = 0; i < groups.size(); ++i) {
        truth[i] = groups[i] >= 0 && count[static_cast<std::size_t>(groups[i])] >= min_size
                       ? groups[i]
                       : -1;
      }
      std::vector<std::int32_t> valid_truth;
      for (std::size_t i = 0; i < groups.size(); ++i) {
        if (test::valid(xyz.data() + 3 * i)) valid_truth.push_back(truth[i]);
      }
      std::vector<std::vector<std::int32_t>> results;
      for (const Connectivity connectivity : {Connectivity::kEuclidean, Connectivity::kImage}) {
        const ClusterParams params{
            .tolerance = 0.025F, .min_cluster_size = min_size, .connectivity = connectivity};
        const Result r = run(xyz, static_cast<std::uint32_t>(width), params);
        const std::string what = std::to_string(width) + "x" + std::to_string(height) + " frame " +
                                 std::to_string(frame) +
                                 (connectivity == Connectivity::kImage ? " image" : " euclidean");
        VERIFY_MSG(r.clusters == expected, what + ": " + std::to_string(r.clusters) +
                                               " clusters, want " + std::to_string(expected));
        std::vector<std::int32_t> valid_labels;
        for (std::size_t i = 0; i < groups.size(); ++i) {
          if (test::valid(xyz.data() + 3 * i)) valid_labels.push_back(r.labels[i]);
        }
        const double ari = test::adjusted_rand_index(valid_labels, valid_truth);
        VERIFY_MSG(ari >= 0.99, what + ": adjusted Rand index " + std::to_string(ari));
        results.push_back(valid_labels);
      }
      const double agreement = test::adjusted_rand_index(results[0], results[1]);
      VERIFY_MSG(agreement >= 0.99, "image and Euclidean agree: " + std::to_string(agreement));
    }
  }
}

// ---- Determinism and edge cases -----------------------------------------------------------------

void check_determinism() {
  const test::TruthCloud scene = test::test_scene(kTol, 0.15F * kTol, 500, 23U);
  const ClusterParams params{.tolerance = kTol, .min_cluster_size = 3};
  EuclideanClusterer clusterer;
  cudaStream_t stream{};
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
  const Result first = run(clusterer, scene.xyz, 0, params, true);
  for (int i = 0; i < 3; ++i) {
    EuclideanClusterer fresh;
    const Result again =
        run(i == 0 ? clusterer : fresh, scene.xyz, 0, params, true, i == 2 ? stream : nullptr);
    VERIFY_MSG(again.labels == first.labels, "same labels on every run");
    VERIFY_MSG(again.summaries.size() == first.summaries.size() &&
                   same_bits(again.summaries.data(), first.summaries.data(),
                             first.summaries.size() * sizeof(ClusterSummary)),
               "same summary bits on every run");
    VERIFY_MSG(again.grouped_xyz == first.grouped_xyz, "same grouped points on every run");
  }
  cudaStreamDestroy(stream);
  // A smaller cloud next, then the large one again: scratch reuse changes nothing.
  (void)run(clusterer, std::vector<float>(scene.xyz.begin(), scene.xyz.begin() + 3000), 0, params);
  const Result after = run(clusterer, scene.xyz, 0, params, true);
  VERIFY_MSG(
      after.labels == first.labels && same_bits(after.summaries.data(), first.summaries.data(),
                                                first.summaries.size() * sizeof(ClusterSummary)),
      "same results after a smaller cloud");
}

void check_edge_cases() {
  const ClusterParams params{.tolerance = kTol};
  {
    const Result r = run({}, 0, params);
    VERIFY_MSG(r.status == cudaSuccess && r.clusters == 0 && r.clustered == 0, "empty cloud");
  }
  {
    std::vector<float> xyz(3000, std::numeric_limits<float>::quiet_NaN());
    xyz[7] = std::numeric_limits<float>::infinity();
    const Result r = run(xyz, 0, params);
    VERIFY_MSG(r.status == cudaSuccess && r.clusters == 0, "all-invalid cloud: no clusters");
    VERIFY_MSG(
        std::all_of(r.labels.begin(), r.labels.end(), [](std::int32_t l) { return l == -1; }),
        "all-invalid cloud: every label -1");
    const Result image =
        run(xyz, 100, ClusterParams{.tolerance = kTol, .connectivity = Connectivity::kImage});
    VERIFY_MSG(image.status == cudaSuccess && image.clusters == 0, "all-invalid organized cloud");
  }
  {
    // All noise: points a meter apart.
    std::vector<float> xyz;
    for (int i = 0; i < 400; ++i) {
      xyz.insert(xyz.end(), {static_cast<float>(i % 20), static_cast<float>(i / 20), 0.5F});
    }
    const Result none = run(xyz, 0, ClusterParams{.tolerance = kTol, .min_cluster_size = 2});
    VERIFY_MSG(none.status == cudaSuccess && none.clusters == 0, "all noise: no clusters");
    const Result each = run(xyz, 0, ClusterParams{.tolerance = kTol, .max_clusters = 100});
    VERIFY_MSG(each.clusters == 100, "all noise with min 1: max_clusters singletons");
    bool order = true;
    for (std::size_t i = 0; i < 400; ++i) {
      order = order && each.labels[i] == (i < 100 ? static_cast<std::int32_t>(i) : -1);
    }
    VERIFY_MSG(order, "equal sizes go by lowest point index");
  }
  {
    // One point; and one huge cluster of a million points.
    const Result one = run({0.1F, 0.2F, 0.3F}, 0, params);
    VERIFY_MSG(one.clusters == 1 && one.labels == std::vector<std::int32_t>{0} &&
                   one.summaries[0].point_count == 1.0F && one.summaries[0].centroid[1] == 0.2F,
               "a single point");
    std::vector<float> plane;
    plane.reserve(3U * 1000U * 1000U);
    for (int i = 0; i < 1000; ++i) {
      for (int j = 0; j < 1000; ++j) {
        plane.insert(plane.end(), {0.01F * static_cast<float>(i), 0.01F * static_cast<float>(j),
                                   1.0F + 0.001F * static_cast<float>((i * 7 + j * 3) % 5)});
      }
    }
    const Result huge = run(plane, 1000, params, true);
    VERIFY_MSG(huge.status == cudaSuccess && huge.clusters == 1 && huge.clustered == 1000000U,
               "one huge cluster");
    if (huge.clusters == 1) {
      const ClusterSummary& s = huge.summaries[0];
      VERIFY(s.point_count == 1.0e6F);
      VERIFY(std::abs(s.centroid[0] - 4.995F) < 1e-4F && std::abs(s.centroid[1] - 4.995F) < 1e-4F);
      VERIFY(s.min[0] == 0.0F && s.max[0] == 0.01F * 999.0F);
      VERIFY(std::abs(std::abs(s.obb_axes[2][2]) - 1.0F) < 1e-4F);
    }
    check_outputs("huge cluster", plane, huge);
    const Result image =
        run(plane, 1000, ClusterParams{.tolerance = kTol, .connectivity = Connectivity::kImage});
    VERIFY_MSG(image.clusters == 1 && image.labels == huge.labels, "one huge image cluster");
    const Result capped =
        run(plane, 1000, ClusterParams{.tolerance = kTol, .max_cluster_size = 999999});
    VERIFY_MSG(capped.clusters == 0, "too large a cluster is dropped");
  }
  {
    // Points beyond the grid's range are not clustered.
    const Result r = run({1.0e30F, 0.0F, 0.0F, 1.0e30F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F}, 0, params);
    VERIFY_MSG(r.labels == (std::vector<std::int32_t>{-1, -1, 0}), "points beyond the grid");
  }
  {
    EuclideanClusterer clusterer;
    std::uint32_t k = 0;
    Device<float> xyz(std::vector<float>(30, 0.0F));
    const auto refused = [&](const ClusterParams& p, std::uint32_t count, std::uint32_t width) {
      return clusterer.cluster(xyz.get(), count, width, p, k, nullptr) == cudaErrorInvalidValue;
    };
    VERIFY(refused(ClusterParams{.tolerance = 0.0F}, 10, 0));
    VERIFY(refused(ClusterParams{.tolerance = std::numeric_limits<float>::quiet_NaN()}, 10, 0));
    VERIFY(refused(ClusterParams{.tolerance = std::numeric_limits<float>::infinity()}, 10, 0));
    VERIFY(refused(ClusterParams{.min_cluster_size = 0}, 10, 0));
    VERIFY(refused(ClusterParams{.min_cluster_size = 5, .max_cluster_size = 4}, 10, 0));
    VERIFY(refused(ClusterParams{.max_clusters = 0}, 10, 0));
    VERIFY(refused(ClusterParams{.max_clusters = kMaxClustersLimit + 1}, 10, 0));
    VERIFY(refused(ClusterParams{.connectivity = Connectivity::kImage}, 10, 0));
    VERIFY(refused(ClusterParams{.connectivity = Connectivity::kImage, .image_window = 0}, 10, 5));
    VERIFY(refused(ClusterParams{.connectivity = Connectivity::kImage, .image_window = 9}, 10, 5));
    VERIFY(refused(ClusterParams{}, 10, 3));
    VERIFY(refused(ClusterParams{}, kMaxPointsLimit + 1U, 0));
    VERIFY(!valid_params(ClusterParams{.tolerance = -1.0F}) && valid_params(ClusterParams{}));
  }
}

void check_render() {
  // Labels to colors, and the boxes drawn over them.
  const std::vector<std::int32_t> labels{-1, 0, 1, 2, 0};
  Device<std::int32_t> device_labels(labels);
  Device<std::uint8_t> rgb(15);
  check_cuda(colorize_labels(device_labels.get(), 5, rgb.get(), nullptr), "colorize_labels");
  const std::vector<std::uint8_t> got = rgb.host(15);
  for (std::size_t i = 0; i < labels.size(); ++i) {
    std::uint8_t want[3];
    label_color(labels[i], want);
    VERIFY_MSG(std::equal(want, want + 3, got.begin() + 3 * i), "label colors");
  }
  std::uint8_t a[3];
  std::uint8_t b[3];
  label_color(0, a);
  label_color(1, b);
  VERIFY_MSG(!std::equal(a, a + 3, b), "neighboring ids have different colors");
  // A box straight ahead of a 64x48 camera draws pixels, and only inside the image.
  ClusterSummary box{};
  box.min[0] = -0.25F;
  box.min[1] = -0.25F;
  box.min[2] = 1.0F;
  box.max[0] = 0.25F;
  box.max[1] = 0.25F;
  box.max[2] = 2.0F;
  for (int j = 11; j < kSummaryFields; ++j) {
    reinterpret_cast<float*>(&box)[j] = std::numeric_limits<float>::quiet_NaN();
  }
  Device<ClusterSummary> summary(std::vector<ClusterSummary>{box});
  Device<std::uint8_t> image(64 * 48 * 3);
  check_cuda(cudaMemset(image.get(), 0, 64 * 48 * 3), "cudaMemset");
  check_cuda(draw_cluster_boxes(reinterpret_cast<const float*>(summary.get()), 1, 50.0F, 50.0F,
                                31.5F, 23.5F, 64, 48, image.get(), nullptr),
             "draw_cluster_boxes");
  const std::vector<std::uint8_t> pixels = image.host(64 * 48 * 3);
  const auto lit = [&](int u, int v) { return pixels[3 * (v * 64 + u)] != 0U; };
  // The front face's corners project to (19, 11) and (44, 36).
  VERIFY_MSG(lit(31, 11) && lit(31, 36) && lit(19, 23) && lit(44, 23), "the front face is drawn");
  VERIFY_MSG(!lit(31, 23) && !lit(2, 2), "inside and outside are not");
  VERIFY(draw_cluster_boxes(nullptr, 0, 0.0F, 1.0F, 0.0F, 0.0F, 64, 48, image.get(), nullptr) ==
         cudaErrorInvalidValue);
}

}  // namespace
}  // namespace holoscan::clustering

int main() {
  using holoscan::clustering::test::TestCase;
  namespace c = holoscan::clustering;
  return holoscan::clustering::test::run({
      {"scene: exact objects, CPU reference", c::check_scene_exact},
      {"scene: jitter and noise", c::check_scene_noisy},
      {"scene: far from the origin", c::check_scene_far_away},
      {"scene: summaries and oriented boxes", c::check_scene_summaries},
      {"size limits and max_clusters", c::check_size_limits},
      {"random clouds against the CPU reference", c::check_random_clouds},
      {"lattices on the tolerance", c::check_lattice},
      {"image connectivity against the CPU reference", c::check_image_connectivity},
      {"synthetic depth camera against its truth", c::check_depth_scene},
      {"determinism", c::check_determinism},
      {"edge cases", c::check_edge_cases},
      {"rendering", c::check_render},
  });
}
