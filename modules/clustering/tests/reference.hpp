// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The CPU references and test scenes: Euclidean clustering by breadth-first
// search, in space and over an image window; the clusters' ids, sizes and
// summaries as the module defines them; the adjusted Rand index; and clouds
// of boxes and spheres with their true objects, made at test time. Compiled
// with -ffp-contract=off, so that distances have the GPU's bits.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <numbers>
#include <random>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <holoscan/clustering/clustering.hpp>

namespace holoscan::clustering::test {

inline bool valid(const float* p) {
  return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

// As the GPU computes it: left to right, nothing contracted.
inline float squared_distance(const float* a, const float* b) {
  const float dx = a[0] - b[0];
  const float dy = a[1] - b[1];
  const float dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

// The connected component of every point (the lowest point index in it), -1
// for an invalid point: breadth-first search over a hash grid whose cells are
// a little wider than the tolerance, so that every neighbor is in the 27
// cells around a point.
inline std::vector<std::int64_t> reference_components(const std::vector<float>& xyz,
                                                      float tolerance) {
  const std::size_t n = xyz.size() / 3;
  const double cell = static_cast<double>(tolerance) * 1.001;
  const float tol2 = tolerance * tolerance;
  using Key = std::tuple<std::int64_t, std::int64_t, std::int64_t>;
  std::map<Key, std::vector<std::uint32_t>> grid;
  std::vector<Key> keys(n);
  for (std::size_t i = 0; i < n; ++i) {
    const float* p = xyz.data() + 3 * i;
    if (!valid(p)) {
      continue;
    }
    keys[i] = {static_cast<std::int64_t>(std::floor(p[0] / cell)),
               static_cast<std::int64_t>(std::floor(p[1] / cell)),
               static_cast<std::int64_t>(std::floor(p[2] / cell))};
    grid[keys[i]].push_back(static_cast<std::uint32_t>(i));
  }
  std::vector<std::int64_t> component(n, -1);
  std::deque<std::uint32_t> queue;
  for (std::size_t seed = 0; seed < n; ++seed) {
    if (component[seed] >= 0 || !valid(xyz.data() + 3 * seed)) {
      continue;
    }
    component[seed] = static_cast<std::int64_t>(seed);
    queue.push_back(static_cast<std::uint32_t>(seed));
    while (!queue.empty()) {
      const std::uint32_t i = queue.front();
      queue.pop_front();
      const float* p = xyz.data() + 3 * i;
      const auto [kx, ky, kz] = keys[i];
      for (std::int64_t dx = -1; dx <= 1; ++dx) {
        for (std::int64_t dy = -1; dy <= 1; ++dy) {
          for (std::int64_t dz = -1; dz <= 1; ++dz) {
            const auto found = grid.find({kx + dx, ky + dy, kz + dz});
            if (found == grid.end()) {
              continue;
            }
            for (const std::uint32_t j : found->second) {
              if (component[j] < 0 && squared_distance(p, xyz.data() + 3 * j) <= tol2) {
                component[j] = static_cast<std::int64_t>(seed);
                queue.push_back(j);
              }
            }
          }
        }
      }
    }
  }
  return component;
}

// The same over an organized cloud's image window, as Connectivity::kImage
// links points.
inline std::vector<std::int64_t> reference_image_components(const std::vector<float>& xyz,
                                                            int width, float tolerance,
                                                            int window) {
  const std::size_t n = xyz.size() / 3;
  const int height = static_cast<int>(n / static_cast<std::size_t>(width));
  const float tol2 = tolerance * tolerance;
  std::vector<std::int64_t> component(n, -1);
  std::deque<std::uint32_t> queue;
  for (std::size_t seed = 0; seed < n; ++seed) {
    if (component[seed] >= 0 || !valid(xyz.data() + 3 * seed)) {
      continue;
    }
    component[seed] = static_cast<std::int64_t>(seed);
    queue.push_back(static_cast<std::uint32_t>(seed));
    while (!queue.empty()) {
      const std::uint32_t i = queue.front();
      queue.pop_front();
      const float* p = xyz.data() + 3 * i;
      const int u = static_cast<int>(i % static_cast<std::uint32_t>(width));
      const int v = static_cast<int>(i / static_cast<std::uint32_t>(width));
      for (int dv = -window; dv <= window; ++dv) {
        for (int du = -window; du <= window; ++du) {
          const int uu = u + du;
          const int vv = v + dv;
          if (uu < 0 || uu >= width || vv < 0 || vv >= height) {
            continue;
          }
          const std::size_t j = static_cast<std::size_t>(vv) * width + uu;
          const float* q = xyz.data() + 3 * j;
          // The GPU tests each pair from its first point; the distance is symmetric.
          if (component[j] < 0 && valid(q) && squared_distance(p, q) <= tol2) {
            component[j] = static_cast<std::int64_t>(seed);
            queue.push_back(static_cast<std::uint32_t>(j));
          }
        }
      }
    }
  }
  return component;
}

// The labels the module gives these components: sizes within the limits,
// the max_clusters largest, by decreasing size and then lowest point index.
inline std::vector<std::int32_t> reference_labels(const std::vector<std::int64_t>& component,
                                                  const ClusterParams& params,
                                                  std::vector<std::uint32_t>* sizes = nullptr) {
  std::unordered_map<std::int64_t, std::uint32_t> count;
  for (const std::int64_t c : component) {
    if (c >= 0) {
      ++count[c];
    }
  }
  // (size, lowest index): the component's id is its lowest index, as BFS seeds it.
  std::vector<std::pair<std::uint32_t, std::int64_t>> kept;
  for (const auto& [c, size] : count) {
    if (size >= params.min_cluster_size && size <= params.max_cluster_size) {
      kept.emplace_back(size, c);
    }
  }
  std::sort(kept.begin(), kept.end(), [](const auto& a, const auto& b) {
    return a.first != b.first ? a.first > b.first : a.second < b.second;
  });
  if (kept.size() > params.max_clusters) {
    kept.resize(params.max_clusters);
  }
  std::unordered_map<std::int64_t, std::int32_t> id;
  for (std::size_t k = 0; k < kept.size(); ++k) {
    id[kept[k].second] = static_cast<std::int32_t>(k);
  }
  if (sizes != nullptr) {
    sizes->clear();
    for (const auto& [size, c] : kept) {
      sizes->push_back(size);
    }
  }
  std::vector<std::int32_t> labels(component.size(), -1);
  for (std::size_t i = 0; i < component.size(); ++i) {
    if (component[i] >= 0) {
      const auto found = id.find(component[i]);
      labels[i] = found == id.end() ? -1 : found->second;
    }
  }
  return labels;
}

// The summary of each cluster, centroid in double.
struct ReferenceCluster {
  std::uint32_t count{};
  std::array<double, 3> centroid{};
  std::array<float, 3> lo{std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::infinity()};
  std::array<float, 3> hi{-std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity()};
};

inline std::vector<ReferenceCluster> reference_clusters(const std::vector<float>& xyz,
                                                        const std::vector<std::int32_t>& labels) {
  std::int32_t k = 0;
  for (const std::int32_t label : labels) {
    k = std::max(k, label + 1);
  }
  std::vector<ReferenceCluster> clusters(static_cast<std::size_t>(k));
  for (std::size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] < 0) {
      continue;
    }
    ReferenceCluster& c = clusters[static_cast<std::size_t>(labels[i])];
    ++c.count;
    for (int axis = 0; axis < 3; ++axis) {
      const float value = xyz[3 * i + axis];
      c.centroid[axis] += value;
      c.lo[axis] = std::min(c.lo[axis], value);
      c.hi[axis] = std::max(c.hi[axis], value);
    }
  }
  for (ReferenceCluster& c : clusters) {
    for (double& value : c.centroid) {
      value /= static_cast<double>(c.count);
    }
  }
  return clusters;
}

// The adjusted Rand index of two labelings of the same points, each label
// (-1 included) one class.
inline double adjusted_rand_index(const std::vector<std::int32_t>& a,
                                  const std::vector<std::int32_t>& b) {
  std::map<std::pair<std::int32_t, std::int32_t>, double> table;
  std::map<std::int32_t, double> rows;
  std::map<std::int32_t, double> columns;
  for (std::size_t i = 0; i < a.size(); ++i) {
    table[{a[i], b[i]}] += 1.0;
    rows[a[i]] += 1.0;
    columns[b[i]] += 1.0;
  }
  const auto pairs = [](double n) { return n * (n - 1.0) / 2.0; };
  double index = 0.0;
  for (const auto& [key, n] : table) {
    index += pairs(n);
  }
  double row_sum = 0.0;
  double column_sum = 0.0;
  for (const auto& [key, n] : rows) {
    row_sum += pairs(n);
  }
  for (const auto& [key, n] : columns) {
    column_sum += pairs(n);
  }
  const double total = pairs(static_cast<double>(a.size()));
  const double expected = total == 0.0 ? 0.0 : row_sum * column_sum / total;
  const double maximum = 0.5 * (row_sum + column_sum);
  return maximum == expected ? 1.0 : (index - expected) / (maximum - expected);
}

// Whether two labelings are the same partition: a one-to-one map between
// their labels, -1 to -1.
inline bool same_partition(const std::vector<std::int32_t>& a, const std::vector<std::int32_t>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  std::unordered_map<std::int32_t, std::int32_t> forward;
  std::unordered_map<std::int32_t, std::int32_t> backward;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if ((a[i] < 0) != (b[i] < 0)) {
      return false;
    }
    const auto [f, f_new] = forward.emplace(a[i], b[i]);
    const auto [r, r_new] = backward.emplace(b[i], a[i]);
    if (f->second != b[i] || r->second != a[i]) {
      return false;
    }
  }
  return true;
}

// ---- Test scenes --------------------------------------------------------------------------------

// A 3x3 rotation, row-major, from yaw (about z), pitch (about y) and roll
// (about x), in degrees: R = Rz * Ry * Rx.
inline std::array<float, 9> rotation(double yaw, double pitch, double roll) {
  const double y = yaw * std::numbers::pi / 180.0;
  const double p = pitch * std::numbers::pi / 180.0;
  const double r = roll * std::numbers::pi / 180.0;
  const double cy = std::cos(y), sy = std::sin(y);
  const double cp = std::cos(p), sp = std::sin(p);
  const double cr = std::cos(r), sr = std::sin(r);
  const double m[9] = {cy * cp,
                       cy * sp * sr - sy * cr,
                       cy * sp * cr + sy * sr,
                       sy * cp,
                       sy * sp * sr + cy * cr,
                       sy * sp * cr - cy * sr,
                       -sp,
                       cp * sr,
                       cp * cr};
  std::array<float, 9> out{};
  for (int i = 0; i < 9; ++i) {
    out[i] = static_cast<float>(m[i]);
  }
  return out;
}

// A cloud with each point's true object (-1 for noise).
struct TruthCloud {
  std::vector<float> xyz;
  std::vector<std::int32_t> truth;

  void add(float x, float y, float z, std::int32_t object) {
    xyz.insert(xyz.end(), {x, y, z});
    truth.push_back(object);
  }
  [[nodiscard]] std::size_t size() const { return truth.size(); }
};

// The surface of a box of half extents `half`, rotated by `r` (rows: the
// box's axes in the world are its columns) about `center`, sampled on a grid
// of about `spacing`, each point moved by `jitter` at most on each axis.
inline void add_box(TruthCloud& cloud, const std::array<float, 3>& center,
                    const std::array<float, 3>& half, const std::array<float, 9>& r, float spacing,
                    float jitter, std::int32_t object, std::mt19937& rng) {
  std::uniform_real_distribution<float> noise(-jitter, jitter);
  for (int axis = 0; axis < 3; ++axis) {
    const int a = (axis + 1) % 3;
    const int b = (axis + 2) % 3;
    const int na = std::max(1, static_cast<int>(std::ceil(2.0F * half[a] / spacing)));
    const int nb = std::max(1, static_cast<int>(std::ceil(2.0F * half[b] / spacing)));
    for (const float side : {-1.0F, 1.0F}) {
      for (int i = 0; i <= na; ++i) {
        for (int j = 0; j <= nb; ++j) {
          std::array<float, 3> local{};
          local[axis] = side * half[axis];
          local[a] = -half[a] + 2.0F * half[a] * static_cast<float>(i) / static_cast<float>(na);
          local[b] = -half[b] + 2.0F * half[b] * static_cast<float>(j) / static_cast<float>(nb);
          float world[3];
          for (int k = 0; k < 3; ++k) {
            world[k] = center[k] + r[3 * k] * local[0] + r[3 * k + 1] * local[1] +
                       r[3 * k + 2] * local[2] + noise(rng);
          }
          cloud.add(world[0], world[1], world[2], object);
        }
      }
    }
  }
}

// The surface of a sphere, a Fibonacci lattice of about `spacing`.
inline void add_sphere(TruthCloud& cloud, const std::array<float, 3>& center, float radius,
                       float spacing, float jitter, std::int32_t object, std::mt19937& rng) {
  std::uniform_real_distribution<float> noise(-jitter, jitter);
  const int n =
      std::max(8, static_cast<int>(4.0 * std::numbers::pi * radius * radius / (spacing * spacing)));
  const double golden = std::numbers::pi * (3.0 - std::sqrt(5.0));
  for (int i = 0; i < n; ++i) {
    const double z = 1.0 - 2.0 * (i + 0.5) / n;
    const double rho = std::sqrt(1.0 - z * z);
    const double phi = golden * i;
    cloud.add(center[0] + static_cast<float>(radius * rho * std::cos(phi)) + noise(rng),
              center[1] + static_cast<float>(radius * rho * std::sin(phi)) + noise(rng),
              center[2] + static_cast<float>(radius * z) + noise(rng), object);
  }
}

// Isolated points: uniform in [lo, hi], each kept only when no point of the
// cloud is within `clearance`.
inline void add_noise(TruthCloud& cloud, int count, const std::array<float, 3>& lo,
                      const std::array<float, 3>& hi, float clearance, std::mt19937& rng) {
  std::uniform_real_distribution<float> unit(0.0F, 1.0F);
  const float c2 = clearance * clearance;
  for (int added = 0, tries = 0; added < count && tries < 100 * count; ++tries) {
    float p[3];
    for (int k = 0; k < 3; ++k) {
      p[k] = lo[k] + (hi[k] - lo[k]) * unit(rng);
    }
    bool clear = true;
    for (std::size_t i = 0; i < cloud.size() && clear; ++i) {
      clear = squared_distance(p, cloud.xyz.data() + 3 * i) > c2;
    }
    if (clear) {
      cloud.add(p[0], p[1], p[2], -1);
      ++added;
    }
  }
}

// The objects of the test scene; touching objects share one.
inline constexpr int kSceneObjects = 7;
inline constexpr int kSmallClump = 6;  // Five points, below most minimum sizes.

// Boxes and spheres on a removed ground, at tolerance `tol`: a rotated box; a
// box with a small box resting on it (touching: one object); a sphere, and a
// smaller one 1.6 tolerances from it (nearly touching: two objects); a small
// box; a large flat box; a clump of five points; and `noise` isolated points.
// Surfaces are sampled at half the tolerance and moved by up to `jitter`.
inline TruthCloud test_scene(float tol, float jitter, int noise, std::uint32_t seed,
                             const std::array<float, 3>& offset = {0.0F, 0.0F, 0.0F}) {
  std::mt19937 rng(seed);
  TruthCloud cloud;
  const float s = 0.5F * tol;
  const auto at = [&offset](float x, float y, float z) {
    return std::array<float, 3>{x + offset[0], y + offset[1], z + offset[2]};
  };
  const std::array<float, 9> identity = rotation(0.0, 0.0, 0.0);
  add_box(cloud, at(0.0F, 0.0F, 0.5F), {0.2F, 0.15F, 0.1F}, rotation(30.0, 10.0, 5.0), s, jitter, 0,
          rng);
  add_box(cloud, at(1.5F, 0.0F, 0.25F), {0.25F, 0.2F, 0.25F}, identity, s, jitter, 1, rng);
  add_box(cloud, at(1.5F, 0.0F, 0.6F), {0.1F, 0.1F, 0.1F}, identity, s, jitter, 1, rng);
  add_sphere(cloud, at(0.0F, 1.5F, 0.3F), 0.3F, s, jitter, 2, rng);
  add_sphere(cloud, at(0.0F, 1.5F + 0.3F + 0.2F + 1.6F * tol, 0.2F), 0.2F, s, jitter, 3, rng);
  add_box(cloud, at(1.5F, 1.5F, 0.03F), {0.03F, 0.03F, 0.03F}, identity, s, jitter, 4, rng);
  add_box(cloud, at(0.8F, -1.2F, 0.025F), {1.0F, 0.5F, 0.025F}, rotation(15.0, 0.0, 0.0), s, jitter,
          5, rng);
  for (int i = 0; i < 5; ++i) {
    const auto p = at(-1.0F + 0.3F * tol * static_cast<float>(i), 1.0F, 1.0F);
    cloud.add(p[0], p[1], p[2], kSmallClump);
  }
  add_noise(cloud, noise, at(-1.5F, -2.0F, 0.0F), at(2.5F, 3.0F, 1.5F), 2.0F * tol, rng);
  return cloud;
}

}  // namespace holoscan::clustering::test
