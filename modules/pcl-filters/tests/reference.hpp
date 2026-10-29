// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// CPU references of the filters, written plainly from their definitions, and
// test clouds. Compiled with -ffp-contract=off, so the same expressions give
// the same bits as the kernels, which are compiled without fused
// multiply-adds.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include <holoscan/pcl_filters/filters.hpp>

namespace holoscan::pcl_filters::test {

inline bool valid(const float* p) {
  return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

inline std::vector<std::uint8_t> crop_reference(const std::vector<float>& xyz,
                                                const CropBoxParams& params) {
  const std::size_t n = xyz.size() / 3;
  std::vector<std::uint8_t> keep(n, 0U);
  for (std::size_t i = 0; i < n; ++i) {
    const float* q = xyz.data() + 3 * i;
    if (!valid(q)) {
      continue;
    }
    float p[3] = {q[0], q[1], q[2]};
    if (params.use_transform) {
      const float* m = params.transform.matrix;
      for (int row = 0; row < 3; ++row) {
        p[row] = m[4 * row] * q[0] + m[4 * row + 1] * q[1] + m[4 * row + 2] * q[2] + m[4 * row + 3];
      }
    }
    bool inside = true;
    for (int axis = 0; axis < 3; ++axis) {
      inside = inside && p[axis] >= params.min[axis] && p[axis] <= params.max[axis];
    }
    const float squared = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    inside = inside && squared >= params.min_distance * params.min_distance &&
             squared <= params.max_distance * params.max_distance;
    keep[i] = inside != params.keep_outside ? 1U : 0U;
  }
  return keep;
}

// The voxel of a point, as a sortable tuple, or false when it has none.
inline bool voxel_of(const float* p, float size, std::int64_t index[3]) {
  if (!valid(p)) {
    return false;
  }
  for (int axis = 0; axis < 3; ++axis) {
    const float cell = std::floor(p[axis] / size);
    if (!(cell >= -static_cast<float>(kVoxelIndexRange) &&
          cell < static_cast<float>(kVoxelIndexRange))) {
      return false;
    }
    index[axis] = static_cast<std::int64_t>(cell);
  }
  return true;
}

struct VoxelResult {
  std::vector<float> xyz;
  std::vector<std::uint8_t> rgb;
};

inline VoxelResult voxel_reference(const std::vector<float>& xyz,
                                   const std::vector<std::uint8_t>& rgb, float size,
                                   VoxelMode mode) {
  struct Entry {
    std::int64_t voxel[3];
    std::uint32_t index;
  };
  std::vector<Entry> entries;
  for (std::uint32_t i = 0; i < xyz.size() / 3; ++i) {
    Entry entry{};
    if (voxel_of(xyz.data() + 3 * i, size, entry.voxel)) {
      entry.index = i;
      entries.push_back(entry);
    }
  }
  std::stable_sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
    return std::lexicographical_compare(a.voxel, a.voxel + 3, b.voxel, b.voxel + 3);
  });
  VoxelResult result;
  for (std::size_t start = 0; start < entries.size();) {
    std::size_t end = start;
    double sum[3] = {};
    std::uint32_t color[3] = {};
    while (end < entries.size() &&
           std::equal(entries[end].voxel, entries[end].voxel + 3, entries[start].voxel)) {
      const std::uint32_t i = entries[end].index;
      for (int axis = 0; axis < 3; ++axis) {
        sum[axis] += static_cast<double>(xyz[3 * i + axis]);
      }
      if (!rgb.empty()) {
        for (int channel = 0; channel < 3; ++channel) {
          color[channel] += rgb[3 * i + channel];
        }
      }
      ++end;
    }
    const auto points = static_cast<std::uint32_t>(end - start);
    const std::uint32_t first = entries[start].index;
    for (int axis = 0; axis < 3; ++axis) {
      result.xyz.push_back(mode == VoxelMode::kCentroid
                               ? static_cast<float>(sum[axis] / static_cast<double>(points))
                               : xyz[3 * first + axis]);
    }
    if (!rgb.empty()) {
      for (int channel = 0; channel < 3; ++channel) {
        result.rgb.push_back(static_cast<std::uint8_t>((color[channel] + points / 2U) / points));
      }
    }
    start = end;
  }
  return result;
}

// Each valid point's mean distance to its k nearest other points within
// `radius`, NaN without k of them; by brute force.
inline std::vector<float> knn_mean_reference(const std::vector<float>& xyz, std::uint32_t k,
                                             float radius) {
  const std::size_t n = xyz.size() / 3;
  std::vector<float> mean(n, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> distances;
  for (std::size_t i = 0; i < n; ++i) {
    const float* p = xyz.data() + 3 * i;
    if (!valid(p)) {
      continue;
    }
    distances.clear();
    for (std::size_t j = 0; j < n; ++j) {
      const float* q = xyz.data() + 3 * j;
      if (j == i || !valid(q)) {
        continue;
      }
      const float ex = q[0] - p[0];
      const float ey = q[1] - p[1];
      const float ez = q[2] - p[2];
      const float distance = std::sqrt(ex * ex + ey * ey + ez * ez);
      if (distance <= radius) {
        distances.push_back(distance);
      }
    }
    if (distances.size() < k) {
      continue;
    }
    std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
    float sum = 0.0F;
    for (std::uint32_t j = 0; j < k; ++j) {
      sum += distances[j];
    }
    mean[i] = sum / static_cast<float>(k);
  }
  return mean;
}

// mean + multiplier * sample deviation of the finite values.
inline double threshold_reference(const std::vector<float>& mean, double multiplier) {
  double count = 0.0;
  double sum = 0.0;
  double squares = 0.0;
  for (const float value : mean) {
    if (std::isfinite(value)) {
      count += 1.0;
      sum += value;
      squares += static_cast<double>(value) * value;
    }
  }
  if (count < 1.0) {
    return 0.0;
  }
  const double variance = count > 1.0 ? (squares - sum * sum / count) / (count - 1.0) : 0.0;
  return sum / count + multiplier * std::sqrt(std::max(variance, 0.0));
}

inline std::vector<std::uint8_t> radius_reference(const std::vector<float>& xyz, float radius,
                                                  std::uint32_t min_neighbors) {
  const std::size_t n = xyz.size() / 3;
  std::vector<std::uint8_t> keep(n, 0U);
  const float squared_radius = radius * radius;
  for (std::size_t i = 0; i < n; ++i) {
    const float* p = xyz.data() + 3 * i;
    if (!valid(p)) {
      continue;
    }
    std::uint32_t neighbors = 0;
    for (std::size_t j = 0; j < n && neighbors < min_neighbors; ++j) {
      const float* q = xyz.data() + 3 * j;
      if (j == i || !valid(q)) {
        continue;
      }
      const float ex = q[0] - p[0];
      const float ey = q[1] - p[1];
      const float ez = q[2] - p[2];
      neighbors += ex * ex + ey * ey + ez * ez <= squared_radius ? 1U : 0U;
    }
    keep[i] = neighbors >= min_neighbors ? 1U : 0U;
  }
  return keep;
}

// A deterministic generator for test clouds.
class Random {
 public:
  explicit Random(std::uint64_t seed) : state_(seed) {}
  std::uint64_t next() {
    state_ += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state_;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
  }
  // Uniform in [lo, hi).
  float uniform(float lo, float hi) {
    return lo + (hi - lo) * static_cast<float>(next() >> 40U) * (1.0F / 16777216.0F);
  }

 private:
  std::uint64_t state_;
};

}  // namespace holoscan::pcl_filters::test
