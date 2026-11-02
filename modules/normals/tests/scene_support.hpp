// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What the kernel tests share: device buffers, clouds of the synthetic scene
// with their truth, and the statistics they are judged by.

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/normals/normals.hpp>
#include <holoscan/normals/planes.hpp>
#include <holoscan/normals/synthetic_scene.hpp>

namespace holoscan::normals::test {

inline void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

// A device copy of a host vector, or `count` uninitialized elements.
template <typename T>
class Device {
 public:
  explicit Device(std::size_t count) : count_(count) {
    cuda_check(cudaMalloc(&data_, std::max<std::size_t>(count, 1) * sizeof(T)), "cudaMalloc");
  }
  explicit Device(const std::vector<T>& host) : Device(host.size()) {
    if (!host.empty()) {
      cuda_check(cudaMemcpy(data_, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice),
                 "cudaMemcpy");
    }
  }
  ~Device() { cudaFree(data_); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  [[nodiscard]] T* get() const { return data_; }
  [[nodiscard]] std::vector<T> host() const {
    std::vector<T> out(count_);
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    if (count_ != 0) {
      cuda_check(cudaMemcpy(out.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost),
                 "cudaMemcpy");
    }
    return out;
  }

 private:
  T* data_{};
  std::size_t count_{};
};

// A frame of the scene, organized, with its truth.
struct SceneCloud {
  SyntheticScene scene;
  std::vector<float> xyz;
  std::vector<std::uint8_t> rgb;
  std::vector<std::uint8_t> surfaces;
  std::vector<float> normals;  // The true normals.
  [[nodiscard]] std::size_t size() const { return surfaces.size(); }
};

inline SceneCloud scene_cloud(const SyntheticScene& scene) {
  SceneCloud cloud;
  cloud.scene = scene;
  synthetic_scene_cloud(scene, cloud.xyz, cloud.rgb, cloud.surfaces, cloud.normals);
  return cloud;
}

inline SceneCloud scene_cloud(int width, int height, std::uint32_t frame = 0) {
  return scene_cloud(synthetic_scene(width, height, frame));
}

// The normals and curvature of `xyz` on the GPU.
struct Estimate {
  std::vector<float> normals;
  std::vector<float> curvature;
};

inline Estimate estimate(NormalEstimator& estimator, const std::vector<float>& xyz, int width,
                         int height, const NormalParams& params) {
  const std::size_t n = xyz.size() / 3;
  Device<float> points(xyz);
  Device<float> normals(3 * n);
  Device<float> curvature(n);
  cuda_check(estimator.estimate(points.get(), static_cast<std::uint32_t>(n), width, height, params,
                                normals.get(), curvature.get(), nullptr),
             "estimate");
  return Estimate{normals.host(), curvature.host()};
}

inline Estimate estimate(const std::vector<float>& xyz, int width, int height,
                         const NormalParams& params) {
  NormalEstimator estimator;
  return estimate(estimator, xyz, width, height, params);
}

// Pixels whose window of `radius` pixels shows only their own surface (or
// no depth): the neighborhood the normal sees is one surface. `radius_of`
// gives each pixel's radius from its depth when set.
template <typename RadiusOf>
std::vector<std::uint8_t> interior_mask(const SceneCloud& cloud, RadiusOf&& radius_of) {
  const int width = cloud.scene.width;
  const int height = cloud.scene.height;
  std::vector<std::uint8_t> mask(cloud.size(), 0U);
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * width + u;
      const std::uint8_t surface = cloud.surfaces[i];
      if (surface == kSurfaceNone) {
        continue;
      }
      const int radius = radius_of(cloud.xyz[3 * i + 2]);
      bool alone = true;
      for (int dv = -radius; dv <= radius && alone; ++dv) {
        for (int du = -radius; du <= radius && alone; ++du) {
          const int row = v + dv;
          const int column = u + du;
          if (row < 0 || row >= height || column < 0 || column >= width) {
            continue;
          }
          const std::uint8_t other = cloud.surfaces[static_cast<std::size_t>(row) * width + column];
          alone = other == kSurfaceNone || other == surface;
        }
      }
      mask[i] = alone ? 1U : 0U;
    }
  }
  return mask;
}

inline std::vector<std::uint8_t> interior_mask(const SceneCloud& cloud, int radius) {
  return interior_mask(cloud, [radius](float) { return radius; });
}

inline double angle_deg(const float* a, const float* b, bool unsigned_angle = false) {
  double dot = 0.0;
  for (int k = 0; k < 3; ++k) {
    dot += static_cast<double>(a[k]) * b[k];
  }
  if (unsigned_angle) {
    dot = std::fabs(dot);
  }
  return std::acos(std::clamp(dot, -1.0, 1.0)) * 180.0 / std::numbers::pi;
}

struct Stats {
  std::size_t count{};
  double median{};
  double p95{};
  double max{};
};

inline Stats stats(std::vector<double> values) {
  Stats s;
  s.count = values.size();
  if (values.empty()) {
    return s;
  }
  std::sort(values.begin(), values.end());
  s.median = values[values.size() / 2];
  s.p95 = values[std::min(values.size() - 1, static_cast<std::size_t>(0.95 * values.size()))];
  s.max = values.back();
  return s;
}

// Angle errors of the estimated normals against the truth on the pixels
// `mask` selects (and their orientation: how many point away from the true
// normal).
struct AngleReport {
  Stats angles;
  std::size_t flipped{};
  std::size_t missing{};  // Selected pixels without a normal.
};

inline AngleReport angle_report(const std::vector<float>& normals, const std::vector<float>& truth,
                                const std::vector<std::uint8_t>& mask) {
  AngleReport report;
  std::vector<double> angles;
  for (std::size_t i = 0; i < mask.size(); ++i) {
    if (mask[i] == 0U) {
      continue;
    }
    const float* n = normals.data() + 3 * i;
    if (!std::isfinite(n[0])) {
      ++report.missing;
      continue;
    }
    const double angle = angle_deg(n, truth.data() + 3 * i);
    report.flipped += angle > 90.0 ? 1U : 0U;
    angles.push_back(angle);
  }
  report.angles = stats(std::move(angles));
  return report;
}

}  // namespace holoscan::normals::test
