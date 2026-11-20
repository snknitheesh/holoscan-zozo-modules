// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Clouds of the synthetic terrain made on the host, for the tests: points on
// a regular x-y lattice at the terrain's height, with optional Gaussian
// height noise from a seeded generator, and device scratch to run the
// kernels directly.

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <holoscan/height_map/grid.hpp>
#include <holoscan/height_map/height_mapper.hpp>
#include <holoscan/height_map/synthetic_terrain.hpp>

namespace holoscan::height_map::test {

// Points every `spacing` meters over [x0, x1) x [y0, y1), offset by half a
// spacing, at the terrain's height plus noise of standard deviation `noise`;
// in the map frame.
inline std::vector<float> terrain_cloud(const Terrain& terrain, float x0, float x1, float y0,
                                        float y1, float spacing, float noise = 0.0F,
                                        std::uint32_t seed = 1U) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> gauss(0.0F, 1.0F);
  std::vector<float> xyz;
  const int nx = static_cast<int>(std::floor((x1 - x0) / spacing));
  const int ny = static_cast<int>(std::floor((y1 - y0) / spacing));
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const float x = x0 + (static_cast<float>(i) + 0.5F) * spacing;
      const float y = y0 + (static_cast<float>(j) + 0.5F) * spacing;
      const float z = terrain_height(terrain, x, y) + (noise > 0.0F ? noise * gauss(rng) : 0.0F);
      xyz.insert(xyz.end(), {x, y, z});
    }
  }
  return xyz;
}

// The inverse of a rigid transform.
inline Transform inverse(const Transform& t) {
  Transform out;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      out.m[4 * r + c] = t.m[4 * c + r];
    }
  }
  for (int r = 0; r < 3; ++r) {
    out.m[4 * r + 3] =
        -(out.m[4 * r] * t.m[3] + out.m[4 * r + 1] * t.m[7] + out.m[4 * r + 2] * t.m[11]);
  }
  return out;
}

// `xyz` moved by `t`.
inline std::vector<float> transformed(const std::vector<float>& xyz, const Transform& t) {
  std::vector<float> out(xyz.size());
  for (std::size_t i = 0; i + 2 < xyz.size(); i += 3) {
    const double x = xyz[i];
    const double y = xyz[i + 1];
    const double z = xyz[i + 2];
    for (int r = 0; r < 3; ++r) {
      out[i + r] = static_cast<float>(t.m[4 * r] * x + t.m[4 * r + 1] * y + t.m[4 * r + 2] * z +
                                      t.m[4 * r + 3]);
    }
  }
  return out;
}

inline Transform translation(double x, double y, double z) {
  Transform t;
  t.m[3] = x;
  t.m[7] = y;
  t.m[11] = z;
  return t;
}

// Device scratch for running the kernels directly.
template <typename T>
struct Scratch {
  T* data{};
  std::size_t count{};
  explicit Scratch(std::size_t n) : count(n) {
    if (cudaMalloc(&data, std::max<std::size_t>(n, 1) * sizeof(T)) != cudaSuccess) {
      throw std::runtime_error("cudaMalloc failed");
    }
  }
  explicit Scratch(const std::vector<T>& host) : Scratch(host.size()) {
    if (!host.empty() && cudaMemcpy(data, host.data(), host.size() * sizeof(T),
                                    cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("cudaMemcpy failed");
    }
  }
  ~Scratch() { cudaFree(data); }
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  [[nodiscard]] std::vector<T> host() const {
    std::vector<T> out(count);
    if (cudaDeviceSynchronize() != cudaSuccess) {
      throw std::runtime_error("cudaDeviceSynchronize failed");
    }
    if (count != 0 &&
        cudaMemcpy(out.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error("cudaMemcpy failed");
    }
    return out;
  }
};

// Every layer of a mapper, on the host.
struct HostLayers {
  std::vector<float> elevation, variance, count, min_elevation, max_elevation, time;
};

inline HostLayers read_layers(const HeightMapper& mapper) {
  const std::size_t cells = mapper.params().geometry.cells();
  Scratch<float> e(cells), v(cells), n(cells), lo(cells), hi(cells), t(cells);
  if (mapper.write_layers(MapLayers{e.data, v.data, n.data, lo.data, hi.data, t.data}, nullptr) !=
      cudaSuccess) {
    throw std::runtime_error("write_layers failed");
  }
  return HostLayers{e.host(), v.host(), n.host(), lo.host(), hi.host(), t.host()};
}

// Integrates a host cloud.
inline void update(HeightMapper& mapper, const std::vector<float>& xyz, const Transform& pose,
                   std::int64_t stamp) {
  Scratch<float> points(xyz);
  if (mapper.update(points.data, static_cast<std::uint32_t>(xyz.size() / 3), pose, stamp,
                    nullptr) != cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess) {
    throw std::runtime_error("update failed");
  }
}

// Bitwise equality, any NaN equal to any NaN.
inline bool same_floats(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!(std::isnan(a[i]) && std::isnan(b[i])) && std::memcmp(&a[i], &b[i], 4) != 0) {
      return false;
    }
  }
  return true;
}

// The center of cell (r, c).
inline double cell_x(const MapGeometry& g, int c) {
  return g.origin_x + (c + 0.5) * static_cast<double>(g.resolution);
}
inline double cell_y(const MapGeometry& g, int r) {
  return g.origin_y + (r + 0.5) * static_cast<double>(g.resolution);
}

inline double percentile(std::vector<double> values, double q) {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  return values[std::min(values.size() - 1, static_cast<std::size_t>(q * values.size()))];
}

}  // namespace holoscan::height_map::test
