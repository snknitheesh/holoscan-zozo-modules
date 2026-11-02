// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Test helpers: device buffers, frames of the synthetic scene rendered at
// test time, meshes copied to the host, and their errors against the scene's
// exact surfaces.

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/tsdf/marching_cubes.hpp>
#include <holoscan/tsdf/synthetic_scene.hpp>
#include <holoscan/tsdf/tsdf_volume.hpp>

namespace holoscan::tsdf::test {

inline void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

template <typename T>
class Device {
 public:
  explicit Device(std::size_t count) : count_(count) {
    cuda_check(
        cudaMalloc(reinterpret_cast<void**>(&data_), std::max<std::size_t>(count, 1) * sizeof(T)),
        "cudaMalloc");
  }
  ~Device() { cudaFree(data_); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  [[nodiscard]] T* get() const noexcept { return data_; }
  [[nodiscard]] std::vector<T> host(std::size_t count) const {
    std::vector<T> out(count);
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    if (count != 0) {
      cuda_check(cudaMemcpy(out.data(), data_, count * sizeof(T), cudaMemcpyDeviceToHost),
                 "cudaMemcpy");
    }
    return out;
  }
  [[nodiscard]] std::vector<T> host() const { return host(count_); }
  void upload(const std::vector<T>& values) {
    cuda_check(cudaMemcpy(data_, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice),
               "cudaMemcpy");
  }

 private:
  T* data_{};
  std::size_t count_{};
};

// One rendered frame on the GPU (float32 meters) and on the host.
struct Frame {
  int width{};
  int height{};
  CameraIntrinsics intrinsics;
  Matrix4 pose{};
  Device<float> depth;
  Device<std::uint8_t> rgb;
  std::vector<float> host_depth;
  std::vector<std::uint8_t> host_rgb;

  Frame(int w, int h)
      : width(w),
        height(h),
        depth(static_cast<std::size_t>(w) * h),
        rgb(static_cast<std::size_t>(w) * h * 3) {}
};

inline constexpr DepthNoise kNoise{.sigma = 0.001F, .sigma_per_m2 = 0.002F, .seed = 0};

// Frame `frame` of `frames` of the orbit around `kind`, with noise seeded by
// the frame.
inline void render(Frame& f, SceneKind kind, std::uint32_t frame, std::uint32_t frames,
                   const DepthNoise& noise) {
  f.intrinsics = synthetic_intrinsics(f.width, f.height);
  f.pose = orbit_pose(kind, frame, frames);
  DepthNoise seeded = noise;
  seeded.seed = noise.seed * 7919U + frame;
  cuda_check(render_scene(synthetic_scene(kind), f.pose, f.intrinsics, f.width, f.height, 3.0F,
                          seeded, DepthType::kFloat32, f.depth.get(), f.rgb.get(), nullptr),
             "render_scene");
  f.host_depth = f.depth.host();
  f.host_rgb = f.rgb.host();
}

inline DepthImage depth_image(const Frame& f) {
  return DepthImage{f.depth.get(), DepthType::kFloat32, f.width, f.height, 1.0F};
}

inline IntegrateParams integrate_params(const Frame& f) {
  return IntegrateParams{.intrinsics = f.intrinsics, .depth_min = 0.1F, .depth_max = 3.0F};
}

// Integrates `frames` frames of the orbit into `volume`.
inline void integrate_orbit(TsdfVolume& volume, SceneKind kind, std::uint32_t first,
                            std::uint32_t count, std::uint32_t frames, int width, int height,
                            const DepthNoise& noise) {
  Frame f(width, height);
  for (std::uint32_t i = first; i < first + count; ++i) {
    render(f, kind, i, frames, noise);
    const ColorImage color{f.rgb.get(), 3, false};
    cuda_check(volume.integrate(depth_image(f), &color, f.pose, integrate_params(f), nullptr),
               "integrate");
  }
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

struct HostMesh {
  std::uint32_t counts[kMeshCountsSize]{};
  std::vector<float> vertices;
  std::vector<float> normals;
  std::vector<std::uint8_t> colors;
  std::vector<std::uint32_t> triangles;
};

inline HostMesh extract(const TsdfVolume& volume, std::uint32_t max_vertices = 1U << 20,
                        std::uint32_t max_triangles = 1U << 21) {
  MeshExtractor extractor(max_vertices, max_triangles);
  Device<float> vertices(std::size_t{max_vertices} * 3);
  Device<float> normals(std::size_t{max_vertices} * 3);
  Device<std::uint8_t> colors(std::size_t{max_vertices} * 3);
  Device<std::uint32_t> triangles(std::size_t{max_triangles} * 3);
  Device<std::uint32_t> counts(kMeshCountsSize);
  const MeshBuffers out{vertices.get(), normals.get(),
                        volume.config().color ? colors.get() : nullptr, triangles.get(),
                        counts.get()};
  cuda_check(extractor.extract(volume, out, nullptr), "extract");
  HostMesh mesh;
  const std::vector<std::uint32_t> c = counts.host();
  std::copy(c.begin(), c.end(), mesh.counts);
  mesh.vertices = vertices.host(std::size_t{mesh.counts[kCountVertices]} * 3);
  mesh.normals = normals.host(std::size_t{mesh.counts[kCountVertices]} * 3);
  if (volume.config().color) {
    mesh.colors = colors.host(std::size_t{mesh.counts[kCountVertices]} * 3);
  }
  mesh.triangles = triangles.host(std::size_t{mesh.counts[kCountTriangles]} * 3);
  return mesh;
}

struct ErrorStats {
  std::size_t count{};
  double mean{};
  double max{};
  double p95{};
};

inline ErrorStats stats(std::vector<double> errors) {
  ErrorStats s;
  s.count = errors.size();
  if (errors.empty()) {
    return s;
  }
  double sum = 0.0;
  for (const double e : errors) {
    sum += e;
  }
  s.mean = sum / static_cast<double>(errors.size());
  std::sort(errors.begin(), errors.end());
  s.max = errors.back();
  s.p95 = errors[std::min(errors.size() - 1, errors.size() * 95 / 100)];
  return s;
}

// The distances of points (x, y, z packed) to the scene's surfaces.
inline ErrorStats surface_error(const SyntheticScene& scene, const std::vector<float>& points) {
  std::vector<double> errors;
  errors.reserve(points.size() / 3);
  for (std::size_t i = 0; i + 2 < points.size(); i += 3) {
    errors.push_back(std::abs(scene_distance(scene, &points[i])));
  }
  return stats(std::move(errors));
}

}  // namespace holoscan::tsdf::test
