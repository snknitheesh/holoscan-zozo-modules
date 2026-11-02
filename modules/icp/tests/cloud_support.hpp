// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Synthetic clouds in device memory for the tests and the benchmark.

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

#include <holoscan/icp/registration.hpp>
#include <holoscan/icp/synthetic_scene.hpp>

namespace holoscan::icp::test {

inline void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

// Packed float32 x, y, z points in device memory.
class DeviceCloud {
 public:
  DeviceCloud() = default;
  explicit DeviceCloud(std::size_t points) : count_(points) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_),
                          std::max<std::size_t>(points, 1U) * 3U * sizeof(float)),
               "cudaMalloc");
  }
  explicit DeviceCloud(const std::vector<float>& host) : DeviceCloud(host.size() / 3U) {
    cuda_check(cudaMemcpy(data_, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy");
  }
  DeviceCloud(DeviceCloud&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0U)) {}
  DeviceCloud& operator=(DeviceCloud&& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(count_, other.count_);
    return *this;
  }
  DeviceCloud(const DeviceCloud&) = delete;
  DeviceCloud& operator=(const DeviceCloud&) = delete;
  ~DeviceCloud() {
    if (data_ != nullptr) cudaFree(data_);
  }

  [[nodiscard]] float* data() const { return data_; }
  [[nodiscard]] std::uint32_t count() const { return static_cast<std::uint32_t>(count_); }

  [[nodiscard]] std::vector<float> to_host() const {
    std::vector<float> host(count_ * 3U);
    cuda_check(cudaMemcpy(host.data(), data_, host.size() * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy");
    return host;
  }

 private:
  float* data_{};
  std::size_t count_{};
};

// The organized cloud of `scene` seen by a width x height synthetic camera at
// `pose`, within 8 m.
inline DeviceCloud render(const SyntheticScene& scene, const Matrix4& pose, int width, int height,
                          const DepthNoise& noise = {}, cudaStream_t stream = nullptr) {
  DeviceCloud cloud(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
  cuda_check(render_scene_cloud(scene, pose, synthetic_camera(width, height), 8.0F, noise,
                                cloud.data(), stream),
             "render_scene_cloud");
  cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  return cloud;
}

inline double degrees(double radians) { return radians * 180.0 / 3.14159265358979323846; }
inline double radians(double degrees) { return degrees * 3.14159265358979323846 / 180.0; }

}  // namespace holoscan::icp::test
