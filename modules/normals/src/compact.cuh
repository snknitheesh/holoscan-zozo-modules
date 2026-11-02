// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The two output layouts of NormalEstimationOp: the points that got a normal
// kept in place (the others NaN), or packed in input order with their colors,
// normals and curvature.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>

namespace holoscan::normals::detail {

// Writes the points that have a normal into an organized cloud of the same
// size, the others NaN with black. `rgb` and `out_rgb` may both be null.
cudaError_t keep_with_normals(const float* xyz, const std::uint8_t* rgb, const float* normals,
                              std::uint32_t count, float* out_xyz, std::uint8_t* out_rgb,
                              cudaStream_t stream);

// Packs the points that have a normal, in their input order.
class Compactor {
 public:
  Compactor();
  ~Compactor();
  Compactor(const Compactor&) = delete;
  Compactor& operator=(const Compactor&) = delete;

  // Counts the points with a normal into device_count().
  cudaError_t count(const float* normals, std::uint32_t count, cudaStream_t stream);
  // The number counted, in device memory, once count() has run.
  [[nodiscard]] const std::uint32_t* device_count() const noexcept;
  // Writes the counted points, their normals and curvature (and colors, when
  // both are given) into buffers of at least that many points.
  cudaError_t write(const float* xyz, const std::uint8_t* rgb, const float* normals,
                    const float* curvature, float* out_xyz, std::uint8_t* out_rgb,
                    float* out_normals, float* out_curvature, cudaStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::normals::detail
