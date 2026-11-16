// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The two output layouts of NormalEstimationOp, one cloud field at a time:
// the points that got a normal kept in place (the others NaN, or black for
// colors), or packed in input order.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>

namespace holoscan::normals::detail {

// Copies `channels` values per point of `in` into `out` for the points with a
// finite normal, and NaN for the others.
cudaError_t keep_with_normals(const float* in, int channels, const float* normals,
                              std::uint32_t count, float* out, cudaStream_t stream);
// The same for colors: black for the points without a normal.
cudaError_t keep_with_normals(const std::uint8_t* in, int channels, const float* normals,
                              std::uint32_t count, std::uint8_t* out, cudaStream_t stream);

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
  // Writes `channels` values per counted point of `in` into `out`, which
  // holds at least that many points.
  cudaError_t pack(const float* in, int channels, float* out, cudaStream_t stream);
  cudaError_t pack(const std::uint8_t* in, int channels, std::uint8_t* out, cudaStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::normals::detail
