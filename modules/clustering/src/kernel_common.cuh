// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Device helpers of the clusterer: launch sizes, point validity, and the CUB
// scans and sorts with their scratch.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include <cub/device/device_radix_sort.cuh>  // NOLINT(build/include_order): CUB
#include <cub/device/device_scan.cuh>        // NOLINT(build/include_order): CUB

#include "device_buffer.hpp"

namespace holoscan::clustering::detail {

inline constexpr int kBlock = 256;

inline unsigned int blocks_for(std::uint64_t count) {
  return static_cast<unsigned int>((count + kBlock - 1U) / kBlock);
}

__device__ inline bool valid_point(float x, float y, float z) {
  return isfinite(x) && isfinite(y) && isfinite(z);
}

// The squared distance between two points, as the CPU reference computes it:
// nothing contracted (the file is compiled with --fmad=false), left to right.
__device__ inline float squared_distance(const float* a, const float* b) {
  const float dx = a[0] - b[0];
  const float dy = a[1] - b[1];
  const float dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

// Exclusive prefix sum of `count` values, with CUB's scratch in `temp`.
inline cudaError_t exclusive_sum(const std::uint32_t* in, std::uint32_t* out, std::uint32_t count,
                                 DeviceBuffer& temp, cudaStream_t stream) {
  std::size_t bytes = 0;
  if (const cudaError_t status =
          cub::DeviceScan::ExclusiveSum(nullptr, bytes, in, out, static_cast<int>(count), stream);
      status != cudaSuccess) {
    return status;
  }
  if (const cudaError_t status = temp.reserve(bytes, stream); status != cudaSuccess) {
    return status;
  }
  return cub::DeviceScan::ExclusiveSum(temp.as<void>(), bytes, in, out, static_cast<int>(count),
                                       stream);
}

// Sorts (key, value) pairs by the low `bits` bits of the key, stably.
template <typename Key>
inline cudaError_t sort_pairs(const Key* keys_in, Key* keys_out, const std::uint32_t* values_in,
                              std::uint32_t* values_out, std::uint32_t count, int bits,
                              DeviceBuffer& temp, cudaStream_t stream) {
  std::size_t bytes = 0;
  if (const cudaError_t status =
          cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys_in, keys_out, values_in, values_out,
                                          static_cast<int>(count), 0, bits, stream);
      status != cudaSuccess) {
    return status;
  }
  if (const cudaError_t status = temp.reserve(bytes, stream); status != cudaSuccess) {
    return status;
  }
  return cub::DeviceRadixSort::SortPairs(temp.as<void>(), bytes, keys_in, keys_out, values_in,
                                         values_out, static_cast<int>(count), 0, bits, stream);
}

}  // namespace holoscan::clustering::detail
