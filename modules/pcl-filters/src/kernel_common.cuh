// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Device helpers shared by the filters: point validity, grid cell keys, launch
// sizes, and the CUB scans and sorts with their scratch.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include <cub/device/device_radix_sort.cuh>  // NOLINT(build/include_order): CUB
#include <cub/device/device_scan.cuh>        // NOLINT(build/include_order): CUB
#include <holoscan/pcl_filters/filters.hpp>

#include "device_buffer.hpp"

namespace holoscan::pcl_filters::detail {

inline constexpr int kBlock = 256;

inline unsigned int blocks_for(std::uint32_t count) { return (count + kBlock - 1U) / kBlock; }

__device__ inline bool valid_point(float x, float y, float z) {
  return isfinite(x) && isfinite(y) && isfinite(z);
}

// A grid cell key: the cell's (x, y, z) index, each biased by
// kVoxelIndexRange into 21 bits, x in the highest. Keys sort like the indices.
inline constexpr std::uint64_t kNoCell = ~std::uint64_t{0};
inline constexpr int kCellBits = 21;
inline constexpr int kKeyBits = 3 * kCellBits;

// The cell index of coordinate `value` in cells of `size`, or false when it is
// out of range (or not finite).
__device__ inline bool cell_index(float value, float size, int& index) {
  const float cell = floorf(value / size);
  if (!(cell >= -static_cast<float>(kVoxelIndexRange) &&
        cell < static_cast<float>(kVoxelIndexRange))) {
    return false;
  }
  index = static_cast<int>(cell);
  return true;
}

__device__ inline std::uint64_t cell_key(int ix, int iy, int iz) {
  return (static_cast<std::uint64_t>(ix + kVoxelIndexRange) << (2 * kCellBits)) |
         (static_cast<std::uint64_t>(iy + kVoxelIndexRange) << kCellBits) |
         static_cast<std::uint64_t>(iz + kVoxelIndexRange);
}

__device__ inline bool in_range(int index) {
  return index >= -kVoxelIndexRange && index < kVoxelIndexRange;
}

// The key of the cell of point p, or kNoCell for an invalid or far point.
__device__ inline std::uint64_t point_cell(float x, float y, float z, float size) {
  int ix = 0;
  int iy = 0;
  int iz = 0;
  if (!valid_point(x, y, z) || !cell_index(x, size, ix) || !cell_index(y, size, iy) ||
      !cell_index(z, size, iz)) {
    return kNoCell;
  }
  return cell_key(ix, iy, iz);
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
inline cudaError_t sort_pairs(const std::uint64_t* keys_in, std::uint64_t* keys_out,
                              const std::uint32_t* values_in, std::uint32_t* values_out,
                              std::uint32_t count, int bits, DeviceBuffer& temp,
                              cudaStream_t stream) {
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

// SplitMix64: the deterministic random numbers of RANSAC.
__device__ inline std::uint64_t mix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}

}  // namespace holoscan::pcl_filters::detail
