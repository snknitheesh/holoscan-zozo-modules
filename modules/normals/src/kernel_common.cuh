// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Device helpers shared by the kernels: point validity, launch sizes, grid
// cell keys, the CUB scans and sorts with their scratch, the random numbers
// of RANSAC, and the symmetric 3x3 eigensolver of the normals and plane fits.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include <cub/device/device_radix_sort.cuh>  // NOLINT(build/include_order): CUB
#include <cub/device/device_scan.cuh>        // NOLINT(build/include_order): CUB

#include "device_buffer.hpp"

namespace holoscan::normals::detail {

inline constexpr int kBlock = 256;

inline unsigned int blocks_for(std::uint32_t count) {
  return (count + static_cast<std::uint32_t>(kBlock) - 1U) / static_cast<std::uint32_t>(kBlock);
}

__device__ inline float nan_float() { return __int_as_float(0x7FC00000); }

__device__ inline bool valid_point(float x, float y, float z) {
  return isfinite(x) && isfinite(y) && isfinite(z);
}

// A grid cell key: the cell's (x, y, z) index, each biased by kCellRange into
// 21 bits, x in the highest. Keys sort like the indices. A cell index is
// limited to [-kCellRange, kCellRange) on each axis; points beyond have no
// cell.
inline constexpr std::int32_t kCellRange = std::int32_t{1} << 20;
inline constexpr std::uint64_t kNoCell = ~std::uint64_t{0};
inline constexpr int kCellBits = 21;
inline constexpr int kKeyBits = 3 * kCellBits;

// The cell index of coordinate `value` in cells of `size`, or false when it is
// out of range (or not finite).
__device__ inline bool cell_index(float value, float size, int& index) {
  const float cell = floorf(value / size);
  if (!(cell >= -static_cast<float>(kCellRange) && cell < static_cast<float>(kCellRange))) {
    return false;
  }
  index = static_cast<int>(cell);
  return true;
}

__device__ inline std::uint64_t cell_key(int ix, int iy, int iz) {
  return (static_cast<std::uint64_t>(ix + kCellRange) << (2 * kCellBits)) |
         (static_cast<std::uint64_t>(iy + kCellRange) << kCellBits) |
         static_cast<std::uint64_t>(iz + kCellRange);
}

__device__ inline bool in_range(int index) { return index >= -kCellRange && index < kCellRange; }

__device__ inline void decode_key(std::uint64_t key, int& ix, int& iy, int& iz) {
  constexpr std::uint64_t mask = (std::uint64_t{1} << kCellBits) - 1U;
  ix = static_cast<int>((key >> (2 * kCellBits)) & mask) - kCellRange;
  iy = static_cast<int>((key >> kCellBits) & mask) - kCellRange;
  iz = static_cast<int>(key & mask) - kCellRange;
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
__host__ __device__ inline std::uint64_t mix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}

// The eigenvalues of symmetric `a` (overwritten), in `values`, and the unit
// eigenvectors as the columns of `vectors`, by cyclic Jacobi rotations.
template <typename T>
__host__ __device__ inline void symmetric_eigen3(T a[3][3], T values[3], T vectors[3][3]) {
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      vectors[row][column] = row == column ? T(1) : T(0);
    }
  }
  for (int sweep = 0; sweep < 16; ++sweep) {
    const T off = fabs(a[0][1]) + fabs(a[0][2]) + fabs(a[1][2]);
    const T diagonal = fabs(a[0][0]) + fabs(a[1][1]) + fabs(a[2][2]);
    if (!(off > T(0)) || off <= diagonal * T(1e-12)) {
      break;
    }
    for (int p = 0; p < 2; ++p) {
      for (int q = p + 1; q < 3; ++q) {
        if (a[p][q] == T(0)) {
          continue;
        }
        const T theta = (a[q][q] - a[p][p]) / (T(2) * a[p][q]);
        const T t = (theta >= T(0) ? T(1) : T(-1)) / (fabs(theta) + sqrt(theta * theta + T(1)));
        const T c = T(1) / sqrt(t * t + T(1));
        const T s = t * c;
        for (int k = 0; k < 3; ++k) {
          const T akp = a[k][p];
          const T akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < 3; ++k) {
          const T apk = a[p][k];
          const T aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 3; ++k) {
          const T vkp = vectors[k][p];
          const T vkq = vectors[k][q];
          vectors[k][p] = c * vkp - s * vkq;
          vectors[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  for (int k = 0; k < 3; ++k) {
    values[k] = a[k][k];
  }
}

// The normal and surface variation of a neighborhood from its covariance
// (overwritten): the unit eigenvector of the smallest eigenvalue, and
// lambda0 / (lambda0 + lambda1 + lambda2). False when the points are one
// point or lie on a line, which have no normal.
template <typename T>
__host__ __device__ inline bool pca_normal(T covariance[3][3], T normal[3], T& curvature) {
  T values[3];
  T vectors[3][3];
  symmetric_eigen3(covariance, values, vectors);
  int order[3] = {0, 1, 2};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2 - i; ++j) {
      if (values[order[j + 1]] < values[order[j]]) {
        const int swap = order[j];
        order[j] = order[j + 1];
        order[j + 1] = swap;
      }
    }
  }
  const T smallest = values[order[0]] > T(0) ? values[order[0]] : T(0);
  const T middle = values[order[1]];
  const T largest = values[order[2]];
  const T sum = smallest + middle + largest;
  if (!(largest > T(0)) || !(middle > largest * T(1e-5)) || !(sum > T(0))) {
    return false;
  }
  T length = T(0);
  for (int k = 0; k < 3; ++k) {
    normal[k] = vectors[k][order[0]];
    length += normal[k] * normal[k];
  }
  length = sqrt(length);
  if (!(length > T(0))) {
    return false;
  }
  for (int k = 0; k < 3; ++k) {
    normal[k] /= length;
  }
  curvature = smallest / sum;
  return true;
}

}  // namespace holoscan::normals::detail
