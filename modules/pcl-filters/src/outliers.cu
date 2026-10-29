// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Outlier removal over a uniform grid. The points are sorted by cell key (a
// stable radix sort), the occupied cells listed with where their points start,
// and each point looks its neighbor cells up by binary search over the sorted
// cell keys. The statistical mode searches cells in rings of growing Chebyshev
// distance and stops once its k nearest are closer than any point of the next
// ring can be; its mean and deviation are summed in double in a fixed order.
// Compiled without fused multiply-adds, so a host reference that computes the
// same distances gets the same bits.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <holoscan/pcl_filters/filters.hpp>

#include "device_buffer.hpp"
#include "kernel_common.cuh"

namespace holoscan::pcl_filters {
namespace {

using detail::blocks_for;
using detail::kBlock;
using detail::kNoCell;
using detail::valid_point;

constexpr unsigned int kStatBlocks = 64;
// The statistical mode's cells are a quarter of the search radius, so four
// rings of cells cover it; on a dense surface the search mostly stops after
// the first ring.
constexpr int kStatisticalRings = 4;

__global__ void cell_keys_kernel(const float* __restrict__ xyz, std::uint32_t count, float size,
                                 std::uint64_t* __restrict__ keys,
                                 std::uint32_t* __restrict__ indices) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    keys[i] = detail::point_cell(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2], size);
    indices[i] = i;
  }
}

__global__ void heads_kernel(const std::uint64_t* __restrict__ keys, std::uint32_t count,
                             std::uint32_t* __restrict__ heads) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    heads[i] = keys[i] != kNoCell && (i == 0U || keys[i] != keys[i - 1U]) ? 1U : 0U;
  }
}

// Cell c's key and the start of its points; starts[cells] is the end of the
// last cell, and `cells` the number of cells.
__global__ void cells_kernel(const std::uint64_t* __restrict__ keys,
                             const std::uint32_t* __restrict__ heads,
                             const std::uint32_t* __restrict__ offsets, std::uint32_t count,
                             std::uint64_t* __restrict__ cell_keys,
                             std::uint32_t* __restrict__ starts,
                             std::uint32_t* __restrict__ cells) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  if (heads[i] != 0U) {
    cell_keys[offsets[i]] = keys[i];
    starts[offsets[i]] = i;
  }
  const bool last_valid = keys[i] != kNoCell && (i + 1U == count || keys[i + 1U] == kNoCell);
  if (last_valid) {
    starts[offsets[i] + heads[i]] = i + 1U;
  }
  if (i == count - 1U) {
    *cells = offsets[i] + heads[i];
  }
}

// The cell with `key`, or -1.
__device__ inline int find_cell(const std::uint64_t* __restrict__ cell_keys, std::uint32_t cells,
                                std::uint64_t key) {
  std::uint32_t lo = 0;
  std::uint32_t hi = cells;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2U;
    if (cell_keys[mid] < key) {
      lo = mid + 1U;
    } else {
      hi = mid;
    }
  }
  return lo < cells && cell_keys[lo] == key ? static_cast<int>(lo) : -1;
}

__device__ inline void decode(std::uint64_t key, int& ix, int& iy, int& iz) {
  constexpr std::uint64_t mask = (std::uint64_t{1} << detail::kCellBits) - 1U;
  ix = static_cast<int>((key >> (2 * detail::kCellBits)) & mask) - kVoxelIndexRange;
  iy = static_cast<int>((key >> detail::kCellBits) & mask) - kVoxelIndexRange;
  iz = static_cast<int>(key & mask) - kVoxelIndexRange;
}

struct Grid {
  const std::uint64_t* sorted_keys;
  const std::uint32_t* sorted_indices;
  const std::uint64_t* cell_keys;
  const std::uint32_t* starts;
  const std::uint32_t* cells;
};

// Calls visit(index, squared distance) for every other point in the cells of
// Chebyshev ring `ring` around cell (ix, iy, iz); visit returns false to stop.
template <typename Visit>
__device__ inline bool visit_ring(const Grid& grid, std::uint32_t cells, const float* xyz,
                                  std::uint32_t self, float px, float py, float pz, int ix, int iy,
                                  int iz, int ring, Visit&& visit) {
  for (int dx = -ring; dx <= ring; ++dx) {
    for (int dy = -ring; dy <= ring; ++dy) {
      for (int dz = -ring; dz <= ring; ++dz) {
        if (max(abs(dx), max(abs(dy), abs(dz))) != ring) {
          continue;
        }
        const int nx = ix + dx;
        const int ny = iy + dy;
        const int nz = iz + dz;
        if (!detail::in_range(nx) || !detail::in_range(ny) || !detail::in_range(nz)) {
          continue;
        }
        const int cell = find_cell(grid.cell_keys, cells, detail::cell_key(nx, ny, nz));
        if (cell < 0) {
          continue;
        }
        for (std::uint32_t j = grid.starts[cell]; j < grid.starts[cell + 1]; ++j) {
          const std::uint32_t other = grid.sorted_indices[j];
          if (other == self) {
            continue;
          }
          const float ex = xyz[3 * other] - px;
          const float ey = xyz[3 * other + 1] - py;
          const float ez = xyz[3 * other + 2] - pz;
          if (!visit(ex * ex + ey * ey + ez * ez)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

__global__ void fill_kernel(std::uint32_t count, float* __restrict__ mean,
                            std::uint8_t* __restrict__ keep) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    if (mean != nullptr) {
      mean[i] = __int_as_float(0x7FC00000);
    }
    keep[i] = 0U;
  }
}

// Thread t takes the t-th point in cell order, for locality; invalid points,
// sorted last, keep the NaN that fill_kernel wrote.
__global__ void knn_kernel(const float* __restrict__ xyz, std::uint32_t count, Grid grid,
                           float cell_size, std::uint32_t k, float search_radius,
                           float* __restrict__ mean) {
  const std::uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= count || grid.sorted_keys[t] == kNoCell) {
    return;
  }
  const std::uint32_t cells = *grid.cells;
  const std::uint32_t i = grid.sorted_indices[t];
  const float px = xyz[3 * i];
  const float py = xyz[3 * i + 1];
  const float pz = xyz[3 * i + 2];
  int ix = 0;
  int iy = 0;
  int iz = 0;
  decode(grid.sorted_keys[t], ix, iy, iz);
  float best[kMaxNeighbors] = {};
  std::uint32_t found = 0;
  const auto visit = [&](float squared) {
    const float distance = sqrtf(squared);
    if (!(distance <= search_radius) || (found == k && !(distance < best[k - 1U]))) {
      return true;
    }
    // Insert in ascending order, dropping the k-th when full.
    std::uint32_t at = found < k ? found++ : k - 1U;
    while (at > 0U && best[at - 1U] > distance) {
      best[at] = best[at - 1U];
      --at;
    }
    best[at] = distance;
    return true;
  };
  for (int ring = 0; ring <= kStatisticalRings; ++ring) {
    visit_ring(grid, cells, xyz, i, px, py, pz, ix, iy, iz, ring, visit);
    // Every point beyond ring `ring` is farther than ring * cell_size.
    if (ring >= 1 && found == k && best[k - 1U] <= static_cast<float>(ring) * cell_size) {
      break;
    }
  }
  if (found < k) {
    return;
  }
  float sum = 0.0F;
  for (std::uint32_t j = 0; j < k; ++j) {
    sum += best[j];
  }
  mean[i] = sum / static_cast<float>(k);
}

__global__ void radius_kernel(const float* __restrict__ xyz, std::uint32_t count, Grid grid,
                              float radius, std::uint32_t min_neighbors,
                              std::uint8_t* __restrict__ keep) {
  const std::uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= count || grid.sorted_keys[t] == kNoCell) {
    return;
  }
  const std::uint32_t cells = *grid.cells;
  const std::uint32_t i = grid.sorted_indices[t];
  int ix = 0;
  int iy = 0;
  int iz = 0;
  decode(grid.sorted_keys[t], ix, iy, iz);
  const float squared_radius = radius * radius;
  std::uint32_t neighbors = 0;
  const auto count_near = [&](float squared) {
    neighbors += squared <= squared_radius ? 1U : 0U;
    return neighbors < min_neighbors;
  };
  const float px = xyz[3 * i];
  const float py = xyz[3 * i + 1];
  const float pz = xyz[3 * i + 2];
  // Cells of one radius a side: every neighbor is within the first ring.
  if (min_neighbors != 0U &&
      visit_ring(grid, cells, xyz, i, px, py, pz, ix, iy, iz, 0, count_near)) {
    visit_ring(grid, cells, xyz, i, px, py, pz, ix, iy, iz, 1, count_near);
  }
  keep[i] = neighbors >= min_neighbors ? 1U : 0U;
}

// Per-block count, sum and sum of squares of the finite mean distances.
__global__ void stats_kernel(const float* __restrict__ mean, std::uint32_t count,
                             double* __restrict__ partials) {
  __shared__ double shared[3][kBlock];
  double sums[3] = {};
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    const float value = mean[i];
    if (isfinite(value)) {
      sums[0] += 1.0;
      sums[1] += value;
      sums[2] += static_cast<double>(value) * value;
    }
  }
  for (int k = 0; k < 3; ++k) {
    shared[k][threadIdx.x] = sums[k];
  }
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride /= 2U) {
    if (threadIdx.x < stride) {
      for (int k = 0; k < 3; ++k) {
        shared[k][threadIdx.x] += shared[k][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (int k = 0; k < 3; ++k) {
      partials[blockIdx.x * 3 + k] = shared[k][0];
    }
  }
}

// mean + multiplier * sample standard deviation, as PCL computes it.
__global__ void threshold_kernel(const double* __restrict__ partials, unsigned int blocks,
                                 double multiplier, double* __restrict__ threshold) {
  double s[3] = {};
  for (unsigned int b = 0; b < blocks; ++b) {
    for (int k = 0; k < 3; ++k) {
      s[k] += partials[b * 3 + k];
    }
  }
  if (s[0] < 1.0) {
    *threshold = 0.0;
    return;
  }
  const double average = s[1] / s[0];
  const double variance = s[0] > 1.0 ? (s[2] - s[1] * s[1] / s[0]) / (s[0] - 1.0) : 0.0;
  *threshold = average + multiplier * sqrt(variance > 0.0 ? variance : 0.0);
}

__global__ void statistical_keep_kernel(const float* __restrict__ mean, std::uint32_t count,
                                        const double* __restrict__ threshold,
                                        std::uint8_t* __restrict__ keep) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    const float value = mean[i];
    keep[i] = isfinite(value) && static_cast<double>(value) <= *threshold ? 1U : 0U;
  }
}

}  // namespace

struct OutlierFilter::Impl {
  detail::DeviceBuffer keys;
  detail::DeviceBuffer sorted_keys;
  detail::DeviceBuffer indices;
  detail::DeviceBuffer sorted_indices;
  detail::DeviceBuffer heads;
  detail::DeviceBuffer offsets;
  detail::DeviceBuffer cell_keys;
  detail::DeviceBuffer starts;
  detail::DeviceBuffer cells;
  detail::DeviceBuffer mean;
  detail::DeviceBuffer partials;
  detail::DeviceBuffer threshold;
  detail::DeviceBuffer temp;
};

OutlierFilter::OutlierFilter() : impl_(std::make_unique<Impl>()) {}
OutlierFilter::~OutlierFilter() = default;

cudaError_t OutlierFilter::filter(const float* xyz, std::uint32_t count,
                                  const OutlierParams& params, std::uint8_t* keep,
                                  cudaStream_t stream) {
  const bool statistical = params.mode == OutlierMode::kStatistical;
  if (count > kMaxPointsLimit ||
      (statistical &&
       (params.k < 1U || params.k > kMaxNeighbors || !(params.search_radius > 0.0F) ||
        !std::isfinite(params.search_radius) || !std::isfinite(params.std_multiplier))) ||
      (!statistical && (!(params.radius > 0.0F) || !std::isfinite(params.radius)))) {
    return cudaErrorInvalidValue;
  }
  Impl& s = *impl_;
  const std::size_t n = count;
  for (auto [buffer, bytes] :
       {std::pair{&s.keys, n * 8U}, std::pair{&s.sorted_keys, n * 8U},
        std::pair{&s.indices, n * 4U}, std::pair{&s.sorted_indices, n * 4U},
        std::pair{&s.heads, n * 4U}, std::pair{&s.offsets, n * 4U}, std::pair{&s.cell_keys, n * 8U},
        std::pair{&s.starts, n * 4U + 4U}, std::pair{&s.cells, std::size_t{4}},
        std::pair{&s.mean, n * 4U},
        std::pair{&s.partials, std::size_t{kStatBlocks} * 3U * sizeof(double)},
        std::pair{&s.threshold, sizeof(double)}}) {
    if (const cudaError_t status = buffer->reserve(bytes, stream); status != cudaSuccess) {
      return status;
    }
  }
  if (count == 0U) {
    return cudaMemsetAsync(s.threshold.as<void>(), 0, sizeof(double), stream);
  }
  const float cell_size =
      statistical ? params.search_radius / static_cast<float>(kStatisticalRings) : params.radius;
  cell_keys_kernel<<<blocks_for(count), kBlock, 0, stream>>>(
      xyz, count, cell_size, s.keys.as<std::uint64_t>(), s.indices.as<std::uint32_t>());
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return status;
  }
  if (const cudaError_t status =
          detail::sort_pairs(s.keys.as<std::uint64_t>(), s.sorted_keys.as<std::uint64_t>(),
                             s.indices.as<std::uint32_t>(), s.sorted_indices.as<std::uint32_t>(),
                             count, detail::kKeyBits, s.temp, stream);
      status != cudaSuccess) {
    return status;
  }
  heads_kernel<<<blocks_for(count), kBlock, 0, stream>>>(s.sorted_keys.as<std::uint64_t>(), count,
                                                         s.heads.as<std::uint32_t>());
  if (const cudaError_t status = detail::exclusive_sum(
          s.heads.as<std::uint32_t>(), s.offsets.as<std::uint32_t>(), count, s.temp, stream);
      status != cudaSuccess) {
    return status;
  }
  cells_kernel<<<blocks_for(count), kBlock, 0, stream>>>(
      s.sorted_keys.as<std::uint64_t>(), s.heads.as<std::uint32_t>(), s.offsets.as<std::uint32_t>(),
      count, s.cell_keys.as<std::uint64_t>(), s.starts.as<std::uint32_t>(),
      s.cells.as<std::uint32_t>());
  const Grid grid{s.sorted_keys.as<std::uint64_t>(), s.sorted_indices.as<std::uint32_t>(),
                  s.cell_keys.as<std::uint64_t>(), s.starts.as<std::uint32_t>(),
                  s.cells.as<std::uint32_t>()};
  fill_kernel<<<blocks_for(count), kBlock, 0, stream>>>(
      count, statistical ? s.mean.as<float>() : nullptr, keep);
  if (!statistical) {
    radius_kernel<<<blocks_for(count), kBlock, 0, stream>>>(xyz, count, grid, params.radius,
                                                            params.min_neighbors, keep);
    return cudaGetLastError();
  }
  knn_kernel<<<blocks_for(count), kBlock, 0, stream>>>(xyz, count, grid, cell_size, params.k,
                                                       params.search_radius, s.mean.as<float>());
  stats_kernel<<<kStatBlocks, kBlock, 0, stream>>>(s.mean.as<float>(), count,
                                                   s.partials.as<double>());
  threshold_kernel<<<1, 1, 0, stream>>>(s.partials.as<double>(), kStatBlocks,
                                        static_cast<double>(params.std_multiplier),
                                        s.threshold.as<double>());
  statistical_keep_kernel<<<blocks_for(count), kBlock, 0, stream>>>(s.mean.as<float>(), count,
                                                                    s.threshold.as<double>(), keep);
  return cudaGetLastError();
}

const float* OutlierFilter::device_mean_distances() const noexcept {
  return impl_->mean.as<float>();
}

const double* OutlierFilter::device_threshold() const noexcept {
  return impl_->threshold.as<double>();
}

}  // namespace holoscan::pcl_filters
