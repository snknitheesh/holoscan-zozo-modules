// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The voxel grid: each point's voxel key, a stable radix sort of (key, index),
// a scan of the voxel heads, and one thread per voxel that sums its points in
// input order. Sums are in double and nothing contracts (--fmad=false), so a
// host reference that sums in the same order gets the same bits.

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

__global__ void voxel_keys_kernel(const float* __restrict__ xyz, std::uint32_t count, float size,
                                  std::uint64_t* __restrict__ keys,
                                  std::uint32_t* __restrict__ indices) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    keys[i] = detail::point_cell(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2], size);
    indices[i] = i;
  }
}

// 1 where a voxel starts in the sorted keys.
__global__ void heads_kernel(const std::uint64_t* __restrict__ keys, std::uint32_t count,
                             std::uint32_t* __restrict__ heads) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    heads[i] = keys[i] != kNoCell && (i == 0U || keys[i] != keys[i - 1U]) ? 1U : 0U;
  }
}

__global__ void starts_kernel(const std::uint32_t* __restrict__ heads,
                              const std::uint32_t* __restrict__ offsets, std::uint32_t count,
                              std::uint32_t* __restrict__ starts,
                              std::uint32_t* __restrict__ total) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  if (heads[i] != 0U) {
    starts[offsets[i]] = i;
  }
  if (i == count - 1U) {
    *total = offsets[i] + heads[i];
  }
}

__global__ void zero_kernel(std::uint32_t* value) { *value = 0U; }

__global__ void voxel_write_kernel(const float* __restrict__ xyz,
                                   const std::uint8_t* __restrict__ rgb,
                                   const std::uint64_t* __restrict__ keys,
                                   const std::uint32_t* __restrict__ indices,
                                   const std::uint32_t* __restrict__ starts,
                                   const std::uint32_t* __restrict__ total, std::uint32_t count,
                                   VoxelMode mode, float* __restrict__ out_xyz,
                                   std::uint8_t* __restrict__ out_rgb) {
  const std::uint32_t voxel = blockIdx.x * blockDim.x + threadIdx.x;
  if (voxel >= *total) {
    return;
  }
  const std::uint32_t start = starts[voxel];
  const std::uint64_t key = keys[start];
  double sum[3] = {0.0, 0.0, 0.0};
  std::uint32_t color[3] = {0U, 0U, 0U};
  std::uint32_t points = 0;
  for (std::uint32_t j = start; j < count && keys[j] == key; ++j) {
    const std::uint32_t index = indices[j];
    for (int axis = 0; axis < 3; ++axis) {
      sum[axis] += static_cast<double>(xyz[3 * index + axis]);
    }
    if (out_rgb != nullptr) {
      for (int channel = 0; channel < 3; ++channel) {
        color[channel] += rgb[3 * index + channel];
      }
    }
    ++points;
  }
  // The sort is stable over ascending indices, so the voxel's first entry is
  // its point with the lowest index.
  const std::uint32_t first = indices[start];
  for (int axis = 0; axis < 3; ++axis) {
    out_xyz[3 * voxel + axis] = mode == VoxelMode::kCentroid
                                    ? static_cast<float>(sum[axis] / static_cast<double>(points))
                                    : xyz[3 * first + axis];
  }
  if (out_rgb != nullptr) {
    for (int channel = 0; channel < 3; ++channel) {
      out_rgb[3 * voxel + channel] =
          static_cast<std::uint8_t>((color[channel] + points / 2U) / points);
    }
  }
}

}  // namespace

struct VoxelGrid::Impl {
  detail::DeviceBuffer keys;
  detail::DeviceBuffer sorted_keys;
  detail::DeviceBuffer indices;
  detail::DeviceBuffer sorted_indices;
  detail::DeviceBuffer heads;
  detail::DeviceBuffer offsets;
  detail::DeviceBuffer starts;
  detail::DeviceBuffer total;
  detail::DeviceBuffer temp;
  std::uint32_t count{};
};

VoxelGrid::VoxelGrid() : impl_(std::make_unique<Impl>()) {}
VoxelGrid::~VoxelGrid() = default;

cudaError_t VoxelGrid::group(const float* xyz, std::uint32_t count, float voxel_size,
                             cudaStream_t stream) {
  if (!(voxel_size > 0.0F) || count > kMaxPointsLimit) {
    return cudaErrorInvalidValue;
  }
  Impl& s = *impl_;
  s.count = count;
  const std::size_t n = count;
  for (auto [buffer, bytes] : {std::pair{&s.keys, n * 8U}, std::pair{&s.sorted_keys, n * 8U},
                               std::pair{&s.indices, n * 4U}, std::pair{&s.sorted_indices, n * 4U},
                               std::pair{&s.heads, n * 4U}, std::pair{&s.offsets, n * 4U},
                               std::pair{&s.starts, n * 4U}, std::pair{&s.total, std::size_t{4}}}) {
    if (const cudaError_t status = buffer->reserve(bytes, stream); status != cudaSuccess) {
      return status;
    }
  }
  if (count == 0U) {
    zero_kernel<<<1, 1, 0, stream>>>(s.total.as<std::uint32_t>());
    return cudaGetLastError();
  }
  voxel_keys_kernel<<<blocks_for(count), kBlock, 0, stream>>>(
      xyz, count, voxel_size, s.keys.as<std::uint64_t>(), s.indices.as<std::uint32_t>());
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return status;
  }
  // kNoCell has all 63 low bits set, so it sorts after every voxel.
  if (const cudaError_t status =
          detail::sort_pairs(s.keys.as<std::uint64_t>(), s.sorted_keys.as<std::uint64_t>(),
                             s.indices.as<std::uint32_t>(), s.sorted_indices.as<std::uint32_t>(),
                             count, detail::kKeyBits, s.temp, stream);
      status != cudaSuccess) {
    return status;
  }
  heads_kernel<<<blocks_for(count), kBlock, 0, stream>>>(s.sorted_keys.as<std::uint64_t>(), count,
                                                         s.heads.as<std::uint32_t>());
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return status;
  }
  if (const cudaError_t status = detail::exclusive_sum(
          s.heads.as<std::uint32_t>(), s.offsets.as<std::uint32_t>(), count, s.temp, stream);
      status != cudaSuccess) {
    return status;
  }
  starts_kernel<<<blocks_for(count), kBlock, 0, stream>>>(
      s.heads.as<std::uint32_t>(), s.offsets.as<std::uint32_t>(), count,
      s.starts.as<std::uint32_t>(), s.total.as<std::uint32_t>());
  return cudaGetLastError();
}

const std::uint32_t* VoxelGrid::device_count() const noexcept {
  return impl_->total.as<std::uint32_t>();
}

cudaError_t VoxelGrid::write(const float* xyz, const std::uint8_t* rgb, VoxelMode mode,
                             float* out_xyz, std::uint8_t* out_rgb, cudaStream_t stream) {
  const Impl& s = *impl_;
  if (out_rgb != nullptr && rgb == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (s.count == 0U) {
    return cudaSuccess;
  }
  // As many threads as points, the most voxels there can be; each reads the
  // actual count on the device.
  voxel_write_kernel<<<blocks_for(s.count), kBlock, 0, stream>>>(
      xyz, rgb, s.sorted_keys.as<std::uint64_t>(), s.sorted_indices.as<std::uint32_t>(),
      s.starts.as<std::uint32_t>(), s.total.as<std::uint32_t>(), s.count, mode, out_xyz, out_rgb);
  return cudaGetLastError();
}

}  // namespace holoscan::pcl_filters
