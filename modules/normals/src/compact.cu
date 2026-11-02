// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <memory>

#include "compact.cuh"
#include "device_buffer.hpp"
#include "kernel_common.cuh"

namespace holoscan::normals::detail {
namespace {

__device__ inline bool has_normal(const float* normals, std::uint32_t i) {
  return isfinite(normals[3 * i]);
}

__global__ void keep_kernel(const float* __restrict__ xyz, const std::uint8_t* __restrict__ rgb,
                            const float* __restrict__ normals, std::uint32_t count,
                            float* __restrict__ out_xyz, std::uint8_t* __restrict__ out_rgb) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const bool keep = has_normal(normals, i);
  const float nan = nan_float();
  for (int c = 0; c < 3; ++c) {
    out_xyz[3 * i + c] = keep ? xyz[3 * i + c] : nan;
    if (out_rgb != nullptr) {
      out_rgb[3 * i + c] = keep ? rgb[3 * i + c] : 0U;
    }
  }
}

__global__ void flags_kernel(const float* __restrict__ normals, std::uint32_t count,
                             std::uint32_t* __restrict__ flags) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    flags[i] = has_normal(normals, i) ? 1U : 0U;
  }
}

__global__ void total_kernel(const std::uint32_t* __restrict__ flags,
                             const std::uint32_t* __restrict__ offsets, std::uint32_t count,
                             std::uint32_t* __restrict__ total) {
  *total = count == 0U ? 0U : offsets[count - 1U] + flags[count - 1U];
}

__global__ void pack_kernel(const std::uint32_t* __restrict__ flags,
                            const std::uint32_t* __restrict__ offsets, std::uint32_t count,
                            const float* __restrict__ xyz, const std::uint8_t* __restrict__ rgb,
                            const float* __restrict__ normals, const float* __restrict__ curvature,
                            float* __restrict__ out_xyz, std::uint8_t* __restrict__ out_rgb,
                            float* __restrict__ out_normals, float* __restrict__ out_curvature) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count || flags[i] == 0U) {
    return;
  }
  const std::uint32_t at = offsets[i];
  for (int c = 0; c < 3; ++c) {
    out_xyz[3 * at + c] = xyz[3 * i + c];
    out_normals[3 * at + c] = normals[3 * i + c];
    if (out_rgb != nullptr) {
      out_rgb[3 * at + c] = rgb[3 * i + c];
    }
  }
  out_curvature[at] = curvature[i];
}

}  // namespace

cudaError_t keep_with_normals(const float* xyz, const std::uint8_t* rgb, const float* normals,
                              std::uint32_t count, float* out_xyz, std::uint8_t* out_rgb,
                              cudaStream_t stream) {
  if (count == 0U) {
    return cudaSuccess;
  }
  if ((rgb == nullptr) != (out_rgb == nullptr)) {
    return cudaErrorInvalidValue;
  }
  keep_kernel<<<blocks_for(count), kBlock, 0, stream>>>(xyz, rgb, normals, count, out_xyz, out_rgb);
  return cudaGetLastError();
}

struct Compactor::Impl {
  DeviceBuffer flags;
  DeviceBuffer offsets;
  DeviceBuffer total;
  DeviceBuffer temp;
  std::uint32_t count{};
};

Compactor::Compactor() : impl_(std::make_unique<Impl>()) {}
Compactor::~Compactor() = default;

cudaError_t Compactor::count(const float* normals, std::uint32_t count, cudaStream_t stream) {
  Impl& s = *impl_;
  const std::size_t n = count == 0U ? 1U : count;
  for (DeviceBuffer* buffer : {&s.flags, &s.offsets}) {
    if (const cudaError_t status = buffer->reserve(n * 4U, stream); status != cudaSuccess) {
      return status;
    }
  }
  if (const cudaError_t status = s.total.reserve(4U, stream); status != cudaSuccess) {
    return status;
  }
  s.count = count;
  if (count == 0U) {
    return cudaMemsetAsync(s.total.as<void>(), 0, 4U, stream);
  }
  flags_kernel<<<blocks_for(count), kBlock, 0, stream>>>(normals, count,
                                                         s.flags.as<std::uint32_t>());
  if (const cudaError_t status = exclusive_sum(
          s.flags.as<std::uint32_t>(), s.offsets.as<std::uint32_t>(), count, s.temp, stream);
      status != cudaSuccess) {
    return status;
  }
  total_kernel<<<1, 1, 0, stream>>>(s.flags.as<std::uint32_t>(), s.offsets.as<std::uint32_t>(),
                                    count, s.total.as<std::uint32_t>());
  return cudaGetLastError();
}

const std::uint32_t* Compactor::device_count() const noexcept {
  return impl_->total.as<std::uint32_t>();
}

cudaError_t Compactor::write(const float* xyz, const std::uint8_t* rgb, const float* normals,
                             const float* curvature, float* out_xyz, std::uint8_t* out_rgb,
                             float* out_normals, float* out_curvature, cudaStream_t stream) {
  const std::uint32_t count = impl_->count;
  if (count == 0U) {
    return cudaSuccess;
  }
  if ((rgb == nullptr) != (out_rgb == nullptr)) {
    return cudaErrorInvalidValue;
  }
  pack_kernel<<<blocks_for(count), kBlock, 0, stream>>>(
      impl_->flags.as<std::uint32_t>(), impl_->offsets.as<std::uint32_t>(), count, xyz, rgb,
      normals, curvature, out_xyz, out_rgb, out_normals, out_curvature);
  return cudaGetLastError();
}

}  // namespace holoscan::normals::detail
