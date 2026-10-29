// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Keep masks to clouds: organized, removed points NaN, or packed.

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

__global__ void apply_mask_kernel(const float* __restrict__ xyz,
                                  const std::uint8_t* __restrict__ rgb,
                                  const std::uint8_t* __restrict__ keep, std::uint32_t count,
                                  float* __restrict__ out_xyz, std::uint8_t* __restrict__ out_rgb) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const bool kept = keep[i] != 0U;
  const float nan = __int_as_float(0x7FC00000);
  for (int axis = 0; axis < 3; ++axis) {
    out_xyz[3 * i + axis] = kept ? xyz[3 * i + axis] : nan;
  }
  if (out_rgb != nullptr) {
    for (int channel = 0; channel < 3; ++channel) {
      out_rgb[3 * i + channel] = kept ? rgb[3 * i + channel] : std::uint8_t{0};
    }
  }
}

__global__ void flags_kernel(const std::uint8_t* __restrict__ keep, std::uint32_t count,
                             std::uint32_t* __restrict__ flags) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    flags[i] = keep[i] != 0U ? 1U : 0U;
  }
}

// The kept count: the last offset plus the last flag.
__global__ void total_kernel(const std::uint32_t* __restrict__ flags,
                             const std::uint32_t* __restrict__ offsets, std::uint32_t count,
                             std::uint32_t* __restrict__ total) {
  *total = count == 0U ? 0U : offsets[count - 1U] + flags[count - 1U];
}

__global__ void scatter_kernel(const float* __restrict__ xyz, const std::uint8_t* __restrict__ rgb,
                               const std::uint32_t* __restrict__ flags,
                               const std::uint32_t* __restrict__ offsets, std::uint32_t count,
                               float* __restrict__ out_xyz, std::uint8_t* __restrict__ out_rgb) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count || flags[i] == 0U) {
    return;
  }
  const std::uint32_t o = offsets[i];
  for (int axis = 0; axis < 3; ++axis) {
    out_xyz[3 * o + axis] = xyz[3 * i + axis];
  }
  if (out_rgb != nullptr) {
    for (int channel = 0; channel < 3; ++channel) {
      out_rgb[3 * o + channel] = rgb[3 * i + channel];
    }
  }
}

}  // namespace

cudaError_t apply_mask(const float* xyz, const std::uint8_t* rgb, const std::uint8_t* keep,
                       std::uint32_t count, float* out_xyz, std::uint8_t* out_rgb,
                       cudaStream_t stream) {
  if (count == 0U) {
    return cudaSuccess;
  }
  if (out_rgb != nullptr && rgb == nullptr) {
    return cudaErrorInvalidValue;
  }
  apply_mask_kernel<<<blocks_for(count), kBlock, 0, stream>>>(xyz, rgb, keep, count, out_xyz,
                                                              out_rgb);
  return cudaGetLastError();
}

struct Compactor::Impl {
  detail::DeviceBuffer flags;
  detail::DeviceBuffer offsets;
  detail::DeviceBuffer total;
  detail::DeviceBuffer temp;
  std::uint32_t count{};
};

Compactor::Compactor() : impl_(std::make_unique<Impl>()) {}
Compactor::~Compactor() = default;

cudaError_t Compactor::count(const std::uint8_t* keep, std::uint32_t count, cudaStream_t stream) {
  Impl& s = *impl_;
  s.count = count;
  for (auto [buffer, bytes] :
       {std::pair{&s.flags, std::size_t{count} * 4U},
        std::pair{&s.offsets, std::size_t{count} * 4U}, std::pair{&s.total, std::size_t{4}}}) {
    if (const cudaError_t status = buffer->reserve(bytes, stream); status != cudaSuccess) {
      return status;
    }
  }
  if (count != 0U) {
    flags_kernel<<<blocks_for(count), kBlock, 0, stream>>>(keep, count,
                                                           s.flags.as<std::uint32_t>());
    if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
      return status;
    }
    if (const cudaError_t status = detail::exclusive_sum(
            s.flags.as<std::uint32_t>(), s.offsets.as<std::uint32_t>(), count, s.temp, stream);
        status != cudaSuccess) {
      return status;
    }
  }
  total_kernel<<<1, 1, 0, stream>>>(s.flags.as<std::uint32_t>(), s.offsets.as<std::uint32_t>(),
                                    count, s.total.as<std::uint32_t>());
  return cudaGetLastError();
}

const std::uint32_t* Compactor::device_count() const noexcept {
  return impl_->total.as<std::uint32_t>();
}

cudaError_t Compactor::write(const float* xyz, const std::uint8_t* rgb, float* out_xyz,
                             std::uint8_t* out_rgb, cudaStream_t stream) {
  const Impl& s = *impl_;
  if (s.count == 0U) {
    return cudaSuccess;
  }
  if (out_rgb != nullptr && rgb == nullptr) {
    return cudaErrorInvalidValue;
  }
  scatter_kernel<<<blocks_for(s.count), kBlock, 0, stream>>>(xyz, rgb, s.flags.as<std::uint32_t>(),
                                                             s.offsets.as<std::uint32_t>(), s.count,
                                                             out_xyz, out_rgb);
  return cudaGetLastError();
}

}  // namespace holoscan::pcl_filters
