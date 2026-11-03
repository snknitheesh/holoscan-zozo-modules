// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Keep masks to clouds: the input's size, removed points NaN, or packed; and
// the padding after a packed cloud.

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

__global__ void apply_mask_kernel(CloudFields in, const std::uint8_t* __restrict__ keep,
                                  std::uint32_t count, CloudOutput out) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  if (keep[i] != 0U) {
    detail::copy_point(in, i, out, i);
  } else {
    detail::remove_point(out, i);
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

__global__ void scatter_kernel(CloudFields in, const std::uint32_t* __restrict__ flags,
                               const std::uint32_t* __restrict__ offsets, std::uint32_t count,
                               CloudOutput out) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count && flags[i] != 0U) {
    detail::copy_point(in, i, out, offsets[i]);
  }
}

__global__ void pad_kernel(CloudOutput out, const std::uint32_t* __restrict__ device_count,
                           std::uint32_t capacity) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < capacity && i >= *device_count) {
    detail::remove_point(out, i);
  }
}

}  // namespace

cudaError_t pad_cloud(const CloudOutput& out, const std::uint32_t* device_count,
                      std::uint32_t capacity, cudaStream_t stream) {
  if (capacity == 0U) {
    return cudaSuccess;
  }
  if (out.xyz == nullptr || device_count == nullptr) {
    return cudaErrorInvalidValue;
  }
  pad_kernel<<<blocks_for(capacity), kBlock, 0, stream>>>(out, device_count, capacity);
  return cudaGetLastError();
}

cudaError_t apply_mask(const CloudFields& in, const std::uint8_t* keep, std::uint32_t count,
                       const CloudOutput& out, cudaStream_t stream) {
  if (count == 0U) {
    return cudaSuccess;
  }
  if (!detail::fields_match(in, out)) {
    return cudaErrorInvalidValue;
  }
  apply_mask_kernel<<<blocks_for(count), kBlock, 0, stream>>>(in, keep, count, out);
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

cudaError_t Compactor::write(const CloudFields& in, const CloudOutput& out, cudaStream_t stream) {
  const Impl& s = *impl_;
  if (s.count == 0U) {
    return cudaSuccess;
  }
  if (!detail::fields_match(in, out)) {
    return cudaErrorInvalidValue;
  }
  scatter_kernel<<<blocks_for(s.count), kBlock, 0, stream>>>(
      in, s.flags.as<std::uint32_t>(), s.offsets.as<std::uint32_t>(), s.count, out);
  return cudaGetLastError();
}

}  // namespace holoscan::pcl_filters
