// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <memory>

#include "compact.cuh"
#include "device_buffer.hpp"
#include "kernel_common.cuh"
#include "nvtx_range.hpp"

namespace holoscan::normals::detail {
namespace {

__device__ inline bool has_normal(const float* normals, std::uint32_t i) {
  return isfinite(normals[3 * i]);
}

template <typename T>
__device__ inline T missing();
template <>
__device__ inline float missing<float>() {
  return nan_float();
}
template <>
__device__ inline std::uint8_t missing<std::uint8_t>() {
  return 0U;
}

template <typename T>
__global__ void keep_kernel(const T* __restrict__ in, int channels,
                            const float* __restrict__ normals, std::uint32_t count,
                            T* __restrict__ out) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const bool keep = has_normal(normals, i);
  const std::size_t at = static_cast<std::size_t>(i) * static_cast<std::size_t>(channels);
  for (int c = 0; c < channels; ++c) {
    out[at + c] = keep ? in[at + c] : missing<T>();
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

template <typename T>
__global__ void pack_kernel(const std::uint32_t* __restrict__ flags,
                            const std::uint32_t* __restrict__ offsets, std::uint32_t count,
                            const T* __restrict__ in, int channels, T* __restrict__ out) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count || flags[i] == 0U) {
    return;
  }
  const std::size_t from = static_cast<std::size_t>(i) * static_cast<std::size_t>(channels);
  const std::size_t to = static_cast<std::size_t>(offsets[i]) * static_cast<std::size_t>(channels);
  for (int c = 0; c < channels; ++c) {
    out[to + c] = in[from + c];
  }
}

template <typename T>
cudaError_t keep(const T* in, int channels, const float* normals, std::uint32_t count, T* out,
                 cudaStream_t stream) {
  if (count == 0U) {
    return cudaSuccess;
  }
  if (in == nullptr || normals == nullptr || out == nullptr || channels < 1) {
    return cudaErrorInvalidValue;
  }
  keep_kernel<T><<<blocks_for(count), kBlock, 0, stream>>>(in, channels, normals, count, out);
  return cudaGetLastError();
}

}  // namespace

cudaError_t keep_with_normals(const float* in, int channels, const float* normals,
                              std::uint32_t count, float* out, cudaStream_t stream) {
  return keep(in, channels, normals, count, out, stream);
}

cudaError_t keep_with_normals(const std::uint8_t* in, int channels, const float* normals,
                              std::uint32_t count, std::uint8_t* out, cudaStream_t stream) {
  return keep(in, channels, normals, count, out, stream);
}

struct Compactor::Impl {
  DeviceBuffer flags;
  DeviceBuffer offsets;
  DeviceBuffer total;
  DeviceBuffer temp;
  std::uint32_t count{};

  template <typename T>
  cudaError_t pack(const T* in, int channels, T* out, cudaStream_t stream) const {
    if (count == 0U) {
      return cudaSuccess;
    }
    if (in == nullptr || out == nullptr || channels < 1) {
      return cudaErrorInvalidValue;
    }
    pack_kernel<T><<<blocks_for(count), kBlock, 0, stream>>>(
        flags.as<std::uint32_t>(), offsets.as<std::uint32_t>(), count, in, channels, out);
    return cudaGetLastError();
  }
};

Compactor::Compactor() : impl_(std::make_unique<Impl>()) {}
Compactor::~Compactor() = default;

cudaError_t Compactor::count(const float* normals, std::uint32_t count, cudaStream_t stream) {
  const NvtxRange range("holoscan_normals: compaction count");
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

cudaError_t Compactor::pack(const float* in, int channels, float* out, cudaStream_t stream) {
  return impl_->pack(in, channels, out, stream);
}

cudaError_t Compactor::pack(const std::uint8_t* in, int channels, std::uint8_t* out,
                            cudaStream_t stream) {
  return impl_->pack(in, channels, out, stream);
}

}  // namespace holoscan::normals::detail
