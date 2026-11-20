// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Device scratch for the kernels: a buffer that grows in stream order,
// without the host waiting for the GPU.

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <utility>

namespace holoscan::height_map::detail {

// Whether `stream`'s device has CUDA's stream-ordered allocator (memory pools).
inline bool stream_ordered_allocation(cudaStream_t stream) {
  int device = 0;
  int supported = 0;
  return cudaStreamGetDevice(stream, &device) == cudaSuccess &&
         cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device) ==
             cudaSuccess &&
         supported != 0;
}

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() {
    if (data_ == nullptr) {
      return;
    }
    // cudaFree waits for nothing with stream-ordered memory: wait for the
    // device first, as the stream that last used the buffer may not exist
    // any more. Only when the operator stops or is destroyed.
    if (stream_ordered_) {
      cudaDeviceSynchronize();
    }
    cudaFree(data_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  // Makes the buffer at least `bytes` long. Grows by at least half, so that a
  // slowly growing cloud does not reallocate every frame. With the
  // stream-ordered allocator (every GPU the SDK supports has it) the new
  // buffer is allocated and the old one freed on `stream`, after the work
  // queued on it: the host never waits. Without it, growing waits for
  // `stream` to finish with the old buffer.
  cudaError_t reserve(std::size_t bytes, cudaStream_t stream) {
    if (bytes <= bytes_) {
      return cudaSuccess;
    }
    const std::size_t grown = std::max(bytes, bytes_ + bytes_ / 2U);
    if (data_ == nullptr) {
      stream_ordered_ = stream_ordered_allocation(stream);
    }
    if (stream_ordered_) {
      void* fresh = nullptr;
      if (const cudaError_t status = cudaMallocAsync(&fresh, grown, stream);
          status != cudaSuccess) {
        return status;
      }
      if (data_ != nullptr) {
        if (const cudaError_t status = cudaFreeAsync(data_, stream); status != cudaSuccess) {
          cudaFreeAsync(fresh, stream);
          return status;
        }
      }
      data_ = fresh;
      bytes_ = grown;
      return cudaSuccess;
    }
    if (data_ != nullptr) {
      if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
        return status;
      }
      cudaFree(data_);
      data_ = nullptr;
      bytes_ = 0;
    }
    if (const cudaError_t status = cudaMalloc(&data_, grown); status != cudaSuccess) {
      data_ = nullptr;
      return status;
    }
    bytes_ = grown;
    return cudaSuccess;
  }

  // Swaps two buffers grown on the same stream.
  void swap(DeviceBuffer& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(bytes_, other.bytes_);
    std::swap(stream_ordered_, other.stream_ordered_);
  }

  template <typename T>
  [[nodiscard]] T* as() const noexcept {
    return static_cast<T*>(data_);
  }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

 private:
  void* data_{};
  std::size_t bytes_{};
  bool stream_ordered_{};
};

}  // namespace holoscan::height_map::detail
