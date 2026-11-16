// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Device scratch for the clusterer and the operator: a buffer that grows in
// stream order, without waiting for the host.

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>

namespace holoscan::clustering::detail {

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() {
    // cudaFree waits for the device to be done with stream-ordered memory.
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  // Makes the buffer at least `bytes` long, on the device of `stream` (from
  // its device's memory pool). Grows by at least half, so that a slowly
  // growing cloud does not reallocate every frame. The old buffer is freed in
  // `stream`'s order, after the work queued on it so far, and the new one is
  // usable by what is queued next: neither waits for the host. The contents
  // are not kept.
  cudaError_t reserve(std::size_t bytes, cudaStream_t stream) {
    if (bytes <= bytes_) {
      return cudaSuccess;
    }
    const std::size_t grown = std::max(bytes, bytes_ + bytes_ / 2U);
    if (data_ != nullptr) {
      if (const cudaError_t status = cudaFreeAsync(data_, stream); status != cudaSuccess) {
        return status;
      }
      data_ = nullptr;
      bytes_ = 0;
    }
    if (const cudaError_t status = cudaMallocAsync(&data_, grown, stream); status != cudaSuccess) {
      data_ = nullptr;
      return status;
    }
    bytes_ = grown;
    return cudaSuccess;
  }

  template <typename T>
  [[nodiscard]] T* as() const noexcept {
    return static_cast<T*>(data_);
  }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

 private:
  void* data_{};
  std::size_t bytes_{};
};

}  // namespace holoscan::clustering::detail
