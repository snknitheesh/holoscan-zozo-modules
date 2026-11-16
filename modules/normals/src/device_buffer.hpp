// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Device scratch for the kernels: a buffer that grows, and is freed only after
// the stream is done with it.

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>

namespace holoscan::normals::detail {

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  // Makes the buffer at least `bytes` long. Grows by at least half, so that a
  // slowly growing cloud reallocates a few times rather than every frame. A
  // buffer that is replaced is freed only after `stream` is done with it,
  // which waits for the stream; the first allocation waits for nothing.
  cudaError_t reserve(std::size_t bytes, cudaStream_t stream) {
    if (bytes <= bytes_) {
      return cudaSuccess;
    }
    const std::size_t grown = std::max(bytes, bytes_ + bytes_ / 2U);
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

  template <typename T>
  [[nodiscard]] T* as() const noexcept {
    return static_cast<T*>(data_);
  }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

 private:
  void* data_{};
  std::size_t bytes_{};
};

}  // namespace holoscan::normals::detail
