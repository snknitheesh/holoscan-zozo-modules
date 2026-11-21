// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// CUDA error checking and owning device and pinned host buffers.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace holoscan::fiducials {

// @throws std::runtime_error naming `what`, the error and the GPU's free memory, unless
//         `status` is cudaSuccess. The free memory tells a GPU that other processes have nearly
//         filled from other failures.
inline void cuda_check(cudaError_t status, const char* what) {
  if (status == cudaSuccess) return;
  std::string message = std::string(what) + ": " + cudaGetErrorString(status);
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
    message += " (GPU memory: " + std::to_string(free_bytes >> 20U) + " MiB free of " +
               std::to_string(total_bytes >> 20U) + " MiB)";
  }
  throw std::runtime_error(message);
}

// `count` elements of device memory, or of pinned host memory, released on destruction.
template <typename T, bool kHost = false>
class CudaBuffer {
 public:
  CudaBuffer() = default;
  explicit CudaBuffer(std::size_t count) {
    void* data = nullptr;
    const std::size_t bytes = (count > 0 ? count : 1) * sizeof(T);
    if constexpr (kHost) {
      cuda_check(cudaMallocHost(&data, bytes), "cudaMallocHost");
    } else {
      cuda_check(cudaMalloc(&data, bytes), "cudaMalloc");
    }
    data_ = static_cast<T*>(data);
  }
  ~CudaBuffer() { release(); }
  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;
  CudaBuffer(CudaBuffer&& other) noexcept : data_(other.data_) { other.data_ = nullptr; }
  CudaBuffer& operator=(CudaBuffer&& other) noexcept {
    if (this != &other) {
      release();
      data_ = other.data_;
      other.data_ = nullptr;
    }
    return *this;
  }
  [[nodiscard]] T* get() const noexcept { return data_; }

 private:
  void release() noexcept {
    if constexpr (kHost) {
      cudaFreeHost(data_);
    } else {
      cudaFree(data_);
    }
    data_ = nullptr;
  }

  T* data_{};
};

template <typename T>
using DeviceBuffer = CudaBuffer<T, false>;
template <typename T>
using PinnedBuffer = CudaBuffer<T, true>;

}  // namespace holoscan::fiducials
