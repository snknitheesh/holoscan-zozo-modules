// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Tensors for the operator tests: copies of host data in device, pinned host or
// pageable host memory.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>

namespace holoscan::isp::test {

// Where a test tensor lives: CUDA device memory, pinned host memory
// (cudaMallocHost), or pageable host memory (malloc).
enum class Memory : std::uint8_t { kDevice, kPinned, kPageable };

// A packed tensor of `shape` holding `size` bytes from `bytes`, in `memory`.
inline holoscan::Tensor make(const std::vector<std::int64_t>& shape, DLDataType dtype,
                             const void* bytes, std::size_t size, Memory memory) {
  const holoscan::MemoryStorageType storage =
      memory == Memory::kDevice   ? holoscan::MemoryStorageType::kDevice
      : memory == Memory::kPinned ? holoscan::MemoryStorageType::kHost
                                  : holoscan::MemoryStorageType::kSystem;
  auto tensor = holoscan::make_tensor(shape, dtype, std::nullopt, storage,
                                      std::make_shared<holoscan::UnboundedAllocator>());
  if (!tensor) {
    throw std::runtime_error(tensor.error().message);
  }
  auto destination = tensor->mutable_data();
  if (!destination) {
    throw std::runtime_error(destination.error().message);
  }
  if (size == 0) {
    return std::move(*tensor);
  }
  if (memory == Memory::kPageable) {
    std::memcpy(*destination, bytes, size);
    return std::move(*tensor);
  }
  // A copy from pageable memory may return before it has landed; the operator
  // reads on a stream of its own, so wait for it.
  if (cudaMemcpy(*destination, bytes, size, cudaMemcpyDefault) != cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed");
  }
  return std::move(*tensor);
}

// In device memory, or else pinned host memory.
inline holoscan::Tensor make(const std::vector<std::int64_t>& shape, DLDataType dtype,
                             const void* bytes, std::size_t size, bool device) {
  return make(shape, dtype, bytes, size, device ? Memory::kDevice : Memory::kPinned);
}

}  // namespace holoscan::isp::test
