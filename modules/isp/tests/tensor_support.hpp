// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Tensors for the operator tests: copies of host data in host or device memory.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
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

// A tensor of `shape` holding `size` bytes from `bytes`, in host or device memory.
inline holoscan::Tensor make(const std::vector<std::int64_t>& shape, DLDataType dtype,
                             const void* bytes, std::size_t size, bool device) {
  auto tensor = holoscan::make_tensor(
      shape, dtype, std::nullopt,
      device ? holoscan::MemoryStorageType::kDevice : holoscan::MemoryStorageType::kHost,
      std::make_shared<holoscan::UnboundedAllocator>());
  if (!tensor) {
    throw std::runtime_error(tensor.error().message);
  }
  auto destination = tensor->mutable_data();
  if (!destination) {
    throw std::runtime_error(destination.error().message);
  }
  // A copy from pageable memory may return before it has landed; the operator
  // reads on a stream of its own, so wait for it.
  if (size != 0 && (cudaMemcpy(*destination, bytes, size, cudaMemcpyDefault) != cudaSuccess ||
                    cudaDeviceSynchronize() != cudaSuccess)) {
    throw std::runtime_error("cudaMemcpy failed");
  }
  return std::move(*tensor);
}

}  // namespace holoscan::isp::test
