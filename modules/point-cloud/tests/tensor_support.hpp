// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Tensors for the operator tests: copies of host data in host or device
// memory, and tensors over memory whose allocator the test plays, to see how
// an operator holds and waits for its input.

#pragma once

#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>

namespace holoscan::point_cloud::test {

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
  if (size != 0 && cudaMemcpy(*destination, bytes, size, cudaMemcpyDefault) != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed");
  }
  return std::move(*tensor);
}

// Pinned host or device memory whose allocator the test plays: it counts the
// streams it is asked to wait for before reuse (and can refuse to wait), and
// the host waits for the data to be ready. Without `ready_on_stream`, it is
// host-written memory that no stream makes ready.
class TestMemory final : public holoscan::TensorMemoryReference {
 public:
  TestMemory(std::size_t bytes, bool device, bool waits_for_streams, bool ready_on_stream = true)
      : device_(device), waits_(waits_for_streams), ready_on_stream_(ready_on_stream) {
    if ((device ? cudaMalloc(&base_, bytes) : cudaMallocHost(&base_, bytes)) != cudaSuccess) {
      throw std::runtime_error("allocation failed");
    }
  }
  ~TestMemory() override { device_ ? cudaFree(base_) : cudaFreeHost(base_); }
  TestMemory(const TestMemory&) = delete;
  TestMemory& operator=(const TestMemory&) = delete;

  bool set_deallocation_stream(cudaStream_t stream) override {
    if (waits_ && stream != nullptr) {
      streams_.fetch_add(1);
    }
    return waits_;
  }
  [[nodiscard]] bool tracks_deallocation_stream() const noexcept override { return waits_; }
  // The data is written before the tensor is made, so readiness is immediate.
  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override {
    return ready_on_stream_;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> synchronize_for_host() override {
    host_waits_.fetch_add(1);
    return {};
  }

  [[nodiscard]] std::uint8_t* base() const { return static_cast<std::uint8_t*>(base_); }
  [[nodiscard]] int streams() const { return streams_.load(); }
  [[nodiscard]] int host_waits() const { return host_waits_.load(); }

 private:
  void* base_{};
  bool device_{};
  bool waits_{};
  bool ready_on_stream_{};
  std::atomic<int> streams_{0};
  std::atomic<int> host_waits_{0};
};

// A packed tensor of `shape` at `offset` bytes into `memory`, labeled as on
// GPU `device_id` when `device`.
inline holoscan::Tensor wrap(const std::shared_ptr<TestMemory>& memory, std::size_t offset,
                             const std::vector<std::int64_t>& shape, DLDataType dtype, bool device,
                             std::int32_t device_id = 0) {
  auto context = std::make_shared<holoscan::DLManagedTensorContext>();
  context->dl_shape = shape;
  context->dl_strides.assign(shape.size(), 1);
  for (std::size_t axis = shape.size() - 1; axis-- > 0;) {
    context->dl_strides[axis] = context->dl_strides[axis + 1] * shape[axis + 1];
  }
  DLTensor& tensor = context->tensor.dl_tensor;
  tensor.data = memory->base() + offset;
  tensor.device = DLDevice{device ? kDLCUDA : kDLCUDAHost, device_id};
  tensor.ndim = static_cast<std::int32_t>(shape.size());
  tensor.dtype = dtype;
  tensor.shape = context->dl_shape.data();
  tensor.strides = context->dl_strides.data();
  tensor.byte_offset = 0;
  context->memory_reference = memory;
  return holoscan::Tensor(std::move(context));
}

// Sleeps on `stream`, so that work queued after it is still pending a while.
inline bool stall(cudaStream_t stream) {
  return cudaLaunchHostFunc(
             stream, [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds(300)); },
             nullptr) == cudaSuccess;
}

}  // namespace holoscan::point_cloud::test
