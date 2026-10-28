// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Images for the operator tests: host data copied into a host or device tensor,
// wrapped in an image descriptor; and the pixels of a captured output image.

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

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

namespace holoscan::rectify::test {

// A tensor of `shape` holding `size` bytes from `bytes`, in host or device memory.
inline holoscan::Tensor make(const std::vector<std::int64_t>& shape, DLDataType dtype,
                             const void* bytes, std::size_t size, bool device) {
  auto tensor = holoscan::make_tensor(
      shape, dtype, std::nullopt,
      device ? holoscan::MemoryStorageType::kDevice : holoscan::MemoryStorageType::kHost,
      std::make_shared<holoscan::UnboundedAllocator>());
  if (!tensor) throw std::runtime_error(tensor.error().message);
  auto destination = tensor->mutable_data();
  if (!destination) throw std::runtime_error(destination.error().message);
  if (size != 0 && cudaMemcpy(*destination, bytes, size, cudaMemcpyDefault) != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed");
  }
  return std::move(*tensor);
}

// A packed image of `channels` 8-bit (or, for MONO16, 16-bit) values per pixel.
template <typename T>
holoscan::schema::ImageT image(const std::vector<T>& pixels, int width, int height, int channels,
                               holoscan::schema::ImageEncoding encoding, bool device,
                               std::uint64_t sequence = 0) {
  const DLDataType dtype{kDLUInt, static_cast<std::uint8_t>(8 * sizeof(T)), 1};
  std::vector<std::int64_t> shape{height, width};
  if (channels > 1) shape.push_back(channels);
  holoscan::schema::ImageT result;
  result.width = width;
  result.height = height;
  result.encoding = encoding;
  result.header = std::make_shared<holoscan::schema::HeaderT>();
  result.header->device_sequence = sequence;
  result.header->frame_id = "camera";
  result.data = std::make_shared<holoscan::Tensor>(
      make(shape, dtype, pixels.data(), pixels.size() * sizeof(T), device));
  return result;
}

// The pixels of a captured image.
template <typename T>
std::vector<T> pixels(const holoscan::testing::Snapshot<holoscan::schema::ImageT>& snapshot) {
  const auto& tensor = snapshot.tensor(holoscan::schema::ImageTensorFields::data);
  std::vector<T> values(tensor.bytes().size() / sizeof(T));
  std::memcpy(values.data(), tensor.bytes().data(), values.size() * sizeof(T));
  return values;
}

}  // namespace holoscan::rectify::test
