// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Stereo pairs of the synthetic scene as image messages, in device memory,
// the runtime's pinned or pageable host memory, or imported pinned memory,
// and frames of them with their camera info, for the operator tests.

#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/foundation_stereo/model_input.hpp>
#include <holoscan/foundation_stereo/synthetic_scene.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/messages/stereo_camera_info_schema_traits.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

namespace holoscan::foundation_stereo::test {

using holoscan::schema::ImageEncoding;

inline std::chrono::nanoseconds capture_time(std::uint32_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

// kHost: the runtime's pinned host memory (MemoryStorageType::kHost); kPageable:
// plain system memory (kSystem); kPinned: pinned memory imported through
// DLPack, which the runtime cannot hold for the GPU.
enum class Memory { kDevice, kHost, kPageable, kPinned };

inline PixelFormat format_of(ImageEncoding encoding) {
  switch (encoding) {
    case holoscan::schema::ImageEncoding_MONO8:
      return PixelFormat::kMono8;
    case holoscan::schema::ImageEncoding_BGR8:
      return PixelFormat::kBgr8;
    case holoscan::schema::ImageEncoding_RGBA8:
      return PixelFormat::kRgba8;
    case holoscan::schema::ImageEncoding_BGRA8:
      return PixelFormat::kBgra8;
    default:
      return PixelFormat::kRgb8;
  }
}

// A tensor of `shape` holding `bytes`, in `memory`.
inline std::shared_ptr<holoscan::Tensor> make_tensor(std::vector<std::uint8_t> bytes,
                                                     std::vector<std::int64_t> shape,
                                                     Memory memory) {
  if (memory != Memory::kPinned) {
    const holoscan::MemoryStorageType storage =
        memory == Memory::kDevice ? holoscan::MemoryStorageType::kDevice
        : memory == Memory::kHost ? holoscan::MemoryStorageType::kHost
                                  : holoscan::MemoryStorageType::kSystem;
    auto tensor = holoscan::make_tensor(shape, DLDataType{kDLUInt, 8, 1}, std::nullopt, storage,
                                        std::make_shared<holoscan::UnboundedAllocator>());
    if (!tensor) throw std::runtime_error(tensor.error().message);
    auto destination = tensor->mutable_data();
    if (!destination ||
        cudaMemcpy(*destination, bytes.data(), bytes.size(), cudaMemcpyDefault) != cudaSuccess) {
      throw std::runtime_error("cannot fill the tensor");
    }
    return std::make_shared<holoscan::Tensor>(std::move(*tensor));
  }
  struct Storage {
    DLManagedTensor managed{};
    std::vector<std::int64_t> shape;
    void* pinned{};
    ~Storage() {
      if (pinned != nullptr) cudaFreeHost(pinned);
    }
  };
  auto storage = std::make_unique<Storage>();
  storage->shape = std::move(shape);
  if (cudaMallocHost(&storage->pinned, bytes.size()) != cudaSuccess) {
    throw std::runtime_error("cudaMallocHost failed");
  }
  std::memcpy(storage->pinned, bytes.data(), bytes.size());
  storage->managed.dl_tensor = {.data = storage->pinned,
                                .device = {kDLCUDAHost, 0},
                                .ndim = static_cast<std::int32_t>(storage->shape.size()),
                                .dtype = {kDLUInt, 8, 1},
                                .shape = storage->shape.data(),
                                .strides = nullptr,
                                .byte_offset = 0};
  storage->managed.manager_ctx = storage.get();
  storage->managed.deleter = [](DLManagedTensor* tensor) {
    delete static_cast<Storage*>(tensor->manager_ctx);
  };
  return std::make_shared<holoscan::Tensor>(&storage.release()->managed);
}

// One eye's packed pixels and its image message.
struct Eye {
  std::vector<std::uint8_t> pixels;
  holoscan::schema::ImageT image;
};

// The synthetic scene's two views in `encoding`, from `memory`.
inline std::array<Eye, 2> scene_pair(const SyntheticScene& scene, ImageEncoding encoding,
                                     Memory memory) {
  const std::size_t count = static_cast<std::size_t>(scene.width) * scene.height;
  std::uint8_t* gpu = nullptr;
  if (cudaMalloc(&gpu, 2 * count * 3) != cudaSuccess) throw std::runtime_error("cudaMalloc");
  std::vector<std::uint8_t> rgb(2 * count * 3);
  const cudaError_t rendered = render_scene(scene, 3, gpu, gpu + count * 3, nullptr);
  const cudaError_t copied = rendered == cudaSuccess
                                 ? cudaMemcpy(rgb.data(), gpu, rgb.size(), cudaMemcpyDeviceToHost)
                                 : rendered;
  cudaFree(gpu);
  if (copied != cudaSuccess) throw std::runtime_error("render_scene failed");
  const PixelFormat format = format_of(encoding);
  const int n = channels(format);
  const bool bgr = format == PixelFormat::kBgr8 || format == PixelFormat::kBgra8;
  std::array<Eye, 2> pair;
  for (int side = 0; side < 2; ++side) {
    const std::uint8_t* from = rgb.data() + side * count * 3;
    std::vector<std::uint8_t> bytes(count * n);
    for (std::size_t i = 0; i < count; ++i) {
      for (int c = 0; c < n; ++c) {
        // MONO8 is the green channel; a pixel's alpha is opaque.
        const int source = n == 1 ? 1 : c == 3 ? -1 : bgr ? 2 - c : c;
        bytes[i * n + c] = source < 0 ? 255 : from[i * 3 + source];
      }
    }
    std::vector<std::int64_t> shape{scene.height, scene.width};
    if (n > 1) shape.push_back(n);
    pair[side].pixels = bytes;
    pair[side].image.width = scene.width;
    pair[side].image.height = scene.height;
    pair[side].image.encoding = encoding;
    pair[side].image.data = make_tensor(std::move(bytes), std::move(shape), memory);
  }
  return pair;
}

// The scene's camera info as the operator's input takes it.
inline holoscan::messages::schema::StereoCameraInfoT scene_info(const SyntheticScene& scene) {
  return holoscan::messages::to_message(scene_camera_info(scene));
}

template <typename T>
void add_field(holoscan::testing::TestFrame& frame, std::string name, T value,
               std::chrono::nanoseconds time, std::uint64_t frame_id) {
  auto sample = holoscan::testing::sample_at(std::move(value), time);
  if (frame_id != 0) sample.metadata.frame_id = frame_id;
  std::vector<holoscan::testing::TestSample<T>> samples;
  samples.push_back(std::move(sample));
  frame.fields.push_back(
      {std::move(name), holoscan::testing::staged_samples<T>(std::move(samples))});
}

// One stereo frame at `time`: its camera info first, when given, as a
// producer sending it with every pair does, then the left and right images;
// all with frame id `frame_id` when it is not 0.
inline holoscan::testing::TestFrame stereo_frame(
    std::chrono::nanoseconds time, std::array<Eye, 2> pair,
    std::optional<holoscan::messages::schema::StereoCameraInfoT> info, std::uint64_t frame_id = 0) {
  holoscan::testing::TestFrame frame;
  if (info) add_field(frame, "camera_info", std::move(*info), time, frame_id);
  add_field(frame, "left", std::move(pair[0].image), time, frame_id);
  add_field(frame, "right", std::move(pair[1].image), time, frame_id);
  return frame;
}

template <typename T>
std::vector<T> values_of(const holoscan::testing::TensorSnapshot& tensor) {
  std::vector<T> out(tensor.bytes().size() / sizeof(T));
  std::memcpy(out.data(), tensor.bytes().data(), out.size() * sizeof(T));
  return out;
}

}  // namespace holoscan::foundation_stereo::test
