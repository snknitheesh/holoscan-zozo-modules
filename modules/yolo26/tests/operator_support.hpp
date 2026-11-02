// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What the operator tests share: images in GPU, host or pinned memory, the
// synthetic scenes as images, and the values of a captured tensor.

#pragma once

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#include <holoscan/yolo26/letterbox.hpp>
#include <holoscan/yolo26/synthetic_scene.hpp>

#include "test_support.hpp"

namespace holoscan::yolo26::test {

using holoscan::schema::ImageEncoding;
using holoscan::testing::TensorSnapshot;

enum class Memory { kDevice, kHost, kPinned };

// A tensor of `dtype` holding `bytes`: in plain host memory, or copied into
// pinned host or GPU memory.
inline std::shared_ptr<holoscan::Tensor> image_tensor(std::vector<std::uint8_t> bytes,
                                                      std::vector<std::int64_t> shape,
                                                      Memory memory,
                                                      DLDataType dtype = {kDLUInt, 8, 1}) {
  if (memory == Memory::kDevice) {
    // From the SDK's allocator, whose memory says when it is ready to read.
    auto tensor =
        holoscan::make_tensor(shape, dtype, std::nullopt, holoscan::MemoryStorageType::kDevice,
                              std::make_shared<holoscan::UnboundedAllocator>());
    if (!tensor) {
      throw std::runtime_error(tensor.error().message);
    }
    auto destination = tensor->mutable_data();
    if (!destination || cudaMemcpy(*destination, bytes.data(), bytes.size(),
                                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("cannot copy the image to the GPU");
    }
    return std::make_shared<holoscan::Tensor>(std::move(*tensor));
  }
  struct Storage {
    DLManagedTensor managed{};
    std::vector<std::int64_t> shape;
    std::vector<std::uint8_t> bytes;
    void* pinned{};
    ~Storage() {
      if (pinned != nullptr) cudaFreeHost(pinned);
    }
  };
  auto storage = std::make_unique<Storage>();
  storage->shape = std::move(shape);
  storage->bytes = std::move(bytes);
  void* data = storage->bytes.data();
  DLDeviceType type = kDLCPU;
  if (memory == Memory::kPinned) {
    if (cudaMallocHost(&storage->pinned, storage->bytes.size()) != cudaSuccess) {
      throw std::runtime_error("cudaMallocHost failed");
    }
    std::memcpy(storage->pinned, storage->bytes.data(), storage->bytes.size());
    data = storage->pinned;
    type = kDLCUDAHost;
  }
  storage->managed.dl_tensor = {.data = data,
                                .device = {type, 0},
                                .ndim = static_cast<std::int32_t>(storage->shape.size()),
                                .dtype = dtype,
                                .shape = storage->shape.data(),
                                .strides = nullptr,
                                .byte_offset = 0};
  storage->managed.manager_ctx = storage.get();
  storage->managed.deleter = [](DLManagedTensor* tensor) {
    delete static_cast<Storage*>(tensor->manager_ctx);
  };
  return std::make_shared<holoscan::Tensor>(&storage.release()->managed);
}

inline ImageEncoding encoding_of(PixelFormat format) {
  switch (format) {
    case PixelFormat::kBgr8:
      return holoscan::schema::ImageEncoding_BGR8;
    case PixelFormat::kRgba8:
      return holoscan::schema::ImageEncoding_RGBA8;
    case PixelFormat::kBgra8:
      return holoscan::schema::ImageEncoding_BGRA8;
    default:
      return holoscan::schema::ImageEncoding_RGB8;
  }
}

// `scene` drawn in `format` and labeled `label` (by default, as drawn).
inline holoscan::schema::ImageT scene_image(const SyntheticScene& scene, PixelFormat format,
                                            Memory memory = Memory::kHost,
                                            std::optional<ImageEncoding> label = std::nullopt) {
  const int n = channels(format);
  const std::size_t pitch = static_cast<std::size_t>(scene.width) * n;
  std::vector<std::uint8_t> pixels(pitch * scene.height);
  render_scene(scene, format, pixels.data(), pitch);
  holoscan::schema::ImageT image;
  image.width = scene.width;
  image.height = scene.height;
  image.encoding = label.value_or(encoding_of(format));
  image.data = image_tensor(std::move(pixels), {scene.height, scene.width, n}, memory);
  return image;
}

// The harness stages the images in GPU memory, `max_bytes` for the largest.
template <typename Op>
void stage_on_gpu(holoscan::testing::OperatorTestHarness<Op>& h, std::uint64_t max_bytes) {
  h.input_source_tensor_spec("image",
                             holoscan::TensorOutputSpec{
                                 .representation =
                                     holoscan::TensorRepresentation{
                                         .memory_kind = holoscan::MemoryKind::kCudaDevice,
                                         .dtype = DLDataType{kDLUInt, 8, 1},
                                         .rank = 3U,
                                         .strides = holoscan::TensorStridePolicy::kContiguousOnly,
                                     },
                                 .bounds = holoscan::tensor_bounds(max_bytes),
                                 .storage = holoscan::TensorOutputStorage::kRuntimePool,
                             });
  h.input_source_device("image", holoscan::DeviceId{0});
}

template <typename T>
std::vector<T> values(const TensorSnapshot& snapshot, const std::vector<std::int64_t>& shape) {
  VERIFY(snapshot.shape() == shape);
  std::vector<T> out(snapshot.bytes().size() / sizeof(T));
  VERIFY(snapshot.bytes().size() == out.size() * sizeof(T));
  std::memcpy(out.data(), snapshot.bytes().data(), out.size() * sizeof(T));
  return out;
}

struct Frame {
  std::vector<BoxDetection> detections;
  holoscan::SampleMetadata metadata;
};

// Each frame's detections from the four outputs, after checking their shapes,
// that slots past the count are zero, and that all four carry one capture
// time and frame id.
inline std::vector<Frame> frames_of(const holoscan::testing::TestResult& result, int capacity) {
  const auto boxes = result.snapshots<holoscan::Tensor>("boxes");
  const auto scores = result.snapshots<holoscan::Tensor>("scores");
  const auto class_ids = result.snapshots<holoscan::Tensor>("class_ids");
  const auto counts = result.snapshots<holoscan::Tensor>("count");
  VERIFY(boxes.size() == counts.size() && scores.size() == counts.size() &&
         class_ids.size() == counts.size());
  std::vector<Frame> frames;
  for (std::size_t f = 0;
       f < counts.size() && f < boxes.size() && f < scores.size() && f < class_ids.size(); ++f) {
    const auto b = values<float>(boxes[f], {capacity, 4});
    const auto s = values<float>(scores[f], {capacity});
    const auto c = values<std::int32_t>(class_ids[f], {capacity});
    const auto n = values<std::int32_t>(counts[f], {1});
    VERIFY(boxes[f].dtype().code == kDLFloat && class_ids[f].dtype().code == kDLInt);
    Frame frame;
    frame.metadata = counts[f].sample_metadata();
    for (const TensorSnapshot* other : {&boxes[f], &scores[f], &class_ids[f]}) {
      VERIFY(other->sample_metadata().capture_timestamp_ns == frame.metadata.capture_timestamp_ns &&
             other->sample_metadata().frame_id == frame.metadata.frame_id);
    }
    if (n.empty() || b.size() != static_cast<std::size_t>(capacity) * 4U || s.size() != c.size()) {
      continue;
    }
    VERIFY(n[0] >= 0 && n[0] <= capacity);
    for (int i = 0; i < capacity; ++i) {
      const auto k = static_cast<std::size_t>(i);
      if (i < n[0]) {
        frame.detections.push_back(
            BoxDetection{b[4 * k], b[4 * k + 1], b[4 * k + 2], b[4 * k + 3], s[k], c[k]});
      } else {
        VERIFY_MSG(b[4 * k] == 0 && b[4 * k + 1] == 0 && b[4 * k + 2] == 0 && b[4 * k + 3] == 0 &&
                       s[k] == 0 && c[k] == 0,
                   "slot " + std::to_string(i) + " past the count is not zero");
      }
    }
    frames.push_back(std::move(frame));
  }
  return frames;
}

inline void check_truth(const Frame& frame, const SyntheticScene& scene, int input_size) {
  const float scale = make_letterbox(scene.width, scene.height, input_size).scale;
  const TruthMatch match =
      match_truth(frame.detections, scene_truth(scene), 2.0F / scale + 1.0F, 1e-6F);
  VERIFY_MSG(match.missed == 0 && match.extra == 0,
             std::to_string(scene.width) + "x" + std::to_string(scene.height) + ": missed " +
                 std::to_string(match.missed) + ", extra " + std::to_string(match.extra) +
                 ", edge error " + std::to_string(match.max_edge_error));
}

inline std::chrono::nanoseconds capture_time(std::uint32_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

}  // namespace holoscan::yolo26::test
