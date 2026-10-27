// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "frame_source_op.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/apriltag/synthetic_scene.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

namespace apriltag_app {
namespace {

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

holoscan::Error failure(std::string message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, std::move(message)};
}

}  // namespace

void image_output(holoscan::OperatorSpec& spec, holoscan::Output<holoscan::schema::ImageT>& image,
                  const char* name, holoscan::MemoryKind memory, int width, int height,
                  int channels) {
  const std::uint64_t bytes = std::uint64_t{static_cast<std::uint32_t>(width)} * height *
                              static_cast<std::uint32_t>(channels);
  spec.output(image, name)
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = memory,
                  .dtype = kUInt8,
                  .rank = 3U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(bytes),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      })
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
}

FrameSourceOp::FrameSourceOp(FrameSourceConfig config) : config_(std::move(config)) {
  (void)holoscan::apriltag::make_scene(config_.family, 0, config_.width,
                                       config_.height);  // Family, minimum size.
  if (config_.width > 4096 || config_.height > 4096) {
    throw std::invalid_argument("synthetic images must be 320x240 to 4096x4096");
  }
  if (config_.count == 0 || config_.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("frame source needs a positive count and period");
  }
}

void FrameSourceOp::setup(holoscan::OperatorSpec& spec) {
  image_output(
      spec, image, "image",
      config_.device_output ? holoscan::MemoryKind::kCudaDevice : holoscan::MemoryKind::kHost,
      config_.width, config_.height, 3);
}

holoscan::Contract FrameSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void FrameSourceOp::start() { emitted_ = 0; }

holoscan::expected<void, holoscan::Error> FrameSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= config_.count) {
    return {};  // Done: idle until the consumers request completion.
  }
  const std::vector<std::uint8_t> pixels = holoscan::apriltag::render_scene(
      holoscan::apriltag::make_scene(config_.family, emitted_, config_.width, config_.height), 3);
  const std::array<std::int64_t, 3> shape{config_.height, config_.width, 3};
  auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  if (config_.device_output) {
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    if (writer->byte_size() != pixels.size()) {
      return holoscan::make_unexpected(failure("image loan has an unexpected size"));
    }
    const cudaError_t copied =
        cudaMemcpyAsync(writer->data_as<std::uint8_t>(), pixels.data(), pixels.size(),
                        cudaMemcpyHostToDevice, context.cuda_stream().get());
    if (copied != cudaSuccess) {
      return holoscan::make_unexpected(failure(cudaGetErrorString(copied)));
    }
    if (auto committed = std::move(*writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
  } else {
    auto committed = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                     -> holoscan::expected<void, holoscan::Error> {
      if (writer.byte_size() != pixels.size()) {
        return holoscan::make_unexpected(failure("image loan has an unexpected size"));
      }
      std::memcpy(writer.data_as<std::uint8_t>(), pixels.data(), pixels.size());
      return {};
    });
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
  }
  holoscan::schema::ImageT descriptor;
  descriptor.width = config_.width;
  descriptor.height = config_.height;
  descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
  descriptor.header = std::make_shared<holoscan::schema::HeaderT>();
  descriptor.header->device_sequence = std::uint64_t{emitted_} + 1U;
  auto emitted =
      image.emit_tensor(std::move(*loan), descriptor,
                        holoscan::EmitOptions{.capture_time = context.activation_time()});
  if (emitted) {
    ++emitted_;
    return {};
  }
  // A busy consumer fails the emit with kBackpressured before anything is
  // delivered: send the same frame again on the next tick.
  return emitted.error().code == holoscan::ErrorCode::kBackpressured
             ? holoscan::expected<void, holoscan::Error>{}
             : emitted;
}

}  // namespace apriltag_app
