// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "camera_op.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

namespace image_quality_app {
namespace {

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

holoscan::Error failure(std::string message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "SyntheticCameraOp: " + std::move(message)};
}

}  // namespace

SyntheticCameraOp::SyntheticCameraOp(CameraConfig config) : config_(std::move(config)) {
  const Schedule& s = config_.schedule;
  if (s.width < 16 || s.height < 16 || s.width > 4096 || s.height > 4096) {
    throw std::invalid_argument("synthetic frames must be 16x16 to 4096x4096");
  }
  if (s.segment_frames <= s.obstruction_frames) {
    throw std::invalid_argument("segments must be longer than the obstruction window");
  }
  if (config_.count == 0 || config_.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("the camera needs a positive count and period");
  }
}

void SyntheticCameraOp::setup(holoscan::OperatorSpec& spec) {
  const auto bytes = static_cast<std::uint64_t>(config_.schedule.width) *
                     static_cast<std::uint64_t>(config_.schedule.height) * 3U;
  spec.output(image, "image")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = config_.device_output ? holoscan::MemoryKind::kCudaDevice
                                                       : holoscan::MemoryKind::kHost,
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

holoscan::Contract SyntheticCameraOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void SyntheticCameraOp::start() { sent_ = 0; }

holoscan::expected<void, holoscan::Error> SyntheticCameraOp::compute(
    holoscan::ExecutionContext& context) {
  if (sent_ >= config_.count) {
    return {};  // Done: idle until the reporter requests completion.
  }
  const holoscan::image_quality::HostImage frame = config_.schedule.render(sent_);
  const std::array<std::int64_t, 3> shape{frame.height, frame.width, 3};
  auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  if (config_.device_output) {
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    if (writer->byte_size() != frame.pixels.size()) {
      return holoscan::make_unexpected(failure("image loan has an unexpected size"));
    }
    // From pageable memory: staged before the call returns, so `frame` may go.
    const cudaError_t copied =
        cudaMemcpyAsync(writer->data_as<std::uint8_t>(), frame.pixels.data(), frame.pixels.size(),
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
      if (writer.byte_size() != frame.pixels.size()) {
        return holoscan::make_unexpected(failure("image loan has an unexpected size"));
      }
      std::memcpy(writer.data_as<std::uint8_t>(), frame.pixels.data(), frame.pixels.size());
      return {};
    });
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
  }
  holoscan::schema::ImageT descriptor;
  descriptor.width = frame.width;
  descriptor.height = frame.height;
  descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
  auto emitted = image.emit_tensor(
      std::move(*loan), descriptor,
      holoscan::EmitOptions{.capture_time = context.activation_time(), .frame_id = sent_});
  if (emitted) {
    ++sent_;
    return {};
  }
  // A busy consumer fails the emit with kBackpressured before anything is
  // delivered: send the same frame again on the next tick.
  return emitted.error().code == holoscan::ErrorCode::kBackpressured
             ? holoscan::expected<void, holoscan::Error>{}
             : emitted;
}

}  // namespace image_quality_app
