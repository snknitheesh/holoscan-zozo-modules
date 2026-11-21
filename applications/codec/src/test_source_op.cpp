// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "test_source_op.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/codec/test_video.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

namespace codec_app {
namespace {

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

holoscan::Error failure(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "TestSourceOp: " + message};
}

}  // namespace

TestSourceOp::TestSourceOp(TestSourceConfig config) : config_(std::move(config)) {
  if (config_.width < 160 || config_.height < 160 || config_.width > 4096 ||
      config_.height > 4096) {
    throw std::invalid_argument("frames must be 160x160 to 4096x4096");
  }
  if (config_.period <= std::chrono::nanoseconds::zero() || config_.in_flight == 0) {
    throw std::invalid_argument("the source needs a positive period and frames in flight");
  }
}

void TestSourceOp::setup(holoscan::OperatorSpec& spec) {
  const std::uint64_t bytes =
      static_cast<std::uint64_t>(config_.width) * static_cast<std::uint64_t>(config_.height) * 3U;
  spec.output(image, "image")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kCudaDevice,
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

holoscan::Contract TestSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void TestSourceOp::start() {
  emitted_ = 0;
  pending_capture_.reset();
}

holoscan::expected<void, holoscan::Error> TestSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (config_.count != 0 && emitted_ >= config_.count) {
    return {};  // Done: idle until the run completes.
  }
  if (config_.done &&
      emitted_ >= config_.done->load(std::memory_order_acquire) + config_.in_flight) {
    return {};  // Too many frames out: try again on the next tick.
  }
  if (!pending_capture_) {
    pending_capture_ = context.activation_time();
  }
  pixels_ = holoscan::codec::render_test_frame(config_.width, config_.height, emitted_);
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(failure("the runtime gave no CUDA stream"));
  }
  auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::array<std::int64_t, 3>{config_.height, config_.width, 3}, .dtype = kUInt8});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  // From pageable memory the copy is staged before it returns, so the pixels may change after.
  if (const cudaError_t status = cudaMemcpyAsync(writer->data_as<std::uint8_t>(), pixels_.data(),
                                                 pixels_.size(), cudaMemcpyHostToDevice, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(failure(cudaGetErrorString(status)));
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  holoscan::schema::ImageT descriptor;
  descriptor.width = config_.width;
  descriptor.height = config_.height;
  descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
  descriptor.header = std::make_shared<holoscan::schema::HeaderT>();
  descriptor.header->device_sequence = std::uint64_t{emitted_} + 1U;
  descriptor.header->capture_timestamp_ns = pending_capture_->timestamp_ns;
  auto sent = image.emit_tensor(
      std::move(*loan), descriptor,
      holoscan::EmitOptions{.capture_time = *pending_capture_, .frame_id = emitted_ + 1U});
  if (sent) {
    pending_capture_.reset();
    ++emitted_;
    return {};
  }
  // A full queue refuses the frame before delivering it: send it again on the next tick.
  return sent.error().code == holoscan::ErrorCode::kBackpressured
             ? holoscan::expected<void, holoscan::Error>{}
             : sent;
}

}  // namespace codec_app
