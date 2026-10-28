// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "stereo_source_op.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

namespace rectify_app {
namespace {

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

holoscan::Error failure(std::string message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "StereoSourceOp: " + std::move(message)};
}

void declare(holoscan::OperatorSpec& spec, holoscan::Output<holoscan::schema::ImageT>& port,
             const char* name, bool device, int width, int height) {
  spec.output(port, name)
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind =
                      device ? holoscan::MemoryKind::kCudaDevice : holoscan::MemoryKind::kHost,
                  .dtype = kUInt8,
                  .rank = 2U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(static_cast<std::uint64_t>(width) *
                                            static_cast<std::uint64_t>(height)),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      })
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
}

}  // namespace

StereoSourceOp::StereoSourceOp(StereoSourceConfig config) : config_(std::move(config)) {
  if (config_.count == 0 || config_.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("the source needs a positive count and period");
  }
  if (config_.scene) {
    config_.width = config_.scene->calibration.left.width;
    config_.height = config_.scene->calibration.left.height;
  } else {
    frames_ = read_index(config_.recording);
    const GrayImage first = read_pgm(frames_.front().left);
    config_.width = first.width;
    config_.height = first.height;
  }
}

void StereoSourceOp::setup(holoscan::OperatorSpec& spec) {
  declare(spec, left, "left", config_.device_output, config_.width, config_.height);
  declare(spec, right, "right", config_.device_output, config_.width, config_.height);
}

holoscan::Contract StereoSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void StereoSourceOp::start() {
  emitted_ = 0;
  pending_.reset();
  left_sent_ = false;
  // The views undistort every pixel once; their frames are then cheap.
  if (config_.scene && !left_view_) {
    left_view_.emplace(holoscan::rectify::left_view(*config_.scene));
    right_view_.emplace(holoscan::rectify::right_view(*config_.scene));
  }
}

holoscan::expected<bool, holoscan::Error> StereoSourceOp::send(
    holoscan::Output<holoscan::schema::ImageT>& port, const std::vector<std::uint8_t>& pixels,
    holoscan::ExecutionContext& context, const holoscan::EmitOptions& emit) {
  const std::array<std::int64_t, 2> shape{config_.height, config_.width};
  auto loan = port.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
  if (!loan) return holoscan::make_unexpected(std::move(loan).error());
  if (config_.device_output) {
    auto writer = loan->write(context.cuda_stream());
    if (!writer) return holoscan::make_unexpected(std::move(writer).error());
    if (const cudaError_t copied =
            cudaMemcpyAsync(writer->data(), pixels.data(), pixels.size(), cudaMemcpyHostToDevice,
                            context.cuda_stream().get());
        copied != cudaSuccess) {
      return holoscan::make_unexpected(failure(cudaGetErrorString(copied)));
    }
    if (auto committed = std::move(*writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
  } else {
    auto committed = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                     -> holoscan::expected<void, holoscan::Error> {
      std::memcpy(writer.data(), pixels.data(), pixels.size());
      return {};
    });
    if (!committed) return holoscan::make_unexpected(std::move(committed).error());
  }
  holoscan::schema::ImageT descriptor;
  descriptor.width = config_.width;
  descriptor.height = config_.height;
  descriptor.encoding = holoscan::schema::ImageEncoding_MONO8;
  descriptor.header = std::make_shared<holoscan::schema::HeaderT>();
  descriptor.header->device_sequence = std::uint64_t{emitted_} + 1U;
  descriptor.header->frame_id = "stereo";
  auto sent = port.emit_tensor(std::move(*loan), descriptor, emit);
  if (sent) return true;
  // A busy consumer refuses the image before anything is delivered.
  if (sent.error().code == holoscan::ErrorCode::kBackpressured) return false;
  return holoscan::make_unexpected(std::move(sent).error());
}

holoscan::expected<void, holoscan::Error> StereoSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= config_.count) return {};  // Done: idle until the run completes.
  if (!pending_ && config_.done && config_.done() < emitted_) return {};  // One in flight.
  if (!pending_) {
    if (config_.scene) {
      left_pixels_ = left_view_->render(emitted_);
      right_pixels_ = right_view_->render(emitted_);
    } else {
      const RecordedFrame& frame = frames_[emitted_ % frames_.size()];
      try {
        GrayImage l = read_pgm(frame.left);
        GrayImage r = read_pgm(frame.right);
        if (l.width != config_.width || l.height != config_.height || r.width != config_.width ||
            r.height != config_.height) {
          return holoscan::make_unexpected(failure(frame.left.string() + ": the size changed"));
        }
        left_pixels_ = std::move(l.pixels);
        right_pixels_ = std::move(r.pixels);
      } catch (const std::exception& error) {
        return holoscan::make_unexpected(failure(error.what()));
      }
    }
    pending_ = holoscan::EmitOptions{.capture_time = context.activation_time(),
                                     .frame_id = std::uint64_t{emitted_} + 1U};
    left_sent_ = false;
  }
  if (!left_sent_) {
    auto sent = send(left, left_pixels_, context, *pending_);
    if (!sent) return holoscan::make_unexpected(std::move(sent).error());
    if (!*sent) return {};  // Busy: the whole pair again on the next tick.
    left_sent_ = true;
  }
  auto sent = send(right, right_pixels_, context, *pending_);
  if (!sent) return holoscan::make_unexpected(std::move(sent).error());
  if (!*sent) return {};  // Busy: the right image again on the next tick.
  pending_.reset();
  ++emitted_;
  return {};
}

}  // namespace rectify_app
