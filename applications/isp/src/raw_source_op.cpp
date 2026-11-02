// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "raw_source_op.hpp"

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/isp/synthetic_raw_cuda.hpp>

namespace isp_demo {
namespace {

namespace isp = holoscan::isp;

bool packed(isp::RawFormat format) {
  return format == isp::RawFormat::kRaw10 || format == isp::RawFormat::kRaw12;
}

DLDataType sample_type(isp::RawFormat format) {
  return format == isp::RawFormat::kUint16 ? DLDataType{kDLUInt, 16U, 1U}
                                           : DLDataType{kDLUInt, 8U, 1U};
}

holoscan::schema::ImageEncoding encoding(isp::BayerPattern pattern) {
  switch (pattern) {
    case isp::BayerPattern::kBGGR:
      return holoscan::schema::ImageEncoding_BAYER_BGGR;
    case isp::BayerPattern::kGRBG:
      return holoscan::schema::ImageEncoding_BAYER_GRBG;
    case isp::BayerPattern::kGBRG:
      return holoscan::schema::ImageEncoding_BAYER_GBRG;
    case isp::BayerPattern::kRGGB:
      break;
  }
  return holoscan::schema::ImageEncoding_BAYER_RGGB;
}

}  // namespace

void validate_config(const RawSourceConfig& config) {
  static_cast<void>(isp::make_synthetic_scene(config.camera));
  if (config.count == 0 || config.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("the raw source needs a positive count and period");
  }
}

RawSourceOp::RawSourceOp(RawSourceConfig config) : config_(std::move(config)) {
  validate_config(config_);
}

void RawSourceOp::setup(holoscan::OperatorSpec& spec) {
  const std::uint64_t bytes =
      isp::raw_row_bytes(config_.camera.format, config_.camera.width) * config_.camera.height;
  spec.output(raw, "raw")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kCudaDevice,
                  .dtype = sample_type(config_.camera.format),
                  .rank = 2U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(bytes),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      })
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
}

holoscan::Contract RawSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void RawSourceOp::start() { emitted_ = 0; }

holoscan::expected<void, holoscan::Error> RawSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= config_.count) {
    return {};  // Done: idle until the checker completes the run.
  }
  if (config_.finished && config_.finished->load(std::memory_order_acquire) < emitted_) {
    return {};  // The previous frame is still in flight.
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure, "RawSourceOp: the runtime gave no CUDA stream"});
  }
  isp::SyntheticRawConfig camera = config_.camera;
  camera.frame = emitted_;
  const int width = camera.width;
  const int height = camera.height;
  const std::size_t row_bytes = isp::raw_row_bytes(camera.format, width);
  const std::array<std::int64_t, 2> shape{
      height, packed(camera.format) ? static_cast<std::int64_t>(row_bytes) : width};
  auto loan = raw.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = shape, .dtype = sample_type(camera.format)});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  if (const cudaError_t status = isp::render_synthetic_raw_cuda(
          isp::make_synthetic_scene(camera), writer->data_as<void>(), row_bytes, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure,
                        std::string("RawSourceOp: render: ") + cudaGetErrorString(status)});
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  holoscan::schema::ImageT descriptor;
  descriptor.width = width;
  descriptor.height = height;
  descriptor.encoding = encoding(camera.pattern);
  // The schema bounds significant bits by the sample type; packed samples carry their own.
  descriptor.significant_bits =
      packed(camera.format) ? 0U : static_cast<std::uint8_t>(camera.bit_depth);
  descriptor.header = std::make_shared<holoscan::schema::HeaderT>();
  descriptor.header->frame_id = "synthetic_camera";
  descriptor.exposure_time_ns = kExposureNs;
  descriptor.gain = kGain;
  // Frame ids count from 1; 0 means none.
  auto sent = raw.emit_tensor(
      std::move(*loan), std::move(descriptor),
      holoscan::EmitOptions{.capture_time = context.activation_time(), .frame_id = emitted_ + 1U});
  if (!sent && sent.error().code == holoscan::ErrorCode::kBackpressured) {
    return {};  // A consumer still busy refused it before delivery: send it on the next tick.
  }
  if (!sent) {
    return sent;
  }
  ++emitted_;
  return {};
}

}  // namespace isp_demo
