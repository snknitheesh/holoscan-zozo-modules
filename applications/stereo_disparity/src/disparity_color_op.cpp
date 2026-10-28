// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "disparity_color_op.hpp"

#include <array>
#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>

namespace stereo_disparity_app {
namespace {

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "DisparityColorOp: " + message};
}

}  // namespace

DisparityColorOp::DisparityColorOp(DisparityColorConfig config) : config_(std::move(config)) {
  if (!(std::isfinite(config_.max_disparity) && config_.max_disparity > 0) ||
      config_.max_width < 1 || config_.max_height < 1) {
    throw std::invalid_argument(
        "DisparityColorOp needs a positive max_disparity, max_width and max_height");
  }
}

DisparityColorOp::~DisparityColorOp() { stop(); }

void DisparityColorOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(disparity, "disparity").queue_depth(2U);
  const std::uint64_t bytes = static_cast<std::uint64_t>(config_.max_width) *
                              static_cast<std::uint64_t>(config_.max_height) * 4U;
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

holoscan::Contract DisparityColorOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{disparity});
  return result;
}

void DisparityColorOp::stop() {
  if (own_stream_ != nullptr) {
    cudaStreamSynchronize(own_stream_);
    cudaStreamDestroy(own_stream_);
    own_stream_ = nullptr;
  }
}

holoscan::expected<void, holoscan::Error> DisparityColorOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = disparity.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  const holoscan::schema::ImageT& input = sample->data;
  if (!input.data || input.encoding != holoscan::schema::ImageEncoding_MONO32F) {
    return holoscan::make_unexpected(invalid("the disparity must be a MONO32F image"));
  }
  const holoscan::Tensor& tensor = *input.data;
  const std::span<const std::int64_t> shape = tensor.shape_span();
  const DLDataType dtype = tensor.dtype();
  const DLDevice device = tensor.device();
  if (dtype.code != kDLFloat || dtype.bits != 32 || dtype.lanes != 1 ||
      !(shape.size() == 2 || (shape.size() == 3 && shape[2] == 1)) ||
      tensor.strides()[1] != static_cast<std::int64_t>(sizeof(float)) ||
      (device.device_type != kDLCUDA && device.device_type != kDLCUDAManaged)) {
    return holoscan::make_unexpected(
        invalid("the disparity must be float32 [H, W] with packed rows in CUDA memory"));
  }
  const int height = static_cast<int>(shape[0]);
  const int width = static_cast<int>(shape[1]);
  if (width > config_.max_width || height > config_.max_height) {
    return holoscan::make_unexpected(
        invalid("the disparity is larger than the configured maximum"));
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    if (own_stream_ == nullptr) {
      if (cudaStreamCreateWithFlags(&own_stream_, cudaStreamNonBlocking) != cudaSuccess) {
        own_stream_ = nullptr;
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, "DisparityColorOp: cudaStreamCreate"});
      }
    }
    stream = own_stream_;
  }
  if (tensor.has_readiness_stream_contract()) {
    if (auto ready = tensor.synchronize_for_consumer_stream(stream); !ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
  }
  auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::array<std::int64_t, 3>{height, width, 4}, .dtype = kUInt8});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  if (const cudaError_t status = colorize_disparity(
          tensor.data_as<float>(), static_cast<std::size_t>(tensor.strides()[0]), width, height,
          config_.max_disparity, writer->data_as<std::uint8_t>(), stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure,
        std::string("DisparityColorOp: colorize_disparity: ") + cudaGetErrorString(status)});
  }
  if (stream == own_stream_) {
    cudaStreamSynchronize(stream);
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  holoscan::schema::ImageT descriptor;
  descriptor.width = width;
  descriptor.height = height;
  descriptor.encoding = holoscan::schema::ImageEncoding_RGBA8;
  descriptor.header = input.header;
  return image.emit_tensor(std::move(*loan), descriptor, holoscan::sensor_io::derive_from(*sample));
}

}  // namespace stereo_disparity_app
