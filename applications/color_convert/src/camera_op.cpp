// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "camera_op.hpp"

#include <cuda_runtime_api.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/color_convert/image_view.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

namespace color_convert_app {
namespace {

namespace cc = holoscan::color_convert;

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

holoscan::Error failure(const std::string& what) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "SyntheticCameraOp: " + what};
}

}  // namespace

void validate_config(const SyntheticCameraConfig& config) {
  if (config.format != cc::PixelFormat::kNv12 && config.format != cc::PixelFormat::kYuyv) {
    throw std::invalid_argument("the camera sends nv12 or yuyv");
  }
  if (config.width < 16 || config.height < 16 || config.width > 8192 || config.height > 8192 ||
      config.width % 2 != 0 || config.height % 2 != 0) {
    throw std::invalid_argument("the camera's width and height must be even, 16..8192");
  }
  if (config.period.count() < 1 || config.period.count() > 1000) {
    throw std::invalid_argument("the camera period must be 1..1000 ms");
  }
}

SyntheticCameraOp::SyntheticCameraOp(SyntheticCameraConfig config) : config_(std::move(config)) {
  validate_config(config_);
}

SyntheticCameraOp::~SyntheticCameraOp() { stop(); }

void SyntheticCameraOp::setup(holoscan::OperatorSpec& spec) {
  const bool nv12 = config_.format == cc::PixelFormat::kNv12;
  spec.output(image, "image")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kCudaDevice,
                  .dtype = kUInt8,
                  .rank = nv12 ? 1U : 3U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(
              cc::packed_size(config_.format, config_.width, config_.height)),
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

void SyntheticCameraOp::start() {
  sent_ = 0;
  converter_ = std::make_unique<cc::Converter>(cc::Backend::kCuda);
  if (rgb_ == nullptr && cudaMalloc(&rgb_, cc::packed_size(cc::PixelFormat::kRgb8, config_.width,
                                                           config_.height)) != cudaSuccess) {
    rgb_ = nullptr;
    throw std::runtime_error("SyntheticCameraOp: cudaMalloc failed");
  }
}

void SyntheticCameraOp::stop() {
  if (rgb_ != nullptr) {
    cudaFree(rgb_);  // Waits for the work that may still use it.
    rgb_ = nullptr;
  }
  converter_.reset();
}

holoscan::expected<void, holoscan::Error> SyntheticCameraOp::compute(
    holoscan::ExecutionContext& context) {
  if ((config_.count != 0 && sent_ >= config_.count) || (config_.done && config_.done() < sent_)) {
    return {};
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(failure("the runtime gave no CUDA stream"));
  }
  const int w = config_.width;
  const int h = config_.height;
  const cc::ImageView rgb =
      cc::packed_view(rgb_, cc::PixelFormat::kRgb8, w, h, cc::MemoryLocation::kDevice);
  const bool nv12 = config_.format == cc::PixelFormat::kNv12;
  const std::vector<std::int64_t> shape = nv12
                                              ? std::vector<std::int64_t>{static_cast<std::int64_t>(
                                                    cc::packed_size(config_.format, w, h))}
                                              : std::vector<std::int64_t>{h, w, 2};
  auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  if (const cudaError_t status = cc::launch_test_pattern(rgb, sent_, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(failure(cudaGetErrorString(status)));
  }
  try {
    (void)converter_->convert(
        rgb, cc::packed_view(writer->data(), config_.format, w, h, cc::MemoryLocation::kDevice),
        cc::ConversionParams{
            .matrix = kCameraMatrix, .range = kCameraRange, .mode = cc::ResizeMode::kNearest},
        stream);
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(failure(error.what()));
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  holoscan::schema::ImageT descriptor;
  descriptor.width = w;
  descriptor.height = h;
  descriptor.encoding =
      nv12 ? holoscan::schema::ImageEncoding_NV12 : holoscan::schema::ImageEncoding_YUYV;
  descriptor.color_space = holoscan::schema::ColorSpace_REC709;
  if (nv12) {
    const auto luma = static_cast<std::uint64_t>(w) * static_cast<std::uint64_t>(h);
    descriptor.plane_layouts = {{0U, static_cast<std::uint32_t>(w), luma},
                                {luma, static_cast<std::uint32_t>(w), luma / 2U}};
  }
  descriptor.header = std::make_shared<holoscan::schema::HeaderT>();
  descriptor.header->frame_id = "synthetic_camera";
  descriptor.header->device_sequence = sent_;
  auto emitted = image.emit_tensor(
      std::move(*loan), descriptor,
      holoscan::EmitOptions{.capture_time = context.activation_time(), .frame_id = sent_ + 1U});
  if (emitted) {
    ++sent_;
    return {};
  }
  // A busy consumer on a lossless edge refuses the frame: send it again next tick.
  return emitted.error().code == holoscan::ErrorCode::kBackpressured
             ? holoscan::expected<void, holoscan::Error>{}
             : emitted;
}

}  // namespace color_convert_app
