// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "scene_source_op.hpp"

#include <array>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/pcl_filters/synthetic_scene.hpp>

namespace pcl_filters_app {
namespace {

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};

holoscan::TensorOutputSpec device_image(DLDataType dtype, std::uint32_t rank, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = dtype,
              .rank = rank,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(bytes),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

holoscan::PayloadOptions image_payload() {
  return holoscan::PayloadOptions{
      .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
  };
}

holoscan::schema::ImageT descriptor(int width, int height,
                                    holoscan::schema::ImageEncoding encoding) {
  holoscan::schema::ImageT image;
  image.width = width;
  image.height = height;
  image.encoding = encoding;
  return image;
}

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return holoscan::Error{holoscan::ErrorCode::kFailure,
                         std::string("SceneSourceOp: ") + what + ": " + cudaGetErrorString(status)};
}

}  // namespace

void validate_config(const SceneSourceConfig& config) {
  if (config.width < 64 || config.height < 48 || config.width > 4096 || config.height > 4096) {
    throw std::invalid_argument("scene images must be 64x48 to 4096x4096");
  }
  if (config.count == 0 || config.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("the scene source needs a positive count and period");
  }
}

SceneSourceOp::SceneSourceOp(SceneSourceConfig config) : config_(config) {
  validate_config(config_);
}

void SceneSourceOp::setup(holoscan::OperatorSpec& spec) {
  const auto pixels =
      static_cast<std::uint64_t>(config_.width) * static_cast<std::uint64_t>(config_.height);
  spec.output(depth, "depth")
      .produces_tensor(device_image(kFloat32, 2U, pixels * 4U))
      .payload_options(image_payload());
  spec.output(color, "color")
      .produces_tensor(device_image(kUInt8, 3U, pixels * 3U))
      .payload_options(image_payload());
}

holoscan::Contract SceneSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void SceneSourceOp::start() {
  emitted_ = 0;
  pending_capture_.reset();
  depth_sent_ = false;
}

holoscan::expected<void, holoscan::Error> SceneSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= config_.count) {
    return {};  // Done: idle until the reporter completes the run.
  }
  if (config_.finished && !pending_capture_ &&
      config_.finished->load(std::memory_order_acquire) < emitted_) {
    return {};  // The previous frame is still in flight.
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure, "SceneSourceOp: the runtime gave no CUDA stream"});
  }
  if (!pending_capture_) {
    pending_capture_ = context.activation_time();
    depth_sent_ = false;
  }
  // Frame ids count from 1; 0 means none.
  const holoscan::EmitOptions emit{.capture_time = *pending_capture_, .frame_id = emitted_ + 1U};
  const holoscan::pcl_filters::SyntheticScene scene =
      holoscan::pcl_filters::synthetic_scene(config_.width, config_.height, emitted_);

  // A consumer still busy refuses a frame with kBackpressured before anything
  // is delivered: send the rest of it on the next tick.
  const auto blocked = [](const holoscan::expected<void, holoscan::Error>& sent) {
    return !sent && sent.error().code == holoscan::ErrorCode::kBackpressured;
  };
  if (!depth_sent_) {
    auto loan = depth.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = std::array<std::int64_t, 2>{config_.height, config_.width}, .dtype = kFloat32});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(stream);
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    if (const cudaError_t status = holoscan::pcl_filters::render_synthetic_scene(
            scene, writer->data_as<float>(), nullptr, nullptr, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("render", status));
    }
    if (auto committed = std::move(*writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    auto sent = depth.emit_tensor(
        std::move(*loan),
        descriptor(config_.width, config_.height, holoscan::schema::ImageEncoding_MONO32F), emit);
    if (blocked(sent)) {
      return {};
    }
    if (!sent) {
      return sent;
    }
    depth_sent_ = true;
  }
  auto loan = color.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::array<std::int64_t, 3>{config_.height, config_.width, 3}, .dtype = kUInt8});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  if (const cudaError_t status = holoscan::pcl_filters::render_synthetic_scene(
          scene, nullptr, writer->data_as<std::uint8_t>(), nullptr, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("render", status));
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  auto sent = color.emit_tensor(
      std::move(*loan),
      descriptor(config_.width, config_.height, holoscan::schema::ImageEncoding_RGB8), emit);
  if (blocked(sent)) {
    return {};
  }
  if (!sent) {
    return sent;
  }
  ++emitted_;
  pending_capture_.reset();
  return {};
}

}  // namespace pcl_filters_app
