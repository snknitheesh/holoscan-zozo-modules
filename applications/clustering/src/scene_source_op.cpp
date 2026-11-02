// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "scene_source_op.hpp"

#include <array>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/clustering/synthetic_scene.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

namespace clustering_app {
namespace {

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};

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

SceneSourceOp::SceneSourceOp(SceneSourceConfig config) : config_(std::move(config)) {
  validate_config(config_);
}

void SceneSourceOp::setup(holoscan::OperatorSpec& spec) {
  const auto pixels =
      static_cast<std::uint64_t>(config_.width) * static_cast<std::uint64_t>(config_.height);
  spec.output(depth, "depth")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kCudaDevice,
                  .dtype = kFloat32,
                  .rank = 2U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(pixels * 4U),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      })
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
}

holoscan::Contract SceneSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void SceneSourceOp::start() { emitted_ = 0; }

holoscan::expected<void, holoscan::Error> SceneSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= config_.count) {
    return {};  // Done: idle until the reporter completes the run.
  }
  if (config_.finished && config_.finished->load(std::memory_order_acquire) < emitted_) {
    return {};  // The previous frame is still in flight.
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure, "SceneSourceOp: the runtime gave no CUDA stream"});
  }
  const holoscan::clustering::SyntheticScene scene =
      holoscan::clustering::synthetic_scene(config_.width, config_.height, emitted_);
  auto loan = depth.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::array<std::int64_t, 2>{config_.height, config_.width}, .dtype = kFloat32});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  if (const cudaError_t status = holoscan::clustering::render_synthetic_scene(
          scene, writer->data_as<float>(), nullptr, nullptr, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("render", status));
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  holoscan::schema::ImageT image;
  image.width = config_.width;
  image.height = config_.height;
  image.encoding = holoscan::schema::ImageEncoding_MONO32F;
  // Frame ids count from 1; 0 means none.
  auto sent = depth.emit_tensor(
      std::move(*loan), image,
      holoscan::EmitOptions{.capture_time = context.activation_time(), .frame_id = emitted_ + 1U});
  if (!sent && sent.error().code == holoscan::ErrorCode::kBackpressured) {
    return {};  // A consumer still busy refused the frame before delivery: retry on the next tick.
  }
  if (!sent) {
    return sent;
  }
  ++emitted_;
  return {};
}

}  // namespace clustering_app
