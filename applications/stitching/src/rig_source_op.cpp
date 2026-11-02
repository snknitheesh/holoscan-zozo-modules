// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "rig_source_op.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

namespace stitching_app {
namespace {

namespace st = holoscan::stitching;

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

holoscan::Error failure(const std::string& what) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "RigSourceOp: " + what};
}

}  // namespace

void validate_config(const RigSourceConfig& config) {
  st::validate(config.rig);
  if (!config.gains.empty() && config.gains.size() != config.rig.cameras.size()) {
    throw std::invalid_argument("give one camera gain per camera");
  }
  for (const float gain : config.gains) {
    if (!(std::isfinite(gain) && gain > 0.0F && gain <= 4.0F)) {
      throw std::invalid_argument("camera gains must be in (0, 4]");
    }
  }
  if (config.period.count() < 1 || config.period.count() > 1000) {
    throw std::invalid_argument("the period must be 1..1000 ms");
  }
  if (config.samples < 1 || config.samples > 8) {
    throw std::invalid_argument("samples must be 1..8");
  }
}

st::RigPose frame_pose(std::uint32_t frame) { return st::moving_pose(frame); }

RigSourceOp::RigSourceOp(RigSourceConfig config) : config_(std::move(config)) {
  validate_config(config_);
}

void RigSourceOp::setup(holoscan::OperatorSpec& spec) {
  for (std::size_t c = 0; c < config_.rig.cameras.size(); ++c) {
    const st::Camera& camera = config_.rig.cameras[c];
    spec.output(images[c], camera.name)
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kCudaDevice,
                    .dtype = kUInt8,
                    .rank = 3U,
                    .strides = holoscan::TensorStridePolicy::kContiguousOnly,
                },
            .bounds = holoscan::tensor_bounds(static_cast<std::uint64_t>(camera.width) *
                                              static_cast<std::uint64_t>(camera.height) * 4U),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        })
        .payload_options(holoscan::PayloadOptions{
            .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
        });
  }
}

holoscan::Contract RigSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void RigSourceOp::start() { sent_ = 0; }

holoscan::expected<void, holoscan::Error> RigSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if ((config_.count != 0 && sent_ >= config_.count) || (config_.done && config_.done() < sent_)) {
    return {};
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(failure("the runtime gave no CUDA stream"));
  }
  const st::RigPose pose = frame_pose(sent_);
  const auto capture = context.activation_time();
  for (std::size_t c = 0; c < config_.rig.cameras.size(); ++c) {
    const st::Camera& camera = config_.rig.cameras[c];
    auto loan = images[c].allocate_tensor(holoscan::TensorLoanRequest{
        .shape = std::array<std::int64_t, 3>{camera.height, camera.width, 4}, .dtype = kUInt8});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(stream);
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    const float gain = config_.gains.empty() ? 1.0F : config_.gains[c];
    if (const cudaError_t status = st::render_camera(
            config_.world, camera, pose, gain, st::PixelFormat::kRgba8, writer->data(),
            static_cast<std::size_t>(camera.width) * 4U, config_.samples, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(failure(cudaGetErrorString(status)));
    }
    if (auto committed = std::move(*writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    holoscan::schema::ImageT descriptor;
    descriptor.width = camera.width;
    descriptor.height = camera.height;
    descriptor.encoding = holoscan::schema::ImageEncoding_RGBA8;
    descriptor.header = std::make_shared<holoscan::schema::HeaderT>();
    descriptor.header->frame_id = camera.name;
    descriptor.header->device_sequence = sent_;
    auto emitted = images[c].emit_tensor(
        std::move(*loan), descriptor,
        holoscan::EmitOptions{.capture_time = capture, .frame_id = sent_ + 1U});
    if (!emitted) {
      // A refused first image leaves nothing sent: send the frame again on
      // the next tick. A refusal after it would split the frame set.
      if (c == 0 && emitted.error().code == holoscan::ErrorCode::kBackpressured) {
        return {};
      }
      return emitted;
    }
  }
  ++sent_;
  return {};
}

}  // namespace stitching_app
