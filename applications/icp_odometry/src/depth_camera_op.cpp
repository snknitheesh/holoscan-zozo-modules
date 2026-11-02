// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "depth_camera_op.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

namespace icp_odometry_app {
namespace {

namespace icp = holoscan::icp;

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kFloat64{kDLFloat, 64U, 1U};

holoscan::TensorOutputSpec tensor_spec(holoscan::MemoryKind memory, DLDataType dtype,
                                       std::uint32_t rank, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = memory,
              .dtype = dtype,
              .rank = rank,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(bytes),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return holoscan::Error{
      holoscan::ErrorCode::kFailure,
      std::string("SyntheticDepthCameraOp: ") + what + ": " + cudaGetErrorString(status)};
}

}  // namespace

void validate_config(const DepthCameraConfig& config) {
  if (config.width < 64 || config.height < 48 || config.width > 4096 || config.height > 4096) {
    throw std::invalid_argument("depth images must be 64x48 to 4096x4096");
  }
  if (config.count == 0 || config.path_frames == 0) {
    throw std::invalid_argument("the camera needs a positive count and path length");
  }
  if (config.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("the camera needs a positive period");
  }
  if (!(config.max_depth > 0.0F)) {
    throw std::invalid_argument("the camera needs a positive maximum depth");
  }
}

icp::DepthNoise camera_noise(std::uint32_t frame) {
  return icp::DepthNoise{.sigma = 0.0005F, .sigma_per_m2 = 0.0005F, .seed = frame};
}

SyntheticDepthCameraOp::SyntheticDepthCameraOp(DepthCameraConfig config)
    : config_(std::move(config)), scene_(icp::synthetic_scene(icp::SceneKind::kRoom)) {
  validate_config(config_);
}

void SyntheticDepthCameraOp::setup(holoscan::OperatorSpec& spec) {
  const auto pixels =
      static_cast<std::uint64_t>(config_.width) * static_cast<std::uint64_t>(config_.height);
  spec.output(depth, "depth")
      .produces_tensor(
          tensor_spec(holoscan::MemoryKind::kCudaDevice, kFloat32, 2U, pixels * sizeof(float)))
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
  spec.output(true_pose, "true_pose")
      .produces_tensor(tensor_spec(holoscan::MemoryKind::kHost, kFloat64, 2U, 16U * 8U));
}

holoscan::Contract SyntheticDepthCameraOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void SyntheticDepthCameraOp::start() {
  emitted_ = 0;
  pending_capture_.reset();
  depth_sent_ = false;
}

holoscan::expected<void, holoscan::Error> SyntheticDepthCameraOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= config_.count) {
    return {};  // Done: idle until the reporter requests completion.
  }
  if (config_.done && config_.done->load(std::memory_order_acquire) < emitted_) {
    return {};  // The frame before is still on its way.
  }
  // A frame keeps its capture time until both outputs are out, so a pose
  // sent on a retry still pairs with its depth image.
  if (!pending_capture_) {
    pending_capture_ = context.activation_time();
    depth_sent_ = false;
  }
  // Frame ids count from 1; 0 means none.
  const holoscan::EmitOptions emit{.capture_time = *pending_capture_, .frame_id = emitted_ + 1U};
  const icp::Matrix4 pose = icp::synthetic_trajectory(emitted_, config_.path_frames);

  if (!depth_sent_) {
    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          "SyntheticDepthCameraOp: the runtime gave no CUDA stream for the depth output"});
    }
    auto loan = depth.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = std::array<std::int64_t, 2>{config_.height, config_.width}, .dtype = kFloat32});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(stream);
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    const icp::DepthNoise noise = config_.noise ? camera_noise(emitted_) : icp::DepthNoise{};
    if (const cudaError_t status = icp::render_scene_depth(
            scene_, pose, icp::synthetic_camera(config_.width, config_.height), config_.max_depth,
            noise, writer->data_as<float>(), stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("render_scene_depth", status));
    }
    if (auto committed = std::move(*writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    holoscan::schema::ImageT image;
    image.width = config_.width;
    image.height = config_.height;
    image.encoding = holoscan::schema::ImageEncoding_MONO32F;
    auto sent = depth.emit_tensor(std::move(*loan), image, emit);
    if (!sent) {
      // A busy consumer refuses the frame; send it again on the next tick.
      return sent.error().code == holoscan::ErrorCode::kBackpressured
                 ? holoscan::expected<void, holoscan::Error>{}
                 : sent;
    }
    depth_sent_ = true;
  }

  auto loan = true_pose.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = std::array<std::int64_t, 2>{4, 4}, .dtype = kFloat64});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  if (auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                     -> holoscan::expected<void, holoscan::Error> {
        std::memcpy(writer.data_as<double>(), pose.data(), sizeof(pose));
        return {};
      });
      !written) {
    return written;
  }
  auto sent = true_pose.emit(std::move(*loan), emit);
  if (!sent) {
    return sent.error().code == holoscan::ErrorCode::kBackpressured
               ? holoscan::expected<void, holoscan::Error>{}
               : sent;
  }
  pending_capture_.reset();
  ++emitted_;
  return {};
}

}  // namespace icp_odometry_app
