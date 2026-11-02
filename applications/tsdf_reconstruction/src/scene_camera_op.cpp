// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "scene_camera_op.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/tsdf/synthetic_scene.hpp>

namespace tsdf_reconstruction_app {
namespace {

namespace ht = holoscan::tsdf;

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kFloat64{kDLFloat, 64U, 1U};
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr DLDataType kUInt16{kDLUInt, 16U, 1U};

// The depth noise of a structured-light camera: 1 mm plus 2 mm per square meter.
constexpr ht::DepthNoise kNoise{.sigma = 0.001F, .sigma_per_m2 = 0.002F, .seed = 0};
constexpr float kMaxDepth = 3.0F;

holoscan::TensorOutputSpec tensor_spec(holoscan::MemoryKind memory, std::uint32_t rank,
                                       std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = memory,
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
                         std::string("SceneCameraOp: ") + what + ": " + cudaGetErrorString(status)};
}

}  // namespace

void validate_config(const SceneCameraConfig& config) {
  if (config.width < 64 || config.height < 48 || config.width > 4096 || config.height > 4096) {
    throw std::invalid_argument("camera images must be 64x48 to 4096x4096");
  }
  if (config.count == 0 || config.orbit == 0 || config.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("the camera needs a positive count, orbit and period");
  }
}

SceneCameraOp::SceneCameraOp(SceneCameraConfig config) : config_(std::move(config)) {
  validate_config(config_);
}

void SceneCameraOp::setup(holoscan::OperatorSpec& spec) {
  const auto pixels =
      static_cast<std::uint64_t>(config_.width) * static_cast<std::uint64_t>(config_.height);
  spec.output(depth, "depth")
      .produces_tensor(tensor_spec(holoscan::MemoryKind::kCudaDevice, 2U, pixels * 4U))
      .payload_options(image_payload());
  spec.output(color, "color")
      .produces_tensor(tensor_spec(holoscan::MemoryKind::kCudaDevice, 3U, pixels * 3U))
      .payload_options(image_payload());
  spec.output(pose, "pose").produces_tensor(tensor_spec(holoscan::MemoryKind::kHost, 2U, 128U));
}

holoscan::Contract SceneCameraOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void SceneCameraOp::start() { emitted_ = 0; }

holoscan::expected<void, holoscan::Error> SceneCameraOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= config_.count) {
    return {};  // Done: idle until the reporters complete the run.
  }
  if (config_.finished && config_.finished->load(std::memory_order_acquire) < emitted_) {
    return {};  // The previous frame is still in flight.
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure, "SceneCameraOp: the runtime gave no CUDA stream"});
  }
  const ht::Matrix4 world_from_camera =
      ht::orbit_pose(ht::SceneKind::kTabletop, emitted_ % config_.orbit, config_.orbit);
  const ht::CameraIntrinsics intrinsics = ht::synthetic_intrinsics(config_.width, config_.height);

  const std::array<std::int64_t, 2> image_shape{config_.height, config_.width};
  const std::array<std::int64_t, 3> color_shape{config_.height, config_.width, 3};
  auto depth_loan = depth.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = image_shape, .dtype = config_.millimeters ? kUInt16 : kFloat32});
  if (!depth_loan) {
    return holoscan::make_unexpected(std::move(depth_loan).error());
  }
  auto color_loan =
      color.allocate_tensor(holoscan::TensorLoanRequest{.shape = color_shape, .dtype = kUInt8});
  if (!color_loan) {
    return holoscan::make_unexpected(std::move(color_loan).error());
  }
  auto pose_loan = pose.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = std::array<std::int64_t, 2>{4, 4}, .dtype = kFloat64});
  if (!pose_loan) {
    return holoscan::make_unexpected(std::move(pose_loan).error());
  }
  {
    auto depth_writer = depth_loan->write(stream);
    if (!depth_writer) {
      return holoscan::make_unexpected(std::move(depth_writer).error());
    }
    auto color_writer = color_loan->write(stream);
    if (!color_writer) {
      return holoscan::make_unexpected(std::move(color_writer).error());
    }
    ht::DepthNoise noise = kNoise;
    noise.seed = emitted_;
    if (const cudaError_t status = ht::render_scene(
            ht::synthetic_scene(ht::SceneKind::kTabletop), world_from_camera, intrinsics,
            config_.width, config_.height, kMaxDepth, noise,
            config_.millimeters ? ht::DepthType::kUint16 : ht::DepthType::kFloat32,
            depth_writer->data_as<void>(), color_writer->data_as<std::uint8_t>(), stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("render_scene", status));
    }
    if (auto committed = std::move(*depth_writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    if (auto committed = std::move(*color_writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    auto pose_writer = pose_loan->write_host();
    if (!pose_writer) {
      return holoscan::make_unexpected(std::move(pose_writer).error());
    }
    double* values = pose_writer->data_as<double>();
    for (std::size_t i = 0; i < world_from_camera.size(); ++i) {
      values[i] = world_from_camera[i];
    }
    if (auto committed = std::move(*pose_writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
  }
  // Frame ids count from 1; 0 means none.
  const holoscan::EmitOptions emit{.capture_time = context.activation_time(),
                                   .frame_id = emitted_ + 1U};
  // Every consumer runs only when all three have arrived, so the first emit
  // decides: a refused depth image is sent again on the next tick.
  auto sent =
      depth.emit_tensor(std::move(*depth_loan),
                        descriptor(config_.width, config_.height,
                                   config_.millimeters ? holoscan::schema::ImageEncoding_MONO16
                                                       : holoscan::schema::ImageEncoding_MONO32F),
                        emit);
  if (!sent && sent.error().code == holoscan::ErrorCode::kBackpressured) {
    return {};
  }
  if (!sent) {
    return sent;
  }
  if (auto color_sent = color.emit_tensor(
          std::move(*color_loan),
          descriptor(config_.width, config_.height, holoscan::schema::ImageEncoding_RGB8), emit);
      !color_sent) {
    return color_sent;
  }
  if (auto pose_sent = pose.emit(std::move(*pose_loan), emit); !pose_sent) {
    return pose_sent;
  }
  ++emitted_;
  return {};
}

}  // namespace tsdf_reconstruction_app
