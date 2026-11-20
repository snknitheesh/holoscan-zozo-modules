// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "terrain_camera_op.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

namespace height_map_app {
namespace {

namespace hm = holoscan::height_map;

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kFloat64{kDLFloat, 64U, 1U};

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, std::string("TerrainCameraOp: ") + what +
                                                            ": " + cudaGetErrorString(status)};
}

}  // namespace

hm::TerrainCamera camera_for_frame(int width, int height, std::uint32_t frame) {
  hm::TerrainCamera camera;
  camera.width = width;
  camera.height = height;
  camera.fx = camera.fy = 0.7F * static_cast<float>(width);
  camera.cx = 0.5F * static_cast<float>(width - 1);
  camera.cy = 0.5F * static_cast<float>(height - 1);
  const double f = frame;
  camera.pose = hm::camera_pose(0.5 + 0.01 * f, 0.0, 1.1, 0.14 * std::sin(0.05 * f), 0.61);
  camera.max_depth = 8.0F;
  camera.noise_base = 0.001F;
  camera.noise_quadratic = 0.002F;
  camera.seed = frame + 1U;
  return camera;
}

void validate_config(const TerrainCameraConfig& config) {
  if (config.width < 64 || config.height < 48 || config.width > 4096 || config.height > 4096) {
    throw std::invalid_argument("camera images must be 64x48 to 4096x4096");
  }
  if (config.count == 0 || config.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("the camera needs a positive count and period");
  }
}

TerrainCameraOp::TerrainCameraOp(TerrainCameraConfig config)
    : config_(std::move(config)), terrain_(hm::course_terrain()) {
  validate_config(config_);
}

void TerrainCameraOp::setup(holoscan::OperatorSpec& spec) {
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
  spec.output(pose, "pose")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kHost,
                  .dtype = kFloat64,
                  .rank = 2U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(16U * sizeof(double)),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
}

holoscan::Contract TerrainCameraOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void TerrainCameraOp::start() {
  emitted_ = 0;
  delayed_ = false;
  capture_.reset();
  depth_sent_ = false;
}

namespace {

// Whether a refused send or allocation is backpressure: the consumer has no
// room yet, and the same frame goes again on the next tick.
bool backpressure(const holoscan::Error& error) {
  return error.code == holoscan::ErrorCode::kBackpressured ||
         error.code == holoscan::ErrorCode::kResourceExhausted;
}

}  // namespace

holoscan::expected<void, holoscan::Error> TerrainCameraOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= config_.count) {
    return {};  // Done: idle until the reporter completes the run.
  }
  if (!delayed_) {
    delayed_ = true;
    return {};  // The camera info goes first.
  }
  if (!capture_ && config_.finished &&
      config_.finished->load(std::memory_order_acquire) < emitted_) {
    return {};  // The previous frame is still in flight.
  }
  if (!capture_) {
    capture_ = context.activation_time();
  }
  // Frame numbers count from 1; 0 means none.
  const std::uint64_t number = emitted_ + 1U;
  const holoscan::EmitOptions emit{.capture_time = *capture_, .frame_id = number};
  const hm::TerrainCamera camera = camera_for_frame(config_.width, config_.height, emitted_);
  if (!depth_sent_) {
    if (auto sent = send_depth(context, camera, emit); !sent) {
      return backpressure(sent.error()) ? holoscan::expected<void, holoscan::Error>{} : sent;
    }
    depth_sent_ = true;
  }
  auto pose_loan = pose.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = std::array<std::int64_t, 2>{4, 4}, .dtype = kFloat64});
  if (!pose_loan) {
    return backpressure(pose_loan.error())
               ? holoscan::expected<void, holoscan::Error>{}
               : holoscan::make_unexpected(std::move(pose_loan).error());
  }
  std::array<double, 16> matrix{};
  std::memcpy(matrix.data(), camera.pose.m, sizeof(camera.pose.m));
  matrix[15] = 1.0;
  if (auto written =
          pose_loan->write_host_and_commit([&matrix](const holoscan::TensorOutputWriteGuard& out)
                                               -> holoscan::expected<void, holoscan::Error> {
            std::memcpy(out.data_as<double>(), matrix.data(), sizeof(matrix));
            return {};
          });
      !written) {
    return written;
  }
  if (auto sent = pose.emit(std::move(*pose_loan), emit); !sent) {
    return backpressure(sent.error()) ? holoscan::expected<void, holoscan::Error>{} : sent;
  }
  capture_.reset();
  depth_sent_ = false;
  ++emitted_;
  return {};
}

holoscan::expected<void, holoscan::Error> TerrainCameraOp::send_depth(
    holoscan::ExecutionContext& context, const hm::TerrainCamera& camera,
    const holoscan::EmitOptions& emit) {
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure, "TerrainCameraOp: the runtime gave no CUDA stream"});
  }
  auto depth_loan = depth.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::array<std::int64_t, 2>{config_.height, config_.width}, .dtype = kFloat32});
  if (!depth_loan) {
    return holoscan::make_unexpected(std::move(depth_loan).error());
  }
  auto writer = depth_loan->write(stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  if (const cudaError_t status =
          hm::render_terrain_depth(terrain_, camera, writer->data_as<float>(), nullptr, stream);
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
  image.header = std::make_shared<holoscan::schema::HeaderT>();
  image.header->capture_timestamp_ns = emit.capture_time->timestamp_ns;
  image.header->frame_id = kCameraFrame;
  image.header->device_sequence = *emit.frame_id;
  return depth.emit_tensor(std::move(*depth_loan), image, emit);
}

}  // namespace height_map_app
