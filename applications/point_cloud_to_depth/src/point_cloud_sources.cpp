// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "point_cloud_sources.hpp"

#include <cstring>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/point_cloud/synthetic_scene.hpp>

#include "cloud_io.hpp"

namespace point_cloud_to_depth_app {
namespace {

namespace hm = holoscan::messages;
using holoscan::point_cloud::generate_synthetic_cloud;
using holoscan::point_cloud::synthetic_camera_info;
using holoscan::point_cloud::synthetic_cloud_layout;
using holoscan::point_cloud::synthetic_frame;
using holoscan::point_cloud::SyntheticFrame;

// Which outputs of the frame being sent are out.
constexpr std::uint32_t kSentCameraInfo = 1U;
constexpr std::uint32_t kSentCloud = 2U;

holoscan::Error failure(std::string message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, std::move(message)};
}

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return failure(std::string(what) + ": " + cudaGetErrorString(status));
}

using Result = holoscan::expected<void, holoscan::Error>;

// Records the outcome of sending one output of a frame. Consumers still busy
// with earlier frames make an emit fail with kBackpressured before anything
// is delivered, or an allocation with kResourceExhausted while every slot of
// the output's pool is still held, by frames they have not finished with or
// by a frame they refused (the SDK takes back a refused loan only once this
// source's GPU work on it is done); the source then sends that output again
// on its next tick, with the frame's capture time, rather than failing the
// run or skipping the frame. Returns nothing to go on with the frame's other
// outputs, or what compute() returns now: success when the consumers are
// busy, else the error.
std::optional<Result> settle(Result sent, std::uint32_t output, std::uint32_t& sent_outputs) {
  if (sent) {
    sent_outputs |= output;
    return std::nullopt;
  }
  if (sent.error().code == holoscan::ErrorCode::kBackpressured ||
      sent.error().code == holoscan::ErrorCode::kResourceExhausted) {
    return Result{};
  }
  return Result{holoscan::make_unexpected(std::move(sent).error())};
}

// Fills a host field of `cloud` with `size` bytes from `bytes` and commits it.
Result fill_host(hm::PointCloudLoan& cloud,
                 holoscan::TensorFieldSelector<hm::schema::PointCloudT> field, const void* bytes,
                 std::size_t size) {
  return cloud.write_host_and_commit(
      field, [&](const holoscan::TensorOutputWriteGuard& writer) -> Result {
        if (writer.byte_size() != size) {
          return holoscan::make_unexpected(failure("cloud loan has an unexpected size"));
        }
        if (size != 0U) {
          std::memcpy(writer.data_as<std::uint8_t>(), bytes, size);
        }
        return {};
      });
}

// Sends `cloud`, unorganized, in `frame_id` with the capture time of `emit`.
Result send_cloud(holoscan::Output<hm::schema::PointCloudT>& port, hm::PointCloudLoan cloud,
                  std::size_t points, const std::string& frame_id,
                  const holoscan::EmitOptions& emit) {
  hm::Header header{.frame_id = frame_id};
  if (emit.capture_time) {
    header.capture_timestamp_ns = emit.capture_time->timestamp_ns;
  }
  return hm::emit_point_cloud(port, std::move(cloud), header,
                              hm::unorganized_cloud(static_cast<std::int64_t>(points)), -1, emit);
}

}  // namespace

void validate_config(const SyntheticCloudSourceConfig& config) {
  if (config.width < 64 || config.height < 48 || config.width > 4096 || config.height > 4096) {
    throw std::invalid_argument("synthetic images must be 64x48 to 4096x4096");
  }
  if (config.count == 0 || config.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("synthetic source needs a positive count and period");
  }
}

SyntheticCloudSourceOp::SyntheticCloudSourceOp(SyntheticCloudSourceConfig config)
    : config_(std::move(config)) {
  validate_config(config_);
  point_count_ = synthetic_cloud_layout(config_.width, config_.height).total();
}

SyntheticCloudSourceOp::~SyntheticCloudSourceOp() { release(); }

void SyntheticCloudSourceOp::setup(holoscan::OperatorSpec& spec) {
  hm::declare_point_cloud_output(
      spec, point_cloud, "point_cloud",
      {.max_points = point_count_,
       .colors = config_.color,
       .memory_kind = config_.device_output ? holoscan::MemoryKind::kCudaDevice
                                            : holoscan::MemoryKind::kHost});
  if (config_.camera_info) {
    spec.output(camera_info, "camera_info");
  }
}

holoscan::Contract SyntheticCloudSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void SyntheticCloudSourceOp::start() {
  emitted_ = 0;
  pending_capture_.reset();
  sent_ = 0U;
}

void SyntheticCloudSourceOp::stop() { release(); }

void SyntheticCloudSourceOp::release() noexcept {
  if (own_stream_ != nullptr) {
    cudaStreamSynchronize(own_stream_);
    cudaStreamDestroy(own_stream_);
    own_stream_ = nullptr;
  }
  if (scratch_ != nullptr) {
    cudaFree(scratch_);
    scratch_ = nullptr;
  }
}

holoscan::expected<void, holoscan::Error> SyntheticCloudSourceOp::ensure_scratch() {
  if (scratch_ != nullptr) {
    return {};
  }
  const std::size_t bytes = point_count_ * (3U * sizeof(float) + 3U);
  if (const cudaError_t status = cudaMalloc(&scratch_, bytes); status != cudaSuccess) {
    scratch_ = nullptr;
    return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
  }
  return {};
}

holoscan::expected<void, holoscan::Error> SyntheticCloudSourceOp::fill(hm::PointCloudLoan& cloud,
                                                                       const SyntheticFrame& frame,
                                                                       cudaStream_t stream) {
  const std::size_t point_bytes = point_count_ * 3U * sizeof(float);
  const std::size_t color_bytes = point_count_ * 3U;
  if (config_.device_output) {
    // One kernel makes the points and the colors, so the colors' write runs
    // inside the points'.
    const auto generate = [&](float* points, std::uint8_t* colors) -> Result {
      if (const cudaError_t status =
              generate_synthetic_cloud(frame, config_.millimeter_depth, points, colors, stream);
          status != cudaSuccess) {
        return holoscan::make_unexpected(cuda_failure("generate_synthetic_cloud", status));
      }
      return {};
    };
    return cloud.write_and_commit(hm::PointCloudFields::points, stream,
                                  [&](const holoscan::TensorOutputWriteGuard& points) -> Result {
                                    if (!config_.color) {
                                      return generate(points.data_as<float>(), nullptr);
                                    }
                                    return cloud.write_and_commit(
                                        hm::PointCloudFields::colors, stream,
                                        [&](const holoscan::TensorOutputWriteGuard& colors) {
                                          return generate(points.data_as<float>(),
                                                          colors.data_as<std::uint8_t>());
                                        });
                                  });
  }
  // Make the cloud on the GPU, then fill the host loans from it.
  if (auto scratch = ensure_scratch(); !scratch) {
    return scratch;
  }
  auto* bytes = static_cast<std::uint8_t*>(scratch_);
  if (const cudaError_t status =
          generate_synthetic_cloud(frame, config_.millimeter_depth, reinterpret_cast<float*>(bytes),
                                   bytes + point_bytes, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("generate_synthetic_cloud", status));
  }
  const auto copy_out = [&](holoscan::TensorFieldSelector<hm::schema::PointCloudT> field,
                            std::size_t offset, std::size_t size) {
    return cloud.write_host_and_commit(
        field, [&](const holoscan::TensorOutputWriteGuard& writer) -> Result {
          if (writer.byte_size() != size) {
            return holoscan::make_unexpected(failure("cloud loan has an unexpected size"));
          }
          const cudaError_t status = cudaMemcpyAsync(writer.data_as<std::uint8_t>(), bytes + offset,
                                                     size, cudaMemcpyDeviceToHost, stream);
          const cudaError_t synced = status == cudaSuccess ? cudaStreamSynchronize(stream) : status;
          if (synced != cudaSuccess) {
            return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", synced));
          }
          return {};
        });
  };
  if (auto copied = copy_out(hm::PointCloudFields::points, 0, point_bytes); !copied) {
    return copied;
  }
  if (config_.color) {
    return copy_out(hm::PointCloudFields::colors, point_bytes, color_bytes);
  }
  return {};
}

holoscan::expected<void, holoscan::Error> SyntheticCloudSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= config_.count) {
    return {};  // Done: idle until the consumers request completion.
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    // The runtime writes device loans only on the stream it assigns to an
    // operator with device outputs. One with host outputs gets none, and
    // makes its clouds on its own stream, waiting for each.
    if (config_.device_output) {
      return holoscan::make_unexpected(failure(
          "SyntheticCloudSourceOp: the runtime gave no CUDA stream for the device outputs"));
    }
    if (own_stream_ == nullptr) {
      if (const cudaError_t status = cudaStreamCreateWithFlags(&own_stream_, cudaStreamNonBlocking);
          status != cudaSuccess) {
        own_stream_ = nullptr;
        return holoscan::make_unexpected(cuda_failure("cudaStreamCreate", status));
      }
    }
    stream = own_stream_;
  }
  // A frame keeps its capture time until all its outputs are out, so outputs
  // sent on a retry still pair with those sent before it.
  if (!pending_capture_) {
    pending_capture_ = context.activation_time();
    sent_ = 0U;
  }
  // Frame ids count from 1; 0 means none.
  const holoscan::EmitOptions emit{.capture_time = *pending_capture_, .frame_id = emitted_ + 1U};
  const SyntheticFrame frame =
      synthetic_frame(config_.width, config_.height, emitted_, config_.camera_info);

  if (config_.camera_info && (sent_ & kSentCameraInfo) == 0U) {
    hm::CameraInfo info = synthetic_camera_info(frame);
    info.header.capture_timestamp_ns = pending_capture_->timestamp_ns;
    if (auto stop = settle(camera_info.emit(hm::to_message(info), emit), kSentCameraInfo, sent_)) {
      return *stop;
    }
  }
  auto cloud = hm::allocate_point_cloud(point_cloud, hm::unorganized_cloud(point_count_),
                                        {.colors = config_.color}, ::cuda::stream_ref(stream));
  if (!cloud) {
    return settle(holoscan::make_unexpected(std::move(cloud).error()), kSentCloud, sent_)
        .value_or(Result{});
  }
  if (auto filled = fill(*cloud, frame, stream); !filled) {
    return filled;
  }
  if (auto stop = settle(send_cloud(point_cloud, std::move(*cloud), point_count_, "body", emit),
                         kSentCloud, sent_)) {
    return *stop;
  }
  // Every output of this frame is out.
  pending_capture_.reset();
  ++emitted_;
  return {};
}

void validate_config(const CloudFileSourceConfig& config) {
  if (config.files.empty()) {
    throw std::invalid_argument("point cloud file source needs at least one file");
  }
  if (config.max_points < 1 || config.max_points > kMaxPlyPoints) {
    throw std::invalid_argument("point cloud file source needs max_points of 1.." +
                                std::to_string(kMaxPlyPoints));
  }
  if (config.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("point cloud file source needs a positive period");
  }
  if (!config.loop && config.count > config.files.size()) {
    throw std::invalid_argument("--count exceeds the number of files; add --loop true");
  }
}

CloudFileSourceOp::CloudFileSourceOp(CloudFileSourceConfig config) : config_(std::move(config)) {
  validate_config(config_);
  total_ = config_.count == 0 ? static_cast<std::uint32_t>(config_.files.size()) : config_.count;
}

void CloudFileSourceOp::setup(holoscan::OperatorSpec& spec) {
  hm::declare_point_cloud_output(spec, point_cloud, "point_cloud",
                                 {.max_points = config_.max_points,
                                  .colors = config_.color,
                                  .memory_kind = holoscan::MemoryKind::kHost});
}

holoscan::Contract CloudFileSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void CloudFileSourceOp::start() { emitted_ = 0; }

holoscan::expected<void, holoscan::Error> CloudFileSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= total_) {
    return {};
  }
  const auto& path = config_.files[emitted_ % config_.files.size()];
  PointCloud cloud;
  try {
    cloud = read_ply(path);
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(failure(error.what()));
  }
  if (cloud.size() > config_.max_points) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                        path.string() + ": more than " + std::to_string(config_.max_points) +
                            " points, the most the first look at the files found"});
  }
  if (config_.color && !cloud.color) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, path.string() + ": has no colors"});
  }
  const holoscan::EmitOptions emit{.capture_time = context.activation_time(),
                                   .frame_id = emitted_ + 1U};
  std::uint32_t sent = 0U;
  auto loan = hm::allocate_point_cloud(
      point_cloud, hm::unorganized_cloud(static_cast<std::int64_t>(cloud.size())),
      {.colors = config_.color}, context.cuda_stream());
  if (!loan) {
    return settle(holoscan::make_unexpected(std::move(loan).error()), kSentCloud, sent)
        .value_or(Result{});
  }
  if (auto written = fill_host(*loan, hm::PointCloudFields::points, cloud.xyz.data(),
                               cloud.xyz.size() * sizeof(float));
      !written) {
    return written;
  }
  if (config_.color) {
    if (auto written =
            fill_host(*loan, hm::PointCloudFields::colors, cloud.rgb.data(), cloud.rgb.size());
        !written) {
      return written;
    }
  }
  if (auto stop = settle(send_cloud(point_cloud, std::move(*loan), cloud.size(), "points", emit),
                         kSentCloud, sent)) {
    return *stop;
  }
  ++emitted_;
  return {};
}

}  // namespace point_cloud_to_depth_app
