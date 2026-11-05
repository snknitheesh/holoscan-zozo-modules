// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "stereo_sources.hpp"

#include <array>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/stereo_disparity/synthetic_scene.hpp>

namespace stereo_disparity_app {
namespace {

namespace sd = holoscan::stereo_disparity;

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr DLDataType kUInt16{kDLUInt, 16U, 1U};

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

holoscan::PayloadOptions image_payload() {
  return holoscan::PayloadOptions{.schema = holoscan::schema_identity<holoscan::schema::ImageT>()};
}

holoscan::schema::ImageT image_descriptor(int width, int height,
                                          holoscan::schema::ImageEncoding encoding) {
  holoscan::schema::ImageT image;
  image.width = width;
  image.height = height;
  image.encoding = encoding;
  return image;
}

holoscan::Error failure(std::string message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, std::move(message)};
}

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return failure(std::string(what) + ": " + cudaGetErrorString(status));
}

// Records one part of a frame as sent, or that the consumer is busy (`blocked`,
// to retry on the next tick); any other failure fails the run.
holoscan::expected<void, holoscan::Error> settle(holoscan::expected<void, holoscan::Error> sent,
                                                 std::uint32_t bit, std::uint32_t& sent_bits,
                                                 bool& blocked) {
  if (sent) {
    sent_bits |= bit;
    return {};
  }
  if (sent.error().code == holoscan::ErrorCode::kBackpressured) {
    blocked = true;
    return {};
  }
  return sent;
}

constexpr std::uint32_t kSentLeft = 1U;
constexpr std::uint32_t kSentRight = 2U;
constexpr std::uint32_t kSentDepth = 4U;
constexpr std::uint32_t kSentCameraInfo = 8U;

// A frame on every tick of `period`, with the newest camera info, if any.
holoscan::Contract source_contract(
    std::chrono::nanoseconds period,
    const holoscan::Input<holoscan::messages::schema::StereoCameraInfoT>& camera_info) {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = period});
  result.input(camera_info, holoscan::LatestWithin{
                                .max_age = holoscan::messages::CameraInfoInputOptions{}.max_age});
  result.on_missing(camera_info, holoscan::SafeDefault{});
  return result;
}

}  // namespace

holoscan::expected<bool, holoscan::Error> FrameCameraInfo::update(
    holoscan::Input<holoscan::messages::schema::StereoCameraInfoT>& port) {
  auto info = latest_.update(port);
  if (!info) {
    return holoscan::make_unexpected(std::move(info).error());
  }
  if (*info == nullptr) {
    return false;
  }
  if (latest_.changed()) {
    message_ = holoscan::messages::to_message(**info);
  }
  return true;
}

holoscan::expected<void, holoscan::Error> FrameCameraInfo::send(
    holoscan::Output<holoscan::messages::schema::StereoCameraInfoT>& port,
    const holoscan::EmitOptions& emit) {
  message_.header->capture_timestamp_ns = emit.capture_time ? emit.capture_time->timestamp_ns : 0;
  return port.emit_ref(message_, emit);
}

void validate_config(const SyntheticSourceConfig& config) {
  (void)sd::make_scene(0, config.width, config.height);  // Checks the size.
  if (config.count == 0 || config.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("the synthetic source needs a positive count and period");
  }
}

SyntheticStereoSourceOp::SyntheticStereoSourceOp(SyntheticSourceConfig config)
    : config_(std::move(config)) {
  validate_config(config_);
}

SyntheticStereoSourceOp::~SyntheticStereoSourceOp() { release(); }

void SyntheticStereoSourceOp::setup(holoscan::OperatorSpec& spec) {
  const std::uint64_t bytes = static_cast<std::uint64_t>(config_.width) *
                              static_cast<std::uint64_t>(config_.height) *
                              (config_.color ? 3U : 1U);
  const holoscan::MemoryKind memory =
      config_.device_output ? holoscan::MemoryKind::kCudaDevice : holoscan::MemoryKind::kHost;
  const std::uint32_t rank = config_.color ? 3U : 2U;
  spec.input(camera_info, "camera_info");
  spec.output(stereo_camera_info, "stereo_camera_info");
  spec.output(left, "left")
      .produces_tensor(tensor_spec(memory, kUInt8, rank, bytes))
      .payload_options(image_payload());
  spec.output(right, "right")
      .produces_tensor(tensor_spec(memory, kUInt8, rank, bytes))
      .payload_options(image_payload());
}

holoscan::Contract SyntheticStereoSourceOp::contract() const {
  return source_contract(config_.period, camera_info);
}

void SyntheticStereoSourceOp::start() {
  frame_camera_info_.reset();
  emitted_ = 0;
  pending_capture_.reset();
  sent_ = 0U;
}

void SyntheticStereoSourceOp::stop() { release(); }

void SyntheticStereoSourceOp::release() noexcept {
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

holoscan::expected<void, holoscan::Error> SyntheticStereoSourceOp::compute(
    holoscan::ExecutionContext& context) {
  auto camera_info_came = frame_camera_info_.update(camera_info);
  if (!camera_info_came) {
    return holoscan::make_unexpected(std::move(camera_info_came).error());
  }
  if (emitted_ >= config_.count || !*camera_info_came) {
    return {};  // Done, idle until the consumers request completion; or not started.
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    if (own_stream_ == nullptr) {
      if (const cudaError_t status = cudaStreamCreateWithFlags(&own_stream_, cudaStreamNonBlocking);
          status != cudaSuccess) {
        own_stream_ = nullptr;
        return holoscan::make_unexpected(cuda_failure("cudaStreamCreate", status));
      }
    }
    stream = own_stream_;
  }
  const int channels = config_.color ? 3 : 1;
  const std::size_t bytes = static_cast<std::size_t>(config_.width) * config_.height * channels;
  // A frame keeps its capture time until both images are out, and is rendered
  // once.
  if (!pending_capture_) {
    pending_capture_ = context.activation_time();
    sent_ = 0U;
    if (scratch_ == nullptr) {
      if (const cudaError_t status = cudaMalloc(&scratch_, 2 * bytes); status != cudaSuccess) {
        scratch_ = nullptr;
        return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
      }
    }
    auto* pair = static_cast<std::uint8_t*>(scratch_);
    if (const cudaError_t status =
            sd::render_scene(sd::make_scene(emitted_, config_.width, config_.height), channels,
                             pair, pair + bytes, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("render_scene", status));
    }
  }
  const holoscan::EmitOptions emit{.capture_time = *pending_capture_, .frame_id = emitted_ + 1U};
  const auto encoding =
      config_.color ? holoscan::schema::ImageEncoding_RGB8 : holoscan::schema::ImageEncoding_MONO8;
  bool blocked = false;
  if ((sent_ & kSentCameraInfo) == 0U) {
    if (auto settled = settle(frame_camera_info_.send(stereo_camera_info, emit), kSentCameraInfo,
                              sent_, blocked);
        !settled || blocked) {
      return settled;
    }
  }
  for (int side = 0; side < 2; ++side) {
    const std::uint32_t bit = side == 0 ? kSentLeft : kSentRight;
    if ((sent_ & bit) != 0U) continue;
    holoscan::Output<holoscan::schema::ImageT>& port = side == 0 ? left : right;
    std::vector<std::int64_t> shape{config_.height, config_.width};
    if (config_.color) shape.push_back(3);
    auto loan = port.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
    if (!loan) {
      return settle(holoscan::make_unexpected(std::move(loan).error()), bit, sent_, blocked);
    }
    const std::uint8_t* image = static_cast<const std::uint8_t*>(scratch_) + side * bytes;
    if (config_.device_output) {
      auto writer = loan->write(stream);
      if (!writer) {
        return holoscan::make_unexpected(std::move(writer).error());
      }
      if (const cudaError_t status = cudaMemcpyAsync(writer->data_as<std::uint8_t>(), image, bytes,
                                                     cudaMemcpyDeviceToDevice, stream);
          status != cudaSuccess) {
        return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", status));
      }
      if (stream == own_stream_) {
        if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
          return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
        }
      }
      if (auto committed = std::move(*writer).commit(); !committed) {
        return holoscan::make_unexpected(std::move(committed).error());
      }
    } else {
      auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                     -> holoscan::expected<void, holoscan::Error> {
        const cudaError_t status = cudaMemcpyAsync(writer.data_as<std::uint8_t>(), image, bytes,
                                                   cudaMemcpyDeviceToHost, stream);
        const cudaError_t synced = status == cudaSuccess ? cudaStreamSynchronize(stream) : status;
        if (synced != cudaSuccess) {
          return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", synced));
        }
        return {};
      });
      if (!written) {
        return written;
      }
    }
    if (auto settled = settle(
            port.emit_tensor(std::move(*loan),
                             image_descriptor(config_.width, config_.height, encoding), emit),
            bit, sent_, blocked);
        !settled || blocked) {
      return settled;
    }
  }
  pending_capture_.reset();
  ++emitted_;
  return {};
}

void validate_config(const RecordingSourceConfig& config) {
  if (config.recording.frames.empty()) {
    throw std::invalid_argument("the recording has no frames");
  }
  if (config.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("the recording source needs a positive period");
  }
  if (!config.loop && config.count > config.recording.frames.size()) {
    throw std::invalid_argument("--count " + std::to_string(config.count) + " exceeds the " +
                                std::to_string(config.recording.frames.size()) +
                                " frames of the recording; add --loop true");
  }
  if (config.depth && !config.recording.depth_aligned) {
    throw std::invalid_argument(
        "the recording has no depth aligned with its left image (depth.yaml describing the "
        "left camera)");
  }
}

RecordingSourceOp::RecordingSourceOp(RecordingSourceConfig config) : config_(std::move(config)) {
  validate_config(config_);
  total_ = config_.count == 0 ? static_cast<std::uint32_t>(config_.recording.frames.size())
                              : config_.count;
}

void RecordingSourceOp::setup(holoscan::OperatorSpec& spec) {
  const holoscan::messages::CameraInfo& camera = config_.recording.camera_info.left;
  const std::uint64_t pixels =
      static_cast<std::uint64_t>(camera.width) * static_cast<std::uint64_t>(camera.height);
  spec.input(camera_info, "camera_info");
  spec.output(stereo_camera_info, "stereo_camera_info");
  spec.output(left, "left")
      .produces_tensor(tensor_spec(holoscan::MemoryKind::kHost, kUInt8, 2U, pixels))
      .payload_options(image_payload());
  spec.output(right, "right")
      .produces_tensor(tensor_spec(holoscan::MemoryKind::kHost, kUInt8, 2U, pixels))
      .payload_options(image_payload());
  if (config_.depth) {
    spec.output(depth, "depth")
        .produces_tensor(tensor_spec(holoscan::MemoryKind::kHost, kUInt16, 2U, pixels * 2U))
        .payload_options(image_payload());
  }
}

holoscan::Contract RecordingSourceOp::contract() const {
  return source_contract(config_.period, camera_info);
}

void RecordingSourceOp::start() {
  frame_camera_info_.reset();
  emitted_ = 0;
  pending_capture_.reset();
  sent_ = 0U;
}

holoscan::expected<void, holoscan::Error> RecordingSourceOp::compute(
    holoscan::ExecutionContext& context) {
  auto camera_info_came = frame_camera_info_.update(camera_info);
  if (!camera_info_came) {
    return holoscan::make_unexpected(std::move(camera_info_came).error());
  }
  if (emitted_ >= total_ || !*camera_info_came) {
    return {};  // Done, or not started.
  }
  const int width = static_cast<int>(config_.recording.camera_info.left.width);
  const int height = static_cast<int>(config_.recording.camera_info.left.height);
  const RecordingFrame& frame =
      config_.recording.frames[emitted_ % config_.recording.frames.size()];
  if (!pending_capture_) {
    pending_capture_ = context.activation_time();
    sent_ = 0U;
    try {
      left_pixels_ = read_pgm(frame.left);
      right_pixels_ = read_pgm(frame.right);
      if (config_.depth) {
        if (frame.depth.empty()) {
          throw std::runtime_error("frame " + std::to_string(frame.frame) + " has no depth image");
        }
        depth_pixels_ = read_pgm(frame.depth);
      }
    } catch (const std::exception& error) {
      return holoscan::make_unexpected(failure(error.what()));
    }
    for (const auto* pgm : {&left_pixels_, &right_pixels_, &depth_pixels_}) {
      if (*pgm && ((*pgm)->width != width || (*pgm)->height != height)) {
        return holoscan::make_unexpected(failure("frame " + std::to_string(frame.frame) +
                                                 ": an image is not " + std::to_string(width) +
                                                 "x" + std::to_string(height)));
      }
    }
    if (left_pixels_->max_value > 255 || right_pixels_->max_value > 255) {
      return holoscan::make_unexpected(failure("the eyes must be 8-bit PGM images"));
    }
  }
  const holoscan::EmitOptions emit{.capture_time = *pending_capture_, .frame_id = emitted_ + 1U};
  bool blocked = false;

  const std::array<std::int64_t, 2> shape{height, width};
  const auto send = [&](holoscan::Output<holoscan::schema::ImageT>& port, const Pgm& pgm,
                        bool sixteen,
                        std::uint32_t bit) -> holoscan::expected<void, holoscan::Error> {
    auto loan = port.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = shape, .dtype = sixteen ? kUInt16 : kUInt8});
    if (!loan) {
      return settle(holoscan::make_unexpected(std::move(loan).error()), bit, sent_, blocked);
    }
    auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                   -> holoscan::expected<void, holoscan::Error> {
      if (sixteen) {
        std::memcpy(writer.data_as<std::uint16_t>(), pgm.pixels.data(),
                    pgm.pixels.size() * sizeof(std::uint16_t));
      } else {
        auto* out = writer.data_as<std::uint8_t>();
        for (std::size_t i = 0; i < pgm.pixels.size(); ++i) {
          out[i] = static_cast<std::uint8_t>(pgm.pixels[i]);
        }
      }
      return {};
    });
    if (!written) {
      return written;
    }
    return settle(
        port.emit_tensor(std::move(*loan),
                         image_descriptor(width, height,
                                          sixteen ? holoscan::schema::ImageEncoding_MONO16
                                                  : holoscan::schema::ImageEncoding_MONO8),
                         emit),
        bit, sent_, blocked);
  };
  if ((sent_ & kSentCameraInfo) == 0U) {
    if (auto settled = settle(frame_camera_info_.send(stereo_camera_info, emit), kSentCameraInfo,
                              sent_, blocked);
        !settled || blocked) {
      return settled;
    }
  }
  if (config_.depth && (sent_ & kSentDepth) == 0U) {
    if (auto sent = send(depth, *depth_pixels_, true, kSentDepth); !sent || blocked) {
      return sent;
    }
  }
  if ((sent_ & kSentLeft) == 0U) {
    if (auto sent = send(left, *left_pixels_, false, kSentLeft); !sent || blocked) {
      return sent;
    }
  }
  if ((sent_ & kSentRight) == 0U) {
    if (auto sent = send(right, *right_pixels_, false, kSentRight); !sent || blocked) {
      return sent;
    }
  }
  pending_capture_.reset();
  ++emitted_;
  return {};
}

}  // namespace stereo_disparity_app
