// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/stereo_camera/stereo_camera_op.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>

namespace holoscan::stereo_camera {
namespace {

using Result = holoscan::expected<void, holoscan::Error>;

// How often a live camera is asked for a new frame; a poll without one does nothing.
constexpr std::chrono::milliseconds kLivePoll{2};
// The range of a recording's pace.
constexpr std::chrono::milliseconds kMinPeriod{1};
constexpr std::chrono::seconds kMaxPeriod{1};
// A recording's pace when it cannot be learned from its timestamps.
constexpr std::chrono::milliseconds kDefaultPeriod{33};

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr DLDataType kUInt16{kDLUInt, 16U, 1U};

holoscan::Error failure(std::string message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "StereoCameraOp: " + std::move(message)};
}

holoscan::schema::ImageEncoding encoding_of(PixelFormat format) {
  switch (format) {
    case PixelFormat::kMono8:
      return holoscan::schema::ImageEncoding_MONO8;
    case PixelFormat::kMono16:
      return holoscan::schema::ImageEncoding_MONO16;
    case PixelFormat::kRgb8:
      return holoscan::schema::ImageEncoding_RGB8;
    case PixelFormat::kBgr8:
      return holoscan::schema::ImageEncoding_BGR8;
  }
  return holoscan::schema::ImageEncoding_CUSTOM;
}

// [H, W] for one channel, [H, W, 3] for three.
std::uint32_t rank_of(PixelFormat format) { return channels(format) == 1 ? 2U : 3U; }

DLDataType dtype_of(PixelFormat format) {
  return format == PixelFormat::kMono16 ? kUInt16 : kUInt8;
}

holoscan::TensorOutputSpec image_spec(const StreamLayout& layout, OutputMemory memory) {
  const std::uint64_t bytes = static_cast<std::uint64_t>(layout.width) *
                              static_cast<std::uint64_t>(layout.height) *
                              static_cast<std::uint64_t>(bytes_per_pixel(layout.format));
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = memory == OutputMemory::kDevice ? holoscan::MemoryKind::kCudaDevice
                                                             : holoscan::MemoryKind::kHost,
              .dtype = dtype_of(layout.format),
              .rank = rank_of(layout.format),
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

std::string describe(const StreamLayout& layout) {
  return std::to_string(layout.width) + "x" + std::to_string(layout.height) + " " +
         std::string(pixel_format_name(layout.format));
}

}  // namespace

namespace detail {
// The ports of one frame, in the order they go out.
enum class Port : std::uint8_t { kLeft, kRight, kRgb, kDepth, kCalibration };
}  // namespace detail
using detail::Port;

struct StereoCameraOp::Pending {
  StereoFrameView view;
  holoscan::TimePoint capture;
  std::uint64_t frame_id{};
  // The ports still to send it, in order.
  std::vector<Port> ports;
};

void validate_config(const StereoCameraConfig& config) {
  if (!config.make_source) {
    validate(config.source);
  }
  if (config.period < std::chrono::nanoseconds::zero() ||
      (config.period != std::chrono::nanoseconds::zero() &&
       (config.period < kMinPeriod || config.period > kMaxPeriod))) {
    throw std::invalid_argument("period must be zero (the recorded rate) or 1 ms..1 s");
  }
  if (config.frame_id.empty()) {
    throw std::invalid_argument("frame_id must not be empty");
  }
}

StereoCameraOp::StereoCameraOp(StereoCameraConfig config) : config_(std::move(config)) {
  validate_config(config_);
  if (config_.make_source) {
    source_ = config_.make_source();
    if (!source_) {
      throw std::invalid_argument("make_source returned no source");
    }
  } else {
    source_ = create_source(config_.source);
  }
  layout_ = source_->layout();
  if (layout_.left != layout_.right) {
    throw std::invalid_argument("the eyes must have the same size and format, not " +
                                describe(layout_.left) + " and " + describe(layout_.right));
  }
  if (layout_.live) {
    period_ = kLivePoll;
  } else if (config_.period != std::chrono::nanoseconds::zero()) {
    period_ = config_.period;
  } else {
    period_ = kDefaultPeriod;
    if (!config_.make_source && config_.source.backend == Backend::kPlayback) {
      const std::int64_t recorded =
          RecordingReader(config_.source.playback.directory).median_period_ns();
      if (recorded > 0) {
        period_ =
            std::clamp(std::chrono::nanoseconds(recorded), std::chrono::nanoseconds(kMinPeriod),
                       std::chrono::nanoseconds(kMaxPeriod));
      }
    }
  }
}

StereoCameraOp::~StereoCameraOp() {
  if (source_) {
    source_->close();
  }
}

void StereoCameraOp::setup(holoscan::OperatorSpec& spec) {
  spec.output(left, "left")
      .produces_tensor(image_spec(layout_.left, config_.memory))
      .payload_options(image_payload());
  spec.output(right, "right")
      .produces_tensor(image_spec(layout_.right, config_.memory))
      .payload_options(image_payload());
  // Optional ports exist only when the source has the stream: an emit into an
  // unconnected output fails.
  if (layout_.rgb) {
    spec.output(rgb, "rgb")
        .produces_tensor(image_spec(*layout_.rgb, config_.memory))
        .payload_options(image_payload());
  }
  if (layout_.depth) {
    spec.output(depth, "depth")
        .produces_tensor(image_spec(*layout_.depth, config_.memory))
        .payload_options(image_payload());
  }
  if (config_.calibration) {
    spec.output(calibration, "calibration");
  }
}

holoscan::Contract StereoCameraOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = period_});
  return result;
}

void StereoCameraOp::start() {
  pending_.reset();
  frame_ids_ = 0;
  try {
    source_->open();
    message_ = to_message(source_->calibration());
  } catch (const std::exception& error) {
    source_->close();
    throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure,
                                 std::string("StereoCameraOp: ") + error.what()};
  }
  message_->backend = std::string(backend_name(source_->backend()));
  const StereoCalibration& calibration = source_->calibration();
  std::string camera = calibration.camera.name.empty() ? "camera" : calibration.camera.name;
  if (!calibration.camera.serial.empty()) camera += " " + calibration.camera.serial;
  if (!calibration.camera.firmware.empty()) camera += " (" + calibration.camera.firmware + ")";
  HOLOSCAN_LOG_INFO("StereoCameraOp: {} {}: eyes {}, baseline {:.2f} mm, {}{}{}", message_->backend,
                    camera, describe(layout_.left), 1000.0 * calibration.baseline(),
                    calibration.rectified ? "rectified" : "not rectified",
                    layout_.rgb ? ", rgb " + describe(*layout_.rgb) : "",
                    layout_.depth ? ", depth " + describe(*layout_.depth) : "");
}

void StereoCameraOp::stop() {
  pending_.reset();
  source_->close();
}

holoscan::expected<void, holoscan::Error> StereoCameraOp::compute(
    holoscan::ExecutionContext& context) {
  if (!pending_) {
    if (config_.count != 0 && frame_ids_ >= config_.count) {
      return {};  // Done: idle until the run completes.
    }
    std::optional<StereoFrameView> view;
    try {
      view = source_->grab();
    } catch (const std::exception& error) {
      return holoscan::make_unexpected(failure(error.what()));
    }
    if (!view) {
      return {};  // No new frame yet, or the recording has ended.
    }
    auto frame = std::make_unique<Pending>();
    frame->view = *view;
    // The capture time on the graph's clock: now, less the frame's age when the
    // source knows it.
    frame->capture = context.activation_time();
    if (view->age_ns) {
      frame->capture.timestamp_ns -= *view->age_ns;
    }
    frame->frame_id = ++frame_ids_;
    frame->ports = {Port::kLeft, Port::kRight};
    if (layout_.rgb) frame->ports.push_back(Port::kRgb);
    if (layout_.depth) frame->ports.push_back(Port::kDepth);
    if (config_.calibration) frame->ports.push_back(Port::kCalibration);
    pending_ = std::move(frame);
  }
  return send(*pending_, context);
}

holoscan::expected<void, holoscan::Error> StereoCameraOp::send(
    Pending& frame, holoscan::ExecutionContext& context) {
  const holoscan::EmitOptions options{.capture_time = frame.capture, .frame_id = frame.frame_id};
  cudaStream_t stream = nullptr;
  if (config_.memory == OutputMemory::kDevice) {
    stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return holoscan::make_unexpected(
          failure("the runtime gave no CUDA stream for the device outputs; bind their device"));
    }
  }
  const auto send_image = [&](holoscan::Output<holoscan::schema::ImageT>& port,
                              const ImageView& view, const char* suffix) -> Result {
    const std::size_t row = static_cast<std::size_t>(view.width) * bytes_per_pixel(view.format);
    std::vector<std::int64_t> shape{view.height, view.width};
    if (channels(view.format) == 3) shape.push_back(3);
    auto loan = port.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype_of(view.format)});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    if (config_.memory == OutputMemory::kDevice) {
      auto writer = loan->write(stream);
      if (!writer) {
        return holoscan::make_unexpected(std::move(writer).error());
      }
      // From pageable memory the copy is staged before it returns, so the
      // source may reuse its buffer afterwards.
      if (const cudaError_t status = cudaMemcpy2DAsync(
              writer->data_as<std::uint8_t>(), row, view.data, view.pitch, row,
              static_cast<std::size_t>(view.height), cudaMemcpyHostToDevice, stream);
          status != cudaSuccess) {
        return holoscan::make_unexpected(
            failure(std::string("cudaMemcpy2DAsync: ") + cudaGetErrorString(status)));
      }
      if (auto committed = std::move(*writer).commit(); !committed) {
        return holoscan::make_unexpected(std::move(committed).error());
      }
    } else {
      auto committed = loan->write_host_and_commit(
          [&](const holoscan::TensorOutputWriteGuard& writer) -> Result {
            auto* out = writer.data_as<std::uint8_t>();
            for (int y = 0; y < view.height; ++y) {
              std::memcpy(out + static_cast<std::size_t>(y) * row,
                          view.data + static_cast<std::size_t>(y) * view.pitch, row);
            }
            return {};
          });
      if (!committed) {
        return committed;
      }
    }
    holoscan::schema::ImageT descriptor;
    descriptor.width = view.width;
    descriptor.height = view.height;
    descriptor.encoding = encoding_of(view.format);
    descriptor.header = std::make_shared<holoscan::schema::HeaderT>();
    descriptor.header->capture_timestamp_ns = frame.capture.timestamp_ns;
    descriptor.header->frame_id = config_.frame_id + suffix;
    descriptor.header->device_sequence = frame.view.device_sequence;
    return port.emit_tensor(std::move(*loan), descriptor, options);
  };

  while (!frame.ports.empty()) {
    Result sent;
    switch (frame.ports.front()) {
      case Port::kLeft:
        sent = send_image(left, frame.view.left, "_left");
        break;
      case Port::kRight:
        sent = send_image(right, frame.view.right, "_right");
        break;
      case Port::kRgb:
        sent = send_image(rgb, *frame.view.rgb, "_rgb");
        break;
      case Port::kDepth:
        sent = send_image(depth, *frame.view.depth, "_depth");
        break;
      case Port::kCalibration: {
        schema::StereoCalibrationT message = *message_;
        message.frame = frame.view.index;
        message.timestamp_ns = frame.view.timestamp_ns;
        message.device_sequence = frame.view.device_sequence;
        sent = calibration.emit(std::move(message), options);
        break;
      }
    }
    if (!sent) {
      if (sent.error().code != holoscan::ErrorCode::kBackpressured) {
        pending_.reset();
        return sent;
      }
      if (!layout_.live) {
        return {};  // A consumer is busy: send the rest of the frame on the next tick.
      }
      // A live frame is not worth waiting for: drop the rest of it.
      if (config_.stats) ++config_.stats->dropped;
      pending_.reset();
      return {};
    }
    frame.ports.erase(frame.ports.begin());
  }
  if (config_.stats) ++config_.stats->sent;
  pending_.reset();
  return {};
}

}  // namespace holoscan::stereo_camera
