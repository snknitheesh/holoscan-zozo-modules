// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/codec/video_encoder_op.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/codec/video_codec.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "image_input.hpp"

namespace holoscan::codec {
namespace {

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
// Room for an access unit's headers beyond its pixels' worth of NV12.
constexpr std::uint64_t kAccessUnitSlack = 64U * 1024U;

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "VideoEncoderOp: " + message};
}

holoscan::Error failure(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "VideoEncoderOp: " + message};
}

std::uint64_t max_access_unit_bytes(const VideoEncoderConfig& config) {
  return static_cast<std::uint64_t>(config.max_width) *
             static_cast<std::uint64_t>(config.max_height) * 3U / 2U +
         kAccessUnitSlack;
}

schema::VideoCodec to_schema(Codec codec) {
  return codec == Codec::kH264 ? schema::VideoCodec_H264 : schema::VideoCodec_H265;
}

}  // namespace

struct VideoEncoderOp::State {
  struct StreamDestroy {
    void operator()(cudaStream_t stream) const noexcept {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  };

  std::atomic<Backend> resolved{Backend::kAuto};
  std::unique_ptr<VideoEncoder> encoder;
  // Why start() could not set up, which every compute() then reports.
  std::string start_error;
  std::ofstream file;
  // For activations without a runtime CUDA stream (host images); made on first use.
  std::unique_ptr<CUstream_st, StreamDestroy> own_stream;
  detail::ImageStaging staging;
  bool copy_logged{};
  int width{};
  int height{};
  std::uint64_t frames{};
  std::uint64_t bytes{};
};

void validate_config(const VideoEncoderConfig& config) {
  validate_settings(config.settings);
  if (config.max_width < kMinFrameSide || config.max_height < kMinFrameSide ||
      config.max_width > kMaxFrameSide || config.max_height > kMaxFrameSide) {
    throw std::invalid_argument("max_width and max_height must be " +
                                std::to_string(kMinFrameSide) + " to " +
                                std::to_string(kMaxFrameSide));
  }
}

VideoEncoderOp::VideoEncoderOp(VideoEncoderConfig config)
    : config_(std::move(config)), state_(std::make_unique<State>()) {
  validate_config(config_);
}

VideoEncoderOp::~VideoEncoderOp() = default;

Backend VideoEncoderOp::resolved_backend() const noexcept { return state_->resolved; }

void VideoEncoderOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots: one image in compute() while the next one arrives.
  spec.input(image, "image").queue_depth(2U);
  spec.output(bitstream, "bitstream")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kHost,
                  .dtype = kUInt8,
                  .rank = 1U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(max_access_unit_bytes(config_)),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      })
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<schema::EncodedVideoFrameT>(),
      });
}

holoscan::Contract VideoEncoderOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{image});
  return result;
}

void VideoEncoderOp::start() {
  State& state = *state_;
  state.start_error.clear();
  state.frames = 0;
  state.bytes = 0;
  state.width = 0;
  state.height = 0;
  try {
    state.encoder = std::make_unique<VideoEncoder>(config_.settings);
  } catch (const std::exception& error) {
    state.start_error = error.what();
    HOLOSCAN_LOG_ERROR("VideoEncoderOp: {}", state.start_error);
    return;
  }
  state.resolved = state.encoder->backend();
  const EncoderSettings& s = config_.settings;
  HOLOSCAN_LOG_INFO(
      "VideoEncoderOp: {} on the {} backend ({}){}; {} {}, gop {}, preset {}, {}",
      codec_name(s.codec), backend_name(state.encoder->backend()), state.encoder->backend_detail(),
      s.backend == Backend::kAuto ? ", which auto chose as the first backend that works here" : "",
      rate_control_name(s.rate_control),
      s.rate_control == RateControl::kConstQp ? "qp " + std::to_string(s.qp)
                                              : std::to_string(s.bitrate) + " bit/s",
      s.gop, preset_name(s.preset), s.low_latency ? "low latency" : "high quality");
  if (!config_.output_path.empty()) {
    state.file.open(config_.output_path, std::ios::binary | std::ios::trunc);
    if (!state.file) {
      state.start_error = "cannot write " + config_.output_path.string();
      HOLOSCAN_LOG_ERROR("VideoEncoderOp: {}", state.start_error);
    }
  }
}

void VideoEncoderOp::stop() {
  State& state = *state_;
  if (state.frames > 0) {
    const double seconds = static_cast<double>(state.frames) / config_.settings.frame_rate;
    HOLOSCAN_LOG_INFO("VideoEncoderOp: {} frames, {} bytes, {:.2f} Mbit/s at {} frames/s",
                      state.frames, state.bytes,
                      static_cast<double>(state.bytes) * 8.0 / seconds / 1e6,
                      config_.settings.frame_rate);
  }
  if (state.file.is_open()) {
    state.file.close();
  }
  state.encoder.reset();
  state.staging.release();
  state.own_stream.reset();
}

holoscan::expected<void, holoscan::Error> VideoEncoderOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = image.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  State& state = *state_;
  if (!state.start_error.empty() || !state.encoder) {
    return holoscan::make_unexpected(failure(state.start_error));
  }
  const holoscan::schema::ImageT& input = sample->data;
  auto view = detail::view_image(input);
  if (!view) {
    return holoscan::make_unexpected(invalid(view.error()));
  }
  if (view->image.width > config_.max_width || view->image.height > config_.max_height) {
    return holoscan::make_unexpected(
        invalid("image " + std::to_string(view->image.width) + "x" +
                std::to_string(view->image.height) + " exceeds the configured " +
                std::to_string(config_.max_width) + "x" + std::to_string(config_.max_height)));
  }
  if (state.width != 0 &&
      (view->image.width != state.width || view->image.height != state.height) &&
      !config_.allow_resolution_change) {
    return holoscan::make_unexpected(
        invalid("the image size changed from " + std::to_string(state.width) + "x" +
                std::to_string(state.height) + " to " + std::to_string(view->image.width) + "x" +
                std::to_string(view->image.height) + ", and allow_resolution_change is off"));
  }

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    // Host images and a host output: no runtime stream. Use our own, never the legacy default
    // stream, which serializes with every other stream in the process.
    if (!state.own_stream) {
      cudaStream_t created = nullptr;
      if (const cudaError_t status = cudaStreamCreateWithFlags(&created, cudaStreamNonBlocking);
          status != cudaSuccess) {
        return holoscan::make_unexpected(
            failure(std::string("cudaStreamCreate: ") + cudaGetErrorString(status)));
      }
      state.own_stream.reset(created);
    }
    stream = state.own_stream.get();
  }
  const holoscan::Tensor& tensor = *input.data;
  auto ready = view->on_device && tensor.has_readiness_stream_contract()
                   ? tensor.synchronize_for_consumer_stream(stream)
                   : tensor.synchronize_for_host();
  if (!ready) {
    return holoscan::make_unexpected(std::move(ready).error());
  }

  std::int64_t pts = 0;
  if (input.header && input.header->capture_timestamp_ns != 0) {
    pts = input.header->capture_timestamp_ns;
  } else if (sample->metadata.source_clock_id.valid()) {
    pts = sample->metadata.capture_timestamp_ns;
  } else {
    pts = static_cast<std::int64_t>(static_cast<double>(state.frames) * 1e9 /
                                    config_.settings.frame_rate);
  }

  EncodedFrame encoded;
  try {
    DeviceImage device_image = view->image;
    if (!view->on_device) {
      if (!state.copy_logged) {
        state.copy_logged = true;
        HOLOSCAN_LOG_INFO(
            "VideoEncoderOp: the images are in host memory, so each one is copied to the device "
            "before encoding ({} bytes per {}x{} frame); send CUDA device images to avoid it",
            detail::image_bytes(view->image), view->image.width, view->image.height);
      }
      device_image = state.staging.to_device(view->image, stream);
    }
    encoded = state.encoder->encode(device_image, pts, stream);
  } catch (const std::invalid_argument& error) {
    return holoscan::make_unexpected(invalid(error.what()));
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(failure(error.what()));
  }
  state.width = view->image.width;
  state.height = view->image.height;

  if (encoded.data.size() > max_access_unit_bytes(config_)) {
    return holoscan::make_unexpected(
        failure("an access unit of " + std::to_string(encoded.data.size()) +
                " bytes exceeds the output's capacity; raise max_width and max_height"));
  }
  if (state.file.is_open()) {
    state.file.write(reinterpret_cast<const char*>(encoded.data.data()),
                     static_cast<std::streamsize>(encoded.data.size()));
    if (!state.file) {
      return holoscan::make_unexpected(failure("writing " + config_.output_path.string()));
    }
  }
  const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(encoded.data.size())};
  auto loan =
      bitstream.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto committed = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                   -> holoscan::expected<void, holoscan::Error> {
    if (writer.byte_size() < encoded.data.size()) {
      return holoscan::make_unexpected(failure("the output loan is too small"));
    }
    std::memcpy(writer.data_as<std::uint8_t>(), encoded.data.data(), encoded.data.size());
    return {};
  });
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  schema::EncodedVideoFrameT descriptor;
  if (input.header) {
    descriptor.header = std::make_shared<holoscan::schema::HeaderT>(*input.header);
  }
  descriptor.codec = to_schema(config_.settings.codec);
  descriptor.width = encoded.width;
  descriptor.height = encoded.height;
  descriptor.pts = encoded.pts;
  descriptor.keyframe = encoded.keyframe;
  descriptor.frame_index = encoded.index;
  ++state.frames;
  state.bytes += encoded.data.size();
  return bitstream.emit_tensor(std::move(*loan), descriptor,
                               holoscan::sensor_io::derive_from(*sample));
}

}  // namespace holoscan::codec
