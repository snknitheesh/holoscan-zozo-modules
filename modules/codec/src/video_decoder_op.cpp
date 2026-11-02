// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/codec/video_decoder_op.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <exception>
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

namespace holoscan::codec {
namespace {

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
// Images one access unit may complete: one, or a few while the decoder catches up.
constexpr std::uint32_t kMaxImagesPerAccessUnit = 4;

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "VideoDecoderOp: " + message};
}

holoscan::Error failure(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "VideoDecoderOp: " + message};
}

std::size_t chroma_row_bytes(int width) { return 2 * static_cast<std::size_t>((width + 1) / 2); }

// Bytes of one output image of `width` x `height`.
std::uint64_t image_bytes(PixelFormat format, int width, int height) {
  const auto rows = static_cast<std::uint64_t>(height);
  if (format == PixelFormat::kNv12) {
    return static_cast<std::uint64_t>(width) * rows + chroma_row_bytes(width) * ((rows + 1) / 2);
  }
  return static_cast<std::uint64_t>(width) * rows *
         static_cast<std::uint64_t>(bytes_per_pixel(format));
}

holoscan::schema::ImageEncoding encoding_of(PixelFormat format) {
  switch (format) {
    case PixelFormat::kRgb8:
      return holoscan::schema::ImageEncoding_RGB8;
    case PixelFormat::kBgr8:
      return holoscan::schema::ImageEncoding_BGR8;
    case PixelFormat::kRgba8:
      return holoscan::schema::ImageEncoding_RGBA8;
    case PixelFormat::kBgra8:
      return holoscan::schema::ImageEncoding_BGRA8;
    case PixelFormat::kNv12:
      break;
  }
  return holoscan::schema::ImageEncoding_NV12;
}

Codec from_schema(schema::VideoCodec codec) {
  return codec == schema::VideoCodec_H265 ? Codec::kH265 : Codec::kH264;
}

}  // namespace

struct VideoDecoderOp::State {
  struct StreamDestroy {
    void operator()(cudaStream_t stream) const noexcept {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  };
  // What an access unit said about its frame, until the frame comes out.
  struct Pending {
    std::int64_t pts{};
    std::shared_ptr<holoscan::schema::HeaderT> header;
    int width{};
    int height{};
    holoscan::SampleMetadata metadata;
  };

  std::atomic<Backend> resolved{Backend::kAuto};
  std::unique_ptr<VideoDecoder> decoder;
  int width{};
  int height{};
  std::deque<Pending> pending;
  std::unique_ptr<CUstream_st, StreamDestroy> own_stream;
  // A copy of access units that arrive in device memory.
  std::vector<std::uint8_t> host_copy;
  bool empty_logged{};
  bool device_input_logged{};
  std::uint64_t decoded{};
  std::uint64_t empty{};
};

void validate_config(const VideoDecoderConfig& config) {
  if (backend_name(config.backend) == "unknown") {
    throw std::invalid_argument("backend: unknown backend");
  }
  if (pixel_format_name(config.output_format) == "unknown") {
    throw std::invalid_argument("output_format: unknown pixel format");
  }
  if (config.max_width < kMinFrameSide || config.max_height < kMinFrameSide ||
      config.max_width > kMaxFrameSide || config.max_height > kMaxFrameSide) {
    throw std::invalid_argument("max_width and max_height must be " +
                                std::to_string(kMinFrameSide) + " to " +
                                std::to_string(kMaxFrameSide));
  }
}

VideoDecoderOp::VideoDecoderOp(VideoDecoderConfig config)
    : config_(std::move(config)), state_(std::make_unique<State>()) {
  validate_config(config_);
}

VideoDecoderOp::~VideoDecoderOp() = default;

Backend VideoDecoderOp::resolved_backend() const noexcept { return state_->resolved; }

void VideoDecoderOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(bitstream, "bitstream").queue_depth(2U);
  const bool nv12 = config_.output_format == PixelFormat::kNv12;
  spec.output(image, "image")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kCudaDevice,
                  .dtype = kUInt8,
                  .rank = nv12 ? 1U : 3U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(
              image_bytes(config_.output_format, config_.max_width, config_.max_height)),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      })
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      })
      // A decoder catching up at the start of a stream hands out a few frames at once.
      .max_emits_per_compute(kMaxImagesPerAccessUnit);
}

holoscan::Contract VideoDecoderOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{bitstream});
  return result;
}

void VideoDecoderOp::start() {
  State& state = *state_;
  state.decoder.reset();
  state.pending.clear();
  state.width = 0;
  state.height = 0;
  state.decoded = 0;
  state.empty = 0;
}

void VideoDecoderOp::stop() {
  State& state = *state_;
  if (state.decoder) {
    const std::uint64_t skipped = state.decoder->skipped();
    HOLOSCAN_LOG_INFO(
        "VideoDecoderOp: {} frames decoded; {} access units before the first keyframe and {} "
        "empty ones gave none, and {} were still in the decoder",
        state.decoded, skipped, state.empty, state.pending.size());
  }
  state.decoder.reset();
  state.pending.clear();
  state.own_stream.reset();
}

holoscan::expected<void, holoscan::Error> VideoDecoderOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = bitstream.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  State& state = *state_;
  const schema::EncodedVideoFrameT& message = sample->data;
  const std::size_t size = message.data ? static_cast<std::size_t>(message.data->nbytes()) : 0U;
  if (size == 0) {
    ++state.empty;
    if (!state.empty_logged) {
      state.empty_logged = true;
      HOLOSCAN_LOG_WARN("VideoDecoderOp: an empty access unit gave no image");
    }
    return {};
  }
  if (message.width < 1 || message.height < 1 || message.width > config_.max_width ||
      message.height > config_.max_height) {
    return holoscan::make_unexpected(
        invalid("frame size " + std::to_string(message.width) + "x" +
                std::to_string(message.height) + " is not within the configured " +
                std::to_string(config_.max_width) + "x" + std::to_string(config_.max_height)));
  }
  const holoscan::Tensor& tensor = *message.data;
  const DLDevice device = tensor.device();
  const bool on_device = device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
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
  if (auto ready = tensor.synchronize_for_host(); !ready) {
    return holoscan::make_unexpected(std::move(ready).error());
  }
  std::span<const std::uint8_t> access_unit(static_cast<const std::uint8_t*>(tensor.data()), size);
  if (on_device) {
    if (!state.device_input_logged) {
      state.device_input_logged = true;
      HOLOSCAN_LOG_INFO(
          "VideoDecoderOp: the access units are in device memory, and the decoder reads them on "
          "the host, so each one is copied there; send them in host memory to avoid it");
    }
    state.host_copy.resize(size);
    if (const cudaError_t status =
            cudaMemcpy(state.host_copy.data(), tensor.data(), size, cudaMemcpyDeviceToHost);
        status != cudaSuccess) {
      return holoscan::make_unexpected(failure(cudaGetErrorString(status)));
    }
    access_unit = state.host_copy;
  }

  const Codec codec = from_schema(message.codec);
  std::vector<DecodedFrame> frames;
  try {
    const bool restart = !state.decoder || state.decoder->codec() != codec ||
                         message.width != state.width || message.height != state.height;
    if (restart) {
      if (state.decoder && !state.pending.empty()) {
        HOLOSCAN_LOG_WARN(
            "VideoDecoderOp: the stream changed to {} {}x{}; {} frames still in the decoder are "
            "dropped",
            codec_name(codec), message.width, message.height, state.pending.size());
      }
      const bool first = !state.decoder;
      state.decoder = std::make_unique<VideoDecoder>(codec, config_.backend, config_.low_latency);
      state.pending.clear();
      state.width = message.width;
      state.height = message.height;
      if (first) {
        state.resolved = state.decoder->backend();
        HOLOSCAN_LOG_INFO("VideoDecoderOp: {} on the {} backend ({}){}", codec_name(codec),
                          backend_name(state.decoder->backend()), state.decoder->backend_detail(),
                          config_.backend == Backend::kAuto
                              ? ", which auto chose as the first backend that works here"
                              : "");
      }
    }
    state.pending.push_back(State::Pending{.pts = message.pts,
                                           .header = message.header,
                                           .width = message.width,
                                           .height = message.height,
                                           .metadata = sample->metadata});
    const std::uint64_t skipped_before = state.decoder->skipped();
    frames = state.decoder->decode(access_unit, message.pts, stream);
    if (state.decoder->skipped() != skipped_before) {
      state.pending.pop_back();  // Not decodable: no frame will come for it.
    }
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(failure(error.what()));
  }

  if (frames.size() > kMaxImagesPerAccessUnit) {
    HOLOSCAN_LOG_WARN("VideoDecoderOp: one access unit completed {} frames; {} are dropped",
                      frames.size(), frames.size() - kMaxImagesPerAccessUnit);
    frames.resize(kMaxImagesPerAccessUnit);
  }
  for (const DecodedFrame& frame : frames) {
    // The access unit this frame came from: frames come out in order, so earlier ones are gone.
    while (!state.pending.empty() && state.pending.front().pts != frame.pts) {
      state.pending.pop_front();
    }
    if (state.pending.empty()) {
      return holoscan::make_unexpected(failure("a decoded frame matches no access unit"));
    }
    const State::Pending source = std::move(state.pending.front());
    state.pending.pop_front();
    const int width = std::min(source.width, frame.image.width);
    const int height = std::min(source.height, frame.image.height);

    const PixelFormat format = config_.output_format;
    const bool nv12 = format == PixelFormat::kNv12;
    const std::uint64_t bytes = image_bytes(format, width, height);
    std::array<std::int64_t, 3> shape{height, width, bytes_per_pixel(format)};
    std::array<std::int64_t, 1> flat{static_cast<std::int64_t>(bytes)};
    auto loan =
        nv12 ? image.allocate_tensor(holoscan::TensorLoanRequest{.shape = flat, .dtype = kUInt8})
             : image.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(stream);
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    const std::size_t pitch =
        nv12 ? static_cast<std::size_t>(width)
             : static_cast<std::size_t>(width) * static_cast<std::size_t>(bytes_per_pixel(format));
    auto* out = writer->data_as<std::uint8_t>();
    try {
      if (nv12) {
        // The planes back to back, each packed: chroma rows are a byte wider for odd widths.
        const auto rows = static_cast<std::size_t>(height);
        const cudaError_t luma_copy =
            cudaMemcpy2DAsync(out, pitch, frame.image.data, frame.image.pitch, pitch, rows,
                              cudaMemcpyDeviceToDevice, stream);
        const cudaError_t chroma_copy =
            cudaMemcpy2DAsync(out + pitch * rows, chroma_row_bytes(width), frame.image.chroma,
                              frame.image.chroma_pitch, chroma_row_bytes(width), (rows + 1) / 2,
                              cudaMemcpyDeviceToDevice, stream);
        if (luma_copy != cudaSuccess || chroma_copy != cudaSuccess) {
          throw std::runtime_error(
              cudaGetErrorString(luma_copy != cudaSuccess ? luma_copy : chroma_copy));
        }
      } else {
        convert_from_nv12(frame.image, format, out, pitch, width, height, stream);
      }
    } catch (const std::exception& error) {
      return holoscan::make_unexpected(failure(error.what()));
    }
    if (auto committed = std::move(*writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    holoscan::schema::ImageT descriptor;
    descriptor.width = width;
    descriptor.height = height;
    descriptor.encoding = encoding_of(format);
    descriptor.header = source.header ? std::make_shared<holoscan::schema::HeaderT>(*source.header)
                                      : std::make_shared<holoscan::schema::HeaderT>();
    descriptor.header->capture_timestamp_ns = source.pts;
    if (nv12) {
      descriptor.color_space = holoscan::schema::ColorSpace_REC709;
      const std::uint64_t luma = static_cast<std::uint64_t>(width) * height;
      const auto chroma_rows = static_cast<std::uint64_t>((height + 1) / 2);
      descriptor.plane_layouts = {
          holoscan::schema::ImagePlaneLayout(0, static_cast<std::uint32_t>(width), luma),
          holoscan::schema::ImagePlaneLayout(luma,
                                             static_cast<std::uint32_t>(chroma_row_bytes(width)),
                                             chroma_row_bytes(width) * chroma_rows)};
    }
    ++state.decoded;
    if (auto sent = image.emit_tensor(std::move(*loan), descriptor,
                                      holoscan::sensor_io::derive_from(source.metadata));
        !sent) {
      return sent;
    }
  }
  return {};
}

}  // namespace holoscan::codec
