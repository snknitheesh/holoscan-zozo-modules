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
#include <map>
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
// Images one compute() may emit, which sizes the output's pool. Low latency: one, or a few while
// Jetson Orin's decoder catches up. Otherwise also every frame a stream holds back for
// reordering, handed out together at its end: at most 16 in H.264 and H.265, and the access
// unit's own. More are held (see Held) and emitted first in the next compute().
constexpr std::uint32_t kMaxImagesLowLatency = 4;
constexpr std::uint32_t kMaxImagesReordering = 17;
// Access units whose image has not come out yet; beyond any stream's reordering, so one dropped
// for room gave no image.
constexpr std::size_t kMaxPending = 64;

std::uint32_t max_images(const VideoDecoderConfig& config) {
  if (config.max_images_per_compute != 0) {
    return config.max_images_per_compute;
  }
  return config.low_latency ? kMaxImagesLowLatency : kMaxImagesReordering;
}

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

// What an access unit said about its frame, until the frame comes out.
struct Pending {
  std::shared_ptr<holoscan::schema::HeaderT> header;
  int width{};
  int height{};
  holoscan::SampleMetadata metadata;
};

// Converts `frame` (or copies it, for NV12) into a new output image on `stream`, with `source`'s
// header and provenance, and emits it.
holoscan::expected<void, holoscan::Error> emit_image(
    holoscan::Output<holoscan::schema::ImageT>& image, PixelFormat format,
    const DecodedFrame& frame, const Pending& source, cudaStream_t stream) {
  const int width = std::min(source.width, frame.image.width);
  const int height = std::min(source.height, frame.image.height);
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
      const cudaError_t chroma_copy = cudaMemcpy2DAsync(
          out + pitch * rows, chroma_row_bytes(width), frame.image.chroma, frame.image.chroma_pitch,
          chroma_row_bytes(width), (rows + 1) / 2, cudaMemcpyDeviceToDevice, stream);
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
  // The encoded image's header, unchanged: its capture time is the one the image came with, never
  // the pts, which the encoder may have made up from the frame's index.
  descriptor.header = source.header ? std::make_shared<holoscan::schema::HeaderT>(*source.header)
                                    : std::make_shared<holoscan::schema::HeaderT>();
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
  return image.emit_tensor(std::move(*loan), descriptor,
                           holoscan::sensor_io::derive_from(source.metadata));
}

// Frees device memory from cudaMallocPitch.
struct CudaFree {
  void operator()(std::uint8_t* memory) const noexcept { cudaFree(memory); }
};

// A decoded frame the output had no room for in its compute(): an NV12 copy, its chroma after its
// luma at the same pitch, and what its access unit said.
struct Held {
  std::unique_ptr<std::uint8_t, CudaFree> memory;
  std::size_t pitch{};
  int width{};
  int height{};
  Pending source;
};

// Carries an emit's error out through the decoder, which stops on an exception from its sink.
struct EmitFailed : std::runtime_error {
  explicit EmitFailed(holoscan::Error failed)
      : std::runtime_error(failed.message), error(std::move(failed)) {}
  holoscan::Error error;
};

}  // namespace

struct VideoDecoderOp::State {
  struct StreamDestroy {
    void operator()(cudaStream_t stream) const noexcept {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  };
  std::atomic<Backend> resolved{Backend::kAuto};
  std::unique_ptr<VideoDecoder> decoder;
  int width{};
  int height{};
  // By pts, which the decoder hands back with each frame: frames of a stream with B-frames come
  // out in another order than their access units went in. Equal pts go in arrival order.
  std::multimap<std::int64_t, Pending> pending;
  std::unique_ptr<CUstream_st, StreamDestroy> own_stream;
  // A copy of access units that arrive in device memory.
  std::vector<std::uint8_t> host_copy;
  bool empty_logged{};
  bool skip_logged{};
  bool device_input_logged{};
  std::uint64_t decoded{};
  std::uint64_t empty{};
  std::uint64_t ends{};
  // Frames waiting for room in the output, oldest first, and copies' memory to reuse.
  std::deque<Held> held;
  std::vector<Held> spare;
  bool held_logged{};
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
      .max_emits_per_compute(max_images(config_));
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
  state.ends = 0;
  state.held.clear();
  // The cold part of the setup, here rather than on the first access unit: loading the libraries,
  // opening the devices and choosing the backend. The hardware session itself needs the stream's
  // size, which only its first keyframe tells.
  std::string codecs;
  std::string reasons;
  Backend backend = Backend::kAuto;
  std::string detail;
  for (const Codec codec : {Codec::kH264, Codec::kH265}) {
    try {
      const VideoDecoder probe(codec, config_.backend, config_.low_latency);
      codecs += std::string(codecs.empty() ? "" : " and ") + std::string(codec_name(codec));
      backend = probe.backend();
      detail = probe.backend_detail();
    } catch (const std::exception& error) {
      reasons += std::string(reasons.empty() ? "" : "; ") + error.what();
    }
  }
  if (codecs.empty()) {
    throw std::runtime_error("VideoDecoderOp: " + reasons);
  }
  state.resolved = backend;
  HOLOSCAN_LOG_INFO(
      "VideoDecoderOp: {} on the {} backend ({}){}; the CPU waits for each decoded frame before "
      "the decoder reuses its memory",
      codecs, backend_name(backend), detail,
      config_.backend == Backend::kAuto ? ", which auto chose as the first backend that works here"
                                        : "");
}

void VideoDecoderOp::stop() {
  State& state = *state_;
  if (state.decoder || state.decoded > 0) {
    const std::uint64_t skipped = state.decoder ? state.decoder->skipped() : 0U;
    HOLOSCAN_LOG_INFO(
        "VideoDecoderOp: {} frames decoded, {} streams ended; {} access units before the first "
        "keyframe and {} empty ones gave none, {} were still in the decoder, and {} decoded "
        "frames were still waiting for room in the output",
        state.decoded, state.ends, skipped, state.empty, state.pending.size(), state.held.size());
  }
  state.decoder.reset();
  state.pending.clear();
  state.held.clear();
  state.spare.clear();
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
  const bool end = message.end_of_stream;
  if (size == 0 && !end) {
    ++state.empty;
    if (!state.empty_logged) {
      state.empty_logged = true;
      HOLOSCAN_LOG_WARN("VideoDecoderOp: an empty access unit gave no image");
    }
    return {};
  }
  if (size != 0 && (message.width < 1 || message.height < 1 || message.width > config_.max_width ||
                    message.height > config_.max_height)) {
    return holoscan::make_unexpected(
        invalid("frame size " + std::to_string(message.width) + "x" +
                std::to_string(message.height) + " is not within the configured " +
                std::to_string(config_.max_width) + "x" + std::to_string(config_.max_height)));
  }

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
  std::span<const std::uint8_t> access_unit;
  if (size != 0) {
    const holoscan::Tensor& tensor = *message.data;
    const DLDevice device = tensor.device();
    const bool on_device = device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
    if (auto ready = tensor.synchronize_for_host(); !ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
    access_unit = {static_cast<const std::uint8_t*>(tensor.data()), size};
    if (on_device) {
      if (!state.device_input_logged) {
        state.device_input_logged = true;
        HOLOSCAN_LOG_INFO(
            "VideoDecoderOp: the access units are in device memory, and the decoder reads them on "
            "the host, so each one is copied there and the CPU waits for the copy; send them in "
            "host memory to avoid it");
      }
      state.host_copy.resize(size);
      if (const cudaError_t status =
              cudaMemcpy(state.host_copy.data(), tensor.data(), size, cudaMemcpyDeviceToHost);
          status != cudaSuccess) {
        return holoscan::make_unexpected(failure(cudaGetErrorString(status)));
      }
      access_unit = state.host_copy;
    }
  }

  const std::uint32_t limit = max_images(config_);
  std::uint32_t emitted = 0;
  // Copies `frame` out of the decoder's memory, on `stream`, to emit later. The decoder waits for
  // the stream before it reuses its memory.
  const auto hold = [&](const DecodedFrame& frame, Pending source, cudaStream_t on) {
    if (!state.held_logged) {
      state.held_logged = true;
      HOLOSCAN_LOG_WARN(
          "VideoDecoderOp: the decoder handed out more than {} images in one compute(); the rest "
          "are copied and emitted first in the next ones",
          limit);
    }
    const int width = frame.image.width;
    const int height = frame.image.height;
    Held copy;
    const auto fits = std::find_if(state.spare.begin(), state.spare.end(), [&](const Held& h) {
      return h.width >= width && h.height >= height;
    });
    if (fits != state.spare.end()) {
      copy = std::move(*fits);
      state.spare.erase(fits);
    } else {
      void* memory = nullptr;
      std::size_t pitch = 0;
      const auto rows = static_cast<std::size_t>(height) + static_cast<std::size_t>(height + 1) / 2;
      if (const cudaError_t status =
              cudaMallocPitch(&memory, &pitch, static_cast<std::size_t>(width), rows);
          status != cudaSuccess) {
        throw std::runtime_error(std::string("holding a decoded frame: ") +
                                 cudaGetErrorString(status));
      }
      copy.memory.reset(static_cast<std::uint8_t*>(memory));
      copy.pitch = pitch;
    }
    copy.width = width;
    copy.height = height;
    copy.source = std::move(source);
    std::uint8_t* luma = copy.memory.get();
    const auto rows = static_cast<std::size_t>(height);
    const auto bytes = static_cast<std::size_t>(width);
    const cudaError_t luma_copy =
        cudaMemcpy2DAsync(luma, copy.pitch, frame.image.data, frame.image.pitch, bytes, rows,
                          cudaMemcpyDeviceToDevice, on);
    const cudaError_t chroma_copy = cudaMemcpy2DAsync(
        luma + copy.pitch * rows, copy.pitch, frame.image.chroma, frame.image.chroma_pitch,
        chroma_row_bytes(width), (rows + 1) / 2, cudaMemcpyDeviceToDevice, on);
    if (luma_copy != cudaSuccess || chroma_copy != cudaSuccess) {
      throw std::runtime_error(
          std::string("holding a decoded frame: ") +
          cudaGetErrorString(luma_copy != cudaSuccess ? luma_copy : chroma_copy));
    }
    state.held.push_back(std::move(copy));
  };
  // Emits held frames, oldest first, while the output has room.
  const auto emit_held = [&]() -> holoscan::expected<void, holoscan::Error> {
    bool any = false;
    while (!state.held.empty() && emitted < limit) {
      Held& next = state.held.front();
      const DecodedFrame frame{
          .image = DeviceImage{
              .data = next.memory.get(),
              .pitch = next.pitch,
              .chroma = next.memory.get() + next.pitch * static_cast<std::size_t>(next.height),
              .chroma_pitch = next.pitch,
              .width = next.width,
              .height = next.height,
              .format = PixelFormat::kNv12}};
      if (auto sent = emit_image(image, config_.output_format, frame, next.source, stream); !sent) {
        return sent;
      }
      ++emitted;
      ++state.decoded;
      any = true;
      state.spare.push_back(std::move(next));
      state.held.pop_front();
    }
    if (any) {
      // The copies are reused, possibly by a compute() on another stream: wait for the reads.
      if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
        return holoscan::make_unexpected(failure(cudaGetErrorString(status)));
      }
    }
    return {};
  };
  // Frames held back by an earlier compute() go first, in order.
  if (auto drained = emit_held(); !drained) {
    return drained;
  }

  // Each decoded frame goes straight from the decoder's memory into its output image; one the
  // output has no room for this time is copied out and held for the next compute().
  const DecodedFrameSink sink = [&](const DecodedFrame& frame) {
    const auto found = state.pending.find(frame.pts);
    if (found == state.pending.end()) {
      throw std::runtime_error("a decoded frame matches no access unit");
    }
    Pending source = std::move(found->second);
    state.pending.erase(found);
    if (emitted == limit || !state.held.empty()) {
      hold(frame, std::move(source), stream);
      return;
    }
    if (auto sent = emit_image(image, config_.output_format, frame, source, stream); !sent) {
      throw EmitFailed(std::move(sent).error());
    }
    ++emitted;
    ++state.decoded;
  };

  try {
    const Codec codec = from_schema(message.codec);
    if (size != 0) {
      const bool restart = !state.decoder || state.decoder->codec() != codec ||
                           message.width != state.width || message.height != state.height;
      if (restart) {
        if (state.decoder && !state.pending.empty()) {
          HOLOSCAN_LOG_WARN(
              "VideoDecoderOp: the stream changed to {} {}x{}; {} frames still in the decoder "
              "are dropped",
              codec_name(codec), message.width, message.height, state.pending.size());
        }
        state.decoder = std::make_unique<VideoDecoder>(codec, config_.backend, config_.low_latency);
        state.pending.clear();
        state.width = message.width;
        state.height = message.height;
      }
      const auto entry = state.pending.emplace(message.pts, Pending{.header = message.header,
                                                                    .width = message.width,
                                                                    .height = message.height,
                                                                    .metadata = sample->metadata});
      const std::uint64_t skipped_before = state.decoder->skipped();
      state.decoder->decode(access_unit, message.pts, stream, sink);
      if (state.decoder->skipped() != skipped_before) {
        state.pending.erase(entry);  // Not decodable: no frame will come for it.
        if (!state.skip_logged) {
          state.skip_logged = true;
          HOLOSCAN_LOG_WARN(
              "VideoDecoderOp: the stream does not start with a keyframe; its access units give "
              "no image until one comes");
        }
      }
    }
    if (end && state.decoder) {
      state.decoder->flush(stream, sink);
      state.pending.clear();
      ++state.ends;
    }
    // Room left after the decoding: frames it held, in order.
    if (auto drained = emit_held(); !drained) {
      return drained;
    }
    // An access unit whose frame never came (a damaged one, say) leaves its entry: keep the newest.
    while (state.pending.size() > kMaxPending) {
      state.pending.erase(state.pending.begin());
    }
  } catch (const EmitFailed& failed) {
    return holoscan::make_unexpected(failed.error);
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(failure(error.what()));
  }
  return {};
}

}  // namespace holoscan::codec
