// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/isp/isp_op.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/isp/isp_pipeline.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/provenance.hpp>

namespace holoscan::isp {
namespace {

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr DLDataType kUInt16{kDLUInt, 16U, 1U};
constexpr DLDataType kUInt32{kDLUInt, 32U, 1U};
constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "IspOp: " + message};
}

holoscan::Error failure(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "IspOp: " + message};
}

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return failure(std::string(what) + ": " + cudaGetErrorString(status));
}

bool same_type(const DLDataType& a, const DLDataType& b) {
  return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

DLDataType output_dtype(OutputFormat format) {
  return format == OutputFormat::kRgb16    ? kUInt16
         : format == OutputFormat::kRgb32f ? kFloat32
                                           : kUInt8;
}

holoscan::schema::ImageEncoding output_encoding(OutputFormat format) {
  switch (format) {
    case OutputFormat::kRgba8:
      return holoscan::schema::ImageEncoding_RGBA8;
    case OutputFormat::kRgb16:
      return holoscan::schema::ImageEncoding_RGB16;
    case OutputFormat::kRgb32f:
      return holoscan::schema::ImageEncoding_RGB32F;
    case OutputFormat::kRgb8:
      break;
  }
  return holoscan::schema::ImageEncoding_RGB8;
}

std::optional<BayerPattern> encoding_pattern(holoscan::schema::ImageEncoding encoding) {
  switch (encoding) {
    case holoscan::schema::ImageEncoding_BAYER_RGGB:
      return BayerPattern::kRGGB;
    case holoscan::schema::ImageEncoding_BAYER_BGGR:
      return BayerPattern::kBGGR;
    case holoscan::schema::ImageEncoding_BAYER_GRBG:
      return BayerPattern::kGRBG;
    case holoscan::schema::ImageEncoding_BAYER_GBRG:
      return BayerPattern::kGBRG;
    default:
      return std::nullopt;
  }
}

holoscan::TensorOutputSpec tensor_spec(holoscan::MemoryKind kind, DLDataType dtype,
                                       std::uint32_t rank, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = kind,
              .dtype = dtype,
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

bool on_gpu(const holoscan::Tensor& tensor) {
  const DLDevice device = tensor.device();
  return device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
}

// A raw frame: its view (data still where the tensor is), and the bytes its rows span.
struct RawFrame {
  RawImageView view;
  std::size_t span{};
};

holoscan::expected<RawFrame, holoscan::Error> describe(const holoscan::schema::ImageT& image,
                                                       const IspOpConfig& config) {
  const holoscan::Tensor& tensor = *image.data;
  const DLDevice device = tensor.device();
  if (!on_gpu(tensor) && device.device_type != kDLCPU && device.device_type != kDLCUDAHost) {
    return holoscan::make_unexpected(invalid("raw images must be in host or CUDA memory"));
  }
  std::optional<BayerPattern> pattern = encoding_pattern(image.encoding);
  const bool mono = image.encoding == holoscan::schema::ImageEncoding_MONO8 ||
                    image.encoding == holoscan::schema::ImageEncoding_MONO16;
  if (!pattern && !mono) {
    return holoscan::make_unexpected(
        invalid(std::string("unsupported raw encoding ") +
                holoscan::schema::EnumNameImageEncoding(image.encoding) +
                "; use BAYER_RGGB, BAYER_BGGR, BAYER_GRBG, BAYER_GBRG, MONO8 or MONO16"));
  }
  if (mono && !config.pattern) {
    return holoscan::make_unexpected(
        invalid("a MONO8 or MONO16 raw image needs the Bayer pattern in the configuration"));
  }
  if (pattern && config.pattern && *pattern != *config.pattern) {
    return holoscan::make_unexpected(
        invalid(std::string("the image's pattern ") + std::string(pattern_name(*pattern)) +
                " disagrees with the configured " + std::string(pattern_name(*config.pattern))));
  }
  RawFrame frame;
  RawImageView& view = frame.view;
  view.pattern = pattern ? *pattern : *config.pattern;
  view.data = tensor.data();

  const DLDataType dtype = tensor.dtype();
  const std::span<const std::int64_t> shape = tensor.shape_span();
  const std::vector<std::int64_t> strides = tensor.strides();
  int container_bits = 0;
  std::int64_t row_bytes = 0;
  if (config.packing == Packing::kNone) {
    const bool u8 = same_type(dtype, kUInt8);
    if (!(u8 || same_type(dtype, kUInt16))) {
      return holoscan::make_unexpected(invalid("raw samples must be uint8 or uint16"));
    }
    if ((image.encoding == holoscan::schema::ImageEncoding_MONO8 && !u8) ||
        (image.encoding == holoscan::schema::ImageEncoding_MONO16 && u8)) {
      return holoscan::make_unexpected(invalid("raw element type disagrees with its encoding"));
    }
    if (!(shape.size() == 2 || (shape.size() == 3 && shape[2] == 1))) {
      return holoscan::make_unexpected(invalid("raw tensor must be [H, W] or [H, W, 1]"));
    }
    view.height = static_cast<int>(shape[0]);
    view.width = static_cast<int>(shape[1]);
    view.format = u8 ? RawFormat::kUint8 : RawFormat::kUint16;
    container_bits = u8 ? 8 : 16;
    const std::int64_t element = u8 ? 1 : 2;
    if (view.width > 1 && strides[1] != element) {
      return holoscan::make_unexpected(invalid("raw samples within a row must be packed"));
    }
    row_bytes = element * view.width;
    if ((image.width != 0 && image.width != view.width) ||
        (image.height != 0 && image.height != view.height)) {
      return holoscan::make_unexpected(invalid("the descriptor's size disagrees with its tensor"));
    }
  } else {
    if (!same_type(dtype, kUInt8) || shape.size() != 2) {
      return holoscan::make_unexpected(invalid("packed raw images must be uint8 [H, row bytes]"));
    }
    if (shape[1] > 1 && strides[1] != 1) {
      return holoscan::make_unexpected(invalid("packed raw rows must be contiguous"));
    }
    view.height = static_cast<int>(shape[0]);
    view.width = image.width;
    view.format = config.packing == Packing::kMipiRaw10 ? RawFormat::kRaw10 : RawFormat::kRaw12;
    container_bits = config.packing == Packing::kMipiRaw10 ? 10 : 12;
    row_bytes = shape[1];
    if (image.height != 0 && image.height != view.height) {
      return holoscan::make_unexpected(
          invalid("the descriptor's height disagrees with its tensor"));
    }
    if (static_cast<std::size_t>(row_bytes) < raw_row_bytes(view.format, view.width)) {
      return holoscan::make_unexpected(invalid("packed rows of " + std::to_string(row_bytes) +
                                               " bytes cannot hold " + std::to_string(view.width) +
                                               " samples"));
    }
  }
  if (view.width < 2 || view.height < 2 || view.width > config.max_width ||
      view.height > config.max_height) {
    return holoscan::make_unexpected(
        invalid("raw image " + std::to_string(view.width) + "x" + std::to_string(view.height) +
                " is outside 2x2 to the configured " + std::to_string(config.max_width) + "x" +
                std::to_string(config.max_height)));
  }
  const std::int64_t pitch = view.height > 1 ? strides[0] : row_bytes;
  if (pitch < row_bytes) {
    return holoscan::make_unexpected(invalid("raw rows overlap"));
  }
  view.pitch = static_cast<std::size_t>(pitch);
  if (view.format == RawFormat::kUint16 &&
      (reinterpret_cast<std::uintptr_t>(view.data) % 2U != 0U || view.pitch % 2U != 0U)) {
    return holoscan::make_unexpected(invalid("uint16 raw data must be aligned to 2 bytes"));
  }
  const bool known_bits = image.significant_bits != 0U &&
                          image.significant_bits != holoscan::schema::kUnknownImageSignificantBits;
  view.bit_depth = config.bit_depth > 0 ? config.bit_depth
                   : known_bits && config.packing == Packing::kNone
                       ? static_cast<int>(image.significant_bits)
                       : container_bits;
  frame.span = view.pitch * static_cast<std::size_t>(view.height - 1) +
               raw_row_bytes(view.format, view.width);
  return frame;
}

// Says, once, that every raw frame crosses between host and device memory.
void log_copy(Backend backend, bool to_device, std::size_t bytes) {
  HOLOSCAN_LOG_INFO(
      "IspOp ({} backend): the raw images are in {} memory and the backend reads {} memory, so "
      "each frame is copied {}, {} bytes ({:.1f} MB) per frame; to avoid the copy, {}",
      backend_name(backend), to_device ? "host" : "CUDA device", to_device ? "CUDA device" : "host",
      to_device ? "host to device" : "device to host", bytes, static_cast<double>(bytes) / 1e6,
      to_device ? "send CUDA device images or use the cpu backend"
                : "use the cuda backend or send host images");
}

// Keeps `tensor`'s memory from reuse until the work queued on `stream` so far is done.
holoscan::expected<void, holoscan::Error> hold_until_done(const holoscan::Tensor& tensor,
                                                          cudaStream_t stream) {
  holoscan::Tensor handle = tensor;
  if (handle.set_deallocation_stream(stream)) {
    return {};
  }
  if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
  }
  return {};
}

}  // namespace

void validate_config(const IspOpConfig& config) {
  validate_params(config.params);
  if (config.max_width < 2 || config.max_height < 2 || config.max_width > 16384 ||
      config.max_height > 16384) {
    throw std::invalid_argument("max_width and max_height must be 2..16384");
  }
  if (config.bit_depth < 0 || config.bit_depth > 16) {
    throw std::invalid_argument("bit_depth must be 0 (from the image) or 1..16");
  }
  if ((config.packing == Packing::kMipiRaw10 && config.bit_depth != 0 && config.bit_depth != 10) ||
      (config.packing == Packing::kMipiRaw12 && config.bit_depth != 0 && config.bit_depth != 12)) {
    throw std::invalid_argument("packed RAW10 and RAW12 samples have 10 and 12 bits");
  }
}

struct IspOp::State {
  std::unique_ptr<IspPipeline> pipeline;
  std::string start_error;
  // CUDA: a device copy of host input, grown as needed.
  void* staging{};
  std::size_t staging_bytes{};
  // CPU: a host copy of device input, and the stream that copies it.
  std::vector<unsigned char> host_copy;
  cudaStream_t copy_stream{};
  bool copy_logged{};

  void release() noexcept {
    pipeline.reset();
    if (copy_stream != nullptr) {
      cudaStreamSynchronize(copy_stream);
      cudaStreamDestroy(copy_stream);
      copy_stream = nullptr;
    }
    // cudaFree waits for the work on the runtime's streams that may still read it.
    if (staging != nullptr) {
      cudaFree(staging);
    }
    staging = nullptr;
    staging_bytes = 0;
  }
};

IspOp::IspOp(IspOpConfig config)
    : config_(std::move(config)), backend_(Backend::kCuda), state_(std::make_unique<State>()) {
  validate_config(config_);
  backend_ = resolve_backend(config_.backend);
}

IspOp::~IspOp() { state_->release(); }

void IspOp::setup(holoscan::OperatorSpec& spec) {
  const holoscan::MemoryKind kind =
      backend_ == Backend::kCuda ? holoscan::MemoryKind::kCudaDevice : holoscan::MemoryKind::kHost;
  const auto pixels = static_cast<std::uint64_t>(config_.max_width) *
                      static_cast<std::uint64_t>(config_.max_height);
  const auto pixel_bytes = static_cast<std::uint64_t>(output_channels(config_.output_format) *
                                                      output_channel_bytes(config_.output_format));
  // Two slots: one frame in compute() while the next one arrives.
  spec.input(raw, "raw").queue_depth(2U);
  spec.output(image, "image")
      .produces_tensor(
          tensor_spec(kind, output_dtype(config_.output_format), 3U, pixels * pixel_bytes))
      .payload_options(image_payload());
  // Optional ports exist only when configured: an emit into an unconnected output fails.
  if (config_.stats) {
    spec.output(histogram, "histogram")
        .produces_tensor(tensor_spec(kind, kUInt32, 2U,
                                     sizeof(std::uint32_t) * kHistogramChannels * kHistogramBins));
    spec.output(stats, "stats")
        .produces_tensor(tensor_spec(kind, kFloat32, 1U, sizeof(float) * stats::kStatsSize));
  }
}

holoscan::Contract IspOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{raw});
  return result;
}

void IspOp::start() {
  if (config_.backend == Backend::kAuto) {
    HOLOSCAN_LOG_INFO(
        "IspOp: backend auto chose {}: {}", backend_name(backend_),
        backend_ == Backend::kCuda ? "a CUDA device is usable" : "no CUDA device is usable");
  }
  state_->start_error.clear();
  state_->copy_logged = false;
  try {
    state_->pipeline = std::make_unique<IspPipeline>(backend_, config_.params);
  } catch (const std::exception& error) {
    // Reported by compute(), which can fail the frame.
    state_->start_error = error.what();
  }
}

void IspOp::stop() { state_->release(); }

holoscan::expected<void, holoscan::Error> IspOp::compute(holoscan::ExecutionContext& context) {
  State& state = *state_;
  if (!state.pipeline) {
    return holoscan::make_unexpected(failure("the pipeline could not start: " + state.start_error));
  }
  auto sample = raw.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  const holoscan::schema::ImageT& input = sample->data;
  if (!input.data) {
    return holoscan::make_unexpected(invalid("raw image has no tensor"));
  }
  auto frame = describe(input, config_);
  if (!frame) {
    return holoscan::make_unexpected(std::move(frame).error());
  }
  const holoscan::Tensor& tensor = *input.data;
  const bool cuda = backend_ == Backend::kCuda;
  cudaStream_t stream = context.cuda_stream().get();

  // The raw data where the backend reads it.
  if (cuda) {
    if (stream == nullptr) {
      // Not reached in a compiled graph: the outputs are in device memory, so
      // the runtime always assigns a stream.
      return holoscan::make_unexpected(failure("the runtime gave no CUDA stream"));
    }
    if (!on_gpu(tensor)) {
      if (tensor.device().device_type == kDLCPU) {
        // The runtime orders device and stream-ready pinned memory on our
        // stream; plain host memory it does not, so wait for its producer.
        if (auto ready = tensor.synchronize_for_host(); !ready) {
          return holoscan::make_unexpected(std::move(ready).error());
        }
      }
      if (state.staging_bytes < frame->span) {
        cudaStreamSynchronize(stream);  // The old buffer may still be read.
        cudaFree(state.staging);
        state.staging = nullptr;
        state.staging_bytes = 0;
        if (const cudaError_t status = cudaMalloc(&state.staging, frame->span);
            status != cudaSuccess) {
          state.staging = nullptr;
          return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
        }
        state.staging_bytes = frame->span;
      }
      if (const cudaError_t status = cudaMemcpyAsync(state.staging, frame->view.data, frame->span,
                                                     cudaMemcpyHostToDevice, stream);
          status != cudaSuccess) {
        return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", status));
      }
      // A producer must not reuse pinned memory while the copy still reads it;
      // pageable memory is staged by the time cudaMemcpyAsync returns.
      if (tensor.device().device_type == kDLCUDAHost) {
        if (auto held = hold_until_done(tensor, stream); !held) {
          return holoscan::make_unexpected(std::move(held).error());
        }
      }
      if (!state.copy_logged) {
        state.copy_logged = true;
        log_copy(backend_, true, frame->span);
      }
      frame->view.data = state.staging;
    }
  } else if (on_gpu(tensor)) {
    if (auto ready = tensor.synchronize_for_host(); !ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
    if (state.copy_stream == nullptr) {
      if (const cudaError_t status =
              cudaStreamCreateWithFlags(&state.copy_stream, cudaStreamNonBlocking);
          status != cudaSuccess) {
        state.copy_stream = nullptr;
        return holoscan::make_unexpected(cuda_failure("cudaStreamCreate", status));
      }
    }
    state.host_copy.resize(frame->span);
    if (const cudaError_t status =
            cudaMemcpyAsync(state.host_copy.data(), frame->view.data, frame->span,
                            cudaMemcpyDeviceToHost, state.copy_stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", status));
    }
    if (const cudaError_t status = cudaStreamSynchronize(state.copy_stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
    }
    if (!state.copy_logged) {
      state.copy_logged = true;
      log_copy(backend_, false, frame->span);
    }
    frame->view.data = state.host_copy.data();
  } else if (auto ready = tensor.synchronize_for_host(); !ready) {
    return holoscan::make_unexpected(std::move(ready).error());
  }

  // The outputs.
  const int width = frame->view.width;
  const int height = frame->view.height;
  const int channels = output_channels(config_.output_format);
  const auto begin_write = [&](holoscan::TensorOutputLoan& loan) {
    return cuda ? loan.write(stream) : loan.write_host();
  };
  auto image_loan = image.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = std::array<std::int64_t, 3>{height, width, channels},
                                  .dtype = output_dtype(config_.output_format)});
  if (!image_loan) {
    return holoscan::make_unexpected(std::move(image_loan).error());
  }
  auto image_writer = begin_write(*image_loan);
  if (!image_writer) {
    return holoscan::make_unexpected(std::move(image_writer).error());
  }
  std::optional<holoscan::TensorOutputLoan> histogram_loan;
  std::optional<holoscan::TensorOutputLoan> stats_loan;
  std::optional<holoscan::TensorOutputWriteGuard> histogram_writer;
  std::optional<holoscan::TensorOutputWriteGuard> stats_writer;
  if (config_.stats) {
    auto h = histogram.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = std::array<std::int64_t, 2>{kHistogramChannels, kHistogramBins},
        .dtype = kUInt32});
    if (!h) {
      return holoscan::make_unexpected(std::move(h).error());
    }
    histogram_loan.emplace(std::move(*h));
    auto s = stats.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = std::array<std::int64_t, 1>{stats::kStatsSize}, .dtype = kFloat32});
    if (!s) {
      return holoscan::make_unexpected(std::move(s).error());
    }
    stats_loan.emplace(std::move(*s));
    auto hw = begin_write(*histogram_loan);
    if (!hw) {
      return holoscan::make_unexpected(std::move(hw).error());
    }
    histogram_writer.emplace(std::move(*hw));
    auto sw = begin_write(*stats_loan);
    if (!sw) {
      return holoscan::make_unexpected(std::move(sw).error());
    }
    stats_writer.emplace(std::move(*sw));
  }

  const RgbImageView out{
      .data = image_writer->data_as<void>(),
      .pitch = static_cast<std::size_t>(width) * channels *
               static_cast<std::size_t>(output_channel_bytes(config_.output_format)),
      .format = config_.output_format};
  const StatsOutput stats_out{
      .histogram = histogram_writer ? histogram_writer->data_as<std::uint32_t>() : nullptr,
      .stats = stats_writer ? stats_writer->data_as<float>() : nullptr};
  try {
    state.pipeline->process(
        frame->view, out, stats_out,
        FrameExposure{.exposure_ns = input.exposure_time_ns, .gain = input.gain}, stream);
  } catch (const std::invalid_argument& error) {
    return holoscan::make_unexpected(invalid(error.what()));
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(failure(error.what()));
  }

  holoscan::schema::ImageT descriptor;
  descriptor.width = width;
  descriptor.height = height;
  descriptor.encoding = output_encoding(config_.output_format);
  if (input.header) {
    descriptor.header = std::make_shared<holoscan::schema::HeaderT>(*input.header);
  }
  descriptor.exposure_time_ns = input.exposure_time_ns;
  descriptor.gain = input.gain;
  const holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(*sample);
  if (auto committed = std::move(*image_writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  if (auto sent = image.emit_tensor(std::move(*image_loan), std::move(descriptor), emit); !sent) {
    return sent;
  }
  if (config_.stats) {
    if (auto committed = std::move(*histogram_writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    if (auto committed = std::move(*stats_writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    if (auto sent = histogram.emit(std::move(*histogram_loan), emit); !sent) {
      return sent;
    }
    return stats.emit(std::move(*stats_loan), emit);
  }
  return {};
}

}  // namespace holoscan::isp
