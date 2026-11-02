// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/image_quality/image_quality_op.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/image_quality/analyzer.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/provenance.hpp>

namespace holoscan::image_quality {
namespace {

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kUInt32{kDLUInt, 32U, 1U};
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

holoscan::Error invalid(std::string message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                         "ImageQualityOp: " + std::move(message)};
}

holoscan::TensorOutputSpec host_tensor(DLDataType dtype, std::uint32_t rank, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kHost,
              .dtype = dtype,
              .rank = rank,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(bytes),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

std::optional<PixelFormat> pixel_format(holoscan::schema::ImageEncoding encoding) {
  switch (encoding) {
    case holoscan::schema::ImageEncoding_MONO8:
      return PixelFormat::kMono8;
    case holoscan::schema::ImageEncoding_RGB8:
      return PixelFormat::kRgb8;
    case holoscan::schema::ImageEncoding_BGR8:
      return PixelFormat::kBgr8;
    case holoscan::schema::ImageEncoding_RGBA8:
      return PixelFormat::kRgba8;
    case holoscan::schema::ImageEncoding_BGRA8:
      return PixelFormat::kBgra8;
    default:
      return std::nullopt;
  }
}

struct Input {
  ImageView view;
  bool on_device{};
};

// The image's pixels, after checking that the tensor matches its encoding.
holoscan::expected<Input, holoscan::Error> view_image(const holoscan::schema::ImageT& image,
                                                      const holoscan::Tensor& tensor) {
  const auto format = pixel_format(image.encoding);
  if (!format) {
    return holoscan::make_unexpected(
        invalid(std::string("unsupported image encoding ") +
                holoscan::schema::EnumNameImageEncoding(image.encoding) +
                "; use MONO8, RGB8, BGR8, RGBA8 or BGRA8"));
  }
  const DLDataType dtype = tensor.dtype();
  if (dtype.code != kDLUInt || dtype.bits != 8 || dtype.lanes != 1) {
    return holoscan::make_unexpected(invalid("image tensors must hold uint8 elements"));
  }
  const std::span<const std::int64_t> shape = tensor.shape_span();
  const int ch = channels(*format);
  const bool hwc = shape.size() == 3 && shape[2] == ch;
  const bool hw = shape.size() == 2 && ch == 1;
  if (!hwc && !hw) {
    return holoscan::make_unexpected(
        invalid("image tensor shape must be [height, width, channels] matching its encoding"));
  }
  if (shape[0] < 1 || shape[1] < 1) {
    return holoscan::make_unexpected(invalid("the image is empty"));
  }
  const std::vector<std::int64_t> strides = tensor.strides();
  if (strides[1] != ch || (hwc && strides[2] != 1) || strides[0] < shape[1] * ch) {
    return holoscan::make_unexpected(invalid("image pixels must be packed within each row"));
  }
  if ((image.width != 0 && image.width != shape[1]) ||
      (image.height != 0 && image.height != shape[0])) {
    return holoscan::make_unexpected(invalid("image width and height disagree with its tensor"));
  }
  const DLDevice device = tensor.device();
  const bool on_device = device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
  if (!on_device && device.device_type != kDLCPU && device.device_type != kDLCUDAHost) {
    return holoscan::make_unexpected(invalid("image tensors must be in host or CUDA memory"));
  }
  return Input{ImageView{tensor.data_as<std::uint8_t>(), static_cast<int>(shape[1]),
                         static_cast<int>(shape[0]), static_cast<std::size_t>(strides[0]), *format},
               on_device};
}

}  // namespace

void validate_config(const ImageQualityConfig& config) {
  validate_stats_options(config.stats);
  validate_thresholds(config.thresholds);
  if (config.overlay_cell < 1 || config.overlay_cell > 64) {
    throw std::invalid_argument("overlay_cell must be 1..64");
  }
  if (config.log_interval.count() < 0) {
    throw std::invalid_argument("log_interval must not be negative");
  }
}

void draw_overlay(const QualityReport& report, int cell, std::uint8_t* rgba) {
  const int width = report.tiles_x * cell;
  for (int ty = 0; ty < report.tiles_y; ++ty) {
    for (int tx = 0; tx < report.tiles_x; ++tx) {
      const std::uint32_t tile_flags = report.tile_flags(tx, ty);
      std::uint32_t color = 0U;
      if ((tile_flags & kTileObstructed) != 0U) {
        color = kOverlayObstructed;
      } else if ((tile_flags & kTileBlurred) != 0U) {
        color = kOverlayBlurred;
      } else if ((tile_flags & kTileLowTexture) != 0U) {
        color = kOverlayLowTexture;
      }
      const std::array<std::uint8_t, 4> bytes{
          static_cast<std::uint8_t>(color), static_cast<std::uint8_t>(color >> 8U),
          static_cast<std::uint8_t>(color >> 16U), static_cast<std::uint8_t>(color >> 24U)};
      for (int y = 0; y < cell; ++y) {
        std::uint8_t* row =
            rgba + (static_cast<std::size_t>(ty * cell + y) * width + tx * cell) * 4U;
        for (int x = 0; x < cell; ++x) {
          const bool gap = cell > 2 && (x == cell - 1 || y == cell - 1);
          std::memcpy(row + 4U * x, gap ? std::array<std::uint8_t, 4>{}.data() : bytes.data(), 4U);
        }
      }
    }
  }
}

struct ImageQualityOp::State {
  struct StreamDestroy {
    void operator()(cudaStream_t stream) const noexcept {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  };

  std::unique_ptr<ImageQualityAnalyzer> analyzer;
  // For activations without a runtime CUDA stream; made on first use.
  std::unique_ptr<CUstream_st, StreamDestroy> own_stream;
  bool copy_logged{};
  // Failed frames, and the union of their flags, since the last warning.
  std::uint64_t frames{};
  std::uint64_t failed{};
  std::uint32_t failed_flags{};
  std::optional<std::chrono::steady_clock::time_point> last_warning;
};

ImageQualityOp::ImageQualityOp(ImageQualityConfig config)
    : config_(std::move(config)), state_(std::make_unique<State>()) {
  validate_config(config_);
}

ImageQualityOp::~ImageQualityOp() = default;

void ImageQualityOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(image, "image").queue_depth(2U);
  spec.output(metrics, "metrics")
      .produces_tensor(host_tensor(kFloat32, 1U, kMetricCount * sizeof(float)));
  spec.output(flags, "flags").produces_tensor(host_tensor(kUInt32, 1U, sizeof(std::uint32_t)));
  // Optional ports exist only when configured: an emit into an unconnected output fails.
  const auto tile_count = static_cast<std::uint64_t>(config_.stats.tiles_x) *
                          static_cast<std::uint64_t>(config_.stats.tiles_y);
  if (config_.tiles_output) {
    spec.output(tiles, "tiles")
        .produces_tensor(host_tensor(kFloat32, 3U, tile_count * kTileChannelCount * sizeof(float)));
  }
  if (config_.overlay_output) {
    const auto cell = static_cast<std::uint64_t>(config_.overlay_cell);
    spec.output(overlay, "overlay")
        .produces_tensor(host_tensor(kUInt8, 3U, tile_count * cell * cell * 4U))
        .payload_options(holoscan::PayloadOptions{
            .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
        });
  }
}

holoscan::Contract ImageQualityOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{image});
  return result;
}

void ImageQualityOp::start() {
  State& state = *state_;
  // The analyzer allocates on the device this thread uses, so it is made here,
  // not in the constructor.
  state.analyzer = std::make_unique<ImageQualityAnalyzer>(AnalyzerOptions{.stats = config_.stats},
                                                          config_.thresholds);
  state.frames = state.failed = 0;
  state.failed_flags = 0;
  state.last_warning.reset();
}

void ImageQualityOp::stop() {
  state_->analyzer.reset();
  state_->own_stream.reset();
}

holoscan::expected<void, holoscan::Error> ImageQualityOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = image.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  if (!sample->data.data) {
    return holoscan::make_unexpected(invalid("image has no tensor"));
  }
  const holoscan::Tensor& tensor = *sample->data.data;
  auto input = view_image(sample->data, tensor);
  if (!input) {
    return holoscan::make_unexpected(std::move(input).error());
  }
  State& state = *state_;
  if (!state.analyzer) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, "ImageQualityOp: compute before start"});
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    // Host outputs only, so the runtime may give no stream: use our own, not the
    // legacy default stream, which serializes with every other stream.
    if (!state.own_stream) {
      cudaStream_t created = nullptr;
      const cudaError_t status = cudaStreamCreateWithFlags(&created, cudaStreamNonBlocking);
      if (status != cudaSuccess) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure,
            std::string("ImageQualityOp: cudaStreamCreate: ") + cudaGetErrorString(status)});
      }
      state.own_stream.reset(created);
    }
    stream = state.own_stream.get();
  }
  // Wait for the producer's writes: on our stream for device memory, on the host otherwise.
  auto ready = input->on_device && tensor.has_readiness_stream_contract()
                   ? tensor.synchronize_for_consumer_stream(stream)
                   : tensor.synchronize_for_host();
  if (!ready) {
    return holoscan::make_unexpected(std::move(ready).error());
  }
  if (!input->on_device && !state.copy_logged) {
    state.copy_logged = true;
    const std::size_t bytes = static_cast<std::size_t>(input->view.width) * input->view.height *
                              channels(input->view.format);
    HOLOSCAN_LOG_INFO(
        "ImageQualityOp: the images are in host memory and the checks run on the GPU, so each "
        "frame is copied host to device, {} bytes ({:.1f} MB) per frame; send CUDA device "
        "images to avoid the copy",
        bytes, static_cast<double>(bytes) / 1e6);
  }

  // The analyzer returns once the GPU is done, so nothing reads the image afterwards.
  QualityReport report;
  try {
    report = state.analyzer->analyze(input->view, input->on_device, stream);
  } catch (const std::exception& error) {
    cudaStreamSynchronize(stream);
    const bool bad_input = dynamic_cast<const std::invalid_argument*>(&error) != nullptr;
    return holoscan::make_unexpected(holoscan::Error{
        bad_input ? holoscan::ErrorCode::kInvalidArgument : holoscan::ErrorCode::kFailure,
        std::string("ImageQualityOp: ") + error.what()});
  }

  // The frame's capture time, so its outputs align with the image downstream;
  // an unstamped image's outputs are stamped now.
  holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(*sample);
  if (!sample->metadata.source_clock_id.valid()) {
    emit.capture_time = context.activation_time();
  }

  // Allocates a host tensor on `output` and copies `bytes` into it.
  const auto host_loan =
      [](auto& output, holoscan::TensorShapeView shape, DLDataType dtype,
         const void* bytes) -> holoscan::expected<holoscan::TensorOutputLoan, holoscan::Error> {
    auto loan = output.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                   -> holoscan::expected<void, holoscan::Error> {
      std::memcpy(writer.data_as<std::uint8_t>(), bytes, writer.byte_size());
      return {};
    });
    if (!written) {
      return holoscan::make_unexpected(std::move(written).error());
    }
    return std::move(*loan);
  };

  const std::array<std::int64_t, 1> metric_shape{static_cast<std::int64_t>(kMetricCount)};
  auto metric_loan = host_loan(metrics, metric_shape, kFloat32, report.metrics.data());
  if (!metric_loan) {
    return holoscan::make_unexpected(std::move(metric_loan).error());
  }
  if (auto sent = metrics.emit(std::move(*metric_loan), emit); !sent) {
    return sent;
  }
  const std::array<std::int64_t, 1> flag_shape{1};
  auto flag_loan = host_loan(flags, flag_shape, kUInt32, &report.flags);
  if (!flag_loan) {
    return holoscan::make_unexpected(std::move(flag_loan).error());
  }
  if (auto sent = flags.emit(std::move(*flag_loan), emit); !sent) {
    return sent;
  }
  if (config_.tiles_output) {
    const std::array<std::int64_t, 3> tile_shape{report.tiles_y, report.tiles_x,
                                                 static_cast<std::int64_t>(kTileChannelCount)};
    auto tile_loan = host_loan(tiles, tile_shape, kFloat32, report.tiles.data());
    if (!tile_loan) {
      return holoscan::make_unexpected(std::move(tile_loan).error());
    }
    if (auto sent = tiles.emit(std::move(*tile_loan), emit); !sent) {
      return sent;
    }
  }
  if (config_.overlay_output) {
    const int cell = config_.overlay_cell;
    const int width = report.tiles_x * cell;
    const int height = report.tiles_y * cell;
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * 4U);
    draw_overlay(report, cell, pixels.data());
    const std::array<std::int64_t, 3> overlay_shape{height, width, 4};
    auto overlay_loan = host_loan(overlay, overlay_shape, kUInt8, pixels.data());
    if (!overlay_loan) {
      return holoscan::make_unexpected(std::move(overlay_loan).error());
    }
    holoscan::schema::ImageT descriptor;
    descriptor.width = width;
    descriptor.height = height;
    descriptor.encoding = holoscan::schema::ImageEncoding_RGBA8;
    if (auto sent = overlay.emit_tensor(std::move(*overlay_loan), descriptor, emit); !sent) {
      return sent;
    }
  }

  // At most one warning per interval about the frames that failed since the last.
  ++state.frames;
  if (report.flags != 0U && config_.log_interval.count() > 0) {
    ++state.failed;
    state.failed_flags |= report.flags;
    const auto now = std::chrono::steady_clock::now();
    if (!state.last_warning || now - *state.last_warning >= config_.log_interval) {
      HOLOSCAN_LOG_WARN("ImageQualityOp: {} of {} frame(s) failed since the last warning: {}",
                        state.failed, state.frames, flag_names(state.failed_flags));
      state.last_warning = now;
      state.frames = state.failed = 0;
      state.failed_flags = 0;
    }
  }
  return {};
}

}  // namespace holoscan::image_quality
