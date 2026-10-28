// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/stereo_disparity/stereo_disparity_op.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/stereo_disparity/platform.hpp>

#include "kernels.hpp"
#include "vpi_matcher.hpp"

namespace holoscan::stereo_disparity {
namespace {

using std::chrono_literals::operator""ns;
using detail::LockedImage;
using detail::PixelFormat;

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kUInt16{kDLUInt, 16U, 1U};
constexpr int kMaxSide = 16384;

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "StereoDisparityOp: " + message};
}

holoscan::Error failure(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "StereoDisparityOp: " + message};
}

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return failure(std::string(what) + ": " + cudaGetErrorString(status));
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

struct ImageView {
  const std::uint8_t* data{};
  int width{};
  int height{};
  std::size_t pitch{};
  PixelFormat format{PixelFormat::kMono8};
  bool on_device{};
};

// The image's pixels, after checking that its tensor matches its encoding.
holoscan::expected<ImageView, holoscan::Error> view_image(const holoscan::schema::ImageT& image,
                                                          const std::string& side) {
  if (!image.data) {
    return holoscan::make_unexpected(invalid(side + " image has no tensor"));
  }
  const holoscan::Tensor& tensor = *image.data;
  const auto format = pixel_format(image.encoding);
  if (!format) {
    return holoscan::make_unexpected(
        invalid("unsupported " + side + " image encoding " +
                holoscan::schema::EnumNameImageEncoding(image.encoding) +
                "; use MONO8, RGB8, BGR8, RGBA8 or BGRA8"));
  }
  const DLDataType dtype = tensor.dtype();
  if (dtype.code != kDLUInt || dtype.bits != 8 || dtype.lanes != 1) {
    return holoscan::make_unexpected(invalid(side + " image tensor must hold uint8 elements"));
  }
  const std::span<const std::int64_t> shape = tensor.shape_span();
  const int pixel_channels = detail::channels(*format);
  const bool hwc = shape.size() == 3 && shape[2] == pixel_channels;
  const bool hw = shape.size() == 2 && pixel_channels == 1;
  if (!hwc && !hw) {
    return holoscan::make_unexpected(invalid(
        side + " image tensor shape must be [height, width, channels] matching its encoding"));
  }
  const std::vector<std::int64_t> strides = tensor.strides();
  if (strides[1] != pixel_channels || (hwc && strides[2] != 1) ||
      strides[0] < shape[1] * pixel_channels) {
    return holoscan::make_unexpected(invalid(side + " image pixels must be packed within rows"));
  }
  if ((image.width != 0 && image.width != shape[1]) ||
      (image.height != 0 && image.height != shape[0])) {
    return holoscan::make_unexpected(
        invalid(side + " image width and height disagree with its tensor"));
  }
  const DLDevice device = tensor.device();
  const bool on_device = device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
  if (!on_device && device.device_type != kDLCPU && device.device_type != kDLCUDAHost) {
    return holoscan::make_unexpected(invalid(side + " image must be in host or CUDA memory"));
  }
  return ImageView{.data = tensor.data_as<std::uint8_t>(),
                   .width = static_cast<int>(shape[1]),
                   .height = static_cast<int>(shape[0]),
                   .pitch = static_cast<std::size_t>(strides[0]),
                   .format = *format,
                   .on_device = on_device};
}

holoscan::TensorOutputSpec device_image(DLDataType dtype, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = dtype,
              .rank = 2U,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(bytes),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

holoscan::PayloadOptions image_payload() {
  return holoscan::PayloadOptions{.schema = holoscan::schema_identity<holoscan::schema::ImageT>()};
}

std::string no_backend(Backend backend) {
  return "the " + std::string(backend_name(backend)) + " backend needs " +
         (backend == Backend::kPva ? "a PVA, an OFA and a VIC" : "an OFA and a VIC") +
         " (Jetson Orin or Thor), and VPI reports none on this platform; use cuda or auto";
}

}  // namespace

struct StereoDisparityOp::State {
  struct StreamDestroy {
    void operator()(cudaStream_t stream) const noexcept {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  };
  struct DeviceFree {
    void operator()(void* memory) const noexcept { cudaFree(memory); }
  };
  struct EventDestroy {
    void operator()(cudaEvent_t event) const noexcept {
      cudaEventSynchronize(event);
      cudaEventDestroy(event);
    }
  };

  // The configured backend, or what kAuto chose.
  std::atomic<Backend> resolved{Backend::kAuto};
  // Built on the first frame, and again when the image size changes.
  std::unique_ptr<detail::Matcher> matcher;
  // The operator's own stream, for the work it waits for on the host: the
  // gray pair and the copy of VPI's result. The runtime's stream is shared by
  // the operators of a graph, so waiting for it would wait for theirs too.
  std::unique_ptr<CUstream_st, StreamDestroy> work_stream;
  // Device copies of host images: both eyes, up to four channels each.
  std::unique_ptr<void, DeviceFree> staging;
  std::size_t staging_bytes{};
  // VPI's raw disparity and confidence, copied out so that VPI's images are
  // unlocked before the outputs are written on the runtime's stream; and the
  // event after the last frame's outputs, which the next copy waits for.
  std::unique_ptr<void, DeviceFree> raw;
  std::size_t raw_bytes{};
  std::unique_ptr<CUevent_st, EventDestroy> outputs_written;
  // For frames whose activation has no runtime CUDA stream; made on first use.
  std::unique_ptr<CUstream_st, StreamDestroy> own_stream;
  // Whether host images have been copied to the device, which is logged once.
  bool copy_logged{};
};

void validate_config(const StereoDisparityConfig& config) {
  if (auto problem = params_problem(config.backend, config.params, config.confidence)) {
    throw std::invalid_argument(*problem);
  }
  if (config.geometry && config.geometry_input) {
    // The input's geometry would always win; refuse a setting that would be ignored.
    throw std::invalid_argument("give the geometry or enable the geometry input, not both");
  }
  if (config.geometry) {
    validate_geometry(*config.geometry);
  }
  if (config.depth && !config.geometry && !config.geometry_input) {
    throw std::invalid_argument(
        "the depth output needs the geometry: give it, or enable the geometry input");
  }
  if (config.max_width < 1 || config.max_height < 1 || config.max_width > kMaxSide ||
      config.max_height > kMaxSide) {
    throw std::invalid_argument("max_width and max_height must be 1..16384");
  }
  if (config.backend == Backend::kOfa || config.backend == Backend::kPva) {
    const std::vector<Backend> backends = available_backends();
    if (std::find(backends.begin(), backends.end(), config.backend) == backends.end()) {
      throw std::invalid_argument(no_backend(config.backend));
    }
  }
}

StereoDisparityOp::StereoDisparityOp(StereoDisparityConfig config)
    : config_(std::move(config)), state_(std::make_unique<State>()) {
  validate_config(config_);
  state_->resolved = config_.backend;
}

StereoDisparityOp::~StereoDisparityOp() = default;

Backend StereoDisparityOp::resolved_backend() const noexcept { return state_->resolved; }

void StereoDisparityOp::setup(holoscan::OperatorSpec& spec) {
  const auto pixels = static_cast<std::uint64_t>(config_.max_width) *
                      static_cast<std::uint64_t>(config_.max_height);
  // Two slots per input: one frame in compute() while the next one arrives.
  spec.input(left, "left").queue_depth(2U);
  spec.input(right, "right").queue_depth(2U);
  // Optional ports exist only when configured: an unconnected input would
  // never be satisfied, and an emit into an unconnected output fails.
  if (config_.geometry_input) {
    spec.input(geometry, "geometry").queue_depth(2U);
  }
  spec.output(disparity, "disparity")
      .produces_tensor(device_image(kFloat32, pixels * sizeof(float)))
      .payload_options(image_payload());
  if (config_.depth) {
    spec.output(depth, "depth")
        .produces_tensor(device_image(kFloat32, pixels * sizeof(float)))
        .payload_options(image_payload());
  }
  if (config_.confidence) {
    spec.output(confidence, "confidence")
        .produces_tensor(device_image(kUInt16, pixels * sizeof(std::uint16_t)))
        .payload_options(image_payload());
  }
}

holoscan::Contract StereoDisparityOp::contract() const {
  // Every part of a stereo frame carries the same capture time.
  std::vector<holoscan::PortRef> ports{left, right};
  if (config_.geometry_input) ports.emplace_back(geometry);
  holoscan::Contract result;
  result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{ports}, .window = 0ns});
  result.input(left, holoscan::Exact{});
  result.input(right, holoscan::Exact{});
  if (config_.geometry_input) result.input(geometry, holoscan::Exact{});
  return result;
}

void StereoDisparityOp::start() { state_->copy_logged = false; }

void StereoDisparityOp::stop() {
  state_->matcher.reset();
  // The streams and the event first: their work may still use the buffers.
  state_->outputs_written.reset();
  state_->work_stream.reset();
  state_->own_stream.reset();
  state_->staging.reset();
  state_->staging_bytes = 0;
  state_->raw.reset();
  state_->raw_bytes = 0;
}

holoscan::expected<void, holoscan::Error> StereoDisparityOp::compute(
    holoscan::ExecutionContext& context) {
  auto left_sample = left.receive();
  if (!left_sample) {
    return holoscan::make_unexpected(std::move(left_sample).error());
  }
  auto right_sample = right.receive();
  if (!right_sample) {
    return holoscan::make_unexpected(std::move(right_sample).error());
  }
  const auto started = std::chrono::steady_clock::now();
  State& state = *state_;

  const std::array<const holoscan::schema::ImageT*, 2> images{&left_sample->data,
                                                              &right_sample->data};
  std::array<ImageView, 2> views;
  for (int side = 0; side < 2; ++side) {
    auto view = view_image(*images[side], side == 0 ? "left" : "right");
    if (!view) {
      return holoscan::make_unexpected(std::move(view).error());
    }
    views[side] = *view;
  }
  const int width = views[0].width;
  const int height = views[0].height;
  const std::string size = std::to_string(width) + "x" + std::to_string(height);
  if (views[1].width != width || views[1].height != height) {
    return holoscan::make_unexpected(invalid("the left image is " + size + " and the right " +
                                             std::to_string(views[1].width) + "x" +
                                             std::to_string(views[1].height)));
  }
  if (width > config_.max_width || height > config_.max_height) {
    return holoscan::make_unexpected(invalid("image " + size + " exceeds the configured " +
                                             std::to_string(config_.max_width) + "x" +
                                             std::to_string(config_.max_height)));
  }

  // The frame's geometry, or the configured one.
  std::optional<StereoGeometry> frame_geometry = config_.geometry;
  if (config_.geometry_input) {
    auto sample = geometry.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    try {
      frame_geometry = from_message(sample->data);
    } catch (const std::invalid_argument& error) {
      return holoscan::make_unexpected(invalid(error.what()));
    }
  }
  if (frame_geometry && (frame_geometry->width != width || frame_geometry->height != height)) {
    return holoscan::make_unexpected(
        invalid("the geometry is for " + std::to_string(frame_geometry->width) + "x" +
                std::to_string(frame_geometry->height) + " images, and the images are " + size));
  }

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    // Not reached in a compiled graph: the outputs are in device memory, so
    // the runtime assigns a stream. Kept so the operator never falls back to
    // the legacy default stream, which serializes with every other stream.
    if (!state.own_stream) {
      cudaStream_t created = nullptr;
      if (const cudaError_t status = cudaStreamCreateWithFlags(&created, cudaStreamNonBlocking);
          status != cudaSuccess) {
        return holoscan::make_unexpected(cuda_failure("cudaStreamCreate", status));
      }
      state.own_stream.reset(created);
    }
    stream = state.own_stream.get();
  }
  if (!state.work_stream) {
    cudaStream_t created = nullptr;
    if (const cudaError_t status = cudaStreamCreateWithFlags(&created, cudaStreamNonBlocking);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaStreamCreate", status));
    }
    state.work_stream.reset(created);
    cudaEvent_t event = nullptr;
    if (const cudaError_t status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaEventCreate", status));
    }
    state.outputs_written.reset(event);
  }
  cudaStream_t work = state.work_stream.get();
  // Wait for each image's producer: on the work stream for device memory, on
  // the host otherwise.
  for (int side = 0; side < 2; ++side) {
    const holoscan::Tensor& tensor = *images[side]->data;
    auto ready = views[side].on_device && tensor.has_readiness_stream_contract()
                     ? tensor.synchronize_for_consumer_stream(work)
                     : tensor.synchronize_for_host();
    if (!ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
  }

  const std::array<std::int64_t, 2> shape{height, width};
  std::optional<holoscan::TensorOutputLoan> disparity_loan;
  std::optional<holoscan::TensorOutputLoan> depth_loan;
  std::optional<holoscan::TensorOutputLoan> confidence_loan;
  std::optional<holoscan::TensorOutputWriteGuard> disparity_writer;
  std::optional<holoscan::TensorOutputWriteGuard> depth_writer;
  std::optional<holoscan::TensorOutputWriteGuard> confidence_writer;
  const auto take = [stream, &shape](holoscan::Output<holoscan::schema::ImageT>& output,
                                     DLDataType dtype,
                                     std::optional<holoscan::TensorOutputLoan>& loan,
                                     std::optional<holoscan::TensorOutputWriteGuard>& writer)
      -> holoscan::expected<void, holoscan::Error> {
    auto allocated =
        output.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
    if (!allocated) {
      return holoscan::make_unexpected(std::move(allocated).error());
    }
    loan.emplace(std::move(*allocated));
    auto opened = loan->write(stream);
    if (!opened) {
      return holoscan::make_unexpected(std::move(opened).error());
    }
    writer.emplace(std::move(*opened));
    return {};
  };

  std::chrono::steady_clock::duration build_time{};
  try {
    if (!state.matcher || state.matcher->width() != width || state.matcher->height() != height) {
      const auto building = std::chrono::steady_clock::now();
      state.matcher.reset();
      Backend backend = config_.backend;
      if (backend == Backend::kAuto) {
        const AutoChoice choice =
            resolve_auto(probe_platform(), config_.params, config_.confidence, width, height);
        if (choice.backend != state.resolved) {
          HOLOSCAN_LOG_INFO("StereoDisparityOp: backend auto chose {} for {} images: {}",
                            backend_name(choice.backend), size, choice.reason);
        }
        backend = choice.backend;
      }
      state.resolved = backend;
      state.matcher = std::make_unique<detail::Matcher>(backend, config_.params, config_.confidence,
                                                        width, height);
      build_time = std::chrono::steady_clock::now() - building;
    }
    detail::Matcher& matcher = *state.matcher;
    if (!state.copy_logged && (!views[0].on_device || !views[1].on_device)) {
      state.copy_logged = true;
      const std::size_t bytes =
          static_cast<std::size_t>(width) * height *
          (detail::channels(views[0].format) + detail::channels(views[1].format));
      HOLOSCAN_LOG_INFO(
          "StereoDisparityOp ({} backend): the images are in host memory and VPI's matcher reads "
          "CUDA device memory, so each frame's pair is copied host to device, {} bytes ({:.1f} "
          "MB) per frame; to avoid the copy, send CUDA device images",
          backend_name(matcher.backend()), bytes, static_cast<double>(bytes) / 1e6);
    }

    // The gray pair into VPI's images, on the work stream; VPI reads them
    // unlocked.
    {
      const LockedImage gray_left(matcher.gray(0), VPI_LOCK_WRITE);
      const LockedImage gray_right(matcher.gray(1), VPI_LOCK_WRITE);
      const std::size_t half = static_cast<std::size_t>(width) * 4U * height;
      for (int side = 0; side < 2; ++side) {
        const ImageView& view = views[side];
        const LockedImage& gray = side == 0 ? gray_left : gray_right;
        const std::uint8_t* source = view.data;
        std::size_t pitch = view.pitch;
        if (!view.on_device) {
          // Host rows go to the GPU first, packed.
          if (state.staging_bytes < 2 * half) {
            cudaStreamSynchronize(work);
            state.staging.reset();
            state.staging_bytes = 0;
            void* memory = nullptr;
            if (const cudaError_t status = cudaMalloc(&memory, 2 * half); status != cudaSuccess) {
              cudaStreamSynchronize(work);
              return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
            }
            state.staging.reset(memory);
            state.staging_bytes = 2 * half;
          }
          const std::size_t row = static_cast<std::size_t>(width) * detail::channels(view.format);
          auto* target = static_cast<std::uint8_t*>(state.staging.get()) + side * half;
          if (const cudaError_t status = cudaMemcpy2DAsync(target, row, view.data, view.pitch, row,
                                                           height, cudaMemcpyHostToDevice, work);
              status != cudaSuccess) {
            cudaStreamSynchronize(work);
            return holoscan::make_unexpected(cuda_failure("cudaMemcpy2DAsync", status));
          }
          source = target;
          pitch = row;
        }
        if (const cudaError_t status =
                detail::to_gray(source, pitch, view.format, width, height,
                                gray.data<std::uint8_t>(), gray.pitch(), work);
            status != cudaSuccess) {
          cudaStreamSynchronize(work);
          return holoscan::make_unexpected(cuda_failure("to_gray", status));
        }
      }
      // Done with the inputs and the gray images before VPI reads them, and
      // before the locks go.
      if (const cudaError_t status = cudaStreamSynchronize(work); status != cudaSuccess) {
        return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
      }
    }
    matcher.match();

    if (auto taken = take(disparity, kFloat32, disparity_loan, disparity_writer); !taken) {
      return taken;
    }
    if (config_.depth) {
      if (auto taken = take(depth, kFloat32, depth_loan, depth_writer); !taken) {
        return taken;
      }
    }
    if (config_.confidence) {
      if (auto taken = take(confidence, kUInt16, confidence_loan, confidence_writer); !taken) {
        return taken;
      }
    }
    // VPI's result, copied out on the work stream once the last frame's
    // outputs have read the previous copy.
    const std::size_t raw_plane = static_cast<std::size_t>(width) * height * sizeof(std::int16_t);
    if (state.raw_bytes < 2 * raw_plane) {
      cudaEventSynchronize(state.outputs_written.get());
      state.raw.reset();
      state.raw_bytes = 0;
      void* memory = nullptr;
      if (const cudaError_t status = cudaMalloc(&memory, 2 * raw_plane); status != cudaSuccess) {
        return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
      }
      state.raw.reset(memory);
      state.raw_bytes = 2 * raw_plane;
    }
    auto* raw_disparity = static_cast<std::int16_t*>(state.raw.get());
    auto* raw_confidence =
        reinterpret_cast<std::uint16_t*>(static_cast<std::uint8_t*>(state.raw.get()) + raw_plane);
    const std::size_t raw_pitch = static_cast<std::size_t>(width) * sizeof(std::int16_t);
    {
      const LockedImage locked(matcher.disparity(), VPI_LOCK_READ);
      std::optional<LockedImage> locked_confidence;
      if (config_.confidence) {
        locked_confidence.emplace(matcher.confidence(), VPI_LOCK_READ);
      }
      cudaError_t status = cudaStreamWaitEvent(work, state.outputs_written.get(), 0);
      if (status == cudaSuccess) {
        status =
            cudaMemcpy2DAsync(raw_disparity, raw_pitch, locked.data<std::int16_t>(), locked.pitch(),
                              raw_pitch, height, cudaMemcpyDeviceToDevice, work);
      }
      if (status == cudaSuccess && locked_confidence) {
        status = cudaMemcpy2DAsync(
            raw_confidence, raw_pitch, locked_confidence->data<std::uint16_t>(),
            locked_confidence->pitch(), raw_pitch, height, cudaMemcpyDeviceToDevice, work);
      }
      // VPI's images stay locked until the copies are done.
      const cudaError_t synced = cudaStreamSynchronize(work);
      if (status != cudaSuccess || synced != cudaSuccess) {
        return holoscan::make_unexpected(
            cuda_failure("copying VPI's result", status != cudaSuccess ? status : synced));
      }
    }
    detail::DisparityOutputs outputs{
        .disparity = disparity_writer->data_as<float>(),
        .depth = depth_writer ? depth_writer->data_as<float>() : nullptr,
        .confidence = confidence_writer ? confidence_writer->data_as<std::uint16_t>() : nullptr};
    if (frame_geometry) {
      outputs.fx_baseline = static_cast<float>(frame_geometry->fx * frame_geometry->baseline);
      outputs.doffs = static_cast<float>(frame_geometry->doffs);
    }
    // The outputs, on the runtime's stream, which orders them for their
    // consumers; the host does not wait for them.
    if (const cudaError_t launched = detail::finish_disparity(
            raw_disparity, raw_pitch, config_.confidence ? raw_confidence : nullptr, raw_pitch,
            width, height, static_cast<float>(matcher.searched()), outputs, stream);
        launched != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("finish_disparity", launched));
    }
    if (const cudaError_t recorded = cudaEventRecord(state.outputs_written.get(), stream);
        recorded != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaEventRecord", recorded));
    }
    if (stream == state.own_stream.get()) {
      if (const cudaError_t synced = cudaStreamSynchronize(stream); synced != cudaSuccess) {
        return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", synced));
      }
    }
  } catch (const std::exception& error) {
    // Nothing queued on the work stream may still use the inputs after we return.
    cudaStreamSynchronize(work);
    const bool bad_input = dynamic_cast<const std::invalid_argument*>(&error) != nullptr;
    return holoscan::make_unexpected(bad_input ? invalid(error.what()) : failure(error.what()));
  }

  if (config_.stats) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - started - build_time)
                             .count();
    const auto ns = static_cast<std::uint64_t>(std::max<std::int64_t>(elapsed, 0));
    StereoDisparityStats& stats = *config_.stats;
    stats.backend = state.resolved.load();
    stats.frames.fetch_add(1);
    stats.compute_ns_total.fetch_add(ns);
    std::uint64_t previous = stats.compute_ns_max.load();
    while (ns > previous && !stats.compute_ns_max.compare_exchange_weak(previous, ns)) {
    }
  }

  const holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(*left_sample);
  const auto descriptor = [&](holoscan::schema::ImageEncoding encoding) {
    holoscan::schema::ImageT image;
    image.width = width;
    image.height = height;
    image.encoding = encoding;
    image.header = left_sample->data.header;
    return image;
  };
  const auto send =
      [&](holoscan::Output<holoscan::schema::ImageT>& output, holoscan::TensorOutputLoan& loan,
          holoscan::TensorOutputWriteGuard& writer,
          holoscan::schema::ImageEncoding encoding) -> holoscan::expected<void, holoscan::Error> {
    if (auto committed = std::move(writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    return output.emit_tensor(std::move(loan), descriptor(encoding), emit);
  };
  if (auto sent = send(disparity, *disparity_loan, *disparity_writer,
                       holoscan::schema::ImageEncoding_MONO32F);
      !sent) {
    return sent;
  }
  if (depth_loan) {
    if (auto sent =
            send(depth, *depth_loan, *depth_writer, holoscan::schema::ImageEncoding_MONO32F);
        !sent) {
      return sent;
    }
  }
  if (confidence_loan) {
    return send(confidence, *confidence_loan, *confidence_writer,
                holoscan::schema::ImageEncoding_MONO16);
  }
  return {};
}

}  // namespace holoscan::stereo_disparity
