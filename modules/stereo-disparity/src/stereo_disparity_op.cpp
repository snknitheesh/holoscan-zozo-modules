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
#include <holoscan/messages/conversion.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/stereo_disparity/camera_info.hpp>
#include <holoscan/stereo_disparity/platform.hpp>

#include "kernels.hpp"
#include "vpi_matcher.hpp"

namespace holoscan::stereo_disparity {
namespace {

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
  // The operator's own stream, which the matcher's VPI stream wraps: the
  // frame's host copies and matching, which the operator waits for. The
  // runtime's stream is shared by the operators of an execution partition, so
  // waiting for it would wait for theirs too.
  std::unique_ptr<CUstream_st, StreamDestroy> work_stream;
  // Device copies of host images: both eyes, up to four channels each.
  std::unique_ptr<void, DeviceFree> staging;
  std::size_t staging_bytes{};
  // Built on the first frame, and again when the image size changes.
  std::unique_ptr<detail::Matcher> matcher;
  // After the latest frame's outputs, on the runtime's stream, have read the
  // matcher's result: the next frame's matching waits for it on the GPU, and
  // the matcher is not destroyed before it. Declared last, so destroyed (and
  // waited for) first.
  std::unique_ptr<CUevent_st, EventDestroy> outputs_written;
  // The newest valid stereo camera info, and the depth's camera made from it,
  // with the frame ids it was made from: a camera info whose calibration is
  // the same may still name another frame.
  messages::LatestCameraInfo<messages::schema::StereoCameraInfoT> camera_info;
  messages::schema::CameraInfoT depth_camera_info;
  std::optional<std::pair<std::string, std::string>> depth_frame_ids;
  // Whether host images have been copied to the device, which is logged once.
  bool copy_logged{};
};

void validate_config(const StereoDisparityConfig& config) {
  if (auto problem = params_problem(config.backend, config.params, config.confidence)) {
    throw std::invalid_argument(*problem);
  }
  if (config.depth_camera_info && !config.depth) {
    throw std::invalid_argument("the depth_camera_info output goes with the depth output");
  }
  if (config.camera_info_input.max_age <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("camera_info_input.max_age must be > 0");
  }
  if (config.max_width < 1 || config.max_height < 1 || config.max_width > kMaxSide ||
      config.max_height > kMaxSide) {
    throw std::invalid_argument("max_width and max_height must be 1..16384");
  }
  if (config.backend == Backend::kOfa || config.backend == Backend::kPva) {
    const PlatformFacts facts = probe_platform();
    const std::vector<Backend> backends = available_backends(facts);
    if (std::find(backends.begin(), backends.end(), config.backend) == backends.end()) {
      throw std::invalid_argument(no_backend(config.backend));
    }
    if (auto problem = platform_problem(config.backend, config.params, facts)) {
      throw std::invalid_argument(*problem);
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
  // The outputs are the input's size divided by the downscale factor.
  const int downscale = config_.params.downscale;
  const auto pixels = static_cast<std::uint64_t>(output_side(config_.max_width, downscale)) *
                      static_cast<std::uint64_t>(output_side(config_.max_height, downscale));
  // Two slots per input: one frame in compute() while the next one arrives.
  spec.input(left, "left").queue_depth(2U);
  spec.input(right, "right").queue_depth(2U);
  spec.input(camera_info, "camera_info").queue_depth(2U);
  // Optional outputs exist only when configured: an emit into an unconnected
  // output fails.
  spec.output(disparity, "disparity")
      .produces_tensor(device_image(kFloat32, pixels * sizeof(float)))
      .payload_options(image_payload());
  if (config_.depth) {
    spec.output(depth, "depth")
        .produces_tensor(device_image(kFloat32, pixels * sizeof(float)))
        .payload_options(image_payload());
  }
  if (config_.depth_camera_info) {
    spec.output(depth_camera_info, "depth_camera_info");
  }
  if (config_.confidence) {
    spec.output(confidence, "confidence")
        .produces_tensor(device_image(kUInt16, pixels * sizeof(std::uint16_t)))
        .payload_options(image_payload());
  }
}

holoscan::Contract StereoDisparityOp::contract() const {
  // The left and right images of a pair carry the same capture time.
  return messages::camera_info_contract(config_.camera_info_input, camera_info, left, right);
}

void StereoDisparityOp::start() {
  // A restart chooses kAuto's backend again, on its first frame.
  state_->resolved = config_.backend;
  state_->copy_logged = false;
  state_->camera_info.reset();
  state_->depth_frame_ids.reset();
}

void StereoDisparityOp::stop() {
  // The last outputs read the matcher's result, and the matcher's VPI stream
  // wraps the work stream: the event, then the matcher, then the streams.
  state_->outputs_written.reset();
  state_->matcher.reset();
  state_->work_stream.reset();
  state_->staging.reset();
  state_->staging_bytes = 0;
}

holoscan::expected<void, holoscan::Error> StereoDisparityOp::compute(
    holoscan::ExecutionContext& context) {
  State& state = *state_;
  auto info = state.camera_info.update(camera_info);
  if (!info) {
    return holoscan::make_unexpected(std::move(info).error());
  }
  if (*info == nullptr) {
    return {};  // No camera info yet: the frame is skipped.
  }
  const messages::StereoCameraInfo& pair_info = **info;
  auto left_sample = left.receive();
  if (!left_sample) {
    return holoscan::make_unexpected(std::move(left_sample).error());
  }
  auto right_sample = right.receive();
  if (!right_sample) {
    return holoscan::make_unexpected(std::move(right_sample).error());
  }
  const auto started = std::chrono::steady_clock::now();

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

  if (auto problem = camera_info_problem(pair_info, width, height)) {
    return holoscan::make_unexpected(invalid(*problem));
  }
  if (config_.depth_camera_info) {
    auto frame_ids = std::make_pair(pair_info.header.frame_id, pair_info.left.header.frame_id);
    if (state.camera_info.changed() || state.depth_frame_ids != frame_ids) {
      try {
        state.depth_camera_info = messages::to_message(
            stereo_disparity::depth_camera_info(pair_info, config_.params.downscale));
      } catch (const std::invalid_argument& error) {
        return holoscan::make_unexpected(invalid(error.what()));
      }
      state.depth_frame_ids = std::move(frame_ids);
    }
  }

  // The outputs are in device memory, so the runtime gives every activation a
  // stream; without one, the work would go to the legacy default stream.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(failure("the runtime gave this frame no CUDA stream"));
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

  std::optional<holoscan::TensorOutputLoan> disparity_loan;
  std::optional<holoscan::TensorOutputLoan> depth_loan;
  std::optional<holoscan::TensorOutputLoan> confidence_loan;
  std::optional<holoscan::TensorOutputWriteGuard> disparity_writer;
  std::optional<holoscan::TensorOutputWriteGuard> depth_writer;
  std::optional<holoscan::TensorOutputWriteGuard> confidence_writer;
  std::array<std::int64_t, 2> shape{};
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
      // The last outputs may still be reading the old matcher's result.
      cudaEventSynchronize(state.outputs_written.get());
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
                                                        width, height, work);
      build_time = std::chrono::steady_clock::now() - building;
    }
    detail::Matcher& matcher = *state.matcher;
    const int out_width = matcher.output_width();
    const int out_height = matcher.output_height();
    shape = {out_height, out_width};
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

    // The pair in device memory: device images as they are; host images
    // copied, packed, on the work stream, where VPI has nothing queued.
    const std::size_t half = static_cast<std::size_t>(width) * 4U * height;
    if ((!views[0].on_device || !views[1].on_device) && state.staging_bytes < 2 * half) {
      state.staging.reset();
      state.staging_bytes = 0;
      void* memory = nullptr;
      if (const cudaError_t status = cudaMalloc(&memory, 2 * half); status != cudaSuccess) {
        return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
      }
      state.staging.reset(memory);
      state.staging_bytes = 2 * half;
    }
    std::array<detail::DeviceImage, 2> pair;
    for (int side = 0; side < 2; ++side) {
      const ImageView& view = views[side];
      pair[side] = {.data = view.data, .pitch = view.pitch, .format = view.format};
      if (view.on_device) {
        continue;
      }
      const std::size_t row = static_cast<std::size_t>(width) * detail::channels(view.format);
      auto* target = static_cast<std::uint8_t*>(state.staging.get()) + side * half;
      if (const cudaError_t status = cudaMemcpy2DAsync(target, row, view.data, view.pitch, row,
                                                       height, cudaMemcpyHostToDevice, work);
          status != cudaSuccess) {
        cudaStreamSynchronize(work);
        return holoscan::make_unexpected(cuda_failure("cudaMemcpy2DAsync", status));
      }
      pair[side].data = target;
      pair[side].pitch = row;
    }
    // VPI writes its result again only once the last frame's outputs have
    // read it.
    if (const cudaError_t status = cudaStreamWaitEvent(work, state.outputs_written.get(), 0);
        status != cudaSuccess) {
      cudaStreamSynchronize(work);
      return holoscan::make_unexpected(cuda_failure("cudaStreamWaitEvent", status));
    }
    // The frame's host wait (two on ofa and pva; see Matcher).
    matcher.match(pair);

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
    const detail::DisparityOutputs outputs{
        .disparity = disparity_writer->data_as<float>(),
        .depth = depth_writer ? depth_writer->data_as<float>() : nullptr,
        .fx_baseline = static_cast<float>(messages::fx_baseline(pair_info)),
        .doffs = static_cast<float>(pair_info.doffs),
        .confidence = confidence_writer ? confidence_writer->data_as<std::uint16_t>() : nullptr};
    // The outputs, straight from the matcher's result, on the runtime's
    // stream, which orders them for their consumers; the host does not wait
    // for them.
    if (const cudaError_t launched = detail::finish_disparity(
            matcher.disparity(), matcher.disparity_pitch(), matcher.confidence(),
            matcher.confidence_pitch(), out_width, out_height,
            static_cast<float>(matcher.searched()), outputs, stream);
        launched != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("finish_disparity", launched));
    }
    if (const cudaError_t recorded = cudaEventRecord(state.outputs_written.get(), stream);
        recorded != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaEventRecord", recorded));
    }
  } catch (const std::exception& error) {
    // The matcher leaves nothing queued; host copies may still be.
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
    image.width = static_cast<std::int32_t>(shape[1]);
    image.height = static_cast<std::int32_t>(shape[0]);
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
  if (config_.depth_camera_info) {
    // Before the depth: a consumer taking the newest camera info gets this one.
    state.depth_camera_info.header->capture_timestamp_ns =
        emit.capture_time ? emit.capture_time->timestamp_ns : 0;
    if (auto sent = depth_camera_info.emit_ref(state.depth_camera_info, emit); !sent) {
      return sent;
    }
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
