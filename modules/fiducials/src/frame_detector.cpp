// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "frame_detector.hpp"

#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/fiducials/pose.hpp>
#include <holoscan/logger/logger.hpp>

#include "cuda_buffers.hpp"  // cuda_check

namespace holoscan::fiducials {
namespace {

// Least time between two warnings about frames over capacity.
constexpr std::chrono::seconds kOverflowWarningInterval{5};

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

// Says, once, that every frame's gray image crosses between host and device memory because the
// images are not where `backend` reads them, and how not to.
void log_cross_copy(std::string_view op, std::string_view backend, MemoryLocation images, int width,
                    int height) {
  const bool to_device = images == MemoryLocation::kHost;
  const std::size_t bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  HOLOSCAN_LOG_INFO(
      "{} ({} backend): the images are in {} memory and the backend reads {} memory, so each "
      "frame's gray image is copied {}, {} bytes ({:.1f} MB) per frame; to avoid the copy, {}",
      op, backend, to_device ? "host" : "CUDA device", to_device ? "CUDA device" : "host",
      to_device ? "host to device" : "device to host", bytes, static_cast<double>(bytes) / 1e6,
      to_device ? "send CUDA device images or use the cpu backend"
                : "use the cuda backend or send host images");
}

}  // namespace

FrameDetector::FrameDetector(std::string op, Backend backend, DetectorOptions options)
    : op_(std::move(op)), options_(std::move(options)), resolved_(backend) {
  validate_options(options_);
}

FrameDetector::~FrameDetector() = default;

void FrameDetector::start() {
  if (resolved_ == Backend::kAuto) {
    const PlatformFacts facts = probe_platform();
    const Backend chosen = resolve_auto(facts);
    resolved_ = chosen;
    HOLOSCAN_LOG_INFO("{}: backend auto chose {}: {}", op_, backend_name(chosen),
                      facts.cuda_device ? "a CUDA device is usable" : "no CUDA device is usable");
  }
  frames_ = 0;
  waits_ = 0;
  overflowed_frames_ = 0;
  last_overflow_warning_.reset();
}

void FrameDetector::stop() {
  backend_.reset();
  own_stream_.reset();
}

holoscan::Error FrameDetector::error(const std::exception& exception, cudaStream_t stream) const {
  if (stream != nullptr) cudaStreamSynchronize(stream);
  const bool bad_input = dynamic_cast<const std::invalid_argument*>(&exception) != nullptr;
  return holoscan::Error{
      bad_input ? holoscan::ErrorCode::kInvalidArgument : holoscan::ErrorCode::kFailure,
      op_ + ": " + exception.what()};
}

holoscan::expected<FrameDetector::Frame, holoscan::Error> FrameDetector::prepare(
    const holoscan::schema::ImageT& image, holoscan::ExecutionContext& context,
    const messages::CameraInfo* camera) {
  const auto invalid = [this](const std::string& message) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, op_ + ": " + message});
  };
  if (!image.data) return invalid("image has no tensor");
  const holoscan::Tensor& tensor = *image.data;
  const auto format = pixel_format(image.encoding);
  if (!format) {
    return invalid(std::string("unsupported image encoding ") +
                   holoscan::schema::EnumNameImageEncoding(image.encoding) +
                   "; use MONO8, RGB8, BGR8, RGBA8 or BGRA8");
  }
  const auto dtype = tensor.dtype();
  if (dtype.code != kDLUInt || dtype.bits != 8 || dtype.lanes != 1) {
    return invalid("image tensors must hold uint8 elements");
  }
  const std::span<const std::int64_t> shape = tensor.shape_span();
  const int pixel_channels = channels(*format);
  const bool hwc = shape.size() == 3 && shape[2] == pixel_channels;
  const bool hw = shape.size() == 2 && pixel_channels == 1;
  if (!hwc && !hw) {
    return invalid("image tensor shape must be [height, width, channels] matching its encoding");
  }
  const std::vector<std::int64_t> strides = tensor.strides();
  if (strides[1] != pixel_channels || (hwc && strides[2] != 1) ||
      strides[0] < shape[1] * pixel_channels) {
    return invalid("image pixels must be packed within each row");
  }
  if ((image.width != 0 && image.width != shape[1]) ||
      (image.height != 0 && image.height != shape[0])) {
    return invalid("image width and height disagree with its tensor");
  }
  const DLDevice device = tensor.device();
  const bool on_device = device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
  if (!on_device && device.device_type != kDLCPU && device.device_type != kDLCUDAHost) {
    return invalid("image tensors must be in host or CUDA memory");
  }
  const ImageView view{.data = tensor.data_as<std::uint8_t>(),
                       .width = static_cast<int>(shape[1]),
                       .height = static_cast<int>(shape[0]),
                       .pitch = static_cast<std::size_t>(strides[0]),
                       .format = *format,
                       .location = on_device ? MemoryLocation::kDevice : MemoryLocation::kHost};

  if (camera != nullptr) {
    try {
      check_camera_size(*camera, view.width, view.height);
    } catch (const std::invalid_argument& exception) {
      return invalid(exception.what());
    }
  }

  cudaStream_t stream = nullptr;
  try {
    if (!backend_ || view.width != width_ || view.height != height_) {
      backend_.reset();
      backend_ = create_backend(resolved_, options_, view.width, view.height);
      width_ = view.width;
      height_ = view.height;
    }
    const MemoryLocation target = backend_->input_location();
    // A host image for a host backend needs no CUDA stream, so the cpu backend runs without a
    // GPU; anything on the device or crossing to it does.
    if (view.location == MemoryLocation::kDevice || target == MemoryLocation::kDevice) {
      stream = context.cuda_stream().get();
      if (stream == nullptr) {
        // No device memory on our ports, so no runtime stream: use our own, not the legacy
        // default stream, which serializes with every other stream.
        if (!own_stream_) {
          cudaStream_t created = nullptr;
          cuda_check(cudaStreamCreateWithFlags(&created, cudaStreamNonBlocking),
                     "cudaStreamCreate");
          own_stream_.reset(created);
        }
        stream = own_stream_.get();
      }
    }
    // Wait for the producer's writes: on our stream when the tensor says which stream they were
    // on and the image is on the device; otherwise on the host, which with such a stream is a
    // wait for it.
    const bool readiness = tensor.has_readiness_stream_contract();
    auto ready = view.location == MemoryLocation::kDevice && readiness
                     ? tensor.synchronize_for_consumer_stream(stream)
                     : tensor.synchronize_for_host();
    if (!ready) return holoscan::make_unexpected(std::move(ready).error());
    if (readiness && view.location == MemoryLocation::kHost) ++waits_;
    if (view.location != target && !cross_copy_logged_) {
      cross_copy_logged_ = true;
      log_cross_copy(op_, backend_->name(), view.location, view.width, view.height);
    }
    const auto start = std::chrono::steady_clock::now();
    const GrayImageView gray = converter_.convert(view, target, stream);
    return Frame{gray, backend_.get(), stream, frames_++, start};
  } catch (const std::exception& exception) {
    return holoscan::make_unexpected(error(exception, stream));
  }
}

TransferCounts FrameDetector::take_transfers() noexcept {
  TransferCounts counts{.host_waits = waits_};
  waits_ = 0;
  counts += converter_.take_transfers();
  if (backend_) counts += backend_->take_transfers();
  return counts;
}

void FrameDetector::warn_overflow(const Overflow& overflow) {
  ++overflowed_frames_;
  const auto now = std::chrono::steady_clock::now();
  if (last_overflow_warning_ && now - *last_overflow_warning_ < kOverflowWarningInterval) return;
  HOLOSCAN_LOG_WARN(
      "{} ({} backend): {} frame(s) over capacity since the last warning; the latest dropped {} "
      "candidates and {} markers{}",
      op_, backend_ ? backend_->name() : "no", overflowed_frames_, overflow.candidates,
      overflow.markers, overflow.truncated() ? ", so markers may be missing" : "");
  overflowed_frames_ = 0;
  last_overflow_warning_ = now;
}

}  // namespace holoscan::fiducials
