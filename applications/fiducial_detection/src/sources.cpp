// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "sources.hpp"

#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/messages/conversion.hpp>
#include <linux/videodev2.h>  // NOLINT(build/include_order)
#include <sys/ioctl.h>        // NOLINT(build/include_order)
#include <sys/mman.h>         // NOLINT(build/include_order)

#include "recording.hpp"
#include "yuyv.hpp"

namespace fiducial_app {
namespace {

namespace fd = holoscan::fiducials;

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr int kMaxSide = 4096;

holoscan::Error failure(std::string message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, std::move(message)};
}

// Declares `image` as an RGB8 output of width x height frames in `memory`.
void rgb_output(holoscan::OperatorSpec& spec, holoscan::Output<holoscan::schema::ImageT>& image,
                holoscan::MemoryKind memory, int width, int height) {
  spec.output(image, "image")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = memory,
                  .dtype = kUInt8,
                  .rank = 3U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(std::uint64_t{static_cast<std::uint32_t>(width)} *
                                            static_cast<std::uint32_t>(height) * 3U),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      })
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
}

holoscan::schema::ImageT rgb_descriptor(int width, int height) {
  holoscan::schema::ImageT descriptor;
  descriptor.width = width;
  descriptor.height = height;
  descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
  return descriptor;
}

int xioctl(int fd, std::uint64_t request, void* argument) {
  int result = 0;
  do {
    result = ioctl(fd, request, argument);
  } while (result == -1 && errno == EINTR);
  return result;
}

constexpr std::uint32_t kCameraBuffers = 4;

// Opens `config.device` with YUYV at the configured size; returns the descriptor and the pitch.
int open_camera(const CameraSourceConfig& config, std::size_t& pitch) {
  if (config.width <= 0 || config.height <= 0 || config.width % 2 != 0 || config.width > kMaxSide ||
      config.height > kMaxSide) {
    throw std::invalid_argument("the camera size needs an even width and sides up to " +
                                std::to_string(kMaxSide));
  }
  const int fd = open(config.device.c_str(), O_RDWR | O_NONBLOCK);
  if (fd < 0) {
    throw std::invalid_argument("cannot open camera " + config.device + ": " +
                                std::strerror(errno));
  }
  v4l2_capability capability{};
  if (xioctl(fd, VIDIOC_QUERYCAP, &capability) != 0 ||
      (capability.device_caps & V4L2_CAP_VIDEO_CAPTURE) == 0 ||
      (capability.device_caps & V4L2_CAP_STREAMING) == 0) {
    close(fd);
    throw std::invalid_argument(config.device + " is not a V4L2 capture device with streaming");
  }
  v4l2_format format{};
  format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  format.fmt.pix.width = static_cast<std::uint32_t>(config.width);
  format.fmt.pix.height = static_cast<std::uint32_t>(config.height);
  format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
  format.fmt.pix.field = V4L2_FIELD_NONE;
  if (xioctl(fd, VIDIOC_S_FMT, &format) != 0 || format.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV ||
      format.fmt.pix.width != static_cast<std::uint32_t>(config.width) ||
      format.fmt.pix.height != static_cast<std::uint32_t>(config.height)) {
    close(fd);
    throw std::invalid_argument(config.device + " does not offer YUYV at " +
                                std::to_string(config.width) + "x" + std::to_string(config.height));
  }
  pitch = format.fmt.pix.bytesperline;
  return fd;
}

constexpr std::uint32_t kSentCameraInfo = 1U;
constexpr std::uint32_t kSentImage = 2U;

// After the camera's count, how long the run goes on before it completes, so that the last
// frames pass through the detectors.
constexpr std::chrono::seconds kCameraDrain{1};
constexpr std::chrono::milliseconds kCameraPoll{5};

// Logs what failed in a camera's setup, and throws it.
[[noreturn]] void camera_failure(const std::string& device, const std::string& what) {
  const std::string message = device + ": " + what + ": " + std::strerror(errno);
  HOLOSCAN_LOG_ERROR("CameraSourceOp: {}", message);
  throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure, message};
}

}  // namespace

bool backpressured(const holoscan::Error& error) noexcept {
  return error.code == holoscan::ErrorCode::kBackpressured ||
         error.code == holoscan::ErrorCode::kResourceExhausted;
}

fd::SyntheticScene make_scene(const SceneKind& kind, std::uint32_t frame, int width, int height) {
  return kind.board ? fd::make_board_scene(kind.board_config, frame, width, height)
                    : fd::make_marker_scene(kind.dictionary, frame, width, height);
}

FrameSourceConfig scene_source_config(const SceneKind& kind, int width, int height) {
  (void)make_scene(kind, 0, width, height);  // Dictionary, board, size.
  if (width > kMaxSide || height > kMaxSide) {
    throw std::invalid_argument("synthetic images must be 320x240 to 4096x4096");
  }
  return FrameSourceConfig{.width = width,
                           .height = height,
                           .render =
                               [kind, width, height](std::uint32_t n) {
                                 return fd::render_scene(make_scene(kind, n, width, height), 3);
                               },
                           .sequence = [](std::uint32_t n) { return std::uint64_t{n} + 1U; }};
}

FrameSourceConfig recording_source_config(const std::filesystem::path& clip, std::uint32_t count) {
  std::vector<RecordedFrame> frames = read_index(clip);
  if (count != 0 && count < frames.size()) frames.resize(count);
  const RgbImage first = read_ppm(frames.front().rgb);
  auto shared = std::make_shared<const std::vector<RecordedFrame>>(std::move(frames));
  const int width = first.width;
  const int height = first.height;
  return FrameSourceConfig{
      .width = width,
      .height = height,
      .count = static_cast<std::uint32_t>(shared->size()),
      .render =
          [shared, width, height](std::uint32_t n) {
            RgbImage image = read_ppm((*shared)[n].rgb);
            if (image.width != width || image.height != height) {
              throw std::invalid_argument((*shared)[n].rgb.string() +
                                          ": not the size of the recording's first frame");
            }
            return std::move(image.pixels);
          },
      // The camera's own frame numbers, made nonzero.
      .sequence = [shared](std::uint32_t n) { return (*shared)[n].device_sequence + 1U; }};
}

FrameSourceOp::FrameSourceOp(FrameSourceConfig config) : config_(std::move(config)) {
  if (config_.width <= 0 || config_.height <= 0 || config_.width > kMaxSide ||
      config_.height > kMaxSide) {
    throw std::invalid_argument("frames must be 1x1 to 4096x4096");
  }
  if (config_.count == 0 || config_.period <= std::chrono::nanoseconds::zero() || !config_.render ||
      !config_.sequence) {
    throw std::invalid_argument("a frame source needs a positive count and period, and frames");
  }
}

void FrameSourceOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(camera_info, "camera_info");
  rgb_output(
      spec, image,
      config_.device_output ? holoscan::MemoryKind::kCudaDevice : holoscan::MemoryKind::kHost,
      config_.width, config_.height);
  spec.output(image_camera_info, "image_camera_info");
}

holoscan::Contract FrameSourceOp::contract() const {
  // A frame on every tick, with the newest camera info, if any (none ever holds a tick back).
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  result.input(camera_info, holoscan::LatestWithin{
                                .max_age = holoscan::messages::CameraInfoInputOptions{}.max_age});
  result.on_missing(camera_info, holoscan::SafeDefault{});
  return result;
}

void FrameSourceOp::start() {
  info_.reset();
  emitted_ = 0;
  pending_.reset();
  pixels_.clear();
  sent_ = 0;
}

holoscan::expected<void, holoscan::Error> FrameSourceOp::compute(
    holoscan::ExecutionContext& context) {
  auto info = info_.update(camera_info);
  if (!info) return holoscan::make_unexpected(std::move(info).error());
  if (*info == nullptr || emitted_ >= config_.count) {
    return {};  // No camera info yet; or done, idle until the reports complete the run.
  }
  if (info_.changed()) message_ = holoscan::messages::to_message(**info);
  if (!pending_) {
    pending_ = context.activation_time();
    sent_ = 0;
    try {
      pixels_ = config_.render(emitted_);
    } catch (const std::exception& error) {
      return holoscan::make_unexpected(failure(error.what()));
    }
    if (pixels_.size() != static_cast<std::size_t>(config_.width) * config_.height * 3U) {
      return holoscan::make_unexpected(failure("a frame has an unexpected size"));
    }
  }
  const holoscan::EmitOptions emit{.capture_time = pending_, .frame_id = emitted_ + 1U};
  bool blocked = false;
  // Records one part of the frame as sent, or that consumers are busy (to retry next tick);
  // any other failure fails the run.
  const auto settle = [&](holoscan::expected<void, holoscan::Error> sent,
                          std::uint32_t part) -> holoscan::expected<void, holoscan::Error> {
    if (sent) {
      sent_ |= part;
      return {};
    }
    if (backpressured(sent.error())) {
      blocked = true;
      return {};
    }
    return sent;
  };
  if ((sent_ & kSentCameraInfo) == 0) {
    message_.header->capture_timestamp_ns = pending_->timestamp_ns;
    if (auto r = settle(image_camera_info.emit_ref(message_, emit), kSentCameraInfo); !r) {
      return r;
    }
  }
  if ((sent_ & kSentImage) == 0 && !blocked) {
    const std::array<std::int64_t, 3> shape{config_.height, config_.width, 3};
    auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
    if (!loan) {
      if (auto r = settle(holoscan::make_unexpected(std::move(loan).error()), kSentImage); !r) {
        return r;
      }
    } else {
      if (config_.device_output) {
        auto writer = loan->write(context.cuda_stream());
        if (!writer) return holoscan::make_unexpected(std::move(writer).error());
        if (writer->byte_size() != pixels_.size()) {
          return holoscan::make_unexpected(failure("image loan has an unexpected size"));
        }
        const cudaError_t copied =
            cudaMemcpyAsync(writer->data_as<std::uint8_t>(), pixels_.data(), pixels_.size(),
                            cudaMemcpyHostToDevice, context.cuda_stream().get());
        if (copied != cudaSuccess) {
          return holoscan::make_unexpected(failure(cudaGetErrorString(copied)));
        }
        if (auto committed = std::move(*writer).commit(); !committed) {
          return holoscan::make_unexpected(std::move(committed).error());
        }
      } else {
        auto committed =
            loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                            -> holoscan::expected<void, holoscan::Error> {
              if (writer.byte_size() != pixels_.size()) {
                return holoscan::make_unexpected(failure("image loan has an unexpected size"));
              }
              std::memcpy(writer.data_as<std::uint8_t>(), pixels_.data(), pixels_.size());
              return {};
            });
        if (!committed) return holoscan::make_unexpected(std::move(committed).error());
      }
      holoscan::schema::ImageT descriptor = rgb_descriptor(config_.width, config_.height);
      descriptor.header = std::make_shared<holoscan::schema::HeaderT>();
      descriptor.header->device_sequence = config_.sequence(emitted_);
      descriptor.header->capture_timestamp_ns = pending_->timestamp_ns;
      if (auto r = settle(image.emit_tensor(std::move(*loan), descriptor, emit), kSentImage); !r) {
        return r;
      }
    }
  }
  if (sent_ == (kSentCameraInfo | kSentImage)) {
    ++emitted_;
    pending_.reset();
  }
  return {};
}

void check_camera(const CameraSourceConfig& config) {
  std::size_t pitch = 0;
  close(open_camera(config, pitch));
}

CameraSourceOp::~CameraSourceOp() { release(); }

void CameraSourceOp::release() noexcept {
  for (const Mapping& buffer : buffers_) munmap(buffer.data, buffer.length);
  buffers_.clear();
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
}

void CameraSourceOp::setup(holoscan::OperatorSpec& spec) {
  rgb_output(spec, image, holoscan::MemoryKind::kHost, config_.width, config_.height);
}

holoscan::Contract CameraSourceOp::contract() const {
  // Poll faster than any frame rate; a poll without a new frame does nothing.
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = kCameraPoll});
  return result;
}

void CameraSourceOp::start() {
  release();
  try {
    fd_ = open_camera(config_, pitch_);
  } catch (const std::exception& error) {
    HOLOSCAN_LOG_ERROR("CameraSourceOp: {}", error.what());
    throw;
  }
  emitted_ = 0;
  idle_ticks_ = 0;
  short_frame_logged_ = false;
  v4l2_requestbuffers request{};
  request.count = kCameraBuffers;
  request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  request.memory = V4L2_MEMORY_MMAP;
  if (xioctl(fd_, VIDIOC_REQBUFS, &request) != 0 || request.count == 0) {
    camera_failure(config_.device, "VIDIOC_REQBUFS");
  }
  for (std::uint32_t i = 0; i < request.count; ++i) {
    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = i;
    if (xioctl(fd_, VIDIOC_QUERYBUF, &buffer) != 0)
      camera_failure(config_.device, "VIDIOC_QUERYBUF");
    void* data =
        mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buffer.m.offset);
    if (data == MAP_FAILED) camera_failure(config_.device, "mmap of a camera buffer");
    buffers_.push_back({data, buffer.length});
    if (xioctl(fd_, VIDIOC_QBUF, &buffer) != 0) camera_failure(config_.device, "VIDIOC_QBUF");
  }
  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (xioctl(fd_, VIDIOC_STREAMON, &type) != 0) camera_failure(config_.device, "VIDIOC_STREAMON");
}

void CameraSourceOp::stop() {
  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  xioctl(fd_, VIDIOC_STREAMOFF, &type);
  release();
}

holoscan::expected<void, holoscan::Error> CameraSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (config_.count != 0 && emitted_ >= config_.count) {
    // The count is reached: complete the run once the last frames have passed through, which
    // drop-oldest edges never report.
    if (++idle_ticks_ * kCameraPoll >= kCameraDrain) context.request_completion();
    return {};
  }
  v4l2_buffer buffer{};
  buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buffer.memory = V4L2_MEMORY_MMAP;
  if (xioctl(fd_, VIDIOC_DQBUF, &buffer) != 0) {
    if (errno == EAGAIN) return {};  // No new frame yet.
    return holoscan::make_unexpected(failure("VIDIOC_DQBUF: " + std::string(std::strerror(errno))));
  }
  // Frames that queued up while we were busy are stale: keep only the newest.
  for (v4l2_buffer newer = buffer; xioctl(fd_, VIDIOC_DQBUF, &newer) == 0; buffer = newer) {
    if (xioctl(fd_, VIDIOC_QBUF, &buffer) != 0) {
      return holoscan::make_unexpected(
          failure("VIDIOC_QBUF: " + std::string(std::strerror(errno))));
    }
  }
  // The frame's capture time on the graph's clock: now, less its age on the monotonic clock
  // V4L2 stamps it with.
  holoscan::TimePoint captured = context.activation_time();
  if ((buffer.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC) {
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    const std::int64_t age = (now.tv_sec - buffer.timestamp.tv_sec) * 1'000'000'000LL +
                             (now.tv_nsec - buffer.timestamp.tv_usec * 1000LL);
    if (age > 0) captured.timestamp_ns -= age;
  }
  const auto* yuyv = static_cast<const std::uint8_t*>(buffers_[buffer.index].data);
  // A frame shorter than its rows (a driver's partial frame) is not converted.
  if (buffer.bytesused != 0 &&
      buffer.bytesused < pitch_ * static_cast<std::size_t>(config_.height)) {
    if (!short_frame_logged_) {
      short_frame_logged_ = true;
      HOLOSCAN_LOG_WARN("CameraSourceOp: {}: dropped a frame of {} bytes, short of {} rows of {}",
                        config_.device, buffer.bytesused, config_.height, pitch_);
    }
    if (xioctl(fd_, VIDIOC_QBUF, &buffer) != 0) {
      return holoscan::make_unexpected(
          failure("VIDIOC_QBUF: " + std::string(std::strerror(errno))));
    }
    return {};
  }
  holoscan::schema::ImageT descriptor = rgb_descriptor(config_.width, config_.height);
  descriptor.header = std::make_shared<holoscan::schema::HeaderT>();
  // The camera's frame counter, made nonzero (zero means unnumbered).
  descriptor.header->device_sequence = std::uint64_t{buffer.sequence} + 1U;
  descriptor.header->capture_timestamp_ns = captured.timestamp_ns;
  holoscan::expected<void, holoscan::Error> emitted;
  auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::array<std::int64_t, 3>{config_.height, config_.width, 3}, .dtype = kUInt8});
  if (!loan) {
    emitted = holoscan::make_unexpected(std::move(loan).error());
  } else if (auto committed =
                 loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                 -> holoscan::expected<void, holoscan::Error> {
                   yuyv_to_rgb(yuyv, pitch_, config_.width, config_.height,
                               writer.data_as<std::uint8_t>());
                   return {};
                 });
             !committed) {
    emitted = committed;
  } else {
    emitted = image.emit_tensor(
        std::move(*loan), descriptor,
        holoscan::EmitOptions{.capture_time = captured, .frame_id = emitted_ + 1U});
  }
  // The buffer goes back to the camera whatever happened to the frame.
  if (xioctl(fd_, VIDIOC_QBUF, &buffer) != 0) {
    return holoscan::make_unexpected(failure("VIDIOC_QBUF: " + std::string(std::strerror(errno))));
  }
  if (emitted) {
    ++emitted_;
    return {};
  }
  // A live frame is not worth waiting for: drop one the detectors have no room for.
  return backpressured(emitted.error()) ? holoscan::expected<void, holoscan::Error>{} : emitted;
}

}  // namespace fiducial_app
