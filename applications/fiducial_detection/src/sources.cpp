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
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <linux/videodev2.h>  // NOLINT(build/include_order)
#include <sys/ioctl.h>        // NOLINT(build/include_order)
#include <sys/mman.h>         // NOLINT(build/include_order)

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

// A busy consumer refuses a frame with kBackpressured before anything is delivered.
bool backpressured(const holoscan::expected<void, holoscan::Error>& emitted) {
  return !emitted && emitted.error().code == holoscan::ErrorCode::kBackpressured;
}

}  // namespace

fd::SyntheticScene make_scene(const SceneKind& kind, std::uint32_t frame, int width, int height) {
  return kind.board ? fd::make_board_scene(kind.board_config, frame, width, height)
                    : fd::make_marker_scene(kind.dictionary, frame, width, height);
}

SceneSourceOp::SceneSourceOp(SceneSourceConfig config) : config_(std::move(config)) {
  (void)make_scene(config_.kind, 0, config_.width, config_.height);  // Dictionary, board, size.
  if (config_.width > kMaxSide || config_.height > kMaxSide) {
    throw std::invalid_argument("synthetic images must be 320x240 to 4096x4096");
  }
  if (config_.count == 0 || config_.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("a scene source needs a positive count and period");
  }
}

void SceneSourceOp::setup(holoscan::OperatorSpec& spec) {
  rgb_output(
      spec, image,
      config_.device_output ? holoscan::MemoryKind::kCudaDevice : holoscan::MemoryKind::kHost,
      config_.width, config_.height);
}

holoscan::Contract SceneSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void SceneSourceOp::start() { emitted_ = 0; }

holoscan::expected<void, holoscan::Error> SceneSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= config_.count) return {};  // Done: idle until the reports complete the run.
  const std::vector<std::uint8_t> pixels =
      fd::render_scene(make_scene(config_.kind, emitted_, config_.width, config_.height), 3);
  const std::array<std::int64_t, 3> shape{config_.height, config_.width, 3};
  auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
  if (!loan) return holoscan::make_unexpected(std::move(loan).error());
  if (config_.device_output) {
    auto writer = loan->write(context.cuda_stream());
    if (!writer) return holoscan::make_unexpected(std::move(writer).error());
    if (writer->byte_size() != pixels.size()) {
      return holoscan::make_unexpected(failure("image loan has an unexpected size"));
    }
    const cudaError_t copied =
        cudaMemcpyAsync(writer->data_as<std::uint8_t>(), pixels.data(), pixels.size(),
                        cudaMemcpyHostToDevice, context.cuda_stream().get());
    if (copied != cudaSuccess)
      return holoscan::make_unexpected(failure(cudaGetErrorString(copied)));
    if (auto committed = std::move(*writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
  } else {
    auto committed = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                     -> holoscan::expected<void, holoscan::Error> {
      if (writer.byte_size() != pixels.size()) {
        return holoscan::make_unexpected(failure("image loan has an unexpected size"));
      }
      std::memcpy(writer.data_as<std::uint8_t>(), pixels.data(), pixels.size());
      return {};
    });
    if (!committed) return holoscan::make_unexpected(std::move(committed).error());
  }
  holoscan::schema::ImageT descriptor = rgb_descriptor(config_.width, config_.height);
  descriptor.header = std::make_shared<holoscan::schema::HeaderT>();
  descriptor.header->device_sequence = std::uint64_t{emitted_} + 1U;
  auto emitted =
      image.emit_tensor(std::move(*loan), descriptor,
                        holoscan::EmitOptions{.capture_time = context.activation_time()});
  if (emitted) {
    ++emitted_;
    return {};
  }
  // Send the same frame again on the next tick.
  return backpressured(emitted) ? holoscan::expected<void, holoscan::Error>{} : emitted;
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
  result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{5}});
  return result;
}

void CameraSourceOp::start() {
  release();
  fd_ = open_camera(config_, pitch_);
  emitted_ = 0;
  v4l2_requestbuffers request{};
  request.count = kCameraBuffers;
  request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  request.memory = V4L2_MEMORY_MMAP;
  if (xioctl(fd_, VIDIOC_REQBUFS, &request) != 0 || request.count == 0) {
    throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure, "VIDIOC_REQBUFS failed"};
  }
  for (std::uint32_t i = 0; i < request.count; ++i) {
    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = i;
    if (xioctl(fd_, VIDIOC_QUERYBUF, &buffer) != 0) {
      throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure, "VIDIOC_QUERYBUF failed"};
    }
    void* data =
        mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buffer.m.offset);
    if (data == MAP_FAILED) {
      throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure, "mmap of a camera buffer failed"};
    }
    buffers_.push_back({data, buffer.length});
    if (xioctl(fd_, VIDIOC_QBUF, &buffer) != 0) {
      throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure, "VIDIOC_QBUF failed"};
    }
  }
  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (xioctl(fd_, VIDIOC_STREAMON, &type) != 0) {
    throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure, "VIDIOC_STREAMON failed"};
  }
}

void CameraSourceOp::stop() {
  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  xioctl(fd_, VIDIOC_STREAMOFF, &type);
  release();
}

holoscan::expected<void, holoscan::Error> CameraSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (config_.count != 0 && emitted_ >= config_.count) return {};
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
        std::move(*loan), rgb_descriptor(config_.width, config_.height),
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
  return backpressured(emitted) ? holoscan::expected<void, holoscan::Error>{} : emitted;
}

}  // namespace fiducial_app
