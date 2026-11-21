// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "image_sources.hpp"

#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/mono_depth/synthetic_scene.hpp>
// clang-format groups these Linux headers with the SDK's; cpplint wants them first.
#include <linux/videodev2.h>  // NOLINT(build/include_order)
#include <sys/ioctl.h>        // NOLINT(build/include_order)
#include <sys/mman.h>         // NOLINT(build/include_order)

#include "yuyv.hpp"

namespace mono_depth_app {
namespace {

using holoscan::mono_depth::make_depth_scene;
using holoscan::mono_depth::render_depth_scene;

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

using Result = holoscan::expected<void, holoscan::Error>;

holoscan::Error failure(std::string message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, std::move(message)};
}

holoscan::TensorOutputSpec image_spec(holoscan::MemoryKind memory, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = memory,
              .dtype = kUInt8,
              .rank = 3U,
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

holoscan::schema::ImageEncoding encoding_of(PixelFormat format) {
  switch (format) {
    case PixelFormat::kBgr8:
      return holoscan::schema::ImageEncoding_BGR8;
    case PixelFormat::kRgba8:
      return holoscan::schema::ImageEncoding_RGBA8;
    case PixelFormat::kBgra8:
      return holoscan::schema::ImageEncoding_BGRA8;
    case PixelFormat::kRgb8:
    default:
      return holoscan::schema::ImageEncoding_RGB8;
  }
}

// Whether `error` says the consumers are still busy with earlier frames: an
// emit fails with kBackpressured before anything is delivered, and an
// allocation with kResourceExhausted while every slot of the output's pool is
// still held, by frames the consumers have not finished with or by a frame
// they refused.
bool consumers_busy(const holoscan::Error& error) {
  return error.code == holoscan::ErrorCode::kBackpressured ||
         error.code == holoscan::ErrorCode::kResourceExhausted;
}

// What compute() returns after allocating and sending a frame: on success
// nothing more to do, the frame is out; when the consumers are busy, the frame
// goes again on the next tick; any other failure fails the run.
Result settle(Result sent, bool& delivered) {
  delivered = static_cast<bool>(sent);
  if (sent || consumers_busy(sent.error())) {
    return {};
  }
  return sent;
}

holoscan::schema::ImageT descriptor(int width, int height, PixelFormat format) {
  holoscan::schema::ImageT image;
  image.width = width;
  image.height = height;
  image.encoding = encoding_of(format);
  return image;
}

}  // namespace

void validate_config(const SyntheticImageSourceConfig& config) {
  // make_depth_scene() checks the size.
  (void)make_depth_scene(config.width, config.height, 0);
  if (config.count == 0 || config.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("the synthetic source needs a positive count and period");
  }
}

SyntheticImageSourceOp::SyntheticImageSourceOp(SyntheticImageSourceConfig config)
    : config_(std::move(config)) {
  validate_config(config_);
}

void SyntheticImageSourceOp::setup(holoscan::OperatorSpec& spec) {
  const std::uint64_t bytes = static_cast<std::uint64_t>(config_.width) * config_.height *
                              static_cast<std::uint64_t>(channels(config_.format));
  spec.output(image, "image")
      .produces_tensor(image_spec(
          config_.device_output ? holoscan::MemoryKind::kCudaDevice : holoscan::MemoryKind::kHost,
          bytes))
      .payload_options(image_payload());
}

holoscan::Contract SyntheticImageSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void SyntheticImageSourceOp::start() {
  emitted_ = 0;
  pending_capture_.reset();
}

holoscan::expected<void, holoscan::Error> SyntheticImageSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= config_.count) {
    return {};  // Done: idle until the run completes.
  }
  const int n = channels(config_.format);
  const std::size_t pitch = static_cast<std::size_t>(config_.width) * static_cast<std::size_t>(n);
  // A still scene is drawn once. The frame is captured when it is drawn,
  // which takes a while on the CPU, and not before.
  if (!config_.still || pixels_.empty()) {
    pixels_.resize(pitch * static_cast<std::size_t>(config_.height));
    render_depth_scene(
        make_depth_scene(config_.width, config_.height, config_.still ? 0U : emitted_),
        config_.format, pixels_.data(), pitch);
  }
  if (!pending_capture_) {
    auto now = context.clock().now();
    pending_capture_ = now ? *now : context.activation_time();
  }
  auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::array<std::int64_t, 3>{config_.height, config_.width, n}, .dtype = kUInt8});
  if (!loan) {
    bool unused = false;
    return settle(holoscan::make_unexpected(std::move(loan).error()), unused);
  }
  if (config_.device_output) {
    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return holoscan::make_unexpected(
          failure("SyntheticImageSourceOp: the runtime gave no CUDA stream for the device output"));
    }
    auto writer = loan->write(stream);
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    // From pageable memory the copy is staged before it returns, so the
    // pixels may change afterwards.
    if (!copy_logged_) {
      copy_logged_ = true;
      HOLOSCAN_LOG_INFO(
          "SyntheticImageSourceOp: each frame is drawn on the CPU and copied to the GPU from "
          "pageable memory, a copy that waits until the driver has staged it");
    }
    if (const cudaError_t status = cudaMemcpyAsync(writer->data_as<std::uint8_t>(), pixels_.data(),
                                                   pixels_.size(), cudaMemcpyHostToDevice, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(
          failure(std::string("SyntheticImageSourceOp: ") + cudaGetErrorString(status)));
    }
    if (auto committed = std::move(*writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
  } else {
    auto committed = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                     -> holoscan::expected<void, holoscan::Error> {
      if (writer.byte_size() != pixels_.size()) {
        return holoscan::make_unexpected(failure("image loan has an unexpected size"));
      }
      std::memcpy(writer.data_as<std::uint8_t>(), pixels_.data(), pixels_.size());
      return {};
    });
    if (!committed) {
      return committed;
    }
  }
  bool delivered = false;
  const Result result = settle(
      image.emit_tensor(
          std::move(*loan), descriptor(config_.width, config_.height, config_.format),
          holoscan::EmitOptions{.capture_time = *pending_capture_, .frame_id = emitted_ + 1U}),
      delivered);
  if (delivered) {
    pending_capture_.reset();
    ++emitted_;
  }
  return result;
}

void validate_config(const ImageFileSourceConfig& config) {
  if (config.files.empty()) {
    throw std::invalid_argument("the file source needs at least one image");
  }
  if (config.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("the file source needs a positive period");
  }
  if (!config.loop && config.count > config.files.size()) {
    throw std::invalid_argument("--count exceeds the number of images; add --loop true");
  }
  if (config.infos.size() != config.files.size()) {
    throw std::invalid_argument("the file source needs each file's info (read_image_info())");
  }
  for (const ImageFileInfo& info : config.infos) {
    if (info.width < 1 || info.height < 1 || info.width > kMaxImageSide ||
        info.height > kMaxImageSide) {
      throw std::invalid_argument("the file source's images must be 1.." +
                                  std::to_string(kMaxImageSide) + " pixels a side");
    }
  }
}

ImageFileSourceOp::ImageFileSourceOp(ImageFileSourceConfig config) : config_(std::move(config)) {
  validate_config(config_);
  infos_ = config_.infos;
  for (const ImageFileInfo& info : infos_) {
    max_bytes_ = std::max<std::uint64_t>(max_bytes_,
                                         static_cast<std::uint64_t>(info.width) * info.height * 3U);
  }
  total_ = config_.count == 0 ? static_cast<std::uint32_t>(config_.files.size()) : config_.count;
}

ImageFileSourceOp::~ImageFileSourceOp() { stop(); }

void ImageFileSourceOp::setup(holoscan::OperatorSpec& spec) {
  spec.output(image, "image")
      .produces_tensor(image_spec(holoscan::MemoryKind::kCudaDevice, max_bytes_))
      .payload_options(image_payload());
}

holoscan::Contract ImageFileSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void ImageFileSourceOp::start() {
  stop();
  emitted_ = 0;
  pending_capture_.reset();
  const bool any_jpeg = std::any_of(infos_.begin(), infos_.end(), [](const ImageFileInfo& info) {
    return info.kind == ImageFileKind::kJpeg;
  });
  if (any_jpeg) {
    decoder_ = std::make_unique<JpegDecoder>();
    if (const cudaError_t status = cudaEventCreateWithFlags(&decoded_, cudaEventDisableTiming);
        status != cudaSuccess) {
      decoded_ = nullptr;
      HOLOSCAN_LOG_ERROR("ImageFileSourceOp: cudaEventCreateWithFlags: {}",
                         cudaGetErrorString(status));
      throw std::runtime_error(std::string("ImageFileSourceOp: cudaEventCreateWithFlags: ") +
                               cudaGetErrorString(status));
    }
  }
}

void ImageFileSourceOp::stop() {
  // The last decode may still read the bytes it was given.
  if (decoded_ != nullptr) {
    cudaEventSynchronize(decoded_);
    cudaEventDestroy(decoded_);
    decoded_ = nullptr;
  }
  jpeg_.clear();
  decoder_.reset();
}

holoscan::expected<void, holoscan::Error> ImageFileSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= total_) {
    return {};
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(
        failure("ImageFileSourceOp: the runtime gave no CUDA stream for the device output"));
  }
  const std::size_t index = emitted_ % config_.files.size();
  const std::filesystem::path& path = config_.files[index];
  const ImageFileInfo& info = infos_[index];
  const PixelFormat format = config_.bgr ? PixelFormat::kBgr8 : PixelFormat::kRgb8;
  if (!pending_capture_) {
    pending_capture_ = context.activation_time();
  }
  auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::array<std::int64_t, 3>{info.height, info.width, 3}, .dtype = kUInt8});
  if (!loan) {
    bool unused = false;
    return settle(holoscan::make_unexpected(std::move(loan).error()), unused);
  }
  auto writer = loan->write(stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  const std::size_t pitch = static_cast<std::size_t>(info.width) * 3U;
  try {
    if (info.kind == ImageFileKind::kJpeg) {
      if (!decoder_ || decoded_ == nullptr) {
        return holoscan::make_unexpected(failure("ImageFileSourceOp: not started"));
      }
      // The previous decode may still read the bytes: wait for it (long done
      // by now, a period later) before they are replaced.
      if (!jpeg_logged_) {
        jpeg_logged_ = true;
        HOLOSCAN_LOG_INFO(
            "ImageFileSourceOp: JPEG files are decoded on the GPU; each frame waits for the "
            "previous decode before reading the next file");
      }
      if (const cudaError_t status = cudaEventSynchronize(decoded_); status != cudaSuccess) {
        return holoscan::make_unexpected(
            failure(path.string() + ": " + cudaGetErrorString(status)));
      }
      jpeg_ = read_file(path);
      const std::string error = decoder_->decode(jpeg_, info.width, info.height, config_.bgr,
                                                 writer->data_as<std::uint8_t>(), pitch, stream);
      if (!error.empty()) {
        return holoscan::make_unexpected(failure(path.string() + ": " + error));
      }
      if (const cudaError_t status = cudaEventRecord(decoded_, stream); status != cudaSuccess) {
        return holoscan::make_unexpected(
            failure(path.string() + ": " + cudaGetErrorString(status)));
      }
    } else {
      int width = 0;
      int height = 0;
      std::vector<std::uint8_t> rgb = read_ppm(path, width, height);
      if (width != info.width || height != info.height) {
        return holoscan::make_unexpected(failure(path.string() + ": changed size during the run"));
      }
      if (config_.bgr) {
        for (std::size_t i = 0; i + 2 < rgb.size(); i += 3) {
          std::swap(rgb[i], rgb[i + 2]);
        }
      }
      // Copied from pageable memory: staged before the call returns.
      if (!ppm_logged_) {
        ppm_logged_ = true;
        HOLOSCAN_LOG_INFO(
            "ImageFileSourceOp: PPM files are read on the CPU and copied to the GPU from pageable "
            "memory, a copy that waits until the driver has staged it");
      }
      if (const cudaError_t status = cudaMemcpyAsync(writer->data_as<std::uint8_t>(), rgb.data(),
                                                     rgb.size(), cudaMemcpyHostToDevice, stream);
          status != cudaSuccess) {
        return holoscan::make_unexpected(
            failure(path.string() + ": " + cudaGetErrorString(status)));
      }
    }
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(failure(error.what()));
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  bool delivered = false;
  const Result result = settle(
      image.emit_tensor(
          std::move(*loan), descriptor(info.width, info.height, format),
          holoscan::EmitOptions{.capture_time = *pending_capture_, .frame_id = emitted_ + 1U}),
      delivered);
  if (delivered) {
    pending_capture_.reset();
    ++emitted_;
  }
  return result;
}

namespace {

int xioctl(int fd, std::uint64_t request, void* argument) {
  int result = 0;
  do {
    result = ioctl(fd, request, argument);
  } while (result == -1 && errno == EINTR);
  return result;
}

// The YUYV frame sizes `fd` offers, as "640x480, 1280x720".
std::string yuyv_sizes(int fd) {
  std::string sizes;
  v4l2_frmsizeenum size{};
  size.pixel_format = V4L2_PIX_FMT_YUYV;
  for (; xioctl(fd, VIDIOC_ENUM_FRAMESIZES, &size) == 0; ++size.index) {
    if (size.type != V4L2_FRMSIZE_TYPE_DISCRETE) {
      return "any up to " + std::to_string(size.stepwise.max_width) + "x" +
             std::to_string(size.stepwise.max_height);
    }
    sizes += (sizes.empty() ? "" : ", ") + std::to_string(size.discrete.width) + "x" +
             std::to_string(size.discrete.height);
  }
  return sizes.empty() ? "none" : sizes;
}

constexpr std::uint32_t kCameraBuffers = 4;

// Opens `config.device` and sets YUYV at the configured size; returns the
// descriptor and the row pitch.
int open_camera(const CameraSourceConfig& config, std::size_t& pitch) {
  // YUYV pairs pixels, so the width is even.
  if (config.width <= 0 || config.height <= 0 || config.width % 2 != 0 ||
      config.width > kMaxImageSide || config.height > kMaxImageSide) {
    throw std::invalid_argument("the camera size needs an even width and sides up to " +
                                std::to_string(kMaxImageSide));
  }
  const std::string name = config.device.string();
  const int fd = open(name.c_str(), O_RDWR | O_NONBLOCK);
  if (fd < 0) {
    throw std::invalid_argument("cannot open camera " + name + ": " + std::strerror(errno));
  }
  v4l2_capability capability{};
  if (xioctl(fd, VIDIOC_QUERYCAP, &capability) != 0 ||
      (capability.device_caps & V4L2_CAP_VIDEO_CAPTURE) == 0 ||
      (capability.device_caps & V4L2_CAP_STREAMING) == 0) {
    close(fd);
    throw std::invalid_argument(name + " is not a V4L2 capture device with streaming");
  }
  v4l2_format format{};
  format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  format.fmt.pix.width = static_cast<std::uint32_t>(config.width);
  format.fmt.pix.height = static_cast<std::uint32_t>(config.height);
  format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
  format.fmt.pix.field = V4L2_FIELD_NONE;
  if (xioctl(fd, VIDIOC_S_FMT, &format) != 0) {
    const int error = errno;
    close(fd);
    throw std::invalid_argument("cannot set YUYV " + std::to_string(config.width) + "x" +
                                std::to_string(config.height) + " on " + name + ": " +
                                std::strerror(error));
  }
  if (format.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV ||
      format.fmt.pix.width != static_cast<std::uint32_t>(config.width) ||
      format.fmt.pix.height != static_cast<std::uint32_t>(config.height)) {
    const std::string offered = yuyv_sizes(fd);
    close(fd);
    throw std::invalid_argument(name + " does not offer YUYV at " + std::to_string(config.width) +
                                "x" + std::to_string(config.height) +
                                "; its YUYV sizes: " + offered);
  }
  pitch = format.fmt.pix.bytesperline;
  return fd;
}

}  // namespace

void check_camera(const CameraSourceConfig& config) {
  std::size_t pitch = 0;
  close(open_camera(config, pitch));
}

CameraSourceOp::CameraSourceOp(CameraSourceConfig config) : config_(std::move(config)) {}

CameraSourceOp::~CameraSourceOp() { release(); }

void CameraSourceOp::unmap() noexcept {
  for (const Mapping& buffer : buffers_) {
    munmap(buffer.data, buffer.length);
  }
  buffers_.clear();
}

void CameraSourceOp::release() noexcept {
  unmap();
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
}

void CameraSourceOp::setup(holoscan::OperatorSpec& spec) {
  const std::uint64_t bytes = static_cast<std::uint64_t>(config_.width) * config_.height * 3U;
  spec.output(image, "image")
      .produces_tensor(image_spec(holoscan::MemoryKind::kHost, bytes))
      .payload_options(image_payload());
}

holoscan::Contract CameraSourceOp::contract() const {
  // Poll faster than any frame rate; a poll without a new frame does nothing.
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{5}});
  return result;
}

void CameraSourceOp::start() {
  release();
  // Every failure is logged with its cause before it is thrown.
  const auto fail = [this](const std::string& what) {
    const std::string message =
        "CameraSourceOp: " + config_.device.string() + ": " + what + ": " + std::strerror(errno);
    HOLOSCAN_LOG_ERROR("{}", message);
    release();
    throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure, message};
  };
  try {
    fd_ = open_camera(config_, pitch_);
  } catch (const std::exception& error) {
    HOLOSCAN_LOG_ERROR("CameraSourceOp: {}", error.what());
    throw;
  }
  first_sequence_.reset();
  last_id_ = 0;
  v4l2_requestbuffers request{};
  request.count = kCameraBuffers;
  request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  request.memory = V4L2_MEMORY_MMAP;
  if (xioctl(fd_, VIDIOC_REQBUFS, &request) != 0 || request.count == 0) {
    fail("VIDIOC_REQBUFS failed");
  }
  for (std::uint32_t i = 0; i < request.count; ++i) {
    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = i;
    if (xioctl(fd_, VIDIOC_QUERYBUF, &buffer) != 0) {
      fail("VIDIOC_QUERYBUF failed");
    }
    void* data =
        mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buffer.m.offset);
    if (data == MAP_FAILED) {
      fail("mmap of a camera buffer failed");
    }
    buffers_.push_back({data, buffer.length});
    if (xioctl(fd_, VIDIOC_QBUF, &buffer) != 0) {
      fail("VIDIOC_QBUF failed");
    }
  }
  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (xioctl(fd_, VIDIOC_STREAMON, &type) != 0) {
    fail("VIDIOC_STREAMON failed");
  }
}

void CameraSourceOp::stop() {
  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  xioctl(fd_, VIDIOC_STREAMOFF, &type);
  unmap();
  v4l2_requestbuffers request{};
  request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  request.memory = V4L2_MEMORY_MMAP;
  xioctl(fd_, VIDIOC_REQBUFS, &request);
  release();
}

holoscan::expected<void, holoscan::Error> CameraSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (config_.count != 0 && last_id_ >= config_.count) {
    return {};
  }
  v4l2_buffer buffer{};
  buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buffer.memory = V4L2_MEMORY_MMAP;
  if (xioctl(fd_, VIDIOC_DQBUF, &buffer) != 0) {
    if (errno == EAGAIN) {
      return {};  // No new frame yet.
    }
    return holoscan::make_unexpected(failure("VIDIOC_DQBUF: " + std::string(std::strerror(errno))));
  }
  // Frames that queued up while we were busy are stale: keep only the newest.
  // Their sequence numbers leave a gap, which the reporter counts.
  for (v4l2_buffer newer = buffer; xioctl(fd_, VIDIOC_DQBUF, &newer) == 0; buffer = newer) {
    if (xioctl(fd_, VIDIOC_QBUF, &buffer) != 0) {
      return holoscan::make_unexpected(
          failure("VIDIOC_QBUF: " + std::string(std::strerror(errno))));
    }
  }
  // The frame's id: the camera's frame count since this run's first frame,
  // from 1, so that every frame not sent leaves a gap.
  if (!first_sequence_) {
    first_sequence_ = buffer.sequence;
  }
  const std::uint64_t id = static_cast<std::uint32_t>(buffer.sequence - *first_sequence_) + 1U;
  // The frame's capture time on the graph's clock: the graph clock's now,
  // less the frame's age on the monotonic clock V4L2 stamps it with, both
  // clocks read at the same moment.
  const auto graph_now = context.clock().now();
  timespec now{};
  clock_gettime(CLOCK_MONOTONIC, &now);
  holoscan::TimePoint captured = graph_now ? *graph_now : context.activation_time();
  if ((buffer.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC) {
    const std::int64_t age = (now.tv_sec - buffer.timestamp.tv_sec) * 1'000'000'000LL +
                             (now.tv_nsec - buffer.timestamp.tv_usec * 1000LL);
    if (age > 0) {
      captured.timestamp_ns -= age;
    }
  }
  const auto* yuyv = static_cast<const std::uint8_t*>(buffers_[buffer.index].data);
  Result emitted;
  auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::array<std::int64_t, 3>{config_.height, config_.width, 3}, .dtype = kUInt8});
  if (!loan) {
    emitted = holoscan::make_unexpected(std::move(loan).error());
  } else if (auto committed = loan->write_host_and_commit(
                 [&](const holoscan::TensorOutputWriteGuard& writer) -> Result {
                   yuyv_to_rgb(yuyv, pitch_, config_.width, config_.height,
                               writer.data_as<std::uint8_t>());
                   return {};
                 });
             !committed) {
    emitted = committed;
  } else {
    emitted = image.emit_tensor(std::move(*loan),
                                descriptor(config_.width, config_.height, PixelFormat::kRgb8),
                                holoscan::EmitOptions{.capture_time = captured, .frame_id = id});
  }
  // The buffer goes back to the camera whatever happened to the frame.
  if (xioctl(fd_, VIDIOC_QBUF, &buffer) != 0) {
    return holoscan::make_unexpected(failure("VIDIOC_QBUF: " + std::string(std::strerror(errno))));
  }
  // A live frame is not worth waiting for: one the consumers have no room
  // for is dropped, and its id is a gap the reporter counts as skipped.
  bool delivered = false;
  const Result result = settle(emitted, delivered);
  if (delivered) {
    last_id_ = id;
  }
  return result;
}

}  // namespace mono_depth_app
