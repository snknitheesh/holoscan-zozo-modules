// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The v4l2-sbs backend: a V4L2 camera that sends both eyes side by side in one
// frame, as a ZED does, split into left and right, with the calibration from a
// file.

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/stereo_camera/side_by_side.hpp>
#include <holoscan/stereo_camera/source.hpp>
// clang-format groups these Linux headers with the library's; cpplint wants them first.
#include <linux/videodev2.h>  // NOLINT(build/include_order)
#include <sys/ioctl.h>        // NOLINT(build/include_order)
#include <sys/mman.h>         // NOLINT(build/include_order)

#include "sources.hpp"

namespace holoscan::stereo_camera::detail {
namespace {

constexpr std::uint32_t kBuffers = 4;

int xioctl(int fd, unsigned long request, void* argument) {  // NOLINT(runtime/int): ioctl's type.
  int result = 0;
  do {
    result = ioctl(fd, request, argument);
  } while (result == -1 && errno == EINTR);
  return result;
}

std::string errno_text() { return std::strerror(errno); }

class V4l2SideBySideSource final : public StereoSource {
 public:
  explicit V4l2SideBySideSource(const SourceConfig& config)
      : options_(config.v4l2), calibration_(read_calibration(config.v4l2.calibration_file)) {
    layout_.left = StreamLayout{options_.width / 2, options_.height, options_.output};
    layout_.right = layout_.left;
    layout_.live = true;
  }
  ~V4l2SideBySideSource() override { close(); }

  [[nodiscard]] Backend backend() const noexcept override { return Backend::kV4l2SideBySide; }
  [[nodiscard]] const SourceLayout& layout() const noexcept override { return layout_; }

  [[nodiscard]] const StereoCalibration& calibration() const override {
    if (fd_ < 0) {
      throw std::logic_error("v4l2-sbs: calibration() before open()");
    }
    return calibration_;
  }

  void open() override {
    close();
    const std::string name = options_.device.string();
    fd_ = ::open(name.c_str(), O_RDWR | O_NONBLOCK);
    if (fd_ < 0) {
      throw std::invalid_argument("v4l2-sbs: cannot open " + name + ": " + errno_text());
    }
    try {
      configure(name);
      start_streaming();
    } catch (...) {
      close();
      throw;
    }
    sequence_ = 0;
  }

  [[nodiscard]] std::optional<StereoFrameView> grab() override {
    if (fd_ < 0) {
      throw std::logic_error("v4l2-sbs: grab() before open()");
    }
    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_DQBUF, &buffer) != 0) {
      if (errno == EAGAIN) {
        return std::nullopt;
      }
      throw std::runtime_error("v4l2-sbs: VIDIOC_DQBUF: " + errno_text());
    }
    // Frames that queued up meanwhile are stale: keep only the newest.
    for (v4l2_buffer newer = buffer; xioctl(fd_, VIDIOC_DQBUF, &newer) == 0; buffer = newer) {
      if (xioctl(fd_, VIDIOC_QBUF, &buffer) != 0) {
        throw std::runtime_error("v4l2-sbs: VIDIOC_QBUF: " + errno_text());
      }
    }
    const auto* data = static_cast<const std::uint8_t*>(mappings_[buffer.index].data);
    try {
      split_side_by_side(data, options_.width, options_.height, pitch_, options_.format,
                         options_.output, left_, right_);
    } catch (...) {
      xioctl(fd_, VIDIOC_QBUF, &buffer);
      throw;
    }
    // The buffer goes back as soon as it is copied.
    if (xioctl(fd_, VIDIOC_QBUF, &buffer) != 0) {
      throw std::runtime_error("v4l2-sbs: VIDIOC_QBUF: " + errno_text());
    }
    const std::int64_t stamp =
        static_cast<std::int64_t>(buffer.timestamp.tv_sec) * 1'000'000'000LL +
        static_cast<std::int64_t>(buffer.timestamp.tv_usec) * 1000LL;
    StereoFrameView frame{.left = left_.view(),
                          .right = right_.view(),
                          .index = sequence_++,
                          .timestamp_ns = stamp,
                          .device_sequence = buffer.sequence};
    if ((buffer.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC) {
      timespec now{};
      clock_gettime(CLOCK_MONOTONIC, &now);
      const std::int64_t age = static_cast<std::int64_t>(now.tv_sec) * 1'000'000'000LL +
                               static_cast<std::int64_t>(now.tv_nsec) - stamp;
      if (age >= 0) frame.age_ns = age;
    }
    return frame;
  }

  void close() noexcept override {
    if (fd_ < 0) {
      return;
    }
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(fd_, VIDIOC_STREAMOFF, &type);
    for (const Mapping& mapping : mappings_) {
      munmap(mapping.data, mapping.length);
    }
    mappings_.clear();
    v4l2_requestbuffers request{};
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    xioctl(fd_, VIDIOC_REQBUFS, &request);
    ::close(fd_);
    fd_ = -1;
  }

 private:
  struct Mapping {
    void* data{};
    std::size_t length{};
  };

  void configure(const std::string& name) {
    v4l2_capability capability{};
    if (xioctl(fd_, VIDIOC_QUERYCAP, &capability) != 0 ||
        (capability.device_caps & V4L2_CAP_VIDEO_CAPTURE) == 0 ||
        (capability.device_caps & V4L2_CAP_STREAMING) == 0) {
      throw std::invalid_argument("v4l2-sbs: " + name + " is not a V4L2 capture device");
    }
    const std::uint32_t fourcc =
        options_.format == SideBySideFormat::kYuyv ? V4L2_PIX_FMT_YUYV : V4L2_PIX_FMT_GREY;
    v4l2_format format{};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = static_cast<std::uint32_t>(options_.width);
    format.fmt.pix.height = static_cast<std::uint32_t>(options_.height);
    format.fmt.pix.pixelformat = fourcc;
    format.fmt.pix.field = V4L2_FIELD_NONE;
    const std::string wanted = std::string(side_by_side_format_name(options_.format)) + " " +
                               std::to_string(options_.width) + "x" +
                               std::to_string(options_.height);
    if (xioctl(fd_, VIDIOC_S_FMT, &format) != 0) {
      throw std::invalid_argument("v4l2-sbs: cannot set " + wanted + " on " + name + ": " +
                                  errno_text());
    }
    if (format.fmt.pix.pixelformat != fourcc ||
        format.fmt.pix.width != static_cast<std::uint32_t>(options_.width) ||
        format.fmt.pix.height != static_cast<std::uint32_t>(options_.height)) {
      throw std::invalid_argument("v4l2-sbs: " + name + " does not offer " + wanted);
    }
    pitch_ = format.fmt.pix.bytesperline;
    v4l2_streamparm parameters{};
    parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    v4l2_fract& period = parameters.parm.capture.timeperframe;  // codespell:ignore parm
    period.numerator = 1;
    period.denominator = static_cast<std::uint32_t>(options_.fps);
    // A camera without frame-rate control keeps its own rate; that is not an error.
    xioctl(fd_, VIDIOC_S_PARM, &parameters);
  }

  void start_streaming() {
    v4l2_requestbuffers request{};
    request.count = kBuffers;
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_REQBUFS, &request) != 0 || request.count == 0) {
      throw std::runtime_error("v4l2-sbs: VIDIOC_REQBUFS: " + errno_text());
    }
    for (std::uint32_t i = 0; i < request.count; ++i) {
      v4l2_buffer buffer{};
      buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buffer.memory = V4L2_MEMORY_MMAP;
      buffer.index = i;
      if (xioctl(fd_, VIDIOC_QUERYBUF, &buffer) != 0) {
        throw std::runtime_error("v4l2-sbs: VIDIOC_QUERYBUF: " + errno_text());
      }
      void* data =
          mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buffer.m.offset);
      if (data == MAP_FAILED) {
        throw std::runtime_error("v4l2-sbs: mmap: " + errno_text());
      }
      mappings_.push_back({data, buffer.length});
      if (buffer.length < pitch_ * static_cast<std::size_t>(options_.height)) {
        throw std::runtime_error("v4l2-sbs: the driver's buffers are smaller than a frame");
      }
      if (xioctl(fd_, VIDIOC_QBUF, &buffer) != 0) {
        throw std::runtime_error("v4l2-sbs: VIDIOC_QBUF: " + errno_text());
      }
    }
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_STREAMON, &type) != 0) {
      throw std::runtime_error("v4l2-sbs: VIDIOC_STREAMON: " + errno_text());
    }
  }

  V4l2SideBySideOptions options_;
  StereoCalibration calibration_;
  SourceLayout layout_;
  int fd_{-1};
  std::size_t pitch_{};
  std::vector<Mapping> mappings_;
  std::uint64_t sequence_{};
  Image left_;
  Image right_;
};

}  // namespace

std::unique_ptr<StereoSource> create_v4l2_source(const SourceConfig& config) {
  return std::make_unique<V4l2SideBySideSource>(config);
}

}  // namespace holoscan::stereo_camera::detail
