// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "v4l2_device.hpp"

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <linux/videodev2.h>  // NOLINT(build/include_order)
#include <sys/mman.h>         // NOLINT(build/include_order)

namespace holoscan::codec::detail {
namespace {

// NVIDIA's libv4l2, which routes the codec devices to its codec plugins.
constexpr const char* kLibrary = "libv4l2.so.0";

struct Library {
  int (*open)(const char*, int, ...) = nullptr;
  int (*close)(int) = nullptr;
  int (*ioctl)(int, unsigned long, ...) = nullptr;  // NOLINT(runtime/int)
  void* (*mmap)(void*, std::size_t, int, int, int, std::int64_t) = nullptr;
  int (*munmap)(void*, std::size_t) = nullptr;
  std::string error;
};

Library load() {
  Library library;
  void* handle = dlopen(kLibrary, RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    const char* message = dlerror();
    library.error = std::string(kLibrary) + ": " + (message != nullptr ? message : "not found");
    return library;
  }
  library.open = reinterpret_cast<decltype(library.open)>(dlsym(handle, "v4l2_open"));
  library.close = reinterpret_cast<decltype(library.close)>(dlsym(handle, "v4l2_close"));
  library.ioctl = reinterpret_cast<decltype(library.ioctl)>(dlsym(handle, "v4l2_ioctl"));
  library.mmap = reinterpret_cast<decltype(library.mmap)>(dlsym(handle, "v4l2_mmap"));
  library.munmap = reinterpret_cast<decltype(library.munmap)>(dlsym(handle, "v4l2_munmap"));
  if (library.open == nullptr || library.close == nullptr || library.ioctl == nullptr ||
      library.mmap == nullptr || library.munmap == nullptr) {
    library.error = std::string(kLibrary) + " lacks the v4l2_* entry points";
  }
  return library;
}

const Library& library() {
  static const Library loaded = load();
  return loaded;
}

constexpr std::uint64_t kMicrosecondsPerSecond = 1'000'000;

}  // namespace

const std::string& v4l2_library_error() { return library().error; }

void throw_errno(const char* what) {
  throw std::runtime_error(std::string(what) + ": " + std::strerror(errno));
}

V4l2Device::V4l2Device(const char* path) {
  const Library& lib = library();
  if (!lib.error.empty()) {
    throw std::runtime_error(lib.error);
  }
  fd_ = lib.open(path, O_RDWR | O_NONBLOCK);
  if (fd_ < 0) {
    throw_errno((std::string("open ") + path).c_str());
  }
}

V4l2Device::~V4l2Device() {
  if (fd_ >= 0) {
    library().close(fd_);
  }
}

int V4l2Device::ioctl(unsigned long request, void* argument) const {  // NOLINT(runtime/int)
  return library().ioctl(fd_, request, argument);
}

void V4l2Device::check_ioctl(unsigned long request, void* argument,  // NOLINT(runtime/int)
                             const char* what) const {
  if (ioctl(request, argument) < 0) {
    throw_errno(what);
  }
}

std::vector<std::uint32_t> V4l2Device::formats(v4l2_buf_type type) const {
  std::vector<std::uint32_t> result;
  v4l2_fmtdesc description{};
  description.type = type;
  while (ioctl(VIDIOC_ENUM_FMT, &description) == 0) {
    result.push_back(description.pixelformat);
    ++description.index;
  }
  return result;
}

void V4l2Device::set_control(std::uint32_t id, std::int32_t value, const char* what) const {
  v4l2_ext_control control{};
  control.id = id;
  control.value = value;
  v4l2_ext_controls controls{};
  controls.ctrl_class = V4L2_CTRL_CLASS_MPEG;
  controls.count = 1;
  controls.controls = &control;
  check_ioctl(VIDIOC_S_EXT_CTRLS, &controls, what);
}

void V4l2Device::set_control(std::uint32_t id, void* data, std::uint32_t size,
                             const char* what) const {
  v4l2_ext_control control{};
  control.id = id;
  control.size = size;
  control.string = static_cast<char*>(data);
  v4l2_ext_controls controls{};
  controls.ctrl_class = V4L2_CTRL_CLASS_MPEG;
  controls.count = 1;
  controls.controls = &control;
  check_ioctl(VIDIOC_S_EXT_CTRLS, &controls, what);
}

std::vector<V4l2Buffer> V4l2Device::request_buffers(v4l2_buf_type type, std::uint32_t count,
                                                    std::uint32_t planes) {
  v4l2_requestbuffers request{};
  request.count = count;
  request.type = type;
  request.memory = V4L2_MEMORY_MMAP;
  check_ioctl(VIDIOC_REQBUFS, &request, "VIDIOC_REQBUFS");
  std::vector<V4l2Buffer> buffers(request.count);
  for (std::uint32_t index = 0; index < request.count; ++index) {
    v4l2_plane plane_info[VIDEO_MAX_PLANES] = {};
    v4l2_buffer buffer{};
    buffer.type = type;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = index;
    buffer.m.planes = plane_info;
    buffer.length = planes;
    check_ioctl(VIDIOC_QUERYBUF, &buffer, "VIDIOC_QUERYBUF");
    V4l2Buffer& mapped = buffers[index];
    mapped.planes.resize(planes);
    for (std::uint32_t p = 0; p < planes; ++p) {
      // The NVIDIA plugins map a buffer through its exported dma-buf, not the device.
      v4l2_exportbuffer exported{};
      exported.type = type;
      exported.index = index;
      exported.plane = p;
      check_ioctl(VIDIOC_EXPBUF, &exported, "VIDIOC_EXPBUF");
      V4l2Buffer::Plane& plane = mapped.planes[p];
      plane.dmabuf = exported.fd;
      plane.length = plane_info[p].length;
      void* data = library().mmap(nullptr, plane.length, PROT_READ | PROT_WRITE, MAP_SHARED,
                                  exported.fd, plane_info[p].m.mem_offset);
      if (data == MAP_FAILED) {
        throw_errno("v4l2_mmap");
      }
      plane.data = static_cast<std::uint8_t*>(data);
    }
  }
  return buffers;
}

void V4l2Device::release_buffers(v4l2_buf_type type, std::vector<V4l2Buffer>& buffers) {
  for (V4l2Buffer& buffer : buffers) {
    for (V4l2Buffer::Plane& plane : buffer.planes) {
      if (plane.data != nullptr) {
        library().munmap(plane.data, plane.length);
      }
      if (plane.dmabuf >= 0) {
        ::close(plane.dmabuf);
      }
    }
  }
  buffers.clear();
  v4l2_requestbuffers request{};
  request.count = 0;
  request.type = type;
  request.memory = V4L2_MEMORY_MMAP;
  ioctl(VIDIOC_REQBUFS, &request);
}

void V4l2Device::queue(v4l2_buf_type type, std::uint32_t index,
                       const std::vector<std::uint32_t>& used, std::uint64_t timestamp_us) {
  v4l2_plane planes[VIDEO_MAX_PLANES] = {};
  for (std::size_t p = 0; p < used.size(); ++p) {
    planes[p].bytesused = used[p];
  }
  v4l2_buffer buffer{};
  buffer.type = type;
  buffer.memory = V4L2_MEMORY_MMAP;
  buffer.index = index;
  buffer.m.planes = planes;
  buffer.length = static_cast<std::uint32_t>(used.size());
  buffer.flags = V4L2_BUF_FLAG_TIMESTAMP_COPY;
  buffer.timestamp.tv_sec = static_cast<time_t>(timestamp_us / kMicrosecondsPerSecond);
  buffer.timestamp.tv_usec = static_cast<suseconds_t>(timestamp_us % kMicrosecondsPerSecond);
  check_ioctl(VIDIOC_QBUF, &buffer, "VIDIOC_QBUF");
}

bool V4l2Device::dequeue(v4l2_buf_type type, std::uint32_t planes, Dequeued& out) const {
  v4l2_plane plane_info[VIDEO_MAX_PLANES] = {};
  v4l2_buffer buffer{};
  buffer.type = type;
  buffer.memory = V4L2_MEMORY_MMAP;
  buffer.m.planes = plane_info;
  buffer.length = planes;
  if (ioctl(VIDIOC_DQBUF, &buffer) < 0) {
    if (errno == EAGAIN) {
      return false;
    }
    throw_errno("VIDIOC_DQBUF");
  }
  out.index = buffer.index;
  out.used.assign(planes, 0);
  for (std::uint32_t p = 0; p < planes; ++p) {
    out.used[p] = plane_info[p].bytesused;
  }
  out.timestamp_us = static_cast<std::uint64_t>(buffer.timestamp.tv_sec) * kMicrosecondsPerSecond +
                     static_cast<std::uint64_t>(buffer.timestamp.tv_usec);
  out.flags = buffer.flags;
  return true;
}

void V4l2Device::stream_on(v4l2_buf_type type) const {
  int value = type;
  check_ioctl(VIDIOC_STREAMON, &value, "VIDIOC_STREAMON");
}

void V4l2Device::stream_off(v4l2_buf_type type) const {
  int value = type;
  ioctl(VIDIOC_STREAMOFF, &value);
}

}  // namespace holoscan::codec::detail
