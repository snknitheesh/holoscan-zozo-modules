// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A Jetson V4L2 codec device (/dev/v4l2-nvenc, /dev/v4l2-nvdec) through NVIDIA's libv4l2, loaded
// at run time: memory-to-memory, multi-planar, MMAP buffers on both queues. Module-private.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include <linux/videodev2.h>  // NOLINT(build/include_order)

namespace holoscan::codec::detail {

inline constexpr const char* kV4l2EncoderDevice = "/dev/v4l2-nvenc";
inline constexpr const char* kV4l2DecoderDevice = "/dev/v4l2-nvdec";

// Why libv4l2 cannot be loaded, or empty when it can.
[[nodiscard]] const std::string& v4l2_library_error();

// One buffer of a queue, mapped into this process.
struct V4l2Buffer {
  struct Plane {
    std::uint8_t* data{};
    std::size_t length{};
    // The plane's dma-buf, from VIDIOC_EXPBUF: what the buffer is mapped through and synced with.
    int dmabuf{-1};
  };
  std::vector<Plane> planes;
  bool queued{};
};

class V4l2Device {
 public:
  // Opens `path` non-blocking.
  // @throws std::runtime_error when the library or the device cannot be opened.
  explicit V4l2Device(const char* path);
  ~V4l2Device();
  V4l2Device(const V4l2Device&) = delete;
  V4l2Device& operator=(const V4l2Device&) = delete;

  // ioctl through libv4l2; -1 with errno on failure.
  int ioctl(unsigned long request, void* argument) const;  // NOLINT(runtime/int)
  // As ioctl(), throwing "<what>: <strerror>" on failure.
  void check_ioctl(unsigned long request, void* argument, const char* what) const;  // NOLINT

  // The formats `type` offers, as fourcc codes.
  [[nodiscard]] std::vector<std::uint32_t> formats(v4l2_buf_type type) const;
  // Sets one MPEG-class control; `what` names it in the error.
  void set_control(std::uint32_t id, std::int32_t value, const char* what) const;
  // Sets one MPEG-class control whose value is the structure at `data`.
  void set_control(std::uint32_t id, void* data, std::uint32_t size, const char* what) const;

  // Requests `count` MMAP buffers of `type` with `planes` planes and maps them; count 0 releases
  // them. Returns the buffers the driver gave.
  std::vector<V4l2Buffer> request_buffers(v4l2_buf_type type, std::uint32_t count,
                                          std::uint32_t planes);
  void release_buffers(v4l2_buf_type type, std::vector<V4l2Buffer>& buffers);

  // Queues buffer `index` of `type`, with each plane's bytes used and a timestamp.
  void queue(v4l2_buf_type type, std::uint32_t index, const std::vector<std::uint32_t>& used,
             std::uint64_t timestamp_us);
  // Dequeues a buffer of `type`: its index, each plane's bytes used, timestamp and flags; false
  // when none is ready.
  struct Dequeued {
    std::uint32_t index{};
    std::vector<std::uint32_t> used;
    std::uint64_t timestamp_us{};
    std::uint32_t flags{};
  };
  bool dequeue(v4l2_buf_type type, std::uint32_t planes, Dequeued& out) const;
  void stream_on(v4l2_buf_type type) const;
  void stream_off(v4l2_buf_type type) const;

 private:
  int fd_{-1};
};

// Waits in short sleeps until `ready()` is true or `timeout` passes; returns ready(). libv4l2's
// file descriptors do not work with poll(), so the backends poll the queues this way.
template <typename Ready>
bool wait_until(Ready&& ready, std::chrono::milliseconds timeout) {
  constexpr std::chrono::microseconds kPollInterval{100};
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!ready()) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return ready();
    }
    std::this_thread::sleep_for(kPollInterval);
  }
  return true;
}

// "<what>: <strerror(errno)>" as a runtime_error.
[[noreturn]] void throw_errno(const char* what);

}  // namespace holoscan::codec::detail
