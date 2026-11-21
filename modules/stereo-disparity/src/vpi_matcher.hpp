// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// VPI 4's stereo disparity estimator for one backend and image size, on a VPI
// stream that wraps the operator's CUDA stream (the work stream), or on pva a
// VPI stream of its own.

#pragma once

#include <cuda_runtime_api.h>

// VPI's C API.
extern "C" {
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Types.h>
}

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "holoscan/stereo_disparity/disparity_options.hpp"
#include "kernels.hpp"

namespace holoscan::stereo_disparity::detail {

// @throws std::runtime_error naming `what` and VPI's message unless `status`
//         is VPI_SUCCESS.
void vpi_check(VPIStatus status, const char* what);

// An 8-bit image in CUDA device memory, rows `pitch` bytes apart.
struct DeviceImage {
  const std::uint8_t* data{};
  std::size_t pitch{};
  PixelFormat format{PixelFormat::kMono8};
};

// The estimator for one backend (cuda, ofa or pva) and input size, and the
// images it reads and writes. The host waits once per frame for cuda, twice
// for ofa and pva.
//
// How each backend reads the pair, without copying it first:
// - cuda matches MONO8 as it is (the input is wrapped for VPI), and color
//   after VPI converts it to gray on the GPU.
// - ofa and pva need block-linear gray. VPI's CUDA backend converts color to
//   it directly; MONO8 is first expanded to RGBA8 by a kernel, because VPI
//   4.1.4 has no CUDA conversion from gray to block-linear, and the VIC's is
//   slower (3.6 ms against 0.2 ms per 1280x720 image on Jetson Thor).
// How each backend runs and hands its result to the matcher's own buffers:
// - cuda: on a VPI stream wrapping the work stream, after the work stream's
//   own work, writing the buffers itself; one wait, vpiStreamSync.
// - ofa: on the same kind of stream. The OFA's disparity is block-linear,
//   which VPI 4.1.4's CUDA backend cannot convert; after vpiStreamSync,
//   match() locks it for CUDA, which has VPI convert it to pitch-linear (0.07
//   ms at 1280x720 on Jetson Thor, against 2.6 ms on the VIC), copies it on
//   the work stream, waits, and unlocks it (VPI may move the memory once
//   unlocked).
// - pva: on a VPI stream of its own. On Jetson Thor, VPI 4.1.4 loses the
//   PVA's result on a stream that wraps a CUDA stream: CUDA read it unwritten
//   (all zero) whether it copied it on the stream, after a vpiStreamSync, or
//   through a lock, in most sizes and settings. So the host waits for the work
//   stream first, then VPI converts, matches and copies the result into the
//   buffers on its stream, and vpiStreamSync waits for it.
class Matcher {
 public:
  // `stream` must outlive the matcher.
  // @throws std::invalid_argument for kAuto, or parameters or a size the
  //         backend does not take.
  // @throws std::runtime_error if VPI or CUDA fail to create what it needs
  //         (for instance, the backend's hardware is missing).
  Matcher(Backend backend, const DisparityParams& params, bool confidence, int width, int height,
          cudaStream_t stream);
  ~Matcher();
  Matcher(const Matcher&) = delete;
  Matcher& operator=(const Matcher&) = delete;

  [[nodiscard]] Backend backend() const noexcept { return backend_; }
  [[nodiscard]] int width() const noexcept { return width_; }
  [[nodiscard]] int height() const noexcept { return height_; }
  // The disparity's size: the input's, divided by the downscale factor.
  [[nodiscard]] int output_width() const noexcept { return output_width_; }
  [[nodiscard]] int output_height() const noexcept { return output_height_; }
  // The largest disparity the estimator searches.
  [[nodiscard]] int searched() const noexcept { return searched_; }

  // The latest match's disparity (S16, Q10.5, in input pixels) and, with a
  // confidence map, confidence (U16, else null): output_width() x
  // output_height(), rows disparity_pitch() and confidence_pitch() bytes
  // apart, in device memory. Valid until the next match() writes them.
  [[nodiscard]] const std::int16_t* disparity() const noexcept { return disparity_.data; }
  [[nodiscard]] std::size_t disparity_pitch() const noexcept { return disparity_.pitch; }
  [[nodiscard]] const std::uint16_t* confidence() const noexcept { return confidence_.data; }
  [[nodiscard]] std::size_t confidence_pitch() const noexcept { return confidence_.pitch; }

  // Matches the pair (each width() x height(), in device memory) and waits
  // for the result in disparity() and confidence(). Between calls, VPI has
  // nothing queued, so the caller may queue CUDA work on the work stream
  // before the call (waits for the inputs, copies into them, a wait before the
  // result buffers are written again): on cuda and ofa, the frame's first VPI
  // task is always a CUDA task on that stream, which every later one follows;
  // on pva, the host waits for it first. Do not queue work on the stream from
  // another thread during the call.
  // @throws std::runtime_error on a VPI or CUDA failure; VPI has nothing
  //         queued and nothing locked afterwards either way.
  void match(const std::array<DeviceImage, 2>& pair);

 private:
  struct DeviceFree {
    void operator()(void* memory) const noexcept { cudaFree(memory); }
  };
  // Device memory the matcher owns, and VPI's wrapper of it.
  template <typename T>
  struct OwnedPlane {
    std::unique_ptr<void, DeviceFree> memory;
    T* data{};
    std::size_t pitch{};
    VPIImage image{};
  };
  // Device memory for `width` x `height` pixels of `format`, rows of at least
  // `row_bytes`, wrapped for `backends`.
  template <typename T>
  void allocate(OwnedPlane<T>& plane, int width, int height, std::size_t row_bytes,
                VPIImageFormat format, std::uint64_t backends);
  // VPI's image of side `side`'s input, rewrapped for this frame.
  VPIImage wrap_input(int side, const DeviceImage& image);
  // Queues the frame: each source converted to gray first where `convert`
  // says so, then the matching.
  void submit(const std::array<VPIImage, 2>& sources, const std::array<bool, 2>& convert);
  // ofa: copies VPI's finished disparity into the matcher's buffer.
  void copy_out();
  void release() noexcept;

  Backend backend_;
  DisparityParams params_;
  int width_;
  int height_;
  int output_width_;
  int output_height_;
  int searched_;
  // ofa, pva: SGM passes, num_passes_for() on this platform.
  int num_passes_;
  cudaStream_t cuda_stream_;
  VPIStream stream_{};
  VPIPayload payload_{};
  // The inputs as VPI sees them: the caller's images, wrapped, or for MONO8
  // on ofa and pva the RGBA8 expansion.
  std::array<VPIImage, 2> input_{};
  std::array<PixelFormat, 2> input_format_{};
  std::array<OwnedPlane<std::uint8_t>, 2> expanded_{};
  // cuda: gray converted from color. ofa, pva: block-linear gray.
  std::array<VPIImage, 2> gray_{};
  // ofa, pva: the disparity (block-linear on ofa) and confidence as the OFA or
  // the PVA write them.
  VPIImage accelerator_disparity_{};
  VPIImage accelerator_confidence_{};
  // The disparity and confidence the caller reads.
  OwnedPlane<std::int16_t> disparity_{};
  OwnedPlane<std::uint16_t> confidence_{};
};

}  // namespace holoscan::stereo_disparity::detail
