// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// IspPipeline: the image signal processor without a Holoscan graph. Raw Bayer
// mosaic in, RGB out, with optional statistics, on the GPU (CUDA kernels) or
// the host (the reference implementation), on caller-owned buffers:
//
//   1. black level subtraction, per channel when configured;
//   2. lens shading correction from a gain grid;
//   3. white balance: manual gains, or gray-world / white-patch estimated
//      from the frame and smoothed over frames;
//   4. demosaic: bilinear, or Malvar-He-Cutler;
//   5. unsharp-mask sharpening (optional);
//   6. 3x3 colour correction matrix;
//   7. tone curve: sRGB, gamma, linear, or a table;
//   8. statistics: histograms of the output, mean levels, the auto-exposure
//      error and a suggested next exposure (never applied).
//
// The two backends compute the same output, bit for bit.

#pragma once

#include <cuda_runtime_api.h>

#include <memory>

#include <holoscan/isp/isp_params.hpp>

namespace holoscan::isp {

// Whether a backend reads and writes host or CUDA device memory.
enum class MemoryLocation : std::uint8_t { kHost, kDevice };

// Whether a CUDA device is usable in this process.
[[nodiscard]] PlatformFacts probe_platform() noexcept;
// `backend`, or for kAuto, resolve_auto() of this platform's facts.
[[nodiscard]] Backend resolve_backend(Backend backend) noexcept;

class IspPipeline {
 public:
  // Builds the backend, kAuto resolved; the CUDA backend uses the current device.
  // @throws std::invalid_argument if `params` are not valid.
  // @throws std::runtime_error if CUDA fails.
  IspPipeline(Backend backend, IspParams params);
  ~IspPipeline();
  IspPipeline(const IspPipeline&) = delete;
  IspPipeline& operator=(const IspPipeline&) = delete;

  // kCuda or kCpu.
  [[nodiscard]] Backend backend() const noexcept;
  // Where process() reads and writes its images and statistics.
  [[nodiscard]] MemoryLocation location() const noexcept;
  [[nodiscard]] const IspParams& params() const noexcept;

  // Forgets the white balance estimate, so the next frame starts afresh. Call
  // it when no frame is in flight on the GPU.
  // @throws std::runtime_error if CUDA fails.
  void reset();

  // Processes one frame. `out` is the raw image's size. CUDA: queued on
  // `stream`, all in device memory; CPU: done when this returns, all in host
  // memory, `stream` unused.
  // @throws std::invalid_argument if a view is not usable.
  // @throws std::runtime_error if CUDA fails.
  void process(const RawImageView& raw, const RgbImageView& out, const StatsOutput& stats = {},
               const FrameExposure& exposure = {}, cudaStream_t stream = nullptr);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::isp
