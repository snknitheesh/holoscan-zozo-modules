// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What both detector operators do with an image before and around detection: check it, take a
// CUDA stream, convert it to gray where the backend reads it (logging once when that copies
// between host and device), build the backend, resolve `auto` at start, and warn about frames
// over capacity.

#pragma once

#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/fiducials/detector_backend.hpp>
#include <holoscan/fiducials/types.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "gray_converter.hpp"

namespace holoscan::fiducials {

class FrameDetector {
 public:
  // `op` names the operator in errors and logs.
  FrameDetector(std::string op, Backend backend, DetectorOptions options);
  ~FrameDetector();
  FrameDetector(const FrameDetector&) = delete;
  FrameDetector& operator=(const FrameDetector&) = delete;

  // Resolves Backend::kAuto, once, logging the choice; resets the frame count.
  void start();
  // Releases the backend and the stream.
  void stop();
  // The configured backend, or what kAuto resolved to; kAuto before start. Any thread.
  [[nodiscard]] Backend resolved() const noexcept { return resolved_; }

  // One image, ready to detect in.
  struct Frame {
    GrayImageView gray;
    DetectorBackend* backend;
    cudaStream_t stream;
    std::uint64_t number;                         // Frames prepared before this one since start.
    std::chrono::steady_clock::time_point start;  // When conversion began.
  };

  // Checks `image`, waits for its producer, and converts it for the backend, which is built on
  // the first image and again when the size changes.
  [[nodiscard]] holoscan::expected<Frame, holoscan::Error> prepare(
      const holoscan::schema::ImageT& image, holoscan::ExecutionContext& context);

  // An error from an exception thrown while detecting; nothing queued may still read the image.
  [[nodiscard]] holoscan::Error error(const std::exception& exception, cudaStream_t stream) const;

  // Warns about frames over capacity, at most every few seconds.
  void warn_overflow(const Overflow& overflow);

 private:
  struct StreamDestroy {
    void operator()(cudaStream_t stream) const noexcept {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  };

  std::string op_;
  DetectorOptions options_;
  std::atomic<Backend> resolved_;
  std::unique_ptr<DetectorBackend> backend_;
  int width_{};
  int height_{};
  GrayConverter converter_;
  // For frames whose activation has no runtime CUDA stream; made on first use.
  std::unique_ptr<CUstream_st, StreamDestroy> own_stream_;
  std::uint64_t frames_{};
  bool cross_copy_logged_{};
  std::uint64_t overflowed_frames_{};
  std::optional<std::chrono::steady_clock::time_point> last_overflow_warning_;
};

}  // namespace holoscan::fiducials
