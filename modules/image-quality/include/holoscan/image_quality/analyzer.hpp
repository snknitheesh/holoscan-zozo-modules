// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The GPU analyzer: gathers a frame's statistics with CUDA kernels, four
// launches and one copy back per frame, with integer sums and histograms only,
// so the result is exact and the same on every GPU and run. ImageQualityOp runs
// one; it needs no Holoscan graph.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>

#include <holoscan/image_quality/quality.hpp>

namespace holoscan::image_quality {

struct AnalyzerOptions {
  StatsOptions stats;
  // Blocks each tile is split into, along its rows; 0 picks enough to fill the
  // GPU. Any value gives the same statistics.
  int parts_per_tile{0};
  // The image size, in pixels, whose buffers the constructor allocates, so that
  // gather() allocates nothing for images up to it (from host memory too); 0 x 0
  // allocates on the first frame. A larger image grows the buffers on the frame
  // it arrives with: a cudaFree and a cudaMalloc, which wait for the device.
  int reserve_width{0};
  int reserve_height{0};
};

class GpuStats {
 public:
  // Uses the current CUDA device, and allocates the buffers for the reserved size.
  // @throws std::invalid_argument if `options` is not valid.
  // @throws std::runtime_error if CUDA fails.
  explicit GpuStats(AnalyzerOptions options);
  ~GpuStats();
  GpuStats(const GpuStats&) = delete;
  GpuStats& operator=(const GpuStats&) = delete;

  // Statistics of `image`, compared with the previous image when it had the same
  // size. `image` is in CUDA device (or managed) memory when `on_device`, else in
  // host memory, which is copied to the GPU first. The work runs on `stream`
  // (which may be null for the legacy default stream); the call returns when it
  // is done, so the image may be reused right after.
  // @throws std::invalid_argument for an empty image or a pitch below a row's bytes.
  // @throws std::runtime_error if CUDA fails.
  [[nodiscard]] FrameStats gather(const ImageView& image, bool on_device, cudaStream_t stream);
  // Forgets the previous image.
  void reset() noexcept;
  // The blocks each tile is split into for an image of this size.
  [[nodiscard]] int parts_per_tile(int width, int height) const noexcept;
  // Whether gather() allocates nothing for `image`.
  [[nodiscard]] bool fits(const ImageView& image, bool on_device) const noexcept;

 private:
  struct Buffers;
  // The bytes of the statistics copied back for a width x height image.
  [[nodiscard]] std::size_t results_bytes(int width, int height) const noexcept;
  // Grows the buffers to a width x height image, with a staging copy of `ch`
  // channels per pixel (0 for none).
  void reserve(int width, int height, int ch);

  AnalyzerOptions options_;
  int multiprocessors_{};
  std::unique_ptr<Buffers> buffers_;
};

// GpuStats and a QualityEvaluator: frames in, reports out.
class ImageQualityAnalyzer {
 public:
  // @throws std::invalid_argument if `options` or `thresholds` is not valid.
  // @throws std::runtime_error if CUDA fails.
  ImageQualityAnalyzer(AnalyzerOptions options, QualityThresholds thresholds);

  // See GpuStats::gather.
  [[nodiscard]] QualityReport analyze(const ImageView& image, bool on_device, cudaStream_t stream);
  void reset() noexcept;
  // See GpuStats::fits.
  [[nodiscard]] bool fits(const ImageView& image, bool on_device) const noexcept;

 private:
  GpuStats stats_;
  QualityEvaluator evaluator_;
};

}  // namespace holoscan::image_quality
