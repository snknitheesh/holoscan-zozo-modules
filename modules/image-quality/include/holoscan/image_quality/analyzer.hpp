// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The GPU analyzer: gathers a frame's statistics with CUDA kernels, four
// launches and one copy back per frame, with integer sums and histograms only,
// so the result is exact and the same on every GPU and run. ImageQualityOp runs
// one; it needs no Holoscan graph.

#pragma once

#include <cuda_runtime_api.h>

#include <memory>

#include <holoscan/image_quality/quality.hpp>

namespace holoscan::image_quality {

struct AnalyzerOptions {
  StatsOptions stats;
  // Blocks each tile is split into, along its rows; 0 picks enough to fill the
  // GPU. Any value gives the same statistics.
  int parts_per_tile{0};
};

class GpuStats {
 public:
  // Uses the current CUDA device.
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

 private:
  struct Buffers;
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

 private:
  GpuStats stats_;
  QualityEvaluator evaluator_;
};

}  // namespace holoscan::image_quality
