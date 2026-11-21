// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ExposureFusion and ToneMapper: exposure fusion of brackets, and the global
// tone map of one high-bit-depth frame, on the GPU, without a Holoscan graph.
// ExposureFusionOp runs one of them.

#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <span>

#include <holoscan/denoise/denoise_params.hpp>
#include <holoscan/denoise/fusion_params.hpp>

namespace holoscan::denoise {

class ExposureFusion {
 public:
  // @throws std::invalid_argument if `params` is unusable.
  explicit ExposureFusion(const FusionParams& params);
  ~ExposureFusion();
  ExposureFusion(const ExposureFusion&) = delete;
  ExposureFusion& operator=(const ExposureFusion&) = delete;

  // Fuses 2 or 3 brackets of one scene, MONO8, RGB8 or RGBA8 of the same size and
  // format in CUDA device memory, into `output` of that size and format (alpha
  // 255). Work is queued on `stream`; buffers grow on a larger frame.
  // @throws std::invalid_argument for mismatched or unusable views.
  // @throws std::runtime_error on a CUDA failure.
  void fuse(std::span<const ImageView> brackets, const ImageView& output, cudaStream_t stream);

  class Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

class ToneMapper {
 public:
  // @throws std::invalid_argument if `params` is unusable.
  explicit ToneMapper(const ToneMapParams& params);
  ~ToneMapper();
  ToneMapper(const ToneMapper&) = delete;
  ToneMapper& operator=(const ToneMapper&) = delete;

  // Maps `input` (CUDA device memory) to 8 bits in `output`: MONO8 for one
  // channel, RGB8 for three, RGBA8 for four, of the same size. The frame's
  // statistics are computed on the GPU; nothing waits on the host.
  // @throws std::invalid_argument for mismatched or unusable views.
  // @throws std::runtime_error on a CUDA failure.
  void map(const HdrImageView& input, const ImageView& output, cudaStream_t stream);

  // The last frame's statistics. Waits for `stream`.
  [[nodiscard]] ToneMapStats last_stats(cudaStream_t stream);

  class Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::denoise
