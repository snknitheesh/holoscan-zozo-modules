// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The CUDA backend: one fused kernel for black level, lens shading, white
// balance, demosaic, sharpening, colour correction and tone curve over tiles
// in shared memory, plus small passes for the white balance statistics and the
// histograms. Its device buffers live as long as the object.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <vector>

#include "holoscan/isp/isp_params.hpp"
#include "isp_math.hpp"

namespace holoscan::isp::detail {

class CudaPipeline {
 public:
  // Uploads the tone table and lens shading grid.
  // @throws std::runtime_error if CUDA fails.
  CudaPipeline(const IspParams& params, const std::vector<float>& lut);
  ~CudaPipeline();
  CudaPipeline(const CudaPipeline&) = delete;
  CudaPipeline& operator=(const CudaPipeline&) = delete;

  // Forgets the white balance estimate; no frame may be in flight.
  // @throws std::runtime_error if CUDA fails.
  void reset();

  // Queues a frame on `stream`; `p` from make_frame_params() with null tables,
  // its images and `stats` in device memory.
  [[nodiscard]] cudaError_t process(FrameParams p, const StatsOutput& stats,
                                    const FrameExposure& exposure, cudaStream_t stream);

 private:
  void release() noexcept;
  // Copies host memory to the device and waits for it.
  static void upload(const void* source, std::size_t bytes, void* destination);

  IspParams params_;
  int lut_size_{};
  int sm_count_{1};
  float* lut_{};
  float* lsc_{};
  AwbAccum* awb_accum_{};
  AwbState* awb_state_{};
  StatsAccum* stats_accum_{};
};

}  // namespace holoscan::isp::detail
