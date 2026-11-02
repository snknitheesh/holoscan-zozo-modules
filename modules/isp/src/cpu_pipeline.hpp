// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The CPU backend: the whole pipeline on the host, one pixel at a time with the
// kernels' own arithmetic (isp_math.hpp), so it is also the reference the GPU
// is checked against. Plain C++, for the SDK-free tests as well.

#pragma once

#include <vector>

#include "holoscan/isp/isp_params.hpp"
#include "isp_math.hpp"

namespace holoscan::isp::detail {

// A frame's FrameParams from validated settings and views. `lut` and `lsc`
// are where the backend keeps the tone table and the lens shading grid; the
// CUDA backend passes null and sets its device copies.
[[nodiscard]] FrameParams make_frame_params(const IspParams& params, const RawImageView& raw,
                                            const RgbImageView& out, const float* lut, int lut_size,
                                            const float* lsc);

// The white balance state before the first estimate: the configured gains.
[[nodiscard]] AwbState initial_white_balance(const IspParams& params);

// Buffers the CPU backend reuses from frame to frame.
struct CpuScratch {
  std::vector<float> samples;  // Preprocessed mosaic.
  std::vector<float> rgb;      // Demosaiced RGB with a one-pixel border, for sharpening.
  std::vector<AwbAccum> awb;   // One, kept off the stack.
};

// Runs the pipeline on host memory: `p` from make_frame_params().
void cpu_process(const FrameParams& p, const IspParams& params, const StatsOutput& stats,
                 const FrameExposure& exposure, AwbState& awb, CpuScratch& scratch);

}  // namespace holoscan::isp::detail
