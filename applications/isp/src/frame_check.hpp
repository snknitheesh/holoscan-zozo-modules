// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's per-frame ground truth, on the GPU or the host: the squared error
// of each RGB8 pixel against the synthetic camera's truth through the ISP's
// tone curve, the image's own histograms rebuilt from its pixels and compared
// bin by bin with the ones IspOp sent, and IspOp's statistics, in one small
// result that is read back once per frame.

#pragma once

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <holoscan/isp/isp_params.hpp>
#include <holoscan/isp/synthetic_raw.hpp>

#if defined(__CUDACC__)
#define ISP_DEMO_HD __host__ __device__
#else
#define ISP_DEMO_HD
#endif

namespace isp_demo {

inline constexpr int kHistogramEntries =
    holoscan::isp::kHistogramChannels * holoscan::isp::kHistogramBins;

struct FrameCheckResult {
  // Sum over every pixel and channel of (code - truth code)^2.
  unsigned long long squared_error;
  // Histogram bins that differ from the counts of the image's own pixels.
  unsigned int histogram_mismatches;
  unsigned int reserved;
  float stats[holoscan::isp::stats::kStatsSize];
};

// The tone curve's table at v in 0..1, interpolated linearly, as the ISP evaluates it.
ISP_DEMO_HD inline float tone(const float* lut, int size, float v) {
  const float f = v * static_cast<float>(size - 1);
  const auto i = static_cast<int>(f);
  return i >= size - 1 ? lut[size - 1]
                       : lut[i] + (f - static_cast<float>(i)) * (lut[i + 1] - lut[i]);
}

// The 8-bit code an ideal ISP outputs at (x, y), per channel.
ISP_DEMO_HD inline void truth_codes(const holoscan::isp::SyntheticRawScene& scene,
                                    const float* lut, int lut_size, int x, int y, int codes[3]) {
  float linear[3];
  holoscan::isp::synthetic_truth(scene, x, y, linear);
  for (int k = 0; k < 3; ++k) {
    codes[k] = static_cast<int>(round(255.0 * static_cast<double>(tone(lut, lut_size, linear[k]))));
  }
}

// The luma bin of an RGB8 pixel: Rec. 709 weights, rounded.
ISP_DEMO_HD inline int luma_bin(int r, int g, int b) {
  return (2126 * r + 7152 * g + 722 * b + 5000) / 10000;
}

// Queues the check of one RGB8 frame (rows packed) on `stream`, everything in
// device memory: `scratch` holds kHistogramEntries uint32, `result` is written.
cudaError_t check_frame_cuda(const holoscan::isp::SyntheticRawScene& scene, const float* lut,
                             int lut_size, const std::uint8_t* rgb,
                             const std::uint32_t* histogram, const float* stats,
                             std::uint32_t* scratch, FrameCheckResult* result,
                             cudaStream_t stream);

}  // namespace isp_demo
