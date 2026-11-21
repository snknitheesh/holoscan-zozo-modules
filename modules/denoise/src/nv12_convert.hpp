// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Conversions between the denoiser's formats and full-range NV12 (BT.601), the
// one VPI's Temporal Noise Reduction takes.

#pragma once

#include <cuda_runtime_api.h>

#include <holoscan/denoise/denoise_params.hpp>

namespace holoscan::denoise::detail {

// NV12 planes of an even-sized width x height image.
struct Nv12 {
  Plane y;
  Plane uv;
  int width{};
  int height{};
};

// Writes `input` (MONO8: gray, chroma 128; RGB8, RGBA8: BT.601 full range, chroma
// the mean of each 2x2; NV12: copied) into `output`, which may be one pixel wider
// and taller than `input` for odd sizes: the last column and row repeat.
[[nodiscard]] cudaError_t launch_to_nv12(const ImageView& input, const Nv12& output,
                                         cudaStream_t stream);
// Writes the top-left of `input` into `output` (MONO8: Y; RGB8, RGBA8: BT.601
// full range, chroma interpolated bilinearly, alpha copied from `alpha_source`,
// the frame; NV12: copied).
[[nodiscard]] cudaError_t launch_from_nv12(const Nv12& input, const ImageView& output,
                                           const ImageView& alpha_source, cudaStream_t stream);

}  // namespace holoscan::denoise::detail
