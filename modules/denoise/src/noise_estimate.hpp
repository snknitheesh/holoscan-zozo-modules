// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// GPU passes both denoiser backends share: the noise estimate and the count of
// pixels that changed between two frames. Deterministic: every sum is an
// integer, reduced in a fixed order.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace holoscan::denoise::detail {

// Blocks of the estimate's first pass; its partial sums are reduced in order.
inline constexpr int kNoiseBlocks = 256;

// One 8-bit sample per pixel: byte `channel` of `stride`-byte pixels.
struct SamplePlane {
  const std::uint8_t* data{};
  std::size_t pitch{};
  int width{};
  int height{};
  int stride{1};
  int channel{0};
};

// Queues Immerkaer's noise estimate of `plane` into `*sigma` (code values, at
// least `min_sigma`). `partials` holds kNoiseBlocks uint64 values.
[[nodiscard]] cudaError_t launch_noise_estimate(const SamplePlane& plane, std::uint64_t* partials,
                                                float min_sigma, float* sigma, cudaStream_t stream);

// Queues a count, added to `*count`, of the pixels where the 3x3 means of
// `current` and `previous` differ by more than `threshold` times the standard
// deviation of that difference for noise `*sigma` in `current` and a share
// `previous_variance` of it left in `previous`.
[[nodiscard]] cudaError_t launch_count_changed(const SamplePlane& current,
                                               const SamplePlane& previous, const float* sigma,
                                               float previous_variance, float threshold,
                                               unsigned int* count, cudaStream_t stream);

// Sets `*value` on `stream`.
[[nodiscard]] cudaError_t launch_set(float* value, float to, cudaStream_t stream);

}  // namespace holoscan::denoise::detail
