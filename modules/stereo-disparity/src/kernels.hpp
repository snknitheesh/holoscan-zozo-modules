// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The operator's CUDA kernels: an 8-bit image to gray before matching, and
// VPI's fixed-point disparity to float disparity, depth and confidence after
// it. Each queues its work on `stream` and returns the launch status.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace holoscan::stereo_disparity::detail {

enum class PixelFormat : std::uint8_t { kMono8, kRgb8, kBgr8, kRgba8, kBgra8 };

[[nodiscard]] int channels(PixelFormat format) noexcept;

// `width` x `height` pixels of `format`, rows `pitch` bytes apart, in device
// memory, to gray rows `out_pitch` bytes apart: MONO8 as it is, color as
// BT.601 luma (77 R + 150 G + 29 B + 128) / 256.
[[nodiscard]] cudaError_t to_gray(const std::uint8_t* input, std::size_t pitch, PixelFormat format,
                                  int width, int height, std::uint8_t* out, std::size_t out_pitch,
                                  cudaStream_t stream);

// Where finish_disparity() writes; packed rows of `width`. Null outputs are
// skipped.
struct DisparityOutputs {
  float* disparity{};
  float* depth{};
  float fx_baseline{};  // Focal length in pixels times baseline in meters.
  float doffs{};
  std::uint16_t* confidence{};
};

// VPI's disparity (signed Q10.5 fixed point, rows `raw_pitch` bytes apart) as
// float pixels: valid when above 0 and at most `max_disparity`; invalid ones
// (VPI's 0x7fff, 0 or less, or beyond the search) are written as 0. With
// `depth`, also depth_from_disparity(); with `confidence`, VPI's confidence
// map (rows `confidence_pitch` bytes apart), copied.
[[nodiscard]] cudaError_t finish_disparity(const std::int16_t* raw, std::size_t raw_pitch,
                                           const std::uint16_t* raw_confidence,
                                           std::size_t confidence_pitch, int width, int height,
                                           float max_disparity, const DisparityOutputs& outputs,
                                           cudaStream_t stream);

}  // namespace holoscan::stereo_disparity::detail
