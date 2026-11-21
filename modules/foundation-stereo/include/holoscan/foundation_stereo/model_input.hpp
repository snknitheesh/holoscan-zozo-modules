// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What a Fast-FoundationStereo model takes: the pixel formats
// FoundationStereoOp reads, and the normalization the model expects. No SDK
// or CUDA.

#pragma once

#include <cstdint>

// Lets the kernels share the host's definitions.
#ifndef HOLOSCAN_FOUNDATION_STEREO_HD
#ifdef __CUDACC__
#define HOLOSCAN_FOUNDATION_STEREO_HD __host__ __device__
#else
#define HOLOSCAN_FOUNDATION_STEREO_HD
#endif
#endif

namespace holoscan::foundation_stereo {

enum class PixelFormat : std::uint8_t { kMono8, kRgb8, kBgr8, kRgba8, kBgra8 };

[[nodiscard]] HOLOSCAN_FOUNDATION_STEREO_HD constexpr int channels(PixelFormat format) noexcept {
  switch (format) {
    case PixelFormat::kMono8:
      return 1;
    case PixelFormat::kRgb8:
    case PixelFormat::kBgr8:
      return 3;
    default:
      return 4;
  }
}

// The normalization the model expects, per RGB channel, on 0..255 values:
// (value - kImageMean[c]) / kImageStd[c], ImageNet's. Exports of
// Fast-FoundationStereo as one ONNX file leave it out of the network.
[[nodiscard]] HOLOSCAN_FOUNDATION_STEREO_HD constexpr float image_mean(int channel) noexcept {
  return channel == 0 ? 123.675F : channel == 1 ? 116.28F : 103.53F;
}
[[nodiscard]] HOLOSCAN_FOUNDATION_STEREO_HD constexpr float image_std(int channel) noexcept {
  return channel == 0 ? 58.395F : channel == 1 ? 57.12F : 57.375F;
}
inline constexpr float kImageMean[3] = {image_mean(0), image_mean(1), image_mean(2)};
inline constexpr float kImageStd[3] = {image_std(0), image_std(1), image_std(2)};

}  // namespace holoscan::foundation_stereo
