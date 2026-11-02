// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// How an image fits a depth model's fixed input, and how the model's output
// maps back onto the image. Host code only: no SDK, no CUDA. The kernels that
// apply it are in kernels.hpp.

#pragma once

#include <cstdint>

namespace holoscan::mono_depth {

// Interleaved 8-bit color layouts the operator reads.
enum class PixelFormat : std::uint8_t { kRgb8, kBgr8, kRgba8, kBgra8 };

[[nodiscard]] constexpr int channels(PixelFormat format) noexcept {
  return format == PixelFormat::kRgba8 || format == PixelFormat::kBgra8 ? 4 : 3;
}

// Whether the first channel of a pixel is blue rather than red.
[[nodiscard]] constexpr bool blue_first(PixelFormat format) noexcept {
  return format == PixelFormat::kBgr8 || format == PixelFormat::kBgra8;
}

// Largest image side, and the model input and output sides, accepted.
inline constexpr int kMaxImageSide = 16384;
inline constexpr int kMaxModelSide = 4096;

// How an image of another aspect ratio than the model's input fills it.
enum class ResizeMode : std::uint8_t {
  // Scaled by one factor to fit, centered, the rest padded with the mean
  // color (zero after normalization); the depth comes back from the image's
  // part only. Keeps the geometry the model sees.
  kLetterbox,
  // Scaled along each axis to fill the input. Uses every input pixel, but
  // squeezes the scene when the aspect ratios differ.
  kStretch,
};

// Where a source_width x source_height image lands in an input_width x
// input_height model input: resized to resized_width x resized_height with its
// top-left corner at (pad_left, pad_top). And which pixels of the model's
// output_width x output_height output cover it: columns valid_left..
// valid_right and rows valid_top..valid_bottom, inclusive.
struct ImageFit {
  int source_width{};
  int source_height{};
  int input_width{};
  int input_height{};
  int resized_width{};
  int resized_height{};
  int pad_left{};
  int pad_top{};
  int output_width{};
  int output_height{};
  int valid_left{};
  int valid_top{};
  int valid_right{};
  int valid_bottom{};
};

// The fit of a source image into the model input, and of the model output
// back onto it. A letterbox scales by min(input_width / source_width,
// input_height / source_height), rounds the resized sides half to even
// (at least 1), and splits the padding with the odd pixel, if any, on the
// right or bottom. The output covers the input proportionally: input pixel
// column x is output column x * output_width / input_width.
// @throws std::invalid_argument for a source side outside 1..kMaxImageSide or
// a model side outside 1..kMaxModelSide.
[[nodiscard]] ImageFit make_fit(int source_width, int source_height, int input_width,
                                int input_height, int output_width, int output_height,
                                ResizeMode mode);

}  // namespace holoscan::mono_depth
