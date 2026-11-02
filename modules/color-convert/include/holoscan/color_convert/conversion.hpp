// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What one conversion does: the colour matrix and range of its YCbCr side, how
// it resamples, the part of the source it reads and where in the destination
// that lands. Needs no SDK, CUDA or GPU.
//
// Every backend follows the same definition, which reference_convert()
// (reference.hpp) implements on the CPU:
//
// 1. Each source pixel is decoded to its format's own samples: Y, Cb, Cr for
//    YCbCr formats, whose chroma is taken from the chroma sample covering the
//    pixel (no chroma interpolation); R, G, B for RGB formats; luma for MONO.
//    16-bit samples are divided by 257 onto the 8-bit scale. Alpha is 255
//    unless the format has it.
// 2. Each destination pixel inside the content rectangle samples the source
//    rectangle (nearest, bilinear or area, below) in those samples; pixels
//    outside it, when letterboxing, take the fill colour.
// 3. The samples are converted to the destination's colour model with the
//    matrix and range (nothing is converted between two YCbCr or two RGB
//    formats), rounded half up and clamped. A subsampled destination stores
//    the mean of the chroma of the pixels sharing a chroma sample.
//
// Sampling, with pixel centers at +0.5 and a destination pixel's footprint in
// the source rectangle found by scaling: nearest takes the pixel under the
// footprint's center; bilinear interpolates the four pixels around it,
// clamped to the rectangle; area averages every pixel the footprint covers,
// weighted by how much of it is covered.

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

#include <holoscan/color_convert/image_view.hpp>
#include <holoscan/color_convert/pixel_format.hpp>

namespace holoscan::color_convert {

enum class ResizeMode : std::uint8_t { kNearest, kBilinear, kArea };

// "nearest", "bilinear", "area".
[[nodiscard]] std::string_view resize_mode_name(ResizeMode mode) noexcept;
[[nodiscard]] std::optional<ResizeMode> parse_resize_mode(std::string_view name) noexcept;

// A rectangle of pixels: columns x .. x + width - 1, rows y .. y + height - 1.
struct Rect {
  int x{};
  int y{};
  int width{};
  int height{};

  friend constexpr bool operator==(const Rect&, const Rect&) noexcept = default;
};

struct ConversionParams {
  ColorMatrix matrix{ColorMatrix::kBt601};
  ColorRange range{ColorRange::kLimited};
  ResizeMode mode{ResizeMode::kBilinear};
  // The part of the source to read; the whole source without one.
  std::optional<Rect> crop;
  // Keep the aspect ratio of the source rectangle: scale it to fit, centered,
  // and fill the rest of the destination with `fill_rgb`.
  bool letterbox{false};
  std::array<std::uint8_t, 3> fill_rgb{0, 0, 0};
};

// Where the source rectangle lands in the destination. Without letterboxing
// the content is the whole destination.
struct Mapping {
  Rect source;
  Rect content;
};

// The mapping of a `source_width` x `source_height` image into a
// `width` x `height` one of `destination` format. Letterboxing scales the
// source rectangle by the smaller of the two ratios, rounds the content's size
// to whole pixels and centers it, rounding its offset down. Where the content
// leaves a border on an axis along which `destination` subsamples its chroma,
// its offset and size are made even, so no chroma sample mixes content and
// fill.
// @throws std::invalid_argument for sizes that are not positive or a crop
//         that is not inside the source.
[[nodiscard]] Mapping compute_mapping(int source_width, int source_height, int width, int height,
                                      PixelFormat destination, const ConversionParams& params);

// Checks that `source` can be converted into `destination` with `params`:
// positive sizes, even widths for packed 4:2:2, planes with data and pitches
// that hold a row, and a crop inside the source.
// @throws std::invalid_argument naming the first problem.
void validate_conversion(const ImageView& source, const ImageView& destination,
                         const ConversionParams& params);

}  // namespace holoscan::color_convert
