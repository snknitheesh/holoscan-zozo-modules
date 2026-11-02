// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The pixel formats, colour matrices and ranges the module converts between,
// and the layout of each format in memory. Needs no SDK, CUDA or GPU.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace holoscan::color_convert {

enum class PixelFormat : std::uint8_t {
  kNv12,    // Y plane, then an interleaved U, V plane at half width and half height.
  kNv16,    // Y plane, then an interleaved U, V plane at half width and full height.
  kI420,    // Y, U and V planes, U and V at half width and half height.
  kYuyv,    // Packed 4:2:2: Y0, U, Y1, V for each pair of pixels.
  kUyvy,    // Packed 4:2:2: U, Y0, V, Y1 for each pair of pixels.
  kRgb8,    // R, G, B bytes.
  kBgr8,    // B, G, R bytes.
  kRgba8,   // R, G, B, A bytes.
  kBgra8,   // B, G, R, A bytes.
  kMono8,   // One byte of full-range luma.
  kMono16,  // One 16-bit word of full-range luma.
  kRgb16,   // R, G, B 16-bit words.
};

// Every format, in declaration order.
inline constexpr PixelFormat kAllFormats[] = {
    PixelFormat::kNv12,  PixelFormat::kNv16,  PixelFormat::kI420,   PixelFormat::kYuyv,
    PixelFormat::kUyvy,  PixelFormat::kRgb8,  PixelFormat::kBgr8,   PixelFormat::kRgba8,
    PixelFormat::kBgra8, PixelFormat::kMono8, PixelFormat::kMono16, PixelFormat::kRgb16,
};

// The YCbCr <-> RGB matrix of ITU-R BT.601, BT.709 or BT.2020 (non-constant
// luminance).
enum class ColorMatrix : std::uint8_t { kBt601, kBt709, kBt2020 };

// Limited ("video", Y 16..235, Cb and Cr 16..240) or full ("JPEG", 0..255)
// range, for YCbCr formats. RGB and MONO formats are always full range.
enum class ColorRange : std::uint8_t { kLimited, kFull };

// How a format's samples are interpreted.
enum class ColorModel : std::uint8_t { kYuv, kRgb, kGray };

struct FormatInfo {
  ColorModel model;
  int planes;            // 1, 2 or 3.
  int chroma_shift_x;    // log2 of the horizontal chroma subsampling: 0 or 1.
  int chroma_shift_y;    // log2 of the vertical chroma subsampling: 0 or 1.
  int bytes_per_sample;  // 1, or 2 for the 16-bit formats.
  int channels;          // Samples per pixel in plane 0: 1, 2 (packed 4:2:2), 3 or 4.
};

[[nodiscard]] constexpr FormatInfo format_info(PixelFormat format) noexcept {
  switch (format) {
    case PixelFormat::kNv12:
      return {ColorModel::kYuv, 2, 1, 1, 1, 1};
    case PixelFormat::kNv16:
      return {ColorModel::kYuv, 2, 1, 0, 1, 1};
    case PixelFormat::kI420:
      return {ColorModel::kYuv, 3, 1, 1, 1, 1};
    case PixelFormat::kYuyv:
    case PixelFormat::kUyvy:
      return {ColorModel::kYuv, 1, 1, 0, 1, 2};
    case PixelFormat::kRgb8:
    case PixelFormat::kBgr8:
      return {ColorModel::kRgb, 1, 0, 0, 1, 3};
    case PixelFormat::kRgba8:
    case PixelFormat::kBgra8:
      return {ColorModel::kRgb, 1, 0, 0, 1, 4};
    case PixelFormat::kMono8:
      return {ColorModel::kGray, 1, 0, 0, 1, 1};
    case PixelFormat::kMono16:
      return {ColorModel::kGray, 1, 0, 0, 2, 1};
    case PixelFormat::kRgb16:
      return {ColorModel::kRgb, 1, 0, 0, 2, 3};
  }
  return {ColorModel::kRgb, 1, 0, 0, 1, 3};
}

[[nodiscard]] constexpr bool is_yuv(PixelFormat format) noexcept {
  return format_info(format).model == ColorModel::kYuv;
}
[[nodiscard]] constexpr bool is_planar(PixelFormat format) noexcept {
  return format_info(format).planes > 1;
}
[[nodiscard]] constexpr bool is_subsampled(PixelFormat format) noexcept {
  const FormatInfo info = format_info(format);
  return info.chroma_shift_x != 0 || info.chroma_shift_y != 0;
}
[[nodiscard]] constexpr bool has_alpha(PixelFormat format) noexcept {
  return format == PixelFormat::kRgba8 || format == PixelFormat::kBgra8;
}
// Packed 4:2:2 stores two pixels per chroma pair, so its width must be even.
[[nodiscard]] constexpr bool needs_even_width(PixelFormat format) noexcept {
  return format == PixelFormat::kYuyv || format == PixelFormat::kUyvy;
}

// A plane's packed size: the bytes of one row without padding, and its rows.
// A subsampled extent rounds up, as the SDK's image schema requires: an odd
// width or height keeps a chroma sample for its last column or row.
struct PlaneSize {
  std::size_t row_bytes;
  int rows;
};

[[nodiscard]] constexpr PlaneSize plane_size(PixelFormat format, int width, int height,
                                             int plane) noexcept {
  const FormatInfo info = format_info(format);
  const auto w = static_cast<std::size_t>(width);
  if (plane == 0) {
    return {w * static_cast<std::size_t>(info.channels * info.bytes_per_sample), height};
  }
  const std::size_t chroma_width = (w + ((1U << info.chroma_shift_x) - 1U)) >> info.chroma_shift_x;
  const int chroma_rows = (height + (1 << info.chroma_shift_y) - 1) >> info.chroma_shift_y;
  // Semi-planar formats interleave U and V in their second plane.
  const std::size_t samples = info.planes == 2 ? 2U : 1U;
  return {chroma_width * samples, chroma_rows};
}

// The bytes of a whole packed image, its planes one after the other.
[[nodiscard]] constexpr std::size_t packed_size(PixelFormat format, int width,
                                                int height) noexcept {
  std::size_t total = 0;
  for (int plane = 0; plane < format_info(format).planes; ++plane) {
    const PlaneSize size = plane_size(format, width, height, plane);
    total += size.row_bytes * static_cast<std::size_t>(size.rows);
  }
  return total;
}

// "nv12", "nv16", "i420", "yuyv", "uyvy", "rgb8", "bgr8", "rgba8", "bgra8",
// "mono8", "mono16", "rgb16".
[[nodiscard]] std::string_view format_name(PixelFormat format) noexcept;
[[nodiscard]] std::optional<PixelFormat> parse_format(std::string_view name) noexcept;
// "bt601", "bt709", "bt2020".
[[nodiscard]] std::string_view matrix_name(ColorMatrix matrix) noexcept;
[[nodiscard]] std::optional<ColorMatrix> parse_matrix(std::string_view name) noexcept;
// "limited", "full".
[[nodiscard]] std::string_view range_name(ColorRange range) noexcept;
[[nodiscard]] std::optional<ColorRange> parse_range(std::string_view name) noexcept;

}  // namespace holoscan::color_convert
