// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/codec/test_video.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace holoscan::codec {
namespace {

// 5x7 glyphs, one byte per row with the leftmost pixel in bit 4: the digits 0 to 9, then six
// letter-like shapes for the filler text.
constexpr std::array<std::array<std::uint8_t, 7>, 16> kGlyphs{{
    {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E},  // 0
    {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E},  // 1
    {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F},  // 2
    {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E},  // 3
    {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02},  // 4
    {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E},  // 5
    {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E},  // 6
    {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08},  // 7
    {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E},  // 8
    {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C},  // 9
    {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11},  // A
    {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E},  // B
    {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E},  // C
    {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11},  // H
    {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F},  // L
    {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11},  // M
}};

constexpr int kGlyphScale = 2;
constexpr int kCellWidth = 6 * kGlyphScale;   // 5 pixels and a gap.
constexpr int kCellHeight = 9 * kGlyphScale;  // 7 rows and a gap.

std::uint32_t mix(std::uint32_t value) {
  value ^= value >> 16;
  value *= 0x7FEB352DU;
  value ^= value >> 15;
  value *= 0x846CA68BU;
  value ^= value >> 16;
  return value;
}

// The glyph in text cell (`column`, `row`) of frame `index`: its index in digits at the start of
// the first row, filler elsewhere.
int glyph_at(int column, int row, std::uint32_t index) {
  if (row == 0) {
    const std::string digits = std::to_string(index);
    if (column < static_cast<int>(digits.size())) {
      return digits[static_cast<std::size_t>(column)] - '0';
    }
    if (column == static_cast<int>(digits.size())) {
      return -1;  // A space.
    }
  }
  const std::uint32_t hash = mix(static_cast<std::uint32_t>(column) * 73856093U ^
                                 static_cast<std::uint32_t>(row) * 19349663U ^ (index / 15U));
  return hash % 7U == 0 ? -1 : static_cast<int>(hash % kGlyphs.size());
}

}  // namespace

std::vector<std::uint8_t> render_test_frame(int width, int height, std::uint32_t index,
                                            int channels) {
  if (width < 1 || height < 1 || (channels != 3 && channels != 4)) {
    throw std::invalid_argument("render_test_frame: a positive size and 3 or 4 channels");
  }
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * channels);
  const int shift = static_cast<int>(index % 4096U) * 4;
  const int bar_shift = static_cast<int>(index % 4096U) * 2;
  const int text_shift = static_cast<int>(index % 4096U) * 6;
  // The text band: the middle third of the frame.
  const int text_top = height / 3;
  const int text_bottom = 2 * height / 3;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      std::uint8_t* p = pixels.data() + (static_cast<std::size_t>(y) * width + x) * channels;
      const int gx = x - shift;
      // A smooth gradient: red across, green down, blue along the diagonal.
      int r = (gx * 255 / width) & 0xFF;
      int g = 255 * y / height;
      int b = ((gx + y) * 255 / (width + height)) & 0xFF;
      // Eight soft color bars over the bottom third.
      if (y >= text_bottom) {
        const int bar = ((x / std::max(1, width / 8)) + (y + bar_shift) / 64) % 8;
        r = (r + ((bar & 1) != 0 ? 96 : 0)) / 2 + 32;
        g = (g + ((bar & 2) != 0 ? 160 : 0)) / 2 + 16;
        b = (b + ((bar & 4) != 0 ? 200 : 0)) / 2;
      }
      if (y >= text_top && y < text_bottom) {
        // Dark background, light glyphs: sharp edges, as text has.
        r = 24;
        g = 24;
        b = 40;
        const int cell_y = (y - text_top) / kCellHeight;
        // The first row stands still; the others scroll left like a ticker.
        const int tx = cell_y == 0 ? x : x + text_shift;
        const int cell_x = tx / kCellWidth;
        const int glyph = glyph_at(cell_x, cell_y, index);
        const int gx_in = (tx % kCellWidth) / kGlyphScale;
        const int gy_in = ((y - text_top) % kCellHeight) / kGlyphScale;
        if (glyph >= 0 && gx_in < 5 && gy_in < 7 &&
            ((kGlyphs[static_cast<std::size_t>(glyph)][static_cast<std::size_t>(gy_in)] >>
              (4 - gx_in)) &
             1) != 0) {
          r = 230;
          g = 230;
          b = 200;
        }
      }
      p[0] = static_cast<std::uint8_t>(r);
      p[1] = static_cast<std::uint8_t>(g);
      p[2] = static_cast<std::uint8_t>(b);
      if (channels == 4) {
        p[3] = 255;
      }
    }
  }
  return pixels;
}

double psnr(const std::uint8_t* reference, std::size_t reference_pitch, const std::uint8_t* test,
            std::size_t test_pitch, int width, int height, int channels) {
  if (width < 1 || height < 1 || channels < 1) {
    throw std::invalid_argument("psnr: a positive size and channel count");
  }
  const std::size_t row = static_cast<std::size_t>(width) * channels;
  double squared = 0.0;
  for (int y = 0; y < height; ++y) {
    const std::uint8_t* a = reference + static_cast<std::size_t>(y) * reference_pitch;
    const std::uint8_t* b = test + static_cast<std::size_t>(y) * test_pitch;
    for (std::size_t i = 0; i < row; ++i) {
      const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
      squared += d * d;
    }
  }
  const double mse = squared / (static_cast<double>(row) * height);
  if (mse == 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  return 10.0 * std::log10(255.0 * 255.0 / mse);
}

double psnr(std::span<const std::uint8_t> reference, std::span<const std::uint8_t> test, int width,
            int height, int channels) {
  const std::size_t bytes = static_cast<std::size_t>(width) * height * channels;
  if (reference.size() < bytes || test.size() < bytes) {
    throw std::invalid_argument("psnr: an image is smaller than its size says");
  }
  const std::size_t row = static_cast<std::size_t>(width) * channels;
  return psnr(reference.data(), row, test.data(), row, width, height, channels);
}

}  // namespace holoscan::codec
