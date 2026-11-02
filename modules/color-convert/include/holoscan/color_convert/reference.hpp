// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The CPU reference of a conversion, and the synthetic test pattern. Needs no
// SDK, CUDA or GPU.
//
// reference_convert() follows conversion.hpp's definition with the same
// single-precision arithmetic as the CUDA kernel, so the cuda backend's output
// equals it byte for byte; the SDK-free tests check it against the colour
// science in double precision.

#pragma once

#include <array>
#include <cstdint>

#include <holoscan/color_convert/conversion.hpp>
#include <holoscan/color_convert/image_view.hpp>

namespace holoscan::color_convert {

// Converts `source` into `destination`, both in host memory.
// @throws std::invalid_argument if validate_conversion() does.
void reference_convert(const ImageView& source, const ImageView& destination,
                       const ConversionParams& params);

// The synthetic camera's picture for frame `frame` of a `width` x `height`
// video, as 8-bit R, G, B: in the top two thirds, eight vertical colour bars
// (white, yellow, cyan, green, magenta, red, blue, black at full intensity)
// moving right by four pixels a frame; below them, a horizontal ramp from
// black to white over a vertical red-to-blue ramp, so every value of each
// channel occurs. Integer arithmetic only, so the CPU and the GPU agree.
[[nodiscard]] constexpr std::array<std::uint8_t, 3> test_pattern_rgb(int x, int y, int width,
                                                                     int height,
                                                                     std::uint32_t frame) noexcept {
  const int bars_height = (2 * height) / 3;
  if (y < bars_height) {
    const auto shift = static_cast<int>((frame * 4U) % static_cast<std::uint32_t>(width));
    const int column = (x + width - shift) % width;
    const int bar = (column * 8) / width;
    // White, yellow, cyan, green, magenta, red, blue, black: bit 2 red, bit 1
    // green, bit 0 blue of 7 - bar's colour index.
    constexpr std::uint8_t kBars[8] = {7, 6, 3, 2, 5, 4, 1, 0};
    const std::uint8_t rgb = kBars[bar];
    return {static_cast<std::uint8_t>((rgb & 4U) != 0 ? 255 : 0),
            static_cast<std::uint8_t>((rgb & 2U) != 0 ? 255 : 0),
            static_cast<std::uint8_t>((rgb & 1U) != 0 ? 255 : 0)};
  }
  const int ramp_rows = height - bars_height;
  const int gray = width > 1 ? (x * 255) / (width - 1) : 0;
  const int row = y - bars_height;
  const int blue = ramp_rows > 1 ? (row * 255) / (ramp_rows - 1) : 0;
  return {static_cast<std::uint8_t>((gray + 255 - blue) / 2), static_cast<std::uint8_t>(gray),
          static_cast<std::uint8_t>((gray + blue) / 2)};
}

// Fills `rgb8`, a host RGB8 view, with test_pattern_rgb() for `frame`.
// @throws std::invalid_argument if it is not a host RGB8 view.
void draw_test_pattern(const ImageView& rgb8, std::uint32_t frame);

}  // namespace holoscan::color_convert
