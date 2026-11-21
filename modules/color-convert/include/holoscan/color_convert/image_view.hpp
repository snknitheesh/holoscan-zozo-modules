// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A borrowed image: its format, size and where each plane's rows are. Needs no
// SDK, CUDA or GPU.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <holoscan/color_convert/pixel_format.hpp>

namespace holoscan::color_convert {

enum class MemoryLocation : std::uint8_t { kHost, kDevice };

struct PlaneView {
  std::uint8_t* data{};
  std::size_t pitch{};  // Bytes from one row to the next, at least the packed row.
};

struct ImageView {
  PixelFormat format{PixelFormat::kRgb8};
  int width{};
  int height{};
  std::array<PlaneView, 3> planes{};  // format_info(format).planes of them are used.
  MemoryLocation location{MemoryLocation::kDevice};
};

// A view of a packed image at `base`: each plane's rows without padding, the
// planes one after the other, as packed_size() counts them.
[[nodiscard]] constexpr ImageView packed_view(void* base, PixelFormat format, int width, int height,
                                              MemoryLocation location) noexcept {
  ImageView view{.format = format, .width = width, .height = height, .location = location};
  auto* next = static_cast<std::uint8_t*>(base);
  for (int plane = 0; plane < format_info(format).planes; ++plane) {
    const PlaneSize size = plane_size(format, width, height, plane);
    view.planes[plane] = {next, size.row_bytes};
    next += size.row_bytes * static_cast<std::size_t>(size.rows);
  }
  return view;
}

}  // namespace holoscan::color_convert
