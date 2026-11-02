// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/color_convert/reference.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "kernel_math.hpp"

namespace holoscan::color_convert {

void reference_convert(const ImageView& source, const ImageView& destination,
                       const ConversionParams& params) {
  if (source.location != MemoryLocation::kHost || destination.location != MemoryLocation::kHost) {
    throw std::invalid_argument("reference_convert needs host images");
  }
  const detail::Plan plan = detail::make_plan(source, destination, params);
  const int across = detail::blocks_x(plan);
  const int down = detail::blocks_y(plan);
  for (int by = 0; by < down; ++by) {
    for (int bx = 0; bx < across; ++bx) {
      detail::convert_block(plan, bx, by);
    }
  }
}

void draw_test_pattern(const ImageView& rgb8, std::uint32_t frame) {
  if (rgb8.format != PixelFormat::kRgb8 || rgb8.location != MemoryLocation::kHost ||
      rgb8.width <= 0 || rgb8.height <= 0 || rgb8.planes[0].data == nullptr ||
      rgb8.planes[0].pitch < static_cast<std::size_t>(rgb8.width) * 3U) {
    throw std::invalid_argument("draw_test_pattern needs a host RGB8 image");
  }
  for (int y = 0; y < rgb8.height; ++y) {
    std::uint8_t* row = rgb8.planes[0].data + static_cast<std::size_t>(y) * rgb8.planes[0].pitch;
    for (int x = 0; x < rgb8.width; ++x) {
      const auto rgb = test_pattern_rgb(x, y, rgb8.width, rgb8.height, frame);
      row[3 * x] = rgb[0];
      row[3 * x + 1] = rgb[1];
      row[3 * x + 2] = rgb[2];
    }
  }
}

}  // namespace holoscan::color_convert
