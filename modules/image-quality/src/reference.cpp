// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/image_quality/reference.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

namespace holoscan::image_quality {

void check_image(const ImageView& image) {
  if (image.data == nullptr || image.width < 1 || image.height < 1) {
    throw std::invalid_argument("the image is empty");
  }
  if (image.width > 32768 || image.height > 32768) {
    throw std::invalid_argument("the image is larger than 32768 pixels on a side");
  }
  if (image.pitch < static_cast<std::size_t>(image.width) * channels(image.format)) {
    throw std::invalid_argument("the image pitch is smaller than a row");
  }
}

ReferenceStats::ReferenceStats(StatsOptions options) : options_(std::move(options)) {
  validate_stats_options(options_);
}

void ReferenceStats::reset() noexcept {
  previous_luma_.clear();
  previous_width_ = previous_height_ = 0;
}

FrameStats ReferenceStats::gather(const ImageView& image) {
  check_image(image);
  const int width = image.width;
  const int height = image.height;
  const int ch = channels(image.format);
  const bool bgr = image.format == PixelFormat::kBgr8 || image.format == PixelFormat::kBgra8;
  const TileGrid grid = tile_grid(width, height, options_.tiles_x, options_.tiles_y);

  FrameStats stats;
  stats.width = width;
  stats.height = height;
  stats.tiles_x = grid.tiles_x;
  stats.tiles_y = grid.tiles_y;
  stats.color = ch != 1;
  stats.has_previous = previous_width_ == width && previous_height_ == height;
  stats.tiles.assign(static_cast<std::size_t>(grid.tiles_x) * grid.tiles_y, TileStats{});
  stats.gradient_histogram.assign(kGradientBins, 0U);
  stats.high_pass_histogram.assign(kHighPassBins, 0U);

  const auto tile_of = [&](int x, int y) -> TileStats& {
    // The tile whose bounds hold (x, y): the largest index whose begin is <= x.
    int tx = static_cast<int>(static_cast<std::int64_t>(x) * grid.tiles_x / width);
    while (tx + 1 < grid.tiles_x && tile_begin(tx + 1, grid.tiles_x, width) <= x) ++tx;
    while (tile_begin(tx, grid.tiles_x, width) > x) --tx;
    int ty = static_cast<int>(static_cast<std::int64_t>(y) * grid.tiles_y / height);
    while (ty + 1 < grid.tiles_y && tile_begin(ty + 1, grid.tiles_y, height) <= y) ++ty;
    while (tile_begin(ty, grid.tiles_y, height) > y) --ty;
    return stats.tiles[static_cast<std::size_t>(ty) * grid.tiles_x + tx];
  };

  // Luma, color sums, the hash and the difference to the previous frame.
  std::vector<std::uint8_t> luma(static_cast<std::size_t>(width) * height);
  for (int y = 0; y < height; ++y) {
    const std::uint8_t* row = image.data + static_cast<std::size_t>(y) * image.pitch;
    for (int x = 0; x < width; ++x) {
      const std::uint8_t* p = row + static_cast<std::size_t>(x) * ch;
      std::uint32_t bytes = 0;
      for (int c = 0; c < ch; ++c) bytes |= static_cast<std::uint32_t>(p[c]) << (8U * c);
      int r = p[0];
      int g = p[0];
      int b = p[0];
      if (ch != 1) {
        r = bgr ? p[2] : p[0];
        g = p[1];
        b = bgr ? p[0] : p[2];
      }
      const int y_value = ch == 1 ? p[0] : image_quality::luma(r, g, b);
      const std::size_t index = static_cast<std::size_t>(y) * width + x;
      luma[index] = static_cast<std::uint8_t>(y_value);
      TileStats& tile = tile_of(x, y);
      tile.hash += pixel_hash(index, bytes);
      ++tile.pixels;
      tile.sum_luma += y_value;
      tile.sum_red += r;
      tile.sum_green += g;
      tile.sum_blue += b;
      if (stats.has_previous) {
        tile.sum_abs_diff += std::abs(y_value - previous_luma_[index]);
      }
      ++stats.luma_histogram[y_value];
    }
  }

  // The stencils, on interior pixels.
  const auto at = [&](int x, int y) { return static_cast<int>(luma[y * width + x]); };
  std::vector<std::uint16_t> gradient(luma.size(), 0U);
  for (int y = 1; y + 1 < height; ++y) {
    for (int x = 1; x + 1 < width; ++x) {
      const int laplacian =
          4 * at(x, y) - at(x - 1, y) - at(x + 1, y) - at(x, y - 1) - at(x, y + 1);
      const int gx = (at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)) -
                     (at(x - 1, y - 1) + 2 * at(x - 1, y) + at(x - 1, y + 1));
      const int gy = (at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)) -
                     (at(x - 1, y - 1) + 2 * at(x, y - 1) + at(x + 1, y - 1));
      TileStats& tile = tile_of(x, y);
      ++tile.interior;
      tile.sum_laplacian += laplacian;
      tile.sum_laplacian_sq += static_cast<std::int64_t>(laplacian) * laplacian;
      tile.sum_gradient_sq +=
          static_cast<std::int64_t>(gx) * gx + static_cast<std::int64_t>(gy) * gy;
      const int magnitude = std::abs(gx) + std::abs(gy);
      gradient[static_cast<std::size_t>(y) * width + x] = static_cast<std::uint16_t>(magnitude);
      ++stats.gradient_histogram[magnitude];
    }
  }
  stats.flat_threshold =
      flat_threshold(stats.gradient_histogram.data(), kGradientBins, options_.flat_fraction);
  for (int y = 1; y + 1 < height; ++y) {
    for (int x = 1; x + 1 < width; ++x) {
      if (gradient[static_cast<std::size_t>(y) * width + x] > stats.flat_threshold) continue;
      const int high_pass =
          (at(x - 1, y - 1) + at(x + 1, y - 1) + at(x - 1, y + 1) + at(x + 1, y + 1)) -
          2 * (at(x, y - 1) + at(x - 1, y) + at(x + 1, y) + at(x, y + 1)) + 4 * at(x, y);
      ++stats.high_pass_histogram[std::abs(high_pass)];
    }
  }

  previous_luma_ = std::move(luma);
  previous_width_ = width;
  previous_height_ = height;
  return stats;
}

}  // namespace holoscan::image_quality
