// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/image_quality/synthetic.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace holoscan::image_quality {
namespace {

std::uint64_t mix(std::uint64_t z) {
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31U);
}

std::uint8_t clip(double value) {
  return static_cast<std::uint8_t>(std::clamp(std::lround(value), 0L, 255L));
}

int color_channels(PixelFormat format) { return format == PixelFormat::kMono8 ? 1 : 3; }

// Applies `f` to each color channel value (not alpha).
template <typename F>
void for_each_color(HostImage& image, F&& f) {
  const int ch = channels(image.format);
  const int color = color_channels(image.format);
  for (std::size_t i = 0; i < image.pixels.size(); i += ch) {
    for (int c = 0; c < color; ++c) f(image.pixels[i + c], c);
  }
}

std::vector<double> gaussian_kernel(double sigma) {
  const int radius = static_cast<int>(std::ceil(3.0 * sigma));
  std::vector<double> kernel(2 * radius + 1);
  double sum = 0.0;
  for (int i = -radius; i <= radius; ++i) {
    kernel[i + radius] = std::exp(-0.5 * i * i / (sigma * sigma));
    sum += kernel[i + radius];
  }
  for (double& w : kernel) w /= sum;
  return kernel;
}

}  // namespace

HostImage render_scene(int width, int height, std::uint32_t frame, PixelFormat format) {
  if (width < 1 || height < 1) {
    throw std::invalid_argument("synthetic images need at least one pixel");
  }
  HostImage image{width, height, format, {}};
  const int ch = channels(format);
  const bool bgr = format == PixelFormat::kBgr8 || format == PixelFormat::kBgra8;
  image.pixels.resize(static_cast<std::size_t>(width) * height * ch);
  const std::int64_t shift_x = 5LL * frame;
  const std::int64_t shift_y = 2LL * frame;
  for (int y = 0; y < height; ++y) {
    const std::int64_t cy = (y + shift_y) / kSceneCell;
    for (int x = 0; x < width; ++x) {
      const std::int64_t cx = (x + shift_x) / kSceneCell;
      const std::uint64_t h = mix(static_cast<std::uint64_t>(cx) * 0x100000001B3ULL ^
                                  static_cast<std::uint64_t>(cy) * 0xC2B2AE3D27D4EB4FULL);
      const int r = 40 + static_cast<int>(h % 176U);
      const int g = 40 + static_cast<int>((h >> 16U) % 176U);
      const int b = 40 + static_cast<int>((h >> 32U) % 176U);
      std::uint8_t* p = image.pixels.data() + (static_cast<std::size_t>(y) * width + x) * ch;
      if (ch == 1) {
        p[0] = static_cast<std::uint8_t>(luma(r, g, b));
        continue;
      }
      p[0] = static_cast<std::uint8_t>(bgr ? b : r);
      p[1] = static_cast<std::uint8_t>(g);
      p[2] = static_cast<std::uint8_t>(bgr ? r : b);
      if (ch == 4) p[3] = 255U;
    }
  }
  return image;
}

void gaussian_blur(HostImage& image, double sigma) {
  gaussian_blur(image, sigma, PixelRect{0, 0, image.width, image.height});
}

void gaussian_blur(HostImage& image, double sigma, const PixelRect& region) {
  if (!(sigma > 0)) return;
  const std::vector<double> kernel = gaussian_kernel(sigma);
  const int radius = static_cast<int>(kernel.size() / 2);
  const int width = image.width;
  const int height = image.height;
  const int ch = channels(image.format);
  const int color = color_channels(image.format);
  // Separable: rows into a float buffer, then columns back into the image.
  std::vector<double> rows(image.pixels.size());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < color; ++c) {
        double sum = 0.0;
        for (int k = -radius; k <= radius; ++k) {
          const int xs = std::clamp(x + k, 0, width - 1);
          sum += kernel[k + radius] *
                 image.pixels[(static_cast<std::size_t>(y) * width + xs) * ch + c];
        }
        rows[(static_cast<std::size_t>(y) * width + x) * ch + c] = sum;
      }
    }
  }
  const int x0 = std::max(region.x, 0);
  const int y0 = std::max(region.y, 0);
  const int x1 = std::min(region.x + region.width, width);
  const int y1 = std::min(region.y + region.height, height);
  for (int y = y0; y < y1; ++y) {
    for (int x = x0; x < x1; ++x) {
      for (int c = 0; c < color; ++c) {
        double sum = 0.0;
        for (int k = -radius; k <= radius; ++k) {
          const int ys = std::clamp(y + k, 0, height - 1);
          sum += kernel[k + radius] * rows[(static_cast<std::size_t>(ys) * width + x) * ch + c];
        }
        image.pixels[(static_cast<std::size_t>(y) * width + x) * ch + c] = clip(sum);
      }
    }
  }
}

void apply_gain(HostImage& image, double gain) {
  for_each_color(image, [gain](std::uint8_t& v, int) { v = clip(v * gain); });
}

void apply_channel_gains(HostImage& image, double red, double green, double blue) {
  if (image.format == PixelFormat::kMono8) return;
  const bool bgr = image.format == PixelFormat::kBgr8 || image.format == PixelFormat::kBgra8;
  const double gains[3] = {bgr ? blue : red, green, bgr ? red : blue};
  for_each_color(image, [&gains](std::uint8_t& v, int c) { v = clip(v * gains[c]); });
}

void add_noise(HostImage& image, double sigma, std::uint64_t seed) {
  std::uint64_t state = mix(seed);
  // Box-Muller on SplitMix64 uniforms in (0, 1).
  const auto uniform = [&state] {
    state = mix(state);
    return (static_cast<double>(state >> 11U) + 0.5) / 9007199254740992.0;
  };
  for_each_color(image, [&](std::uint8_t& v, int) {
    const double n =
        std::sqrt(-2.0 * std::log(uniform())) * std::cos(2.0 * std::numbers::pi * uniform());
    v = clip(v + sigma * n);
  });
}

void apply_smudge(HostImage& image, const Smudge& smudge) {
  const int ch = channels(image.format);
  const int color = color_channels(image.format);
  for (int y = 0; y < image.height; ++y) {
    for (int x = 0; x < image.width; ++x) {
      const double dx = (x - smudge.cx) / smudge.rx;
      const double dy = (y - smudge.cy) / smudge.ry;
      if (dx * dx + dy * dy > 1.0) continue;
      std::uint8_t* p = image.pixels.data() + (static_cast<std::size_t>(y) * image.width + x) * ch;
      for (int c = 0; c < color; ++c) p[c] = clip(0.08 * p[c] + 0.92 * 210.0);
    }
  }
}

}  // namespace holoscan::image_quality
