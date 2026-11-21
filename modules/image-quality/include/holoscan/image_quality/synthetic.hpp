// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Synthetic scenes and controlled degradations, on the host, for the tests and
// the demo: a textured scene that moves every frame, and Gaussian blur, an
// exposure gain, Gaussian noise, a hazy smudge and channel gains to spoil it with.
// Everything is deterministic: the same arguments give the same bytes.

#pragma once

#include <cstdint>
#include <vector>

#include <holoscan/image_quality/quality.hpp>

namespace holoscan::image_quality {

// A packed image in host memory.
struct HostImage {
  int width{};
  int height{};
  PixelFormat format{PixelFormat::kRgb8};
  std::vector<std::uint8_t> pixels;

  [[nodiscard]] ImageView view() const noexcept {
    return ImageView{pixels.data(), width, height,
                     static_cast<std::size_t>(width) * channels(format), format};
  }
};

// A rectangle of pixels, [x, x + width) x [y, y + height).
struct PixelRect {
  int x{};
  int y{};
  int width{};
  int height{};
};

// Side of the scene's square cells, in pixels.
inline constexpr int kSceneCell = 12;

// Frame `frame` of the scene: cells of kSceneCell pixels, each a color drawn
// from 40..215 per channel, scrolling by (5, 2) pixels a frame, so every frame
// differs, every area of a few cells has edges, and each channel's mean is about
// the same (gray world). MONO8 gets each cell's luma, the alpha of RGBA8 and
// BGRA8 is 255.
// @throws std::invalid_argument for a size below 1 pixel.
[[nodiscard]] HostImage render_scene(int width, int height, std::uint32_t frame,
                                     PixelFormat format = PixelFormat::kRgb8);

// Gaussian blur of standard deviation `sigma` pixels (radius ceil(3 sigma), edges
// clamped), inside `region`, or over the whole image.
void gaussian_blur(HostImage& image, double sigma);
void gaussian_blur(HostImage& image, double sigma, const PixelRect& region);

// Multiplies every color channel by `gain`, as an exposure change, clipping at 255.
void apply_gain(HostImage& image, double gain);

// Multiplies the red, green and blue channels by their gains (MONO8: the luma of
// the gains' effect is not modeled; it is left unchanged).
void apply_channel_gains(HostImage& image, double red, double green, double blue);

// Adds zero-mean Gaussian noise of standard deviation `sigma` levels to every
// color channel, rounding and clipping; `seed` picks the noise.
void add_noise(HostImage& image, double sigma, std::uint64_t seed);

// A hazy smudge on the lens: inside the ellipse with center (cx, cy) and radii
// (rx, ry), each color channel becomes 0.08 of itself plus 0.92 of 210, which
// keeps a trace of the scene but almost none of its texture.
struct Smudge {
  double cx{};
  double cy{};
  double rx{};
  double ry{};
};
void apply_smudge(HostImage& image, const Smudge& smudge);

}  // namespace holoscan::image_quality
