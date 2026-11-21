// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The per-pixel arithmetic of Sam3SegmentationOp's kernels, shared with their
// host references (reference.hpp), so that both compute the same expressions:
// the module compiles its kernels without fused multiply-adds and its host
// code without floating-point contraction, and the tests compare them bit for
// bit. Host and device code; no SDK.
//
// - The preprocessing stretches an image to the model's square input, as
//   SAM 3's own processor does, and normalizes it.
// - The mask sampling upsamples the model's low-resolution mask logits to the
//   image, bilinearly with half-pixel centers (PyTorch's
//   interpolate(mode="bilinear", align_corners=False)), as SAM 3's
//   postprocessing does.
// - The ranking orders candidate instances by score, best first.

#pragma once

#include <math.h>  // NOLINT(modernize-deprecated-headers): fminf and friends on host and device

#include <array>
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define HOLOSCAN_SAM3_HD __host__ __device__
#else
#define HOLOSCAN_SAM3_HD
#endif

namespace holoscan::sam3 {

// Interleaved 8-bit color layouts the operator reads.
enum class PixelFormat : std::uint8_t { kRgb8, kBgr8, kRgba8, kBgra8 };

[[nodiscard]] constexpr int channels(PixelFormat format) noexcept {
  return format == PixelFormat::kRgba8 || format == PixelFormat::kBgra8 ? 4 : 3;
}

// Whether the first channel of a pixel is blue rather than red.
[[nodiscard]] constexpr bool blue_first(PixelFormat format) noexcept {
  return format == PixelFormat::kBgr8 || format == PixelFormat::kBgra8;
}

// Largest image side, and largest model input and mask side, accepted.
inline constexpr int kMaxImageSide = 16384;
inline constexpr int kMaxModelSide = 4096;

// The normalization of RGB values in [0, 1] that the model expects:
// (value - mean) / std. The defaults are SAM 3's.
struct Normalization {
  std::array<float, 3> mean{0.5F, 0.5F, 0.5F};
  std::array<float, 3> std{0.5F, 0.5F, 0.5F};
};

// The preprocessing kernel's per-launch constants.
struct PreprocessParams {
  std::size_t pitch{};
  int width{};
  int height{};
  int channels{};
  // Offsets of red and blue within a pixel; green is always 1.
  int red{};
  int blue{};
  int input_width{};
  int input_height{};
  // Source pixels per input pixel, along each axis.
  float x_ratio{};
  float y_ratio{};
  float mean[3]{};
  float std[3]{};
};

[[nodiscard]] inline PreprocessParams make_preprocess_params(int width, int height,
                                                             PixelFormat format, std::size_t pitch,
                                                             int input_width, int input_height,
                                                             const Normalization& normalization) {
  const bool bgr = blue_first(format);
  PreprocessParams p;
  p.pitch = pitch;
  p.width = width;
  p.height = height;
  p.channels = channels(format);
  p.red = bgr ? 2 : 0;
  p.blue = bgr ? 0 : 2;
  p.input_width = input_width;
  p.input_height = input_height;
  p.x_ratio = static_cast<float>(width) / static_cast<float>(input_width);
  p.y_ratio = static_cast<float>(height) / static_cast<float>(input_height);
  for (int c = 0; c < 3; ++c) {
    p.mean[c] = normalization.mean[static_cast<std::size_t>(c)];
    p.std[c] = normalization.std[static_cast<std::size_t>(c)];
  }
  return p;
}

// Model input pixel (u, v): `rgb` gets its three normalized values, a
// bilinear sample of the image stretched over the whole input (pixel centers
// aligned, edges clamped, as OpenCV's INTER_LINEAR), divided by 255, less the
// mean, over the standard deviation.
HOLOSCAN_SAM3_HD inline void preprocess_pixel(const PreprocessParams& p, const std::uint8_t* image,
                                              int u, int v, float rgb[3]) {
  const float sx = fmaxf((static_cast<float>(u) + 0.5F) * p.x_ratio - 0.5F, 0.0F);
  const float sy = fmaxf((static_cast<float>(v) + 0.5F) * p.y_ratio - 0.5F, 0.0F);
  const int x0 = static_cast<int>(sx) < p.width - 1 ? static_cast<int>(sx) : p.width - 1;
  const int y0 = static_cast<int>(sy) < p.height - 1 ? static_cast<int>(sy) : p.height - 1;
  const int x1 = x0 + 1 < p.width - 1 ? x0 + 1 : p.width - 1;
  const int y1 = y0 + 1 < p.height - 1 ? y0 + 1 : p.height - 1;
  const float fx = sx - static_cast<float>(x0);
  const float fy = sy - static_cast<float>(y0);
  const std::uint8_t* row0 = image + static_cast<std::size_t>(y0) * p.pitch;
  const std::uint8_t* row1 = image + static_cast<std::size_t>(y1) * p.pitch;
  const int c0 = x0 * p.channels;
  const int c1 = x1 * p.channels;
  const int offsets[3] = {p.red, 1, p.blue};
  for (int c = 0; c < 3; ++c) {
    const int k = offsets[c];
    const float top =
        (1.0F - fx) * static_cast<float>(row0[c0 + k]) + fx * static_cast<float>(row0[c1 + k]);
    const float bottom =
        (1.0F - fx) * static_cast<float>(row1[c0 + k]) + fx * static_cast<float>(row1[c1 + k]);
    const float value = ((1.0F - fy) * top + fy * bottom) / 255.0F;
    rgb[c] = (value - p.mean[c]) / p.std[c];
  }
}

// How a mask_width x mask_height grid of mask logits covers a width x height
// image: the model sees the image stretched over its whole input, so the
// grid covers the whole image too.
struct MaskSampling {
  int mask_width{};
  int mask_height{};
  int width{};
  int height{};
  // Mask cells per image pixel, along each axis.
  float x_scale{};
  float y_scale{};
};

HOLOSCAN_SAM3_HD inline MaskSampling make_mask_sampling(int mask_width, int mask_height, int width,
                                                        int height) {
  MaskSampling s;
  s.mask_width = mask_width;
  s.mask_height = mask_height;
  s.width = width;
  s.height = height;
  s.x_scale = static_cast<float>(mask_width) / static_cast<float>(width);
  s.y_scale = static_cast<float>(mask_height) / static_cast<float>(height);
  return s;
}

// The mask logit at image pixel (x, y): a bilinear sample of the
// mask_height x mask_width `logits`, as PyTorch's interpolate(mode="bilinear",
// align_corners=False) upsamples them.
HOLOSCAN_SAM3_HD inline float sample_mask(const MaskSampling& s, const float* logits, int x,
                                          int y) {
  const float sx = fmaxf((static_cast<float>(x) + 0.5F) * s.x_scale - 0.5F, 0.0F);
  const float sy = fmaxf((static_cast<float>(y) + 0.5F) * s.y_scale - 0.5F, 0.0F);
  const int x0 = static_cast<int>(sx) < s.mask_width - 1 ? static_cast<int>(sx) : s.mask_width - 1;
  const int y0 =
      static_cast<int>(sy) < s.mask_height - 1 ? static_cast<int>(sy) : s.mask_height - 1;
  const int x1 = x0 + 1 < s.mask_width - 1 ? x0 + 1 : s.mask_width - 1;
  const int y1 = y0 + 1 < s.mask_height - 1 ? y0 + 1 : s.mask_height - 1;
  const float fx = sx - static_cast<float>(x0);
  const float fy = sy - static_cast<float>(y0);
  const float* row0 =
      logits + static_cast<std::size_t>(y0) * static_cast<std::size_t>(s.mask_width);
  const float* row1 =
      logits + static_cast<std::size_t>(y1) * static_cast<std::size_t>(s.mask_width);
  const float top = (1.0F - fx) * row0[x0] + fx * row0[x1];
  const float bottom = (1.0F - fx) * row1[x0] + fx * row1[x1];
  return (1.0F - fy) * top + fy * bottom;
}

// Whether candidate a ranks before candidate b: the higher score first, then
// the lower key (prompt, then query), so the order is total and repeatable.
// NaN scores never pass a threshold, so never meet here.
HOLOSCAN_SAM3_HD inline bool ranks_before(float score_a, int key_a, float score_b, int key_b) {
  return score_a > score_b || (score_a == score_b && key_a < key_b);
}

// The logit at which a mask probability of `probability` is reached: masks
// keep the pixels whose logit exceeds it.
[[nodiscard]] inline float logit_of(float probability) {
  return logf(probability / (1.0F - probability));
}

// An instance's box in image pixels, from the model's box in fractions of the
// image ([x1, y1, x2, y2], as the model gives it), clamped to the image.
HOLOSCAN_SAM3_HD inline void box_to_pixels(const float* fraction, int width, int height,
                                           float* pixels) {
  const float w = static_cast<float>(width);
  const float h = static_cast<float>(height);
  pixels[0] = fminf(fmaxf(fraction[0] * w, 0.0F), w);
  pixels[1] = fminf(fmaxf(fraction[1] * h, 0.0F), h);
  pixels[2] = fminf(fmaxf(fraction[2] * w, 0.0F), w);
  pixels[3] = fminf(fmaxf(fraction[3] * h, 0.0F), h);
}

// The colors of instances, by index modulo kOverlayColors.
inline constexpr int kOverlayColors = 10;

// Instance `index`'s color, RGB.
HOLOSCAN_SAM3_HD inline void overlay_color(int index, std::uint8_t rgb[3]) {
  const std::uint8_t palette[kOverlayColors][3] = {
      {230, 25, 75},  {60, 180, 75},  {255, 225, 25}, {0, 130, 200},  {245, 130, 48},
      {145, 30, 180}, {70, 240, 240}, {240, 50, 230}, {210, 245, 60}, {250, 190, 212}};
  const std::uint8_t* color = palette[index % kOverlayColors];
  rgb[0] = color[0];
  rgb[1] = color[1];
  rgb[2] = color[2];
}

// The overlay pixel of label image pixel (x, y): transparent for 0; the
// instance's color, opaque on its outline (a 4-neighbor with another label,
// or the image's edge) and `fill_alpha` inside.
HOLOSCAN_SAM3_HD inline void overlay_pixel(const std::uint16_t* labels, int width, int height,
                                           int x, int y, std::uint8_t fill_alpha,
                                           std::uint8_t* rgba) {
  const std::size_t index = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + x;
  const std::uint16_t label = labels[index];
  if (label == 0) {
    rgba[0] = rgba[1] = rgba[2] = rgba[3] = 0;
    return;
  }
  const bool edge = x == 0 || y == 0 || x == width - 1 || y == height - 1 ||
                    labels[index - 1] != label || labels[index + 1] != label ||
                    labels[index - static_cast<std::size_t>(width)] != label ||
                    labels[index + static_cast<std::size_t>(width)] != label;
  overlay_color(label - 1, rgba);
  rgba[3] = edge ? 255 : fill_alpha;
}

}  // namespace holoscan::sam3
