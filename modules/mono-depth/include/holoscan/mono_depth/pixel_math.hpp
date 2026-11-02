// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The per-pixel arithmetic of the operator's kernels, shared with their host
// references (reference.hpp), so that both compute the same expressions: the
// module compiles its kernels without fused multiply-adds and its host code
// without floating-point contraction, and the tests compare them bit for bit.
// Host and device code; no SDK.

#pragma once

#include <math.h>  // NOLINT(modernize-deprecated-headers): fminf and friends on host and device

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <holoscan/mono_depth/colormap.hpp>
#include <holoscan/mono_depth/image_fit.hpp>

#if defined(__CUDACC__)
#define HOLOSCAN_MONO_DEPTH_HD __host__ __device__
#else
#define HOLOSCAN_MONO_DEPTH_HD
#endif

namespace holoscan::mono_depth {

// What a model's output means.
enum class ModelOutput : std::uint8_t {
  // Relative inverse depth, as Depth Anything's relative models give it:
  // larger is nearer, in units of an unknown scale and shift.
  kRelativeInverseDepth,
  // Depth in meters, as Depth Anything's metric models give it.
  kMetricDepth,
};

// Metric depth from a relative model's inverse depth d: 1 / z = scale * d +
// shift, z in meters. Fit scale and shift to known depths of the scene; they
// hold for one camera and scene type at best.
struct InverseDepthAffine {
  float scale{1.0F};
  float shift{0.0F};
};

// The normalization a model expects, per RGB channel, of values in [0, 1]:
// (value - mean) / std. The defaults are ImageNet's, which Depth Anything uses.
struct Normalization {
  std::array<float, 3> mean{0.485F, 0.456F, 0.406F};
  std::array<float, 3> std{0.229F, 0.224F, 0.225F};
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
  int pad_left{};
  int pad_top{};
  int resized_width{};
  int resized_height{};
  // Source pixels per resized pixel, along each axis.
  float x_ratio{};
  float y_ratio{};
  float mean[3]{};
  float std[3]{};
};

[[nodiscard]] inline PreprocessParams make_preprocess_params(const ImageFit& fit,
                                                             PixelFormat format, std::size_t pitch,
                                                             const Normalization& normalization) {
  const bool bgr = blue_first(format);
  PreprocessParams p;
  p.pitch = pitch;
  p.width = fit.source_width;
  p.height = fit.source_height;
  p.channels = channels(format);
  p.red = bgr ? 2 : 0;
  p.blue = bgr ? 0 : 2;
  p.input_width = fit.input_width;
  p.input_height = fit.input_height;
  p.pad_left = fit.pad_left;
  p.pad_top = fit.pad_top;
  p.resized_width = fit.resized_width;
  p.resized_height = fit.resized_height;
  p.x_ratio = static_cast<float>(fit.source_width) / static_cast<float>(fit.resized_width);
  p.y_ratio = static_cast<float>(fit.source_height) / static_cast<float>(fit.resized_height);
  for (int c = 0; c < 3; ++c) {
    p.mean[c] = normalization.mean[static_cast<std::size_t>(c)];
    p.std[c] = normalization.std[static_cast<std::size_t>(c)];
  }
  return p;
}

// Model input pixel (u, v): `rgb` gets its three normalized values. Inside
// the resized image, a bilinear sample of the image (pixel centers aligned,
// edges clamped, as OpenCV's INTER_LINEAR), divided by 255, less the mean,
// over the standard deviation; outside it, 0, the mean color.
HOLOSCAN_MONO_DEPTH_HD inline void preprocess_pixel(const PreprocessParams& p,
                                                    const std::uint8_t* image, int u, int v,
                                                    float rgb[3]) {
  const int x = u - p.pad_left;
  const int y = v - p.pad_top;
  if (x < 0 || y < 0 || x >= p.resized_width || y >= p.resized_height) {
    rgb[0] = 0.0F;
    rgb[1] = 0.0F;
    rgb[2] = 0.0F;
    return;
  }
  const float sx = fmaxf((static_cast<float>(x) + 0.5F) * p.x_ratio - 0.5F, 0.0F);
  const float sy = fmaxf((static_cast<float>(y) + 0.5F) * p.y_ratio - 0.5F, 0.0F);
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

// The postprocessing kernel's per-launch constants: image pixel (x, y) reads
// the model output at (x * scale_x + offset_x, y * scale_y + offset_y),
// clamped to its valid pixels.
struct PostprocessParams {
  int width{};
  int height{};
  int output_width{};
  float scale_x{};
  float scale_y{};
  float offset_x{};
  float offset_y{};
  float valid_left{};
  float valid_top{};
  float valid_right{};
  float valid_bottom{};
  ModelOutput model_output{ModelOutput::kRelativeInverseDepth};
  bool metric_from_relative{};
  InverseDepthAffine affine{};
};

[[nodiscard]] inline PostprocessParams make_postprocess_params(const ImageFit& fit,
                                                               ModelOutput model_output,
                                                               const InverseDepthAffine* affine) {
  PostprocessParams p;
  p.width = fit.source_width;
  p.height = fit.source_height;
  p.output_width = fit.output_width;
  // Output pixels per input pixel, and input pixels per image pixel.
  const double out_x = static_cast<double>(fit.output_width) / fit.input_width;
  const double out_y = static_cast<double>(fit.output_height) / fit.input_height;
  const double in_x = static_cast<double>(fit.resized_width) / fit.source_width;
  const double in_y = static_cast<double>(fit.resized_height) / fit.source_height;
  // The center of image pixel x lands on input coordinate pad + (x + 0.5) *
  // in_x, which is output coordinate that times out_x, less half a pixel.
  p.scale_x = static_cast<float>(in_x * out_x);
  p.scale_y = static_cast<float>(in_y * out_y);
  p.offset_x = static_cast<float>((fit.pad_left + 0.5 * in_x) * out_x - 0.5);
  p.offset_y = static_cast<float>((fit.pad_top + 0.5 * in_y) * out_y - 0.5);
  p.valid_left = static_cast<float>(fit.valid_left);
  p.valid_top = static_cast<float>(fit.valid_top);
  p.valid_right = static_cast<float>(fit.valid_right);
  p.valid_bottom = static_cast<float>(fit.valid_bottom);
  p.model_output = model_output;
  p.metric_from_relative = affine != nullptr;
  if (affine != nullptr) {
    p.affine = *affine;
  }
  return p;
}

// The model output resampled at image pixel (x, y): bilinear between the
// valid output pixels around it.
HOLOSCAN_MONO_DEPTH_HD inline float sample_output(const PostprocessParams& p, const float* output,
                                                  int x, int y) {
  const float ox =
      fminf(fmaxf(static_cast<float>(x) * p.scale_x + p.offset_x, p.valid_left), p.valid_right);
  const float oy =
      fminf(fmaxf(static_cast<float>(y) * p.scale_y + p.offset_y, p.valid_top), p.valid_bottom);
  const int x0 = static_cast<int>(ox);
  const int y0 = static_cast<int>(oy);
  const int x1 = static_cast<float>(x0) < p.valid_right ? x0 + 1 : x0;
  const int y1 = static_cast<float>(y0) < p.valid_bottom ? y0 + 1 : y0;
  const float fx = ox - static_cast<float>(x0);
  const float fy = oy - static_cast<float>(y0);
  const float* row0 =
      output + static_cast<std::size_t>(y0) * static_cast<std::size_t>(p.output_width);
  const float* row1 =
      output + static_cast<std::size_t>(y1) * static_cast<std::size_t>(p.output_width);
  const float top = (1.0F - fx) * row0[x0] + fx * row0[x1];
  const float bottom = (1.0F - fx) * row1[x0] + fx * row1[x1];
  return (1.0F - fy) * top + fy * bottom;
}

// A resampled model value as inverse depth and, when the operator gives it,
// metric depth in meters. Values with no valid depth (not finite, or not
// positive where a reciprocal is taken) give 0.
struct DepthPair {
  float inverse{};
  float metric{};
};

HOLOSCAN_MONO_DEPTH_HD inline DepthPair depth_pair(const PostprocessParams& p, float value) {
  DepthPair result;
  if (!isfinite(value)) {
    return result;
  }
  if (p.model_output == ModelOutput::kMetricDepth) {
    result.metric = value > 0.0F ? value : 0.0F;
    result.inverse = value > 0.0F ? 1.0F / value : 0.0F;
    return result;
  }
  result.inverse = value;
  if (p.metric_from_relative) {
    const float inverse = p.affine.scale * value + p.affine.shift;
    result.metric = inverse > 0.0F && isfinite(inverse) ? 1.0F / inverse : 0.0F;
  }
  return result;
}

// Floats as unsigned integers in the same order, for atomic minimum and
// maximum; -0 orders before +0.
HOLOSCAN_MONO_DEPTH_HD inline std::uint32_t ordered_bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x80000000U) != 0U ? ~bits : bits | 0x80000000U;
}

HOLOSCAN_MONO_DEPTH_HD inline float from_ordered_bits(std::uint32_t ordered) {
  const std::uint32_t bits = (ordered & 0x80000000U) != 0U ? ordered & 0x7FFFFFFFU : ~ordered;
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// The colormap entry of inverse depth `value` in [low, high]: low (far) is
// entry 0, high (near) entry kColormapSize - 1, outside the range clamped.
// An empty or reversed range gives the middle entry.
HOLOSCAN_MONO_DEPTH_HD inline int colormap_index(float value, float low, float high) {
  if (!(high > low) || !isfinite(value)) {
    return kColormapSize / 2;
  }
  const float t = fminf(fmaxf((value - low) / (high - low), 0.0F), 1.0F);
  return static_cast<int>(t * static_cast<float>(kColormapSize - 1) + 0.5F);
}

}  // namespace holoscan::mono_depth
