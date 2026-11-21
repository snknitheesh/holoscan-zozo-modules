// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <holoscan/mono_depth/colormap.hpp>
#include <holoscan/mono_depth/pixel_math.hpp>
#include <holoscan/mono_depth/reference.hpp>
#include <holoscan/mono_depth/stub_model.hpp>

namespace holoscan::mono_depth {

std::vector<float> reference_preprocess(const std::uint8_t* image, std::size_t pitch,
                                        PixelFormat format, const ImageFit& fit,
                                        const Normalization& normalization) {
  const PreprocessParams params = make_preprocess_params(fit, format, pitch, normalization);
  const std::size_t plane = static_cast<std::size_t>(fit.input_width) * fit.input_height;
  std::vector<float> input(3U * plane);
  for (int v = 0; v < fit.input_height; ++v) {
    for (int u = 0; u < fit.input_width; ++u) {
      float rgb[3];
      preprocess_pixel(params, image, u, v, rgb);
      const std::size_t index = static_cast<std::size_t>(v) * fit.input_width + u;
      input[index] = rgb[0];
      input[plane + index] = rgb[1];
      input[2 * plane + index] = rgb[2];
    }
  }
  return input;
}

std::vector<float> reference_stub(std::span<const float> input, int width, int height) {
  const std::size_t plane = static_cast<std::size_t>(width) * height;
  std::vector<float> output(plane);
  for (std::size_t i = 0; i < plane && 3 * plane <= input.size(); ++i) {
    output[i] = stub_response(input[i], input[plane + i], input[2 * plane + i]);
  }
  return output;
}

ReferenceDepth reference_postprocess(std::span<const float> output, const ImageFit& fit,
                                     ModelOutput model_output, const InverseDepthAffine* affine) {
  const PostprocessParams params = make_postprocess_params(fit, model_output, affine);
  const std::size_t pixels = static_cast<std::size_t>(fit.source_width) * fit.source_height;
  const bool metric = model_output == ModelOutput::kMetricDepth || affine != nullptr;
  ReferenceDepth result;
  result.inverse.resize(pixels);
  if (metric) {
    result.metric.resize(pixels);
  }
  for (int y = 0; y < fit.source_height; ++y) {
    for (int x = 0; x < fit.source_width; ++x) {
      const DepthPair depth = depth_pair(params, sample_output(params, output.data(), x, y));
      const std::size_t index = static_cast<std::size_t>(y) * fit.source_width + x;
      result.inverse[index] = depth.inverse;
      if (metric) {
        result.metric[index] = depth.metric;
      }
    }
  }
  return result;
}

std::array<float, 2> reference_range(std::span<const float> values) {
  bool any = false;
  std::array<float, 2> range{0.0F, 0.0F};
  for (const float value : values) {
    if (!std::isfinite(value)) {
      continue;
    }
    range = any ? std::array<float, 2>{std::min(range[0], value), std::max(range[1], value)}
                : std::array<float, 2>{value, value};
    any = true;
  }
  return range;
}

std::vector<std::uint8_t> reference_colorize(std::span<const float> inverse_depth, float low,
                                             float high) {
  std::vector<std::uint8_t> rgba(inverse_depth.size() * 4U);
  for (std::size_t i = 0; i < inverse_depth.size(); ++i) {
    const auto entry = static_cast<std::size_t>(colormap_index(inverse_depth[i], low, high));
    rgba[4 * i] = kTurbo[3 * entry];
    rgba[4 * i + 1] = kTurbo[3 * entry + 1];
    rgba[4 * i + 2] = kTurbo[3 * entry + 2];
    rgba[4 * i + 3] = 255;
  }
  return rgba;
}

}  // namespace holoscan::mono_depth
