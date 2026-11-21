// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/denoise/fusion_params.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace holoscan::denoise {
namespace {

constexpr std::array<std::pair<ToneCurve, std::string_view>, 3> kCurves{{
    {ToneCurve::kSrgb, "srgb"},
    {ToneCurve::kGamma, "gamma"},
    {ToneCurve::kLinear, "linear"},
}};

constexpr std::array<std::pair<HdrFormat, std::string_view>, 6> kHdrFormats{{
    {HdrFormat::kMono16, "MONO16"},
    {HdrFormat::kRgb16, "RGB16"},
    {HdrFormat::kRgba16, "RGBA16"},
    {HdrFormat::kMono32f, "MONO32F"},
    {HdrFormat::kRgb32f, "RGB32F"},
    {HdrFormat::kRgba32f, "RGBA32F"},
}};

bool in_range(float value, float low, float high) { return value >= low && value <= high; }

// One sample of `view` at (x, y, c), in linear units of its white: a float
// sample as it is (0 when negative or not finite), a code value over 2^bits - 1.
double sample(const HdrImageView& view, int x, int y, int c) {
  const int channels = hdr_channels(view.format);
  const auto* row =
      static_cast<const std::uint8_t*>(view.data) + static_cast<std::size_t>(y) * view.pitch;
  const std::size_t index = static_cast<std::size_t>(x) * channels + c;
  if (hdr_is_float(view.format)) {
    float value = 0.0F;
    std::memcpy(&value, row + index * sizeof(float), sizeof(float));
    return std::isfinite(value) && value > 0.0F ? value : 0.0;
  }
  std::uint16_t code = 0;
  std::memcpy(&code, row + index * sizeof(std::uint16_t), sizeof(code));
  return static_cast<double>(code) / static_cast<double>((1U << view.significant_bits) - 1U);
}

double luminance(const HdrImageView& view, int x, int y) {
  if (hdr_channels(view.format) == 1) return sample(view, x, y, 0);
  return 0.2126 * sample(view, x, y, 0) + 0.7152 * sample(view, x, y, 1) +
         0.0722 * sample(view, x, y, 2);
}

}  // namespace

void validate_fusion_params(const FusionParams& params) {
  if (!in_range(params.contrast_weight, 0.0F, 4.0F) ||
      !in_range(params.saturation_weight, 0.0F, 4.0F) ||
      !in_range(params.exposure_weight, 0.0F, 4.0F)) {
    throw std::invalid_argument("fusion weights must be in 0..4");
  }
  if (params.contrast_weight == 0.0F && params.saturation_weight == 0.0F &&
      params.exposure_weight == 0.0F) {
    throw std::invalid_argument("at least one fusion weight must be positive");
  }
  if (!in_range(params.exposure_sigma, 0.05F, 1.0F)) {
    throw std::invalid_argument("exposure_sigma must be in 0.05..1");
  }
  if (params.levels < 0 || params.levels > 16) {
    throw std::invalid_argument("levels must be 0 (automatic) or 1..16");
  }
}

void validate_tone_map_params(const ToneMapParams& params) {
  if (!in_range(params.key, 0.01F, 1.0F)) {
    throw std::invalid_argument("key must be in 0.01..1");
  }
  if (!(std::isfinite(params.white) && params.white >= 0.0F)) {
    throw std::invalid_argument("white must be 0 (the brightest pixel) or positive");
  }
  if (params.curve != ToneCurve::kSrgb && params.curve != ToneCurve::kGamma &&
      params.curve != ToneCurve::kLinear) {
    throw std::invalid_argument("curve must be srgb, gamma or linear");
  }
  if (!in_range(params.gamma, 1.0F, 5.0F)) {
    throw std::invalid_argument("gamma must be in 1..5");
  }
}

std::string_view tone_curve_name(ToneCurve curve) noexcept {
  for (const auto& [kind, name] : kCurves) {
    if (kind == curve) return name;
  }
  return "unknown";
}

std::optional<ToneCurve> parse_tone_curve(std::string_view name) noexcept {
  for (const auto& [kind, curve_name] : kCurves) {
    if (curve_name == name) return kind;
  }
  return std::nullopt;
}

std::string_view hdr_format_name(HdrFormat format) noexcept {
  for (const auto& [kind, name] : kHdrFormats) {
    if (kind == format) return name;
  }
  return "unknown";
}

int fusion_levels(const FusionParams& params, int width, int height) noexcept {
  // Each level halves the size, rounding up, down to 1 pixel.
  int possible = 1;
  for (int w = width, h = height; (w > 1 || h > 1) && possible < 16; ++possible) {
    w = (w + 1) / 2;
    h = (h + 1) / 2;
  }
  if (params.levels > 0) return std::min(params.levels, possible);
  // Up to kAutoLevels, while the top level keeps 8 pixels on its shorter side.
  int levels = 1;
  for (int side = std::min(width, height); side >= 16 && levels < kAutoFusionLevels;
       side = (side + 1) / 2) {
    ++levels;
  }
  return levels;
}

ToneMapStats tone_map_stats_reference(const HdrImageView& view) {
  double log_sum = 0.0;
  double max_luminance = 0.0;
  for (int y = 0; y < view.height; ++y) {
    for (int x = 0; x < view.width; ++x) {
      const double l = luminance(view, x, y);
      log_sum += std::log(1e-6 + l);
      max_luminance = std::max(max_luminance, l);
    }
  }
  const double count = static_cast<double>(view.width) * static_cast<double>(view.height);
  return ToneMapStats{.log_average = std::exp(log_sum / count), .max_luminance = max_luminance};
}

std::uint8_t encode_reference(double value, const ToneMapParams& params) {
  double v = std::clamp(value, 0.0, 1.0);
  switch (params.curve) {
    case ToneCurve::kSrgb:
      v = v <= 0.0031308 ? 12.92 * v : 1.055 * std::pow(v, 1.0 / 2.4) - 0.055;
      break;
    case ToneCurve::kGamma:
      v = std::pow(v, 1.0 / params.gamma);
      break;
    case ToneCurve::kLinear:
      break;
  }
  return static_cast<std::uint8_t>(std::floor(std::clamp(v, 0.0, 1.0) * 255.0 + 0.5));
}

std::vector<std::uint8_t> tone_map_reference(const HdrImageView& view,
                                             const ToneMapParams& params) {
  const ToneMapStats stats = tone_map_stats_reference(view);
  const double scale = params.key / stats.log_average;
  double white = params.white > 0.0F ? params.white : stats.max_luminance * scale;
  if (!(white > 0.0)) white = 1.0;
  const int channels = hdr_channels(view.format);
  std::vector<std::uint8_t> out(static_cast<std::size_t>(view.width) * view.height * channels);
  for (int y = 0; y < view.height; ++y) {
    for (int x = 0; x < view.width; ++x) {
      std::uint8_t* o = out.data() + (static_cast<std::size_t>(y) * view.width + x) * channels;
      const double l = luminance(view, x, y);
      const double lm = scale * l;
      const double ld = lm * (1.0 + lm / (white * white)) / (1.0 + lm);
      if (channels == 1) {
        o[0] = encode_reference(ld, params);
        continue;
      }
      for (int c = 0; c < 3; ++c) {
        o[c] = encode_reference(l > 0.0 ? sample(view, x, y, c) * ld / l : 0.0, params);
      }
      if (channels == 4) {
        o[3] = static_cast<std::uint8_t>(
            std::floor(std::clamp(sample(view, x, y, 3), 0.0, 1.0) * 255.0 + 0.5));
      }
    }
  }
  return out;
}

double mertens_weight_reference(std::span<const double> channels, double laplacian,
                                const FusionParams& params) {
  double exposedness = 1.0;
  double mean = 0.0;
  for (const double v : channels) {
    exposedness *=
        std::exp(-(v - 0.5) * (v - 0.5) / (2.0 * params.exposure_sigma * params.exposure_sigma));
    mean += v;
  }
  mean /= static_cast<double>(channels.size());
  // The floor of contrast and saturation, as the GPU adds it.
  constexpr double kFloor = 1.0 / 255.0;
  double weight = std::pow(std::abs(laplacian) + kFloor, params.contrast_weight) *
                  std::pow(exposedness, params.exposure_weight);
  if (channels.size() >= 3) {
    double variance = 0.0;
    for (const double v : channels) variance += (v - mean) * (v - mean);
    weight *= std::pow(std::sqrt(variance / static_cast<double>(channels.size())) + kFloor,
                       params.saturation_weight);
  }
  return weight + 1e-12;
}

}  // namespace holoscan::denoise
