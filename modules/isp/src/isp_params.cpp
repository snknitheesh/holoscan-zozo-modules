// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/isp/isp_params.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace holoscan::isp {
namespace {

template <typename Enum, std::size_t N>
std::string_view name_of(const std::array<std::pair<Enum, std::string_view>, N>& names,
                         Enum value) noexcept {
  for (const auto& [kind, name] : names) {
    if (kind == value) {
      return name;
    }
  }
  return "unknown";
}

template <typename Enum, std::size_t N>
std::optional<Enum> parse_of(const std::array<std::pair<Enum, std::string_view>, N>& names,
                             std::string_view text) noexcept {
  for (const auto& [kind, name] : names) {
    if (name == text) {
      return kind;
    }
  }
  return std::nullopt;
}

constexpr std::array<std::pair<Backend, std::string_view>, 3> kBackends{{
    {Backend::kCuda, "cuda"},
    {Backend::kCpu, "cpu"},
    {Backend::kAuto, "auto"},
}};
constexpr std::array<std::pair<DemosaicMethod, std::string_view>, 2> kDemosaics{{
    {DemosaicMethod::kBilinear, "bilinear"},
    {DemosaicMethod::kMalvar, "malvar"},
}};
constexpr std::array<std::pair<BayerPattern, std::string_view>, 4> kPatterns{{
    {BayerPattern::kRGGB, "RGGB"},
    {BayerPattern::kBGGR, "BGGR"},
    {BayerPattern::kGRBG, "GRBG"},
    {BayerPattern::kGBRG, "GBRG"},
}};
constexpr std::array<std::pair<WhiteBalanceMode, std::string_view>, 3> kWhiteBalances{{
    {WhiteBalanceMode::kManual, "manual"},
    {WhiteBalanceMode::kGrayWorld, "gray-world"},
    {WhiteBalanceMode::kWhitePatch, "white-patch"},
}};

bool finite_in(float value, float low, float high) {
  return std::isfinite(value) && value >= low && value <= high;
}

}  // namespace

std::string_view backend_name(Backend backend) noexcept { return name_of(kBackends, backend); }
std::optional<Backend> parse_backend(std::string_view name) noexcept {
  return parse_of(kBackends, name);
}
std::string_view demosaic_name(DemosaicMethod method) noexcept {
  return name_of(kDemosaics, method);
}
std::optional<DemosaicMethod> parse_demosaic(std::string_view name) noexcept {
  return parse_of(kDemosaics, name);
}
std::string_view pattern_name(BayerPattern pattern) noexcept { return name_of(kPatterns, pattern); }
std::optional<BayerPattern> parse_pattern(std::string_view name) noexcept {
  return parse_of(kPatterns, name);
}
std::string_view white_balance_name(WhiteBalanceMode mode) noexcept {
  return name_of(kWhiteBalances, mode);
}
std::optional<WhiteBalanceMode> parse_white_balance(std::string_view name) noexcept {
  return parse_of(kWhiteBalances, name);
}

void validate_params(const IspParams& params) {
  if (!finite_in(params.black_level, 0.0F, 65535.0F)) {
    throw std::invalid_argument("black_level must be 0..65535");
  }
  if (params.channel_black_levels) {
    for (const float level : *params.channel_black_levels) {
      if (!finite_in(level, 0.0F, 65535.0F)) {
        throw std::invalid_argument("channel_black_levels must be 0..65535");
      }
    }
  }
  if (!finite_in(params.white_level, 0.0F, 65535.0F)) {
    throw std::invalid_argument("white_level must be 0 (from the bit depth) or up to 65535");
  }
  if (params.lens_shading) {
    const LensShading& lsc = *params.lens_shading;
    if (lsc.width < 2 || lsc.height < 2 || lsc.width > 256 || lsc.height > 256) {
      throw std::invalid_argument("lens_shading grid must be 2x2 to 256x256 points");
    }
    if (lsc.channels != 1 && lsc.channels != 4) {
      throw std::invalid_argument("lens_shading channels must be 1 or 4");
    }
    if (lsc.gains.size() != static_cast<std::size_t>(lsc.width) * lsc.height * lsc.channels) {
      throw std::invalid_argument("lens_shading gains must hold width x height x channels values");
    }
    for (const float gain : lsc.gains) {
      if (!(std::isfinite(gain) && gain > 0.0F && gain <= 16.0F)) {
        throw std::invalid_argument("lens_shading gains must be in (0, 16]");
      }
    }
  }
  for (const float gain : params.wb_gains) {
    if (!(std::isfinite(gain) && gain > 0.0F && gain <= 64.0F)) {
      throw std::invalid_argument("wb_gains must be in (0, 64]");
    }
  }
  if (!(std::isfinite(params.awb_smoothing) && params.awb_smoothing > 0.0F &&
        params.awb_smoothing <= 1.0F)) {
    throw std::invalid_argument("awb_smoothing must be in (0, 1]");
  }
  if (!(std::isfinite(params.white_patch_percentile) && params.white_patch_percentile >= 0.5F &&
        params.white_patch_percentile < 1.0F)) {
    throw std::invalid_argument("white_patch_percentile must be in [0.5, 1)");
  }
  for (const float value : params.ccm) {
    if (!finite_in(value, -16.0F, 16.0F)) {
      throw std::invalid_argument("ccm entries must be finite, -16..16");
    }
  }
  if (params.tone_curve == ToneCurve::kGamma && !finite_in(params.gamma, 0.1F, 10.0F)) {
    throw std::invalid_argument("gamma must be 0.1..10");
  }
  if (params.tone_curve == ToneCurve::kLut) {
    if (params.lut.size() < 2 || params.lut.size() > 65536) {
      throw std::invalid_argument("lut must have 2..65536 entries");
    }
    for (const float value : params.lut) {
      if (!finite_in(value, 0.0F, 1.0F)) {
        throw std::invalid_argument("lut entries must be 0..1");
      }
    }
  }
  if (!finite_in(params.sharpen, 0.0F, 4.0F)) {
    throw std::invalid_argument("sharpen must be 0..4");
  }
  const AutoExposureParams& ae = params.auto_exposure;
  if (!(std::isfinite(ae.target) && ae.target > 0.0F && ae.target < 1.0F)) {
    throw std::invalid_argument("auto_exposure.target must be in (0, 1)");
  }
  if (!finite_in(ae.damping, 0.0F, 1.0F)) {
    throw std::invalid_argument("auto_exposure.damping must be 0..1");
  }
  if (ae.min_exposure_ns <= 0 || ae.max_exposure_ns < ae.min_exposure_ns) {
    throw std::invalid_argument("auto_exposure needs 0 < min_exposure_ns <= max_exposure_ns");
  }
  if (!(std::isfinite(ae.min_gain) && std::isfinite(ae.max_gain) && ae.min_gain > 0.0F &&
        ae.max_gain >= ae.min_gain)) {
    throw std::invalid_argument("auto_exposure needs 0 < min_gain <= max_gain");
  }
}

std::vector<float> tone_lut(const IspParams& params) {
  if (params.tone_curve == ToneCurve::kLut) {
    return params.lut;
  }
  std::vector<float> lut(kToneLutSize);
  for (int i = 0; i < kToneLutSize; ++i) {
    const double x = static_cast<double>(i) / (kToneLutSize - 1);
    double y = x;
    if (params.tone_curve == ToneCurve::kSrgb) {
      y = x <= 0.0031308 ? 12.92 * x : 1.055 * std::pow(x, 1.0 / 2.4) - 0.055;
    } else if (params.tone_curve == ToneCurve::kGamma) {
      y = std::pow(x, 1.0 / static_cast<double>(params.gamma));
    }
    lut[static_cast<std::size_t>(i)] = static_cast<float>(y < 0.0 ? 0.0 : y > 1.0 ? 1.0 : y);
  }
  return lut;
}

std::size_t raw_row_bytes(RawFormat format, int width) noexcept {
  const auto w = static_cast<std::size_t>(width < 0 ? 0 : width);
  switch (format) {
    case RawFormat::kUint8:
      return w;
    case RawFormat::kUint16:
      return 2U * w;
    case RawFormat::kRaw10:
      return 5U * ((w + 3U) / 4U);
    case RawFormat::kRaw12:
      return 3U * ((w + 1U) / 2U);
  }
  return 0;
}

void validate_raw(const RawImageView& raw, const IspParams& params) {
  if (raw.data == nullptr) {
    throw std::invalid_argument("raw image has no data");
  }
  if (raw.width < 2 || raw.height < 2 || raw.width > 16384 || raw.height > 16384) {
    throw std::invalid_argument("raw image must be 2x2 to 16384x16384");
  }
  if (raw.pitch < raw_row_bytes(raw.format, raw.width)) {
    throw std::invalid_argument("raw image pitch is shorter than a row");
  }
  if (raw.format == RawFormat::kUint16 &&
      (raw.pitch % 2U != 0U || reinterpret_cast<std::uintptr_t>(raw.data) % 2U != 0U)) {
    throw std::invalid_argument("uint16 raw samples must be aligned to 2 bytes, rows too");
  }
  const int max_bits = raw.format == RawFormat::kUint8   ? 8
                       : raw.format == RawFormat::kRaw10 ? 10
                       : raw.format == RawFormat::kRaw12 ? 12
                                                         : 16;
  const bool packed = raw.format == RawFormat::kRaw10 || raw.format == RawFormat::kRaw12;
  if (raw.bit_depth < 1 || raw.bit_depth > max_bits || (packed && raw.bit_depth != max_bits)) {
    throw std::invalid_argument("raw bit depth " + std::to_string(raw.bit_depth) +
                                " does not fit its sample format");
  }
  const float white = params.white_level > 0.0F
                          ? params.white_level
                          : static_cast<float>((1U << static_cast<unsigned>(raw.bit_depth)) - 1U);
  if (white > static_cast<float>((1U << static_cast<unsigned>(raw.bit_depth)) - 1U)) {
    throw std::invalid_argument("white_level exceeds the raw bit depth");
  }
  const std::array<float, 4> black =
      params.channel_black_levels ? *params.channel_black_levels
                                  : std::array<float, 4>{params.black_level, params.black_level,
                                                         params.black_level, params.black_level};
  for (const float level : black) {
    if (!(level < white)) {
      throw std::invalid_argument("black levels must be below the white level");
    }
  }
}

}  // namespace holoscan::isp
