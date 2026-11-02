// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Exposure fusion and tone mapping settings, their high-bit-depth image view,
// and host reference implementations: plain C++, with no SDK or CUDA.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace holoscan::denoise {

// Mertens, Kautz and Van Reeth's exposure fusion: each bracket's pixels weighted
// by contrast^c * saturation^s * well-exposedness^e, blended over Laplacian
// pyramids. Contrast and saturation are floored at 1/255, so that where they are
// 0 in every bracket (flat or gray areas) well-exposedness still decides.
struct FusionParams {
  float contrast_weight{1.0F};    // Exponent c, 0..4.
  float saturation_weight{1.0F};  // Exponent s, 0..4 (no effect on gray brackets).
  float exposure_weight{1.0F};    // Exponent e, 0..4.
  // Well-exposedness is exp(-(v - 0.5)^2 / (2 sigma^2)) per channel, v in 0..1. 0.05..1.
  float exposure_sigma{0.2F};
  // Pyramid levels, 1..16; 0 chooses up to kAutoFusionLevels, while the top
  // level keeps at least 8 pixels on its shorter side. A deeper pyramid blends
  // over wider areas: at an edge between very different exposures it leaves a
  // halo about 2^(levels - 3) pixels wide.
  int levels{0};
};

// The most pyramid levels FusionParams::levels = 0 chooses: halos of at most
// about 8 pixels, at any image size.
inline constexpr int kAutoFusionLevels = 6;

// How tone-mapped values in 0..1 are coded into 8 bits.
enum class ToneCurve : std::uint8_t {
  kSrgb,    // The sRGB transfer function.
  kGamma,   // v^(1/gamma).
  kLinear,  // v.
};

// Reinhard's global operator: each pixel's luminance L, scaled to Lm = key * L /
// Lavg (Lavg the frame's log-average luminance), becomes Lm (1 + Lm / Lw^2) /
// (1 + Lm), and the pixel's colour is scaled by that over L.
struct ToneMapParams {
  float key{0.18F};  // Middle gray, 0.01..1.
  // Lw: the scaled luminance that maps to white; 0 takes the frame's brightest
  // pixel (so that nothing clips), otherwise > 0.
  float white{0.0F};
  ToneCurve curve{ToneCurve::kSrgb};
  float gamma{2.2F};  // For kGamma, 1..5.
};

// @throws std::invalid_argument naming the first unusable field.
void validate_fusion_params(const FusionParams& params);
void validate_tone_map_params(const ToneMapParams& params);

[[nodiscard]] std::string_view tone_curve_name(ToneCurve curve) noexcept;
[[nodiscard]] std::optional<ToneCurve> parse_tone_curve(std::string_view name) noexcept;

// The pyramid levels fusion uses for a width x height image.
[[nodiscard]] int fusion_levels(const FusionParams& params, int width, int height) noexcept;

// High-bit-depth layouts the tone mapper takes.
enum class HdrFormat : std::uint8_t {
  kMono16,
  kRgb16,
  kRgba16,
  kMono32f,
  kRgb32f,
  kRgba32f,
};

[[nodiscard]] constexpr int hdr_channels(HdrFormat format) noexcept {
  return format == HdrFormat::kMono16 || format == HdrFormat::kMono32f ? 1
         : format == HdrFormat::kRgb16 || format == HdrFormat::kRgb32f ? 3
                                                                       : 4;
}
[[nodiscard]] constexpr bool hdr_is_float(HdrFormat format) noexcept {
  return format == HdrFormat::kMono32f || format == HdrFormat::kRgb32f ||
         format == HdrFormat::kRgba32f;
}
[[nodiscard]] std::string_view hdr_format_name(HdrFormat format) noexcept;

// A borrowed high-bit-depth image: uint16 code values of `significant_bits`
// bits (1..16; the white is 2^bits - 1), or float linear values of any scale.
// Alpha, when present, is coded alike and carried to the output.
struct HdrImageView {
  HdrFormat format{HdrFormat::kRgb32f};
  int width{};
  int height{};
  const void* data{};
  std::size_t pitch{};  // Bytes between rows.
  int significant_bits{16};
};

// The frame statistics the tone map uses.
struct ToneMapStats {
  double log_average{};  // exp(mean(log(1e-6 + L))), L in the input's linear units / white.
  double max_luminance{};
};

// Reference implementations on the host, in double precision, for tests and
// for checking a GPU result. `view.data` is host memory.
[[nodiscard]] ToneMapStats tone_map_stats_reference(const HdrImageView& view);
// 8-bit output of the tone map: 1, 3 or 4 bytes per pixel as the input has channels.
[[nodiscard]] std::vector<std::uint8_t> tone_map_reference(const HdrImageView& view,
                                                           const ToneMapParams& params);
// The code value of a linear 0..1 value on `curve`, rounded.
[[nodiscard]] std::uint8_t encode_reference(double value, const ToneMapParams& params);
// Mertens' weight of one pixel, before normalization, from its 0..1 channel
// values (1 or 3) and the magnitude of the Laplacian of its gray value.
[[nodiscard]] double mertens_weight_reference(std::span<const double> channels, double laplacian,
                                              const FusionParams& params);

}  // namespace holoscan::denoise
