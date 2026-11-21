// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The settings without an SDK, CUDA or GPU: names, validation, the auto
// backend's choice, the strength's mapping, pyramid levels, and the tone map
// and fusion weight references against values worked out by hand.

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/denoise/denoise_params.hpp>
#include <holoscan/denoise/fusion_params.hpp>

#include "test_support.hpp"

namespace holoscan::denoise {
namespace {

using test::throws;

void names() {
  for (const Backend backend : {Backend::kCuda, Backend::kVpi, Backend::kAuto}) {
    VERIFY(parse_backend(backend_name(backend)) == backend);
  }
  for (const SpatialFilter filter :
       {SpatialFilter::kNone, SpatialFilter::kBilateral, SpatialFilter::kNlm}) {
    VERIFY(parse_spatial(spatial_name(filter)) == filter);
  }
  for (const VpiEngine engine : {VpiEngine::kAuto, VpiEngine::kVic, VpiEngine::kCuda}) {
    VERIFY(parse_vpi_engine(vpi_engine_name(engine)) == engine);
  }
  for (const ToneCurve curve : {ToneCurve::kSrgb, ToneCurve::kGamma, ToneCurve::kLinear}) {
    VERIFY(parse_tone_curve(tone_curve_name(curve)) == curve);
  }
  VERIFY(!parse_backend("cpu") && !parse_spatial("median") && !parse_tone_curve("aces"));
  VERIFY(format_name(PixelFormat::kNv12) == "NV12");
  VERIFY(hdr_format_name(HdrFormat::kRgba16) == "RGBA16");
  VERIFY(available_backends().front() == Backend::kCuda);
}

void validation() {
  VERIFY(!throws<std::invalid_argument>([] { validate_params(DenoiseParams{}); }));
  const auto rejects = [](DenoiseParams p) {
    return throws<std::invalid_argument>([&] { validate_params(p); });
  };
  VERIFY(rejects(DenoiseParams{.spatial = SpatialFilter::kNone, .temporal = false}));
  VERIFY(rejects(DenoiseParams{.strength = -0.1F}));
  VERIFY(rejects(DenoiseParams{.strength = std::nanf("")}));
  VERIFY(rejects(DenoiseParams{.noise_sigma = -1.0F}));
  VERIFY(rejects(DenoiseParams{.motion_threshold = 0.5F}));
  VERIFY(rejects(DenoiseParams{.max_history = 1}));
  VERIFY(rejects(DenoiseParams{.max_history = 256}));
  VERIFY(rejects(DenoiseParams{.scene_cut_fraction = 0.0F}));
  VERIFY(rejects(DenoiseParams{.scene_cut_fraction = 1.5F}));

  VERIFY(unsupported_reason(Backend::kCuda, DenoiseParams{}).empty());
  VERIFY(!unsupported_reason(Backend::kVpi, DenoiseParams{}).empty());
  VERIFY(unsupported_reason(Backend::kVpi, DenoiseParams{.spatial = SpatialFilter::kNone}).empty());
  VERIFY(!unsupported_reason(Backend::kVpi, DenoiseParams{.temporal = false}).empty());

  VERIFY(!throws<std::invalid_argument>([] { validate_fusion_params(FusionParams{}); }));
  VERIFY(throws<std::invalid_argument>(
      [] { validate_fusion_params(FusionParams{.contrast_weight = 5.0F}); }));
  VERIFY(throws<std::invalid_argument>([] { validate_fusion_params(FusionParams{.levels = 17}); }));
  VERIFY(!throws<std::invalid_argument>([] { validate_tone_map_params(ToneMapParams{}); }));
  VERIFY(throws<std::invalid_argument>(
      [] { validate_tone_map_params(ToneMapParams{.gamma = 0.5F}); }));
  VERIFY(throws<std::invalid_argument>(
      [] { validate_tone_map_params(ToneMapParams{.white = INFINITY}); }));
}

// vpi only with a VIC that runs TNR and settings that suit it.
void auto_choice() {
  const DenoiseParams tnr{.spatial = SpatialFilter::kNone};
  const DenoiseParams bilateral{};
  VERIFY(resolve_auto({.vpi_compiled = true, .vic_tnr = true}, tnr) == Backend::kVpi);
  VERIFY(resolve_auto({.vpi_compiled = true, .vic_tnr = true}, bilateral) == Backend::kCuda);
  VERIFY(resolve_auto({.vpi_compiled = true, .vic_tnr = false}, tnr) == Backend::kCuda);
  VERIFY(resolve_auto({.vpi_compiled = false, .vic_tnr = true}, tnr) == Backend::kCuda);
  VERIFY(resolve_auto({.vpi_compiled = true, .vic_tnr = true},
                      DenoiseParams{.spatial = SpatialFilter::kNlm, .temporal = false}) ==
         Backend::kCuda);
}

void strength_mapping() {
  namespace t = tuning;
  VERIFY(t::history_frames(DenoiseParams{.strength = 0.0F}) == 1);
  VERIFY(t::history_frames(DenoiseParams{.strength = 1.0F}) == 32);
  VERIFY(t::history_frames(DenoiseParams{.strength = 0.5F}) == 17);
  VERIFY(t::history_frames(DenoiseParams{.strength = 1.0F, .max_history = 8}) == 8);
  // A running mean of n frames keeps 1/n of the variance; the recursive filter
  // at weight 1/32 settles at (1/32) / (2 - 1/32).
  VERIFY(std::abs(t::residual_variance(4, 32) - 0.25F) < 1e-7F);
  VERIFY(std::abs(t::residual_variance(1000, 32) - (1.0F / 32) / (2.0F - 1.0F / 32)) < 1e-7F);
  VERIFY(t::residual_variance(1, 1) == 1.0F);
}

void levels() {
  VERIFY(fusion_levels(FusionParams{}, 1920, 1080) == kAutoFusionLevels);
  VERIFY(fusion_levels(FusionParams{}, 3840, 2160) == kAutoFusionLevels);
  VERIFY(fusion_levels(FusionParams{}, 320, 120) == 4);
  VERIFY(fusion_levels(FusionParams{}, 15, 600) == 1);
  VERIFY(fusion_levels(FusionParams{.levels = 3}, 640, 480) == 3);
  // No more levels than halvings down to one pixel.
  VERIFY(fusion_levels(FusionParams{.levels = 16}, 8, 4) == 4);
  VERIFY(fusion_levels(FusionParams{.levels = 16}, 1, 1) == 1);
}

void references() {
  // sRGB: 0 and 1 exact, 0.0031308 on the linear segment, 0.5 at 188.
  VERIFY(encode_reference(0.0, ToneMapParams{}) == 0);
  VERIFY(encode_reference(1.0, ToneMapParams{}) == 255);
  VERIFY(encode_reference(2.0, ToneMapParams{}) == 255);
  VERIFY(encode_reference(0.5, ToneMapParams{}) == 188);
  VERIFY(encode_reference(0.5, ToneMapParams{.curve = ToneCurve::kLinear}) == 128);
  VERIFY(encode_reference(0.25, ToneMapParams{.curve = ToneCurve::kGamma, .gamma = 2.0F}) == 128);

  // A two-pixel gray frame of luminances 1 and 4: log-average 2, so with key
  // 0.18 the scaled values are 0.09 and 0.36; white 0 maps 0.36 to 1.
  const std::array<float, 2> pixels{1.0F, 4.0F};
  const HdrImageView view{.format = HdrFormat::kMono32f,
                          .width = 2,
                          .height = 1,
                          .data = pixels.data(),
                          .pitch = sizeof(pixels)};
  const ToneMapStats stats = tone_map_stats_reference(view);
  VERIFY(std::abs(stats.log_average - 2.0) < 1e-5 && stats.max_luminance == 4.0);
  const auto mapped = tone_map_reference(view, ToneMapParams{.curve = ToneCurve::kLinear});
  const double lw = 0.36;
  const double low = 0.09 * (1 + 0.09 / (lw * lw)) / 1.09;
  VERIFY(mapped.size() == 2 && mapped[1] == 255);
  VERIFY(mapped[0] == static_cast<std::uint8_t>(std::floor(low * 255 + 0.5)));

  // 16-bit codes are divided by 2^bits - 1; alpha is carried, linear.
  const std::array<std::uint16_t, 4> rgba{4095, 4095, 4095, 2048};
  const HdrImageView codes{.format = HdrFormat::kRgba16,
                           .width = 1,
                           .height = 1,
                           .data = rgba.data(),
                           .pitch = sizeof(rgba),
                           .significant_bits = 12};
  VERIFY(std::abs(tone_map_stats_reference(codes).max_luminance - 1.0) < 1e-12);
  VERIFY(tone_map_reference(codes, ToneMapParams{})[3] == 128);

  // Mertens: mid-gray with no contrast weighs 1 x floor; saturation of a gray
  // pixel is 0 plus the floor; exposedness halves 0.5 away by exp(-0.25/0.08).
  const std::array<double, 1> gray{0.5};
  VERIFY(std::abs(mertens_weight_reference(gray, 0.0, FusionParams{}) - (1.0 / 255 + 1e-12)) <
         1e-12);
  const std::array<double, 3> rgb{0.5, 0.5, 0.5};
  VERIFY(std::abs(mertens_weight_reference(rgb, 0.0, FusionParams{}) - (1.0 / 255 / 255 + 1e-12)) <
         1e-12);
  const std::array<double, 1> black{0.0};
  VERIFY(std::abs(mertens_weight_reference(black, 1.0, FusionParams{.contrast_weight = 0.0F}) -
                  std::exp(-0.25 / 0.08)) < 1e-6);
}

}  // namespace
}  // namespace holoscan::denoise

int main() {
  namespace dn = holoscan::denoise;
  return dn::test::run({
      {"names", dn::names},
      {"validation", dn::validation},
      {"auto backend choice", dn::auto_choice},
      {"strength mapping", dn::strength_mapping},
      {"fusion levels", dn::levels},
      {"tone map and fusion references", dn::references},
  });
}
