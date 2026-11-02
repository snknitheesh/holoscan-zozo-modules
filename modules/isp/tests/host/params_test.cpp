// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The settings without an SDK or a GPU: names, the auto backend's choice,
// validation of settings and raw views, packed row sizes and the tone tables.

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <holoscan/isp/isp_params.hpp>

#include "test_support.hpp"

namespace holoscan::isp {
namespace {

using test::throws;

void names() {
  for (const Backend b : {Backend::kCuda, Backend::kCpu, Backend::kAuto}) {
    VERIFY(parse_backend(backend_name(b)) == b);
  }
  for (const DemosaicMethod m : {DemosaicMethod::kBilinear, DemosaicMethod::kMalvar}) {
    VERIFY(parse_demosaic(demosaic_name(m)) == m);
  }
  for (const BayerPattern p :
       {BayerPattern::kRGGB, BayerPattern::kBGGR, BayerPattern::kGRBG, BayerPattern::kGBRG}) {
    VERIFY(parse_pattern(pattern_name(p)) == p);
  }
  for (const WhiteBalanceMode w :
       {WhiteBalanceMode::kManual, WhiteBalanceMode::kGrayWorld, WhiteBalanceMode::kWhitePatch}) {
    VERIFY(parse_white_balance(white_balance_name(w)) == w);
  }
  VERIFY(!parse_backend("vpi") && !parse_demosaic("vng") && !parse_pattern("rggb"));
}

void auto_backend() {
  VERIFY(resolve_auto(PlatformFacts{.cuda_device = true}) == Backend::kCuda);
  VERIFY(resolve_auto(PlatformFacts{.cuda_device = false}) == Backend::kCpu);
}

void validation() {
  VERIFY(!throws<std::invalid_argument>([] { validate_params(IspParams{}); }));
  const auto bad = [](auto tweak) {
    IspParams params;
    tweak(params);
    return throws<std::invalid_argument>([&] { validate_params(params); });
  };
  VERIFY(bad([](IspParams& p) { p.white_level = 70000.0F; }));
  VERIFY(bad([](IspParams& p) { p.channel_black_levels = std::array<float, 4>{0, 0, -1, 0}; }));
  VERIFY(bad([](IspParams& p) { p.lens_shading = LensShading{1, 2, 1, {1.0F, 1.0F}}; }));
  VERIFY(bad(
      [](IspParams& p) { p.lens_shading = LensShading{2, 2, 3, std::vector<float>(12, 1.0F)}; }));
  VERIFY(
      bad([](IspParams& p) { p.lens_shading = LensShading{2, 2, 1, {1.0F, 1.0F, 1.0F, 0.0F}}; }));
  VERIFY(bad([](IspParams& p) { p.white_patch_percentile = 1.0F; }));
  VERIFY(bad([](IspParams& p) {
    p.tone_curve = ToneCurve::kGamma;
    p.gamma = 0.0F;
  }));
  VERIFY(bad([](IspParams& p) {
    p.tone_curve = ToneCurve::kLut;
    p.lut = {0.0F, 1.5F};
  }));
  VERIFY(bad([](IspParams& p) { p.auto_exposure.target = 1.0F; }));
  VERIFY(bad([](IspParams& p) {
    p.auto_exposure.min_gain = 4.0F;
    p.auto_exposure.max_gain = 2.0F;
  }));
  VERIFY(!bad(
      [](IspParams& p) { p.lens_shading = LensShading{2, 2, 4, std::vector<float>(16, 2.0F)}; }));

  std::vector<std::uint8_t> bytes(64, 0);
  const RawImageView good{bytes.data(), 4, 4, 8, RawFormat::kUint16, 12, BayerPattern::kRGGB};
  VERIFY(!throws<std::invalid_argument>([&] { validate_raw(good, IspParams{}); }));
  const auto rejected = [&](auto tweak, IspParams params = {}) {
    RawImageView view = good;
    tweak(view);
    return throws<std::invalid_argument>([&] { validate_raw(view, params); });
  };
  VERIFY(rejected([](RawImageView& v) { v.height = 1; }));
  VERIFY(rejected([](RawImageView& v) { v.pitch = 7; }));
  VERIFY(rejected([](RawImageView& v) { v.format = RawFormat::kUint8; }));  // 12 bits in uint8.
  VERIFY(rejected([](RawImageView& v) {
    v.format = RawFormat::kRaw12;
    v.bit_depth = 10;
  }));
  VERIFY(!rejected([](RawImageView& v) { v.format = RawFormat::kRaw12; }));
  VERIFY(rejected([](RawImageView&) {}, IspParams{.white_level = 5000.0F}));
  VERIFY(rejected([](RawImageView&) {}, IspParams{.black_level = 4095.0F}));
}

void row_bytes() {
  VERIFY(raw_row_bytes(RawFormat::kUint8, 7) == 7);
  VERIFY(raw_row_bytes(RawFormat::kUint16, 7) == 14);
  VERIFY(raw_row_bytes(RawFormat::kRaw10, 4) == 5 && raw_row_bytes(RawFormat::kRaw10, 5) == 10);
  VERIFY(raw_row_bytes(RawFormat::kRaw12, 2) == 3 && raw_row_bytes(RawFormat::kRaw12, 3) == 6);
}

void tone_tables() {
  IspParams params;
  for (const ToneCurve curve : {ToneCurve::kSrgb, ToneCurve::kGamma, ToneCurve::kLinear}) {
    params.tone_curve = curve;
    const std::vector<float> lut = tone_lut(params);
    VERIFY(lut.size() == kToneLutSize && lut.front() == 0.0F && lut.back() == 1.0F);
    bool rising = true;
    for (std::size_t i = 1; i < lut.size(); ++i) rising = rising && lut[i] > lut[i - 1];
    VERIFY(rising);
  }
  params.tone_curve = ToneCurve::kSrgb;
  // sRGB codes 18 % gray as about 0.46.
  VERIFY(std::abs(tone_lut(params)[static_cast<std::size_t>(0.18 * (kToneLutSize - 1))] - 0.4613F) <
         2e-3F);
  params.tone_curve = ToneCurve::kLut;
  params.lut = {0.0F, 0.25F, 1.0F};
  VERIFY(tone_lut(params) == params.lut);
}

}  // namespace
}  // namespace holoscan::isp

int main() {
  using holoscan::isp::test::TestCase;
  return holoscan::isp::test::run({
      TestCase{"names", holoscan::isp::names},
      TestCase{"auto backend", holoscan::isp::auto_backend},
      TestCase{"validation", holoscan::isp::validation},
      TestCase{"row bytes", holoscan::isp::row_bytes},
      TestCase{"tone tables", holoscan::isp::tone_tables},
  });
}
