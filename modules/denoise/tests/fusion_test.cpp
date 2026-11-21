// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ExposureFusion and ToneMapper without a graph, on the synthetic HDR scene
// rendered at test time: three (and two) brackets fused against the scene's
// tone-mapped radiance, with the shadow's and the window's texture kept where
// each single bracket loses one of them; gray brackets; the tone map against
// its double-precision reference for float and 16-bit input; determinism; odd
// and tiny sizes; and unusable settings and views.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/denoise/exposure_fusion.hpp>
#include <holoscan/denoise/fusion_params.hpp>
#include <holoscan/denoise/synthetic_scene.hpp>

#include "image_support.hpp"
#include "test_support.hpp"

namespace holoscan::denoise {
namespace {

using test::DeviceImage;

constexpr int kWidth = 640;
constexpr int kHeight = 360;

// Device memory holding host data.
class DeviceBytes {
 public:
  explicit DeviceBytes(const void* data, std::size_t bytes) {
    test::cuda_check(cudaMalloc(&data_, bytes), "cudaMalloc");
    test::cuda_check(cudaMemcpy(data_, data, bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
  }
  ~DeviceBytes() { cudaFree(data_); }
  DeviceBytes(const DeviceBytes&) = delete;
  DeviceBytes& operator=(const DeviceBytes&) = delete;
  [[nodiscard]] void* get() const { return data_; }

 private:
  void* data_{};
};

// Fuses the brackets of `exposures` of `scene`; packed host output.
std::vector<std::uint8_t> fuse(const HdrScene& scene, std::span<const float> exposures,
                               const FusionParams& params = {}) {
  const PixelFormat format = scene.channels == 1 ? PixelFormat::kMono8 : PixelFormat::kRgb8;
  std::vector<std::unique_ptr<DeviceImage>> brackets;
  std::vector<ImageView> views;
  for (const float exposure : exposures) {
    brackets.push_back(std::make_unique<DeviceImage>(format, scene.width, scene.height));
    brackets.back()->upload(render_bracket(scene, 0, exposure));
    views.push_back(brackets.back()->view());
  }
  DeviceImage out(format, scene.width, scene.height);
  ExposureFusion fusion(params);
  fusion.fuse(views, out.view(), nullptr);
  return out.download();
}

// The tone map of the scene's radiance: the truth fusion is compared with.
// White at twice the brightest luminance, so that no colour clips.
std::vector<std::uint8_t> tone_mapped_truth(const HdrScene& scene) {
  const std::vector<float> radiance = render_radiance(scene, 0);
  const HdrImageView view{
      .format = scene.channels == 1 ? HdrFormat::kMono32f : HdrFormat::kRgb32f,
      .width = scene.width,
      .height = scene.height,
      .data = radiance.data(),
      .pitch = static_cast<std::size_t>(scene.width) * scene.channels * sizeof(float)};
  const ToneMapStats stats = tone_map_stats_reference(view);
  ToneMapParams params;
  params.white = static_cast<float>(2.0 * stats.max_luminance * params.key / stats.log_average);
  return tone_map_reference(view, params);
}

double gray_at(const HdrScene& scene, const std::vector<std::uint8_t>& image, int x, int y) {
  const std::uint8_t* p =
      image.data() + (static_cast<std::size_t>(y) * scene.width + x) * scene.channels;
  return scene.channels == 1 ? p[0] : (p[0] + p[1] + p[2]) / 3.0;
}

// How well `image` reproduces the structure of `truth` in `region`, whatever
// its tone curve: the mean normalized cross-correlation of their gray values
// over the 16x16 blocks inside the region where the truth has detail. A block
// that `image` renders nearly flat (clipped or crushed: under 2 code values of
// standard deviation) scores 0.
double structure_score(const HdrScene& scene, const std::vector<std::uint8_t>& image,
                       const std::vector<std::uint8_t>& truth, HdrRegion region) {
  constexpr int kBlock = 16;
  double total = 0.0;
  int blocks = 0;
  for (int by = 0; by + kBlock <= scene.height; by += kBlock) {
    for (int bx = 0; bx + kBlock <= scene.width; bx += kBlock) {
      if (hdr_region(scene, bx) != region || hdr_region(scene, bx + kBlock - 1) != region) {
        continue;
      }
      double mi = 0.0;
      double mt = 0.0;
      for (int y = by; y < by + kBlock; ++y) {
        for (int x = bx; x < bx + kBlock; ++x) {
          mi += gray_at(scene, image, x, y);
          mt += gray_at(scene, truth, x, y);
        }
      }
      constexpr double kCount = kBlock * kBlock;
      mi /= kCount;
      mt /= kCount;
      double vi = 0.0;
      double vt = 0.0;
      double cov = 0.0;
      for (int y = by; y < by + kBlock; ++y) {
        for (int x = bx; x < bx + kBlock; ++x) {
          const double di = gray_at(scene, image, x, y) - mi;
          const double dt = gray_at(scene, truth, x, y) - mt;
          vi += di * di;
          vt += dt * dt;
          cov += di * dt;
        }
      }
      if (vt / kCount < 4.0) continue;  // No detail in the truth here.
      ++blocks;
      if (vi / kCount >= 4.0) total += cov / std::sqrt(vi * vt);
    }
  }
  return blocks == 0 ? 0.0 : total / blocks;
}

// The structure scores of shadow, mid-tones and window.
std::array<double, 3> structure_scores(const HdrScene& scene,
                                       const std::vector<std::uint8_t>& image,
                                       const std::vector<std::uint8_t>& truth) {
  return {structure_score(scene, image, truth, HdrRegion::kShadow),
          structure_score(scene, image, truth, HdrRegion::kMidtones),
          structure_score(scene, image, truth, HdrRegion::kWindow)};
}

double region_mean(const HdrScene& scene, const std::vector<std::uint8_t>& image,
                   HdrRegion region) {
  double sum = 0.0;
  int count = 0;
  for (int y = 0; y < scene.height; ++y) {
    for (int x = 0; x < scene.width; ++x) {
      if (hdr_region(scene, x) != region) continue;
      sum += gray_at(scene, image, x, y);
      ++count;
    }
  }
  return sum / count;
}

// How an image renders a region of the scene: its clipped and crushed samples,
// and the texture it keeps, the mean step between neighbouring stripes of the
// region (the stripes alternate every 8 pixels: vertically in the shadow,
// horizontally in the window).
struct RegionDetail {
  int clipped{};  // Samples at 254 or more.
  int crushed{};  // Samples at 1 or less.
  double texture{};
};

RegionDetail region_detail(const HdrScene& scene, const std::vector<std::uint8_t>& image,
                           HdrRegion region) {
  RegionDetail detail;
  double steps = 0.0;
  int count = 0;
  const int channels = scene.channels;
  // 16 pixels from a region's edge or the image's: there the blend's halo, where
  // the window meets the mid-tones 13 stops darker, may clip.
  constexpr int kMargin = 16;
  for (int y = kMargin; y + kMargin < scene.height; ++y) {
    for (int x = kMargin; x + kMargin < scene.width; ++x) {
      if (hdr_region(scene, x) != region || hdr_region(scene, x - kMargin) != region ||
          hdr_region(scene, x + kMargin) != region) {
        continue;
      }
      for (int c = 0; c < channels; ++c) {
        const std::uint8_t v =
            image[(static_cast<std::size_t>(y) * scene.width + x) * channels + c];
        detail.clipped += v >= 254;
        detail.crushed += v <= 1;
      }
      // The middle of a stripe against the middle of the next.
      const bool shadow = region == HdrRegion::kShadow;
      if ((shadow ? x : y) % 8 != 4) continue;
      const int nx = shadow ? x + 8 : x;
      const int ny = shadow ? y : y + 8;
      const auto at = [&](int px, int py) {
        return static_cast<double>(
            image[(static_cast<std::size_t>(py) * scene.width + px) * channels]);
      };
      steps += std::abs(at(nx, ny) - at(x, y));
      ++count;
    }
  }
  detail.texture = count == 0 ? 0.0 : steps / count;
  return detail;
}

// Whether the shadow and the window both keep their texture, nothing clipped or crushed.
bool keeps_detail(const HdrScene& scene, const std::vector<std::uint8_t>& image,
                  const std::string& name, bool report) {
  const RegionDetail shadow = region_detail(scene, image, HdrRegion::kShadow);
  const RegionDetail window = region_detail(scene, image, HdrRegion::kWindow);
  if (report) {
    std::printf("  %-20s shadow: texture %5.1f crushed %6d | window: texture %5.1f clipped %6d\n",
                name.c_str(), shadow.texture, shadow.crushed, window.texture, window.clipped);
  }
  return shadow.texture >= 8.0 && shadow.crushed == 0 && window.texture >= 8.0 &&
         window.clipped == 0;
}

// Three brackets fused: the shadow and the window both keep their stripes,
// nothing clips or is crushed, and the whole is close to the tone-mapped
// radiance; each bracket alone loses one end of the range.
void check_fusion_against_truth(int channels) {
  const HdrScene scene{.width = kWidth, .height = kHeight, .channels = channels};
  const auto truth = tone_mapped_truth(scene);
  // Reported only: a luminance tone map scales colours, so the blue window can clip.
  static_cast<void>(keeps_detail(scene, truth, "tone-mapped truth", true));
  for (const float exposure : kBracketExposures) {
    const auto bracket = render_bracket(scene, 0, exposure);
    VERIFY_MSG(!keeps_detail(scene, bracket, "bracket x" + std::to_string(exposure), true),
               "a single bracket loses some detail");
  }
  const auto fused = fuse(scene, kBracketExposures);
  VERIFY_MSG(keeps_detail(scene, fused, "fused, 3 brackets", true),
             "fusion keeps the shadow's and the window's texture");
  // Against the truth: the structure of the shadow and the window, each of
  // which some bracket loses, and the regions in the scene's order of
  // brightness. Fusion is not a global tone curve: it flattens the mid-tones'
  // gentle ramp, and its levels differ from the truth's, so the mid-tones'
  // score and the PSNR are only reported.
  const auto fused_scores = structure_scores(scene, fused, truth);
  std::printf("  %-20s structure: shadow %.3f mid-tones %.3f window %.3f, PSNR %.2f dB\n",
              "fused, 3 brackets", fused_scores[0], fused_scores[1], fused_scores[2],
              psnr(fused.data(), truth.data(), truth.size()));
  VERIFY_MSG(fused_scores[0] >= 0.9 && fused_scores[2] >= 0.9,
             "fused structure in the shadow and the window");
  for (const float exposure : kBracketExposures) {
    const auto bracket = render_bracket(scene, 0, exposure);
    const auto scores = structure_scores(scene, bracket, truth);
    std::printf(
        "  bracket x%-11g structure: shadow %.3f mid-tones %.3f window %.3f, PSNR %.2f dB\n",
        exposure, scores[0], scores[1], scores[2],
        psnr(bracket.data(), truth.data(), truth.size()));
    VERIFY_MSG(std::min(scores[0], scores[2]) < 0.5,
               "every bracket loses the shadow or the window");
  }
  const double shadow = region_mean(scene, fused, HdrRegion::kShadow);
  const double midtones = region_mean(scene, fused, HdrRegion::kMidtones);
  const double window = region_mean(scene, fused, HdrRegion::kWindow);
  std::printf("  fused region means: shadow %.1f, mid-tones %.1f, window %.1f\n", shadow, midtones,
              window);
  VERIFY_MSG(shadow < midtones && midtones < window, "fusion keeps the scene's order");

  // Two brackets, the shortest and the longest, cover the range too.
  const std::array<float, 2> pair{kBracketExposures[0], kBracketExposures[2]};
  const auto fused_pair = fuse(scene, pair);
  VERIFY_MSG(keeps_detail(scene, fused_pair, "fused, 2 brackets", true),
             "two brackets keep the detail");
}

// Fusion's weights follow its settings: with exposure weight only, a flat
// mid-gray bracket outweighs a dark one everywhere.
void check_fusion_weights() {
  const int w = 64;
  const int h = 48;
  DeviceImage dark(PixelFormat::kRgb8, w, h);
  DeviceImage mid(PixelFormat::kRgb8, w, h);
  DeviceImage out(PixelFormat::kRgb8, w, h);
  dark.upload(std::vector<std::uint8_t>(static_cast<std::size_t>(w) * h * 3, 20));
  mid.upload(std::vector<std::uint8_t>(static_cast<std::size_t>(w) * h * 3, 128));
  ExposureFusion fusion(FusionParams{.contrast_weight = 0.0F, .saturation_weight = 0.0F});
  const std::array<ImageView, 2> views{dark.view(), mid.view()};
  fusion.fuse(views, out.view(), nullptr);
  const auto result = out.download();
  // Weights exp(-(v - 0.5)^2 / 0.08)^3: 128 has about 1, 20 about 0.04.
  const double w_mid = 1.0;
  const double w_dark = std::pow(std::exp(-std::pow(20.0 / 255.0 - 0.5, 2) / 0.08), 3);
  const double expected = (128.0 * w_mid + 20.0 * w_dark) / (w_mid + w_dark);
  int wrong = 0;
  for (const std::uint8_t v : result) wrong += std::abs(v - expected) > 1.0;
  VERIFY_MSG(wrong == 0,
             std::to_string(wrong) + " samples off the weighted mean " + std::to_string(expected));
  // The reference weight agrees.
  const std::array<double, 3> gray{0.5, 0.5, 0.5};
  VERIFY(std::abs(mertens_weight_reference(
                      gray, 0.0, FusionParams{.contrast_weight = 0.0F, .saturation_weight = 0.0F}) -
                  1.0) < 1e-9);
}

// Fusion runs on RGBA8 (alpha 255), odd and tiny sizes, and is deterministic.
void check_fusion_sizes_and_determinism() {
  for (const auto& [w, h] :
       {std::pair{1, 1}, std::pair{3, 2}, std::pair{17, 9}, std::pair{301, 157}}) {
    const HdrScene scene{.width = w, .height = h, .channels = 3};
    std::vector<std::unique_ptr<DeviceImage>> brackets;
    std::vector<ImageView> views;
    for (const float exposure : kBracketExposures) {
      const auto rgb = render_bracket(scene, 0, exposure);
      std::vector<std::uint8_t> rgba(static_cast<std::size_t>(w) * h * 4, 255);
      for (std::size_t i = 0; i < static_cast<std::size_t>(w) * h; ++i) {
        std::memcpy(&rgba[4 * i], &rgb[3 * i], 3);
        rgba[4 * i + 3] = 7;
      }
      brackets.push_back(std::make_unique<DeviceImage>(PixelFormat::kRgba8, w, h));
      brackets.back()->upload(rgba);
      views.push_back(brackets.back()->view());
    }
    std::vector<std::uint8_t> results[2];
    for (auto& result : results) {
      DeviceImage out(PixelFormat::kRgba8, w, h);
      ExposureFusion fusion(FusionParams{});
      fusion.fuse(views, out.view(), nullptr);
      result = out.download();
    }
    VERIFY_MSG(results[0] == results[1], "fusion is deterministic");
    int alpha = 0;
    for (std::size_t i = 3; i < results[0].size(); i += 4) alpha += results[0][i] != 255;
    VERIFY_MSG(alpha == 0, "fused alpha is 255");
    std::printf("  %dx%d: %d pyramid levels\n", w, h, fusion_levels(FusionParams{}, w, h));
  }
}

// The tone map matches the double-precision reference within one code value,
// for float and 16-bit input, each layout, white and curve.
void check_tone_map() {
  const HdrScene scene{.width = 200, .height = 120, .channels = 3};
  const std::vector<float> rgb = render_radiance(scene, 0);
  const auto pixels = static_cast<std::size_t>(scene.width) * scene.height;
  struct Case {
    HdrFormat format;
    ToneMapParams params;
    int bits;
  };
  const Case cases[] = {
      {HdrFormat::kRgb32f, ToneMapParams{}, 16},
      {HdrFormat::kMono32f, ToneMapParams{.key = 0.36F}, 16},
      {HdrFormat::kRgba32f, ToneMapParams{.white = 4.0F, .curve = ToneCurve::kGamma}, 16},
      {HdrFormat::kRgb16, ToneMapParams{.curve = ToneCurve::kLinear}, 16},
      {HdrFormat::kRgb16, ToneMapParams{}, 12},
      {HdrFormat::kMono16, ToneMapParams{.gamma = 2.4F}, 10},
      {HdrFormat::kRgba16, ToneMapParams{.key = 0.09F}, 16},
  };
  for (const Case& c : cases) {
    const int channels = hdr_channels(c.format);
    const bool is_float = hdr_is_float(c.format);
    // The radiance, scaled to the code range for 16-bit (the window's peak near white).
    std::vector<float> values(pixels * channels);
    std::vector<std::uint16_t> codes(pixels * channels);
    const double white = std::ldexp(1.0, c.bits) - 1.0;
    for (std::size_t i = 0; i < pixels; ++i) {
      for (int k = 0; k < channels; ++k) {
        const float v = k == 3          ? 0.5F
                        : channels == 1 ? 0.2126F * rgb[3 * i] + 0.7152F * rgb[3 * i + 1] +
                                              0.0722F * rgb[3 * i + 2]
                                        : rgb[3 * i + k];
        values[i * channels + k] = v;
        codes[i * channels + k] = static_cast<std::uint16_t>(
            std::min(white, std::floor((k == 3 ? 0.5 : v / 6.5) * white + 0.5)));
      }
    }
    const std::size_t sample = is_float ? sizeof(float) : sizeof(std::uint16_t);
    const void* host = is_float ? static_cast<const void*>(values.data()) : codes.data();
    const DeviceBytes device(host, pixels * channels * sample);
    HdrImageView view{.format = c.format,
                      .width = scene.width,
                      .height = scene.height,
                      .data = host,
                      .pitch = static_cast<std::size_t>(scene.width) * channels * sample,
                      .significant_bits = c.bits};
    const auto expected = tone_map_reference(view, c.params);
    const ToneMapStats reference = tone_map_stats_reference(view);
    const PixelFormat out_format = channels == 1   ? PixelFormat::kMono8
                                   : channels == 3 ? PixelFormat::kRgb8
                                                   : PixelFormat::kRgba8;
    DeviceImage out(out_format, scene.width, scene.height);
    ToneMapper mapper(c.params);
    view.data = device.get();
    mapper.map(view, out.view(), nullptr);
    const auto result = out.download();
    int off = 0;
    int far = 0;
    for (std::size_t i = 0; i < result.size(); ++i) {
      const int d = std::abs(result[i] - expected[i]);
      off += d > 0;
      far += d > 1;
    }
    const ToneMapStats stats = mapper.last_stats(nullptr);
    std::printf("  %-8s %2d bits %-6s: %d of %zu samples 1 off, log-average %.6f (%.6f)\n",
                std::string(hdr_format_name(c.format)).c_str(), c.bits,
                std::string(tone_curve_name(c.params.curve)).c_str(), off, result.size(),
                stats.log_average, reference.log_average);
    VERIFY_MSG(far == 0, std::to_string(far) + " samples more than 1 off the reference");
    VERIFY(std::abs(stats.log_average / reference.log_average - 1.0) < 1e-5);
    VERIFY(std::abs(stats.max_luminance / reference.max_luminance - 1.0) < 1e-5);
  }
  // With white 0, the brightest pixel maps to white and nothing clips beyond it.
  {
    const HdrImageView view{.format = HdrFormat::kRgb32f,
                            .width = scene.width,
                            .height = scene.height,
                            .data = rgb.data(),
                            .pitch = static_cast<std::size_t>(scene.width) * 3 * sizeof(float)};
    const auto mapped = tone_map_reference(view, ToneMapParams{});
    const std::uint8_t peak = *std::max_element(mapped.begin(), mapped.end());
    VERIFY_MSG(peak >= 250, "the brightest pixel nears white: " + std::to_string(peak));
  }
}

// Unusable settings and views are refused.
void check_rejections() {
  using test::throws;
  VERIFY(throws<std::invalid_argument>(
      [] { ExposureFusion f(FusionParams{.exposure_sigma = 0.0F}); }));
  VERIFY(throws<std::invalid_argument>([] {
    ExposureFusion f(
        FusionParams{.contrast_weight = 0.0F, .saturation_weight = 0.0F, .exposure_weight = 0.0F});
  }));
  VERIFY(throws<std::invalid_argument>([] { ToneMapper m(ToneMapParams{.key = 0.0F}); }));
  VERIFY(throws<std::invalid_argument>([] { ToneMapper m(ToneMapParams{.white = -1.0F}); }));
  ExposureFusion fusion(FusionParams{});
  DeviceImage a(PixelFormat::kRgb8, 16, 16);
  DeviceImage b(PixelFormat::kRgb8, 16, 8);
  DeviceImage m(PixelFormat::kMono8, 16, 16);
  const std::array<ImageView, 2> mismatched{a.view(), b.view()};
  VERIFY(throws<std::invalid_argument>([&] { fusion.fuse(mismatched, a.view(), nullptr); }));
  const std::array<ImageView, 1> one{a.view()};
  VERIFY(throws<std::invalid_argument>([&] { fusion.fuse(one, a.view(), nullptr); }));
  const std::array<ImageView, 2> pair{a.view(), a.view()};
  VERIFY(throws<std::invalid_argument>([&] { fusion.fuse(pair, m.view(), nullptr); }));
  ToneMapper mapper(ToneMapParams{});
  const HdrImageView mono{.format = HdrFormat::kMono32f,
                          .width = 16,
                          .height = 16,
                          .data = a.view().planes[0].data,
                          .pitch = 64};
  VERIFY(throws<std::invalid_argument>([&] { mapper.map(mono, a.view(), nullptr); }));
}

}  // namespace
}  // namespace holoscan::denoise

int main() {
  namespace dn = holoscan::denoise;
  return dn::test::run({
      {"fusion against the tone-mapped truth, RGB", [] { dn::check_fusion_against_truth(3); }},
      {"fusion against the tone-mapped truth, gray", [] { dn::check_fusion_against_truth(1); }},
      {"fusion weights", dn::check_fusion_weights},
      {"fusion sizes and determinism", dn::check_fusion_sizes_and_determinism},
      {"tone map against the reference", dn::check_tone_map},
      {"rejections", dn::check_rejections},
  });
}
