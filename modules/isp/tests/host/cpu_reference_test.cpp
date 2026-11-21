// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The CPU backend without an SDK or a GPU: its bilinear demosaic against an
// independent reference on small random mosaics of every pattern; its
// Malvar-He-Cutler demosaic reproducing linear gradients exactly; packed RAW10
// and RAW12 giving the same image as unpacked samples; and the whole pipeline
// against the synthetic camera's truth.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <holoscan/isp/isp_params.hpp>
#include <holoscan/isp/synthetic_raw.hpp>

#include "cpu_pipeline.hpp"
#include "test_support.hpp"

namespace holoscan::isp {
namespace {

using enum BayerPattern;
constexpr std::array<BayerPattern, 4> kPatterns{kRGGB, kBGGR, kGRBG, kGBRG};

// The colour (0 R, 1 G, 2 B) at (x, y) of `pattern`, from the pattern's name.
int color_at(BayerPattern pattern, int x, int y) {
  const std::string_view name = pattern_name(pattern);
  const char c = name[static_cast<std::size_t>((y & 1) * 2 + (x & 1))];
  return c == 'R' ? 0 : c == 'G' ? 1 : 2;
}

// Runs the CPU backend on a raw view into float RGB.
std::vector<float> process(const IspParams& params, const RawImageView& raw) {
  std::vector<float> out(static_cast<std::size_t>(raw.width) * raw.height * 3U);
  const std::vector<float> lut = tone_lut(params);
  const RgbImageView rgb{out.data(), static_cast<std::size_t>(raw.width) * 12U,
                         OutputFormat::kRgb32f};
  validate_params(params);
  validate_raw(raw, params);
  const detail::FrameParams p =
      detail::make_frame_params(params, raw, rgb, lut.data(), static_cast<int>(lut.size()),
                                params.lens_shading ? params.lens_shading->gains.data() : nullptr);
  detail::AwbState awb = detail::initial_white_balance(params);
  detail::CpuScratch scratch;
  detail::cpu_process(p, params, StatsOutput{}, FrameExposure{}, awb, scratch);
  return out;
}

// Bilinear against an independent reference: each missing colour is the mean
// of the 3x3 neighbours of that colour, reflected at the edges.
void bilinear_reference() {
  for (const BayerPattern pattern : kPatterns) {
    for (const auto& [w, h] : {std::pair{13, 9}, std::pair{2, 2}, std::pair{6, 5}}) {
      std::vector<std::uint16_t> raw(static_cast<std::size_t>(w) * h);
      std::uint32_t seed = 12345U;
      for (std::uint16_t& v : raw) {
        seed = seed * 1664525U + 1013904223U;
        v = static_cast<std::uint16_t>(64U + (seed >> 16) % 4000U);
      }
      IspParams params;
      params.demosaic = DemosaicMethod::kBilinear;
      params.tone_curve = ToneCurve::kLinear;
      params.channel_black_levels = std::array<float, 4>{64.0F, 60.0F, 70.0F, 50.0F};
      params.wb_gains = {1.5F, 1.0F, 1.25F};
      const std::vector<float> got =
          process(params, RawImageView{raw.data(), w, h, 2U * w, RawFormat::kUint16, 12, pattern});

      const auto reflect = [](int v, int n) {
        while (v < 0 || v >= n) v = v < 0 ? -v : 2 * (n - 1) - v;
        return v;
      };
      // Preprocessed samples: black level per channel (R, Gr, Gb, B), gain, clip.
      const auto sample = [&](int x, int y) {
        const int color = color_at(pattern, x, y);
        const bool red_row = color_at(pattern, 0, y) == 0 || color_at(pattern, 1, y) == 0;
        const int channel = color == 0 ? 0 : color == 2 ? 3 : red_row ? 1 : 2;
        const double black = (*params.channel_black_levels)[static_cast<std::size_t>(channel)];
        double v = (raw[static_cast<std::size_t>(y) * w + x] - black) / (4095.0 - black);
        v = std::max(v, 0.0) * params.wb_gains[static_cast<std::size_t>(color)];
        return std::min(v, 1.0);
      };
      double worst = 0.0;
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          for (int c = 0; c < 3; ++c) {
            double want = 0.0;
            if (color_at(pattern, x, y) == c) {
              want = sample(x, y);
            } else {
              double sum = 0.0;
              int n = 0;
              for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                  const int sx = reflect(x + dx, w);
                  const int sy = reflect(y + dy, h);
                  if ((dx != 0 || dy != 0) && color_at(pattern, x + dx, y + dy) == c) {
                    sum += sample(sx, sy);
                    ++n;
                  }
                }
              }
              want = sum / n;
            }
            worst = std::max(worst,
                             std::abs(got[(static_cast<std::size_t>(y) * w + x) * 3U + c] - want));
          }
        }
      }
      VERIFY_MSG(worst < 1e-6, std::string(pattern_name(pattern)) + " " + std::to_string(w) + "x" +
                                   std::to_string(h) + ": off by " + std::to_string(worst));
    }
  }
}

// Malvar-He-Cutler is exact on a linear gradient, away from the edges.
void malvar_gradient() {
  const int w = 24;
  const int h = 20;
  // Each colour a different plane over the image, in counts of 16 bits.
  const auto plane = [](int color, int x, int y) {
    const double a[3] = {9000.0, 20000.0, 14000.0};
    const double bx[3] = {700.0, -300.0, 450.0};
    const double by[3] = {-250.0, 400.0, 600.0};
    return a[color] + bx[color] * x + by[color] * y;
  };
  for (const BayerPattern pattern : kPatterns) {
    std::vector<std::uint16_t> raw(static_cast<std::size_t>(w) * h);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        raw[static_cast<std::size_t>(y) * w + x] =
            static_cast<std::uint16_t>(plane(color_at(pattern, x, y), x, y));
      }
    }
    IspParams params;
    params.demosaic = DemosaicMethod::kMalvar;
    params.tone_curve = ToneCurve::kLinear;
    const std::vector<float> got =
        process(params, RawImageView{raw.data(), w, h, 2U * w, RawFormat::kUint16, 16, pattern});
    double worst = 0.0;
    for (int y = 2; y < h - 2; ++y) {
      for (int x = 2; x < w - 2; ++x) {
        for (int c = 0; c < 3; ++c) {
          worst = std::max(worst, std::abs(got[(static_cast<std::size_t>(y) * w + x) * 3U + c] -
                                           plane(c, x, y) / 65535.0));
        }
      }
    }
    VERIFY_MSG(worst < 2e-6,
               std::string(pattern_name(pattern)) + ": off by " + std::to_string(worst));
  }
}

std::vector<unsigned char> render(const SyntheticRawConfig& config, std::size_t& pitch) {
  pitch = (raw_row_bytes(config.format, config.width) + 1U) & ~std::size_t{1};
  std::vector<unsigned char> bytes(pitch * config.height, 0);
  render_synthetic_raw(make_synthetic_scene(config), bytes.data(), pitch);
  return bytes;
}

// The same samples packed as RAW10 or RAW12 give the same image as uint16.
void packed_formats() {
  for (const auto& [format, bits] :
       {std::pair{RawFormat::kRaw10, 10}, std::pair{RawFormat::kRaw12, 12}}) {
    SyntheticRawConfig config;
    config.width = 70;  // Not a multiple of the packing groups.
    config.height = 30;
    config.bit_depth = bits;
    config.black_levels = {64.0F, 64.0F, 64.0F, 64.0F};
    std::size_t pitch = 0;
    config.format = RawFormat::kUint16;
    const std::vector<unsigned char> unpacked = render(config, pitch);
    const IspParams params = synthetic_isp_params(config);
    const std::vector<float> a =
        process(params, synthetic_raw_view(config, unpacked.data(), pitch));
    config.format = format;
    const std::vector<unsigned char> packed = render(config, pitch);
    const std::vector<float> b = process(params, synthetic_raw_view(config, packed.data(), pitch));
    VERIFY_MSG(a == b, std::string(format == RawFormat::kRaw10 ? "RAW10" : "RAW12") +
                           " differs from the same samples unpacked");
  }
}

// The whole pipeline on the synthetic camera against its truth.
void synthetic_truth_psnr() {
  SyntheticRawConfig config;
  config.width = 320;
  config.height = 240;
  config.vignetting = 0.4F;
  config.black_levels = {256.0F, 260.0F, 252.0F, 258.0F};
  std::size_t pitch = 0;
  const std::vector<unsigned char> raw = render(config, pitch);
  const SyntheticRawScene scene = make_synthetic_scene(config);
  for (const DemosaicMethod method : {DemosaicMethod::kBilinear, DemosaicMethod::kMalvar}) {
    IspParams params = synthetic_isp_params(config);
    params.demosaic = method;
    params.tone_curve = ToneCurve::kLinear;
    const std::vector<float> got = process(params, synthetic_raw_view(config, raw.data(), pitch));
    double squared = 0.0;
    for (int y = 0; y < config.height; ++y) {
      for (int x = 0; x < config.width; ++x) {
        float truth[3];
        synthetic_truth(scene, x, y, truth);
        for (int c = 0; c < 3; ++c) {
          const double d =
              got[(static_cast<std::size_t>(y) * config.width + x) * 3U + c] - truth[c];
          squared += d * d;
        }
      }
    }
    const double psnr = 10.0 * std::log10(3.0 * config.width * config.height / squared);
    std::cout << "  " << demosaic_name(method) << ": linear PSNR " << psnr << " dB\n";
    VERIFY_MSG(psnr > (method == DemosaicMethod::kMalvar ? 32.0 : 26.5),
               std::string(demosaic_name(method)) + ": PSNR " + std::to_string(psnr));
  }
}

}  // namespace
}  // namespace holoscan::isp

int main() {
  using holoscan::isp::test::TestCase;
  return holoscan::isp::test::run({
      TestCase{"bilinear against an independent reference", holoscan::isp::bilinear_reference},
      TestCase{"Malvar-He-Cutler is exact on gradients", holoscan::isp::malvar_gradient},
      TestCase{"packed RAW10 and RAW12", holoscan::isp::packed_formats},
      TestCase{"the synthetic camera's truth", holoscan::isp::synthetic_truth_psnr},
  });
}
