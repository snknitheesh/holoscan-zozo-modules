// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// IspPipeline on synthetic raw frames made at test time: accuracy against the
// ground truth (PSNR, CIEDE2000) for every Bayer pattern, sample format and
// demosaic method, with black levels, white balance gains and vignetting to
// undo; the CUDA backend bit for bit against the CPU backend; white balance
// gains recovered by gray world and white patch, and their smoothing; the
// colour matrix, the tone curves and the statistics; determinism; odd sizes,
// saturation and black frames; and rejected settings and views.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/isp/isp_params.hpp>
#include <holoscan/isp/isp_pipeline.hpp>
#include <holoscan/isp/synthetic_raw.hpp>
#include <holoscan/isp/synthetic_raw_cuda.hpp>

#include "scene_support.hpp"
#include "test_support.hpp"

namespace holoscan::isp {
namespace {

using test::Result;
using test::run;
using test::throws;

using enum BayerPattern;
constexpr std::array<BayerPattern, 4> kPatterns{kRGGB, kBGGR, kGRBG, kGBRG};

struct SampleFormat {
  RawFormat format;
  int bits;
  float black;
};
constexpr std::array<SampleFormat, 6> kFormats{{
    {RawFormat::kUint8, 8, 16.0F},
    {RawFormat::kUint16, 10, 64.0F},
    {RawFormat::kUint16, 12, 256.0F},
    {RawFormat::kUint16, 16, 4096.0F},
    {RawFormat::kRaw10, 10, 64.0F},
    {RawFormat::kRaw12, 12, 256.0F},
}};

std::string format_name(const SampleFormat& f) {
  switch (f.format) {
    case RawFormat::kUint8:
      return "uint8";
    case RawFormat::kRaw10:
      return "RAW10";
    case RawFormat::kRaw12:
      return "RAW12";
    case RawFormat::kUint16:
      break;
  }
  return "uint16/" + std::to_string(f.bits);
}

// A camera with everything to undo: per-channel black levels, white balance
// gains, a colour matrix and vignetting.
SyntheticRawConfig camera(BayerPattern pattern, const SampleFormat& f, int width = 480,
                          int height = 360) {
  SyntheticRawConfig config;
  config.width = width;
  config.height = height;
  config.pattern = pattern;
  config.format = f.format;
  config.bit_depth = f.bits;
  const float step = f.black / 16.0F;
  config.black_levels = {f.black, f.black + step, f.black + 2.0F * step, f.black + 3.0F * step};
  config.vignetting = 0.4F;
  config.frame = 3;
  return config;
}

std::size_t mismatches(const Result& a, const Result& b) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < a.image.size() && i < b.image.size(); ++i) {
    count += a.image[i] != b.image[i] ? 1U : 0U;
  }
  return count + (a.image.size() != b.image.size() ? 1U : 0U);
}

// Every Bayer pattern, sample format and demosaic method against the truth.
void accuracy() {
  double worst_psnr[2] = {99.0, 99.0};
  double worst_mean_de[2] = {0.0, 0.0};
  double worst_patch_de[2] = {0.0, 0.0};
  for (const SampleFormat& f : kFormats) {
    for (const BayerPattern pattern : kPatterns) {
      const SyntheticRawConfig config = camera(pattern, f);
      const test::RawFrame frame = test::render(config);
      double psnr[2] = {};
      for (int method = 0; method < 2; ++method) {
        IspParams params = synthetic_isp_params(config);
        params.demosaic = method == 0 ? DemosaicMethod::kBilinear : DemosaicMethod::kMalvar;
        IspPipeline pipeline(Backend::kCuda, params);
        const Result r = run(pipeline, frame, OutputFormat::kRgb8, false);
        const std::vector<float> lut = tone_lut(params);
        psnr[method] = test::psnr(r, config, lut);
        const test::PatchErrors de = test::checker_delta_e(r, config, lut);
        const std::string what = format_name(f) + " " + std::string(pattern_name(pattern)) + " " +
                                 std::string(demosaic_name(params.demosaic));
        // Most of the error is at edges, where a mosaic cannot know the colours.
        const double min_psnr = method == 0 ? 26.0 : 28.5;
        VERIFY_MSG(psnr[method] > min_psnr,
                   what + ": PSNR " + std::to_string(psnr[method]) + " dB");
        // 8-bit raw loses the dark patches to quantization after the white balance gains.
        VERIFY_MSG(de.mean < (f.bits == 8 ? 0.6 : 0.1) && de.worst < (f.bits == 8 ? 3.5 : 0.5),
                   what + ": patch CIEDE2000 mean " + std::to_string(de.mean) + ", worst " +
                       std::to_string(de.worst));
        worst_psnr[method] = std::min(worst_psnr[method], psnr[method]);
        worst_mean_de[method] = std::max(worst_mean_de[method], de.mean);
        worst_patch_de[method] = std::max(worst_patch_de[method], de.worst);
      }
      VERIFY_MSG(psnr[1] > psnr[0] + 1.5, format_name(f) + " " +
                                              std::string(pattern_name(pattern)) +
                                              ": Malvar-He-Cutler is not clearly better");
    }
  }
  std::cout << std::fixed << std::setprecision(2);
  for (int method = 0; method < 2; ++method) {
    std::cout << "  " << (method == 0 ? "bilinear" : "malvar  ") << ": worst PSNR "
              << worst_psnr[method] << " dB, worst mean patch dE00 " << worst_mean_de[method]
              << ", worst patch dE00 " << worst_patch_de[method] << '\n';
  }
}

// PSNR of each method per bit depth, RGGB, for the README.
void accuracy_table() {
  for (const SampleFormat& f : kFormats) {
    const SyntheticRawConfig config = camera(BayerPattern::kRGGB, f, 640, 480);
    const test::RawFrame frame = test::render(config);
    std::cout << "  " << std::setw(9) << format_name(f);
    for (const DemosaicMethod method : {DemosaicMethod::kBilinear, DemosaicMethod::kMalvar}) {
      IspParams params = synthetic_isp_params(config);
      params.demosaic = method;
      IspPipeline pipeline(Backend::kCuda, params);
      const Result r = run(pipeline, frame, OutputFormat::kRgb8, false);
      const std::vector<float> lut = tone_lut(params);
      const test::PatchErrors de = test::checker_delta_e(r, config, lut);
      std::cout << "  " << demosaic_name(method) << " PSNR " << test::psnr(r, config, lut)
                << " dB, dE00 mean " << de.mean << " worst " << de.worst;
    }
    std::cout << '\n';
  }
}

// The CUDA backend against the CPU backend, bit for bit, in many settings.
void cuda_matches_cpu() {
  struct Case {
    std::string name;
    SyntheticRawConfig config;
    OutputFormat format;
    std::function<void(IspParams&)> tweak;
  };
  const SampleFormat u12 = kFormats[2];
  std::vector<Case> cases;
  for (const DemosaicMethod method : {DemosaicMethod::kBilinear, DemosaicMethod::kMalvar}) {
    const std::string m(demosaic_name(method));
    cases.push_back({m, camera(BayerPattern::kRGGB, u12, 200, 150), OutputFormat::kRgb8,
                     [method](IspParams& p) { p.demosaic = method; }});
    cases.push_back({m + " sharpened", camera(BayerPattern::kGBRG, u12, 200, 150),
                     OutputFormat::kRgba8, [method](IspParams& p) {
                       p.demosaic = method;
                       p.sharpen = 1.5F;
                     }});
    cases.push_back({m + " odd size", camera(BayerPattern::kGRBG, kFormats[4], 67, 35),
                     OutputFormat::kRgb16, [method](IspParams& p) {
                       p.demosaic = method;
                       p.sharpen = 0.5F;
                     }});
    cases.push_back({m + " gray world", camera(BayerPattern::kBGGR, kFormats[5], 160, 120),
                     OutputFormat::kRgb32f, [method](IspParams& p) {
                       p.demosaic = method;
                       p.white_balance = WhiteBalanceMode::kGrayWorld;
                       p.tone_curve = ToneCurve::kGamma;
                     }});
    cases.push_back({m + " white patch, 4-channel shading, table",
                     camera(BayerPattern::kRGGB, kFormats[0], 160, 120), OutputFormat::kRgb8,
                     [method](IspParams& p) {
                       p.demosaic = method;
                       p.white_balance = WhiteBalanceMode::kWhitePatch;
                       p.tone_curve = ToneCurve::kLut;
                       p.lut = {0.0F, 0.6F, 0.85F, 1.0F};
                       LensShading& s = *p.lens_shading;
                       std::vector<float> four;
                       for (const float g : s.gains) {
                         for (int c = 0; c < 4; ++c) four.push_back(g * (1.0F + 0.01F * c));
                       }
                       s.gains = std::move(four);
                       s.channels = 4;
                     }});
  }
  for (Case& c : cases) {
    const test::RawFrame frame = test::render(c.config);
    IspParams params = synthetic_isp_params(c.config, 9);
    c.tweak(params);
    IspPipeline gpu(Backend::kCuda, params);
    IspPipeline cpu(Backend::kCpu, params);
    const FrameExposure exposure{.exposure_ns = 10'000'000, .gain = 2.0F};
    // Two frames, so the white balance smoothing runs too.
    for (int pass = 0; pass < 2; ++pass) {
      const Result g = run(gpu, frame, c.format, true, exposure);
      const Result h = run(cpu, frame, c.format, true, exposure);
      VERIFY_MSG(mismatches(g, h) == 0,
                 c.name + ": " + std::to_string(mismatches(g, h)) + " bytes differ");
      VERIFY_MSG(g.histogram == h.histogram, c.name + ": histograms differ");
      for (int i = 0; i < stats::kStatsSize; ++i) {
        // The auto-exposure fields go through log2 and exp2, whose last bit the
        // GPU and the host may round differently.
        const bool exact = i != stats::kAeErrorEv && i != stats::kExposureScale &&
                           i != stats::kSuggestedExposureNs && i != stats::kSuggestedGain;
        const float tolerance = exact ? 0.0F : 1e-5F * std::max(1.0F, std::abs(h.stats[i]));
        VERIFY_MSG(std::abs(g.stats[i] - h.stats[i]) <= tolerance,
                   c.name + ": stats[" + std::to_string(i) + "] " + std::to_string(g.stats[i]) +
                       " vs " + std::to_string(h.stats[i]));
      }
    }
  }
}

// The GPU renders the synthetic camera exactly as the host does.
void synthetic_render_matches() {
  for (const SampleFormat& f : kFormats) {
    const SyntheticRawConfig config = camera(BayerPattern::kGRBG, f, 101, 37);
    const test::RawFrame host = test::render(config);
    void* device = nullptr;
    test::cuda_check(cudaMalloc(&device, host.bytes.size()), "cudaMalloc");
    test::cuda_check(cudaMemset(device, 0, host.bytes.size()), "cudaMemset");
    test::cuda_check(
        render_synthetic_raw_cuda(make_synthetic_scene(config), device, host.pitch, nullptr),
        "render");
    std::vector<unsigned char> got(host.bytes.size());
    test::cuda_check(cudaMemcpy(got.data(), device, got.size(), cudaMemcpyDeviceToHost),
                     "cudaMemcpy");
    cudaFree(device);
    VERIFY_MSG(got == host.bytes, format_name(f) + ": the GPU render differs from the host's");
  }
}

// Rows padded beyond their samples give the same output as packed rows.
void padded_rows() {
  for (const SampleFormat& f : {kFormats[0], kFormats[2], kFormats[5]}) {
    const SyntheticRawConfig config = camera(BayerPattern::kGBRG, f, 75, 41);
    const test::RawFrame packed = test::render(config);
    test::RawFrame padded = packed;
    padded.pitch = packed.pitch + 38U;
    padded.bytes.assign(padded.pitch * config.height, 0xAB);
    for (int y = 0; y < config.height; ++y) {
      std::memcpy(padded.bytes.data() + y * padded.pitch, packed.bytes.data() + y * packed.pitch,
                  packed.pitch);
    }
    IspPipeline pipeline(Backend::kCuda, synthetic_isp_params(config, 5));
    const Result a = run(pipeline, packed, OutputFormat::kRgb8);
    const Result b = run(pipeline, padded, OutputFormat::kRgb8);
    VERIFY_MSG(mismatches(a, b) == 0 && a.histogram == b.histogram,
               format_name(f) + ": padded rows change the output");
  }
}

// Gray world finds the camera's gains on a gray scene; white patch on the chart.
void auto_white_balance() {
  for (const BayerPattern pattern : kPatterns) {
    for (const auto& [mode, kind] : {std::pair{WhiteBalanceMode::kGrayWorld, SceneKind::kGray},
                                     std::pair{WhiteBalanceMode::kWhitePatch, SceneKind::kChart}}) {
      SyntheticRawConfig config = camera(pattern, kFormats[2]);
      config.kind = kind;
      config.wb_gains = {2.1F, 1.0F, 1.45F};
      const test::RawFrame frame = test::render(config);
      IspParams params = synthetic_isp_params(config);
      params.white_balance = mode;
      params.wb_gains = {1.0F, 1.0F, 1.0F};
      for (const Backend backend : {Backend::kCuda, Backend::kCpu}) {
        IspPipeline pipeline(backend, params);
        const Result r = run(pipeline, frame, OutputFormat::kRgb8);
        const std::string what = std::string(white_balance_name(mode)) + " " +
                                 std::string(pattern_name(pattern)) + " " +
                                 std::string(backend_name(backend));
        VERIFY_MSG(std::abs(r.stats[stats::kWbGainR] / 2.1F - 1.0F) < 0.01F &&
                       r.stats[stats::kWbGainG] == 1.0F &&
                       std::abs(r.stats[stats::kWbGainB] / 1.45F - 1.0F) < 0.01F,
                   what + ": gains " + std::to_string(r.stats[stats::kWbGainR]) + ", " +
                       std::to_string(r.stats[stats::kWbGainB]));
      }
    }
  }
}

// Each frame moves the gains `awb_smoothing` of the way to its estimate.
void white_balance_smoothing() {
  SyntheticRawConfig config = camera(BayerPattern::kRGGB, kFormats[2], 160, 120);
  config.kind = SceneKind::kGray;
  config.wb_gains = {2.0F, 1.0F, 1.5F};
  const test::RawFrame first = test::render(config);
  config.wb_gains = {1.5F, 1.0F, 2.0F};
  const test::RawFrame second = test::render(config);
  IspParams params = synthetic_isp_params(config);
  params.white_balance = WhiteBalanceMode::kGrayWorld;
  params.awb_smoothing = 0.25F;
  IspPipeline pipeline(Backend::kCuda, params);
  const Result a = run(pipeline, first, OutputFormat::kRgb8);
  VERIFY(std::abs(a.stats[stats::kWbGainR] - 2.0F) < 0.02F);  // The first frame takes its estimate.
  const Result b = run(pipeline, second, OutputFormat::kRgb8);
  const float expected = a.stats[stats::kWbGainR] + 0.25F * (1.5F - a.stats[stats::kWbGainR]);
  VERIFY_MSG(std::abs(b.stats[stats::kWbGainR] - expected) < 0.01F,
             "smoothed red gain " + std::to_string(b.stats[stats::kWbGainR]) + ", expected about " +
                 std::to_string(expected));
  float red = b.stats[stats::kWbGainR];
  for (int i = 0; i < 30; ++i)
    red = run(pipeline, second, OutputFormat::kRgb8).stats[stats::kWbGainR];
  VERIFY_MSG(std::abs(red - 1.5F) < 0.015F, "converged red gain " + std::to_string(red));
  pipeline.reset();
  VERIFY(std::abs(run(pipeline, first, OutputFormat::kRgb8).stats[stats::kWbGainR] - 2.0F) < 0.02F);
}

// A flat raw frame: every sample `value` counts, 16-bit, `width` x `height`.
test::RawFrame flat(int width, int height, std::uint16_t r, std::uint16_t g, std::uint16_t b,
                    BayerPattern pattern = BayerPattern::kRGGB) {
  test::RawFrame frame;
  frame.config.width = width;
  frame.config.height = height;
  frame.config.pattern = pattern;
  frame.config.format = RawFormat::kUint16;
  frame.config.bit_depth = 16;
  frame.pitch = 2U * width;
  frame.bytes.assign(frame.pitch * height, 0);
  // The channel at each quad position, per pattern, as the image schema orders them.
  constexpr int kCfa[4][4] = {{kChannelR, kChannelGr, kChannelGb, kChannelB},
                              {kChannelB, kChannelGb, kChannelGr, kChannelR},
                              {kChannelGr, kChannelR, kChannelB, kChannelGb},
                              {kChannelGb, kChannelB, kChannelR, kChannelGr}};
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int channel = kCfa[static_cast<int>(pattern)][((y & 1) << 1) | (x & 1)];
      const std::uint16_t v = channel == kChannelR ? r : channel == kChannelB ? b : g;
      std::memcpy(frame.bytes.data() + y * frame.pitch + 2U * x, &v, 2U);
    }
  }
  return frame;
}

IspParams linear_params() {
  IspParams params;
  params.tone_curve = ToneCurve::kLinear;
  params.white_level = 65535.0F;
  return params;
}

// The colour matrix maps a flat colour as the arithmetic says.
void color_matrix() {
  IspParams params = linear_params();
  params.ccm = {0.9F, 0.2F, -0.1F, 0.05F, 0.8F, 0.15F, -0.2F, 0.1F, 1.1F};
  params.wb_gains = {1.25F, 1.0F, 0.8F};
  const test::RawFrame frame = flat(32, 24, 20000, 30000, 40000, BayerPattern::kGBRG);
  for (const Backend backend : {Backend::kCuda, Backend::kCpu}) {
    IspPipeline pipeline(backend, params);
    const Result r = run(pipeline, frame, OutputFormat::kRgb32f);
    const double cam[3] = {20000.0 / 65535.0 * 1.25, 30000.0 / 65535.0, 40000.0 / 65535.0 * 0.8};
    float worst = 0.0F;
    for (int k = 0; k < 3; ++k) {
      const double want = params.ccm[3 * k] * cam[0] + params.ccm[3 * k + 1] * cam[1] +
                          params.ccm[3 * k + 2] * cam[2];
      for (int y = 0; y < 24; ++y) {
        for (int x = 0; x < 32; ++x) {
          worst = std::max(worst, static_cast<float>(std::abs(test::value(r, x, y, k) - want)));
        }
      }
    }
    VERIFY_MSG(worst < 2e-6F, std::string(backend_name(backend)) + ": colour matrix off by " +
                                  std::to_string(worst));
  }
}

// The tone curves: sRGB and gamma against their formulas, a table as given.
void tone_curves() {
  const int width = 256;
  // A horizontal gray ramp, as raw: each column one level.
  test::RawFrame frame;
  frame.config.width = width;
  frame.config.height = 4;
  frame.config.format = RawFormat::kUint16;
  frame.config.bit_depth = 16;
  frame.pitch = 2U * width;
  frame.bytes.assign(frame.pitch * 4U, 0);
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < width; ++x) {
      const auto v = static_cast<std::uint16_t>(x * 257);
      std::memcpy(frame.bytes.data() + y * frame.pitch + 2U * x, &v, 2U);
    }
  }
  struct Curve {
    ToneCurve curve;
    float gamma;
    std::vector<float> lut;
    double (*formula)(double);
  };
  const std::vector<Curve> curves{
      {ToneCurve::kSrgb,
       2.2F,
       {},
       [](double x) { return x <= 0.0031308 ? 12.92 * x : 1.055 * std::pow(x, 1 / 2.4) - 0.055; }},
      {ToneCurve::kGamma, 2.2F, {}, [](double x) { return std::pow(x, 1 / 2.2); }},
      {ToneCurve::kLinear, 2.2F, {}, [](double x) { return x; }},
      {ToneCurve::kLut, 2.2F, {1.0F, 0.0F}, [](double x) { return 1.0 - x; }},
  };
  for (const Curve& c : curves) {
    IspParams params = linear_params();
    params.demosaic = DemosaicMethod::kBilinear;
    params.tone_curve = c.curve;
    params.gamma = c.gamma;
    params.lut = c.lut;
    IspPipeline pipeline(Backend::kCuda, params);
    const Result r = run(pipeline, frame, OutputFormat::kRgb32f);
    double worst = 0.0;
    // The ramp's bilinear interpolation is exact inside; skip the two border columns.
    for (int x = 1; x < width - 1; ++x) {
      const double want = c.formula(x / 255.0);
      for (int k = 0; k < 3; ++k) worst = std::max(worst, std::abs(test::value(r, x, 1, k) - want));
    }
    VERIFY_MSG(worst < 3e-5, "tone curve " + std::to_string(static_cast<int>(c.curve)) +
                                 " off by " + std::to_string(worst));
  }
}

// Histograms sum to the pixels; means, the auto-exposure error and the suggestion.
void statistics() {
  const int width = 64;
  const int height = 48;
  // Flat mid-gray: every channel at 0.09 of full scale, linear.
  const auto level = static_cast<std::uint16_t>(0.09 * 65535.0 + 0.5);
  const test::RawFrame frame = flat(width, height, level, level, level);
  IspParams params = linear_params();
  params.auto_exposure.target = 0.18F;
  params.auto_exposure.damping = 1.0F;
  params.auto_exposure.max_exposure_ns = 20'000'000;
  for (const Backend backend : {Backend::kCuda, Backend::kCpu}) {
    IspPipeline pipeline(backend, params);
    const std::string what(backend_name(backend));
    const Result r = run(pipeline, frame, OutputFormat::kRgb8, true,
                         FrameExposure{.exposure_ns = 15'000'000, .gain = 1.0F});
    for (int row = 0; row < kHistogramChannels; ++row) {
      std::uint64_t sum = 0;
      for (int b = 0; b < kHistogramBins; ++b) sum += r.histogram[row * kHistogramBins + b];
      VERIFY_MSG(
          sum == static_cast<std::uint64_t>(width) * height,
          what + ": histogram row " + std::to_string(row) + " sums to " + std::to_string(sum));
    }
    // 0.09 linear codes as 23 (0.09 * 255 = 22.95).
    VERIFY_MSG(r.histogram[kHistogramLuma * kHistogramBins + 23] == width * height,
               what + ": a flat frame fills one luma bin");
    const float mean = static_cast<float>(level) / 65535.0F;
    VERIFY(std::abs(r.stats[stats::kMeanR] - mean) < 1e-6F);
    VERIFY(std::abs(r.stats[stats::kMeanLuma] - mean) < 1e-5F);
    VERIFY_MSG(std::abs(r.stats[stats::kAeErrorEv] - 1.0F) < 1e-3F,
               what + ": one stop under, error " + std::to_string(r.stats[stats::kAeErrorEv]));
    VERIFY(std::abs(r.stats[stats::kExposureScale] - 2.0F) < 2e-3F);
    // 30 ms wanted, 20 ms the most: the rest as gain.
    VERIFY_MSG(std::abs(r.stats[stats::kSuggestedExposureNs] - 20e6F) < 1.0F &&
                   std::abs(r.stats[stats::kSuggestedGain] - 1.5F) < 2e-3F,
               what + ": suggested " + std::to_string(r.stats[stats::kSuggestedExposureNs]) +
                   " ns at gain " + std::to_string(r.stats[stats::kSuggestedGain]));
    VERIFY(r.stats[stats::kSaturatedFraction] == 0.0F);
    // Without the frame's exposure, no suggestion.
    const Result unknown = run(pipeline, frame, OutputFormat::kRgb8);
    VERIFY(unknown.stats[stats::kSuggestedExposureNs] == 0.0F);
  }
  // On the chart, every histogram row still sums to the pixels.
  const SyntheticRawConfig config = camera(BayerPattern::kBGGR, kFormats[3], 123, 77);
  IspPipeline pipeline(Backend::kCuda, synthetic_isp_params(config));
  const Result r = run(pipeline, test::render(config), OutputFormat::kRgb16);
  for (int row = 0; row < kHistogramChannels; ++row) {
    std::uint64_t sum = 0;
    for (int b = 0; b < kHistogramBins; ++b) sum += r.histogram[row * kHistogramBins + b];
    VERIFY(sum == 123U * 77U);
  }
}

// The same frame gives the same bytes on every run and in every instance.
void determinism() {
  const SyntheticRawConfig config = camera(BayerPattern::kRGGB, kFormats[2], 640, 480);
  const test::RawFrame frame = test::render(config);
  IspParams params = synthetic_isp_params(config);
  params.white_balance = WhiteBalanceMode::kGrayWorld;
  params.sharpen = 1.0F;
  IspPipeline first(Backend::kCuda, params);
  const Result a = run(first, frame, OutputFormat::kRgb8);
  for (int i = 0; i < 3; ++i) {
    IspPipeline again(Backend::kCuda, params);
    const Result b = run(again, frame, OutputFormat::kRgb8);
    VERIFY(mismatches(a, b) == 0 && a.histogram == b.histogram && a.stats == b.stats);
  }
}

// Odd and tiny sizes, saturation, and black frames.
void edge_cases() {
  // Down to 2x2, every pattern: the CUDA and CPU backends agree.
  for (const auto& [w, h] :
       {std::pair{2, 2}, std::pair{3, 5}, std::pair{33, 17}, std::pair{5, 2}}) {
    for (const BayerPattern pattern : kPatterns) {
      const test::RawFrame frame = flat(w, h, 9000, 21000, 15000, pattern);
      IspParams params = linear_params();
      params.sharpen = 1.0F;
      IspPipeline gpu(Backend::kCuda, params);
      IspPipeline cpu(Backend::kCpu, params);
      const Result g = run(gpu, frame, OutputFormat::kRgb8);
      const Result c = run(cpu, frame, OutputFormat::kRgb8);
      VERIFY_MSG(mismatches(g, c) == 0, std::to_string(w) + "x" + std::to_string(h) + " " +
                                            std::string(pattern_name(pattern)) + " differs");
      // A flat colour stays flat, to the 8-bit rounding.
      bool flat_out = true;
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          flat_out = flat_out && std::abs(test::value(g, x, y, 0) - 9000.0F / 65535.0F) < 0.003F &&
                     std::abs(test::value(g, x, y, 2) - 15000.0F / 65535.0F) < 0.003F;
        }
      }
      VERIFY_MSG(flat_out, std::to_string(w) + "x" + std::to_string(h) + " is not flat");
    }
  }

  // Saturated: four times the light. The white patch and the bright bars are
  // white, without a colour cast, and the statistics count the clipped samples.
  SyntheticRawConfig bright = camera(BayerPattern::kRGGB, kFormats[2]);
  bright.exposure = 4.0F;
  IspParams params = synthetic_isp_params(bright);
  IspPipeline pipeline(Backend::kCuda, params);
  const Result r = run(pipeline, test::render(bright), OutputFormat::kRgb8);
  const SyntheticRawScene scene = make_synthetic_scene(bright);
  const ChartLayout c = chart_layout(scene);
  const int x0 = c.left + scene.shift + c.patch_width / 4;  // Patch 19: white, row 3, column 0.
  const int y0 = c.top + 3 * c.cell_height + c.patch_height / 4;
  int not_white = 0;
  for (int y = y0; y < y0 + c.patch_height / 2; ++y) {
    for (int x = x0; x < x0 + c.patch_width / 2; ++x) {
      for (int k = 0; k < 3; ++k) not_white += test::value(r, x, y, k) < 1.0F ? 1 : 0;
    }
  }
  VERIFY_MSG(not_white == 0,
             std::to_string(not_white) + " values of the clipped white patch are not white");
  const double bright_psnr = test::psnr(r, bright, tone_lut(params));
  VERIFY_MSG(bright_psnr > 23.0,
             "saturated frame far from the truth: PSNR " + std::to_string(bright_psnr) + " dB");
  VERIFY(r.stats[stats::kSaturatedFraction] > 0.05F && r.stats[stats::kAeErrorEv] < 0.0F);

  // Black: every sample at its black level. Black out, the gains kept, the
  // exposure error at its bound.
  SyntheticRawConfig black = camera(BayerPattern::kRGGB, kFormats[2], 64, 48);
  black.exposure = 1e-6F;
  IspParams black_params = synthetic_isp_params(black);
  black_params.white_balance = WhiteBalanceMode::kGrayWorld;
  for (const Backend backend : {Backend::kCuda, Backend::kCpu}) {
    IspPipeline p(backend, black_params);
    const Result b = run(p, test::render(black), OutputFormat::kRgb8);
    bool all_black = true;
    for (const unsigned char v : b.image) all_black = all_black && v == 0;
    VERIFY_MSG(all_black, std::string(backend_name(backend)) + ": black frame is not black");
    VERIFY(b.stats[stats::kWbGainR] == black_params.wb_gains[0] &&
           b.stats[stats::kWbGainB] == black_params.wb_gains[2]);
    VERIFY(b.stats[stats::kMeanLuma] == 0.0F && b.stats[stats::kAeErrorEv] == 8.0F);
    VERIFY(b.histogram[kHistogramLuma * kHistogramBins] == 64U * 48U);
  }
}

void rejects() {
  const auto bad = [](auto tweak) {
    IspParams params;
    tweak(params);
    return throws<std::invalid_argument>([&] { validate_params(params); });
  };
  VERIFY(bad([](IspParams& p) { p.black_level = -1.0F; }));
  VERIFY(bad([](IspParams& p) { p.wb_gains[1] = 0.0F; }));
  VERIFY(bad([](IspParams& p) { p.ccm[4] = std::nanf(""); }));
  VERIFY(bad([](IspParams& p) { p.awb_smoothing = 0.0F; }));
  VERIFY(bad([](IspParams& p) { p.sharpen = 5.0F; }));
  VERIFY(bad([](IspParams& p) { p.tone_curve = ToneCurve::kLut; }));
  VERIFY(bad([](IspParams& p) { p.lens_shading = LensShading{2, 2, 1, {1.0F, 1.0F, 1.0F}}; }));
  VERIFY(bad([](IspParams& p) { p.auto_exposure.max_exposure_ns = 1; }));
  VERIFY(throws<std::invalid_argument>(
      [] { IspPipeline(Backend::kCuda, IspParams{.sharpen = -1.0F}); }));

  IspPipeline pipeline(Backend::kCpu, IspParams{});
  std::vector<std::uint16_t> raw(16, 0);
  std::vector<unsigned char> out(16 * 3, 0);
  const RawImageView good{raw.data(), 4, 4, 8, RawFormat::kUint16, 12, BayerPattern::kRGGB};
  const RgbImageView rgb{out.data(), 12, OutputFormat::kRgb8};
  VERIFY(!throws<std::invalid_argument>([&] { pipeline.process(good, rgb); }));
  const auto rejected = [&](auto tweak) {
    RawImageView view = good;
    tweak(view);
    return throws<std::invalid_argument>([&] { pipeline.process(view, rgb); });
  };
  VERIFY(rejected([](RawImageView& v) { v.width = 1; }));
  VERIFY(rejected([](RawImageView& v) { v.pitch = 6; }));
  VERIFY(rejected([](RawImageView& v) { v.bit_depth = 17; }));
  VERIFY(rejected([](RawImageView& v) { v.format = RawFormat::kRaw10; }));  // RAW10 is 10 bits.
  VERIFY(rejected([](RawImageView& v) { v.data = nullptr; }));
  VERIFY(throws<std::invalid_argument>(
      [&] { pipeline.process(good, RgbImageView{out.data(), 6, OutputFormat::kRgb8}); }));
  IspParams high_black;
  high_black.black_level = 5000.0F;  // Above 12 bits' white.
  IspPipeline black_pipeline(Backend::kCpu, high_black);
  VERIFY(throws<std::invalid_argument>([&] { black_pipeline.process(good, rgb); }));
}

}  // namespace
}  // namespace holoscan::isp

int main() {
  using holoscan::isp::test::TestCase;
  return holoscan::isp::test::run({
      TestCase{"accuracy against the truth, every pattern, format and method",
               holoscan::isp::accuracy},
      TestCase{"accuracy per format, 640x480 RGGB", holoscan::isp::accuracy_table},
      TestCase{"the CUDA backend matches the CPU backend bit for bit",
               holoscan::isp::cuda_matches_cpu},
      TestCase{"the synthetic camera renders the same on the GPU",
               holoscan::isp::synthetic_render_matches},
      TestCase{"padded rows", holoscan::isp::padded_rows},
      TestCase{"auto white balance recovers the camera's gains", holoscan::isp::auto_white_balance},
      TestCase{"white balance smoothing", holoscan::isp::white_balance_smoothing},
      TestCase{"colour correction matrix", holoscan::isp::color_matrix},
      TestCase{"tone curves", holoscan::isp::tone_curves},
      TestCase{"statistics and the exposure suggestion", holoscan::isp::statistics},
      TestCase{"determinism", holoscan::isp::determinism},
      TestCase{"odd sizes, saturation and black frames", holoscan::isp::edge_cases},
      TestCase{"rejected settings and views", holoscan::isp::rejects},
  });
}
