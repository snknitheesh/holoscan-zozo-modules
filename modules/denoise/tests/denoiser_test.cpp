// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The Denoiser without a graph, on the synthetic low-light scene with Poisson
// and Gaussian noise of known levels, rendered at test time: the PSNR gain of
// every method and strength; no ghost trail behind a moving disc; convergence
// of the temporal filter on a still scene; the first frame, a scene cut and a
// reset; every format (NV12 included) and odd sizes; the noise estimate;
// determinism; and the cuda backend against VPI's TNR where VPI is built.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/denoise/denoise_params.hpp>
#include <holoscan/denoise/denoiser.hpp>
#include <holoscan/denoise/synthetic_scene.hpp>

#include "image_support.hpp"
#include "test_support.hpp"

namespace holoscan::denoise {
namespace {

using test::DeviceImage;

constexpr int kWidth = 320;
constexpr int kHeight = 240;

// The low-light noise of the tests: Poisson at 1 code value per photon and
// Gaussian read noise of 6; about 7.5 code values over the scene.
constexpr NoiseModel kNoise{.read_sigma = 6.0F, .shot_gain = 1.0F, .seed = 7};

LowLightScene still_scene(PixelFormat format = PixelFormat::kRgb8, int width = kWidth,
                          int height = kHeight) {
  return LowLightScene{.width = width,
                       .height = height,
                       .format = format,
                       .speed = 0.0F,
                       .object_radius = width / 10};
}

DenoiseParams params_of(SpatialFilter spatial, bool temporal, float strength) {
  DenoiseParams params;
  params.spatial = spatial;
  params.temporal = temporal;
  params.strength = strength;
  return params;
}

std::string label(Backend backend, const DenoiseParams& p) {
  char text[96];
  std::snprintf(text, sizeof(text), "%s %s%s s=%.2f", std::string(backend_name(backend)).c_str(),
                p.temporal ? "temporal+" : "", std::string(spatial_name(p.spatial)).c_str(),
                p.strength);
  return text;
}

// Runs `frames` frames of `scene` through a new denoiser; calls `each(frame,
// noisy, denoised)` with packed host copies.
template <typename Each>
void run_sequence(Backend backend, const DenoiseParams& params, const LowLightScene& scene,
                  int frames, Each&& each, int first_frame = 0) {
  Denoiser denoiser(backend, params);
  DeviceImage noisy(scene.format, scene.width, scene.height);
  DeviceImage out(scene.format, scene.width, scene.height);
  for (int f = first_frame; f < first_frame + frames; ++f) {
    test::cuda_check(render_noisy(scene, kNoise, f, noisy.view(), nullptr), "render_noisy");
    denoiser.process(noisy.view(), out.view(), nullptr);
    each(f, noisy.download(), out.download());
  }
}

struct Quality {
  double input_psnr{};
  double output_psnr{};
  [[nodiscard]] double gain() const { return output_psnr - input_psnr; }
};

// PSNR of the last of `frames` frames of the still scene.
Quality still_quality(Backend backend, const DenoiseParams& params, int frames,
                      PixelFormat format = PixelFormat::kRgb8) {
  const LowLightScene scene = still_scene(format);
  Quality q;
  run_sequence(
      backend, params, scene, frames,
      [&](int f, const std::vector<std::uint8_t>& noisy, const std::vector<std::uint8_t>& out) {
        const auto clean = render_clean(scene, f);
        q.input_psnr = psnr(noisy.data(), clean.data(), clean.size());
        q.output_psnr = psnr(out.data(), clean.data(), clean.size());
      });
  return q;
}

bool vpi_built() {
  const auto backends = available_backends();
  return std::find(backends.begin(), backends.end(), Backend::kVpi) != backends.end();
}

// ---- Cases ----

// Strength 0 leaves every frame as it is, with every method.
void check_strength_zero_is_identity() {
  for (const SpatialFilter spatial :
       {SpatialFilter::kNone, SpatialFilter::kBilateral, SpatialFilter::kNlm}) {
    for (const bool temporal : {false, true}) {
      if (spatial == SpatialFilter::kNone && !temporal) continue;
      const DenoiseParams params = params_of(spatial, temporal, 0.0F);
      int changed = 0;
      run_sequence(Backend::kCuda, params, LowLightScene{.width = 96, .height = 64}, 3,
                   [&](int, const auto& noisy, const auto& out) { changed += noisy != out; });
      VERIFY_MSG(changed == 0, label(Backend::kCuda, params) + " changed a frame at strength 0");
    }
  }
}

// Each method's PSNR gain on the still scene, by strength: more strength, more
// gain, up to a floor at the default strength.
void check_psnr_gains() {
  struct Case {
    SpatialFilter spatial;
    bool temporal;
    int frames;
    double min_gain_at_half;  // dB at strength 0.5
  };
  const Case cases[] = {
      {SpatialFilter::kBilateral, false, 1, 4.0}, {SpatialFilter::kNlm, false, 1, 4.0},
      {SpatialFilter::kNone, true, 16, 7.0},      {SpatialFilter::kBilateral, true, 16, 8.0},
      {SpatialFilter::kNlm, true, 16, 8.0},
  };
  for (const Case& c : cases) {
    double previous = -1.0;
    for (const float strength : {0.25F, 0.5F, 1.0F}) {
      const DenoiseParams params = params_of(c.spatial, c.temporal, strength);
      const Quality q = still_quality(Backend::kCuda, params, c.frames);
      std::printf("  %-34s frames=%2d input=%.2f dB output=%.2f dB gain=%.2f dB\n",
                  label(Backend::kCuda, params).c_str(), c.frames, q.input_psnr, q.output_psnr,
                  q.gain());
      // Alone, each filter gains more with more strength. Together, more
      // temporal averaging leaves the spatial filter less noise to remove, so
      // their sum need not rise at every step.
      if (!c.temporal || c.spatial == SpatialFilter::kNone) {
        VERIFY_MSG(q.gain() > previous - 0.2,
                   label(Backend::kCuda, params) + ": gain falls with strength");
      }
      if (strength == 0.5F) {
        VERIFY_MSG(q.gain() >= c.min_gain_at_half,
                   label(Backend::kCuda, params) + ": gain " + std::to_string(q.gain()) + " dB");
      }
      previous = q.gain();
    }
  }
  if (vpi_built()) {
    for (const float strength : {0.25F, 0.5F, 1.0F}) {
      const DenoiseParams params = params_of(SpatialFilter::kNone, true, strength);
      const Quality q = still_quality(Backend::kVpi, params, 16);
      std::printf("  %-34s frames=16 input=%.2f dB output=%.2f dB gain=%.2f dB\n",
                  label(Backend::kVpi, params).c_str(), q.input_psnr, q.output_psnr, q.gain());
      VERIFY_MSG(q.gain() >= 3.0,
                 label(Backend::kVpi, params) + ": gain " + std::to_string(q.gain()) + " dB");
    }
  }
}

// The temporal filter converges on a still scene: PSNR rises with every frame
// until the history is full, and the residual noise matches the recursive
// filter's variance.
void check_temporal_convergence() {
  const DenoiseParams params = params_of(SpatialFilter::kNone, true, 1.0F);
  const LowLightScene scene = still_scene();
  std::vector<double> gains;
  std::vector<double> noise;
  run_sequence(Backend::kCuda, params, scene, 64, [&](int f, const auto& noisy, const auto& out) {
    const auto clean = render_clean(scene, f);
    gains.push_back(psnr(out.data(), clean.data(), clean.size()) -
                    psnr(noisy.data(), clean.data(), clean.size()));
    double sum = 0.0;
    for (std::size_t i = 0; i < clean.size(); ++i) {
      const double d = static_cast<double>(out[i]) - clean[i];
      sum += d * d;
    }
    noise.push_back(sum / static_cast<double>(clean.size()));
  });
  std::printf("  gain by frame:");
  for (const int f : {0, 1, 3, 7, 15, 31, 63}) std::printf(" %d:%.2f", f + 1, gains[f]);
  std::printf(" dB\n");
  VERIFY_MSG(std::abs(gains[0]) < 1e-9, "the first frame passes through");
  for (std::size_t f = 1; f < 32; ++f) {
    VERIFY_MSG(gains[f] > gains[f - 1] - 0.05,
               "gain falls at frame " + std::to_string(f + 1) + ": " + std::to_string(gains[f]));
  }
  // Settled at weight 1/32: (1/32) / (2 - 1/32) of the input's noise variance,
  // plus the quantization's 1/12 and the scene's edges.
  const double input_variance = noise_variance(kNoise, 45.0);
  const double settled = input_variance * (1.0 / 32.0) / (2.0 - 1.0 / 32.0);
  std::printf("  residual variance at frame 64: %.3f (recursive filter: %.3f)\n", noise.back(),
              settled + 1.0 / 12.0);
  VERIFY_MSG(gains.back() >= 12.0, "settled gain " + std::to_string(gains.back()) + " dB");
  VERIFY_MSG(noise.back() < 2.0 * (settled + 1.0 / 12.0),
             "residual variance " + std::to_string(noise.back()));
}

// A disc moving 6 pixels a frame over a noisy background leaves no trail: where
// it was in the last 4 frames but no longer is, the error is no larger than the
// noise of the input. A plain running average of the same frames trails badly.
void check_no_ghosting(Backend backend, const DenoiseParams& params) {
  const LowLightScene scene{.width = kWidth, .height = kHeight, .speed = 6.0F, .object_radius = 24};
  constexpr int kFrames = 24;
  double trail_error = 0.0;
  double average_error = 0.0;
  double input_error = 0.0;
  std::size_t trail_samples = 0;
  std::vector<double> average;
  run_sequence(backend, params, scene, kFrames, [&](int f, const auto& noisy, const auto& out) {
    if (average.empty()) average.assign(noisy.begin(), noisy.end());
    for (std::size_t i = 0; i < noisy.size(); ++i) average[i] += (noisy[i] - average[i]) / 8.0;
    if (f < 8) return;
    const auto clean = render_clean(scene, f);
    for (int y = 0; y < scene.height; ++y) {
      for (int x = 0; x < scene.width; ++x) {
        if (object_covers(scene, f, x, y)) continue;
        bool trail = false;
        for (int back = 1; back <= 4 && !trail; ++back)
          trail = object_covers(scene, f - back, x, y);
        if (!trail) continue;
        for (int c = 0; c < 3; ++c) {
          const std::size_t i = (static_cast<std::size_t>(y) * scene.width + x) * 3 + c;
          const double truth = clean[i];
          trail_error += (out[i] - truth) * (out[i] - truth);
          average_error += (average[i] - truth) * (average[i] - truth);
          input_error += (noisy[i] - truth) * (noisy[i] - truth);
          ++trail_samples;
        }
      }
    }
  });
  const auto n = static_cast<double>(trail_samples);
  std::printf("  %-34s trail MSE %.2f, input noise %.2f, running average %.2f (%zu samples)\n",
              label(backend, params).c_str(), trail_error / n, input_error / n, average_error / n,
              trail_samples);
  VERIFY(trail_samples > 1000);
  VERIFY_MSG(trail_error / n < input_error / n,
             label(backend, params) + ": trail MSE " + std::to_string(trail_error / n));
  VERIFY_MSG(average_error / n > 4.0 * input_error / n, "the test sees a trail when there is one");
}

// The first frame has no history: with only the temporal filter it passes
// through; with a spatial filter it is what the spatial filter alone makes of it.
void check_first_frame() {
  const LowLightScene scene = still_scene();
  run_sequence(Backend::kCuda, params_of(SpatialFilter::kNone, true, 0.5F), scene, 1,
               [&](int, const auto& noisy, const auto& out) {
                 VERIFY_MSG(noisy == out, "the first frame passes the temporal filter unchanged");
               });
  std::vector<std::uint8_t> temporal;
  std::vector<std::uint8_t> spatial;
  run_sequence(Backend::kCuda, params_of(SpatialFilter::kBilateral, true, 0.5F), scene, 1,
               [&](int, const auto&, const auto& out) { temporal = out; });
  run_sequence(Backend::kCuda, params_of(SpatialFilter::kBilateral, false, 0.5F), scene, 1,
               [&](int, const auto&, const auto& out) { spatial = out; });
  VERIFY_MSG(temporal == spatial, "the first frame is the spatial filter's alone");
}

// A scene cut drops the history: the frame after it comes out as a new
// sequence's first frame would, and a reset() does the same.
void check_scene_cut(Backend backend) {
  constexpr int kCut = 10;
  const DenoiseParams params = params_of(
      backend == Backend::kVpi ? SpatialFilter::kNone : SpatialFilter::kBilateral, true, 0.5F);
  const LowLightScene scene{
      .width = kWidth, .height = kHeight, .speed = 0.0F, .object_radius = 30, .cut_frame = kCut};
  std::vector<std::uint8_t> after_cut;
  double cut_psnr = 0.0;
  run_sequence(backend, params, scene, kCut + 1, [&](int f, const auto&, const auto& out) {
    if (f != kCut) return;
    after_cut = out;
    const auto clean = render_clean(scene, f);
    cut_psnr = psnr(out.data(), clean.data(), clean.size());
  });
  std::vector<std::uint8_t> fresh;
  run_sequence(
      backend, params, scene, 1, [&](int, const auto&, const auto& out) { fresh = out; }, kCut);
  const auto clean = render_clean(scene, kCut);
  const double fresh_psnr = psnr(fresh.data(), clean.data(), clean.size());
  std::printf("  %-34s frame after the cut %.2f dB, as a first frame %.2f dB\n",
              label(backend, params).c_str(), cut_psnr, fresh_psnr);
  if (backend == Backend::kCuda) {
    VERIFY_MSG(after_cut == fresh, "the cut frame is filtered as a first frame");
  } else {
    VERIFY_MSG(cut_psnr > fresh_psnr - 0.5, "no ghost of the old scene after the cut");
  }

  // reset(): the next frame is a first frame.
  Denoiser denoiser(backend, params);
  DeviceImage noisy(scene.format, scene.width, scene.height);
  DeviceImage out(scene.format, scene.width, scene.height);
  for (int f = 0; f < 4; ++f) {
    test::cuda_check(render_noisy(scene, kNoise, kCut, noisy.view(), nullptr), "render_noisy");
    if (f == 3) denoiser.reset();
    denoiser.process(noisy.view(), out.view(), nullptr);
  }
  if (backend == Backend::kCuda) {
    VERIFY_MSG(out.download() == fresh, "a reset starts a new sequence");
  }
}

// Every format: MONO8, RGB8, RGBA8 (alpha kept) and NV12 all gain; odd sizes
// and 1x1 work.
void check_formats_and_sizes(Backend backend) {
  const bool vpi = backend == Backend::kVpi;
  const DenoiseParams params =
      params_of(vpi ? SpatialFilter::kNone : SpatialFilter::kBilateral, true, 0.5F);
  for (const PixelFormat format : {PixelFormat::kMono8, PixelFormat::kRgba8}) {
    const Quality q = still_quality(backend, params, 8, format);
    std::printf("  %-34s %s gain %.2f dB\n", label(backend, params).c_str(),
                std::string(format_name(format)).c_str(), q.gain());
    VERIFY_MSG(q.gain() > 4.0, std::string(format_name(format)) + " gains too little");
  }
  // RGBA8 keeps its alpha.
  {
    const LowLightScene scene = still_scene(PixelFormat::kRgba8, 64, 48);
    std::vector<std::uint8_t> frame = render_clean(scene, 0);
    for (std::size_t i = 3; i < frame.size(); i += 4) frame[i] = static_cast<std::uint8_t>(i / 4);
    Denoiser denoiser(backend, params);
    DeviceImage in(PixelFormat::kRgba8, 64, 48);
    DeviceImage out(PixelFormat::kRgba8, 64, 48);
    in.upload(frame);
    denoiser.process(in.view(), out.view(), nullptr);
    const auto result = out.download();
    int wrong = 0;
    for (std::size_t i = 3; i < frame.size(); i += 4) wrong += result[i] != frame[i];
    VERIFY_MSG(wrong == 0, "RGBA8 alpha changed in " + std::to_string(wrong) + " pixels");
  }
  // NV12 from the RGB scene: luma and chroma both improve.
  {
    const LowLightScene scene = still_scene();
    Denoiser denoiser(backend, params);
    DeviceImage in(PixelFormat::kNv12, kWidth, kHeight);
    DeviceImage out(PixelFormat::kNv12, kWidth, kHeight);
    const auto clean = test::rgb_to_nv12(render_clean(scene, 0), kWidth, kHeight);
    double input_psnr = 0.0;
    double output_psnr = 0.0;
    for (int f = 0; f < 8; ++f) {
      const auto noisy = test::rgb_to_nv12(render_noisy(scene, kNoise, f), kWidth, kHeight);
      in.upload(noisy);
      denoiser.process(in.view(), out.view(), nullptr);
      const auto result = out.download();
      input_psnr = psnr(noisy.data(), clean.data(), clean.size());
      output_psnr = psnr(result.data(), clean.data(), clean.size());
    }
    std::printf("  %-34s NV12 gain %.2f dB\n", label(backend, params).c_str(),
                output_psnr - input_psnr);
    VERIFY_MSG(output_psnr - input_psnr > 3.0, "NV12 gains too little");
  }
  // Odd and tiny sizes: a flat frame comes out flat, the last row and column
  // included (where the vpi backend pads odd sizes).
  for (const auto& [w, h] :
       {std::pair{33, 17}, std::pair{1, 1}, std::pair{2, 3}, std::pair{7, 1}}) {
    for (const PixelFormat format :
         {PixelFormat::kMono8, PixelFormat::kRgb8, PixelFormat::kRgba8}) {
      const int bytes = pixel_bytes(format);
      std::vector<std::uint8_t> flat(static_cast<std::size_t>(w) * h * bytes);
      for (std::size_t i = 0; i < flat.size(); ++i) {
        flat[i] =
            static_cast<std::uint8_t>(bytes == 1 ? 80 : 60 + 20 * static_cast<int>(i % bytes));
      }
      Denoiser denoiser(backend, params);
      DeviceImage in(format, w, h);
      DeviceImage out(format, w, h);
      in.upload(flat);
      std::vector<std::uint8_t> result;
      for (int f = 0; f < 3; ++f) {
        denoiser.process(in.view(), out.view(), nullptr);
        result = out.download();
      }
      int far = 0;
      for (std::size_t i = 0; i < flat.size(); ++i) far += std::abs(result[i] - flat[i]) > 3;
      VERIFY_MSG(far == 0, std::to_string(w) + "x" + std::to_string(h) + " " +
                               std::string(format_name(format)) + ": " + std::to_string(far) +
                               " samples moved");
    }
  }
}

// Two denoisers given the same frames give the same bytes.
void check_determinism(Backend backend) {
  const LowLightScene scene{.width = kWidth, .height = kHeight, .speed = 5.0F, .object_radius = 20};
  std::vector<SpatialFilter> spatials{SpatialFilter::kNone};
  if (backend == Backend::kCuda) {
    spatials.push_back(SpatialFilter::kBilateral);
    spatials.push_back(SpatialFilter::kNlm);
  }
  for (const SpatialFilter spatial : spatials) {
    const DenoiseParams params = params_of(spatial, true, 0.7F);
    std::vector<std::vector<std::uint8_t>> first;
    run_sequence(backend, params, scene, 6,
                 [&](int, const auto&, const auto& out) { first.push_back(out); });
    std::size_t index = 0;
    int differing = 0;
    run_sequence(backend, params, scene, 6,
                 [&](int, const auto&, const auto& out) { differing += out != first[index++]; });
    VERIFY_MSG(differing == 0, label(backend, params) + ": " + std::to_string(differing) +
                                   " frames differ between runs");
  }
}

// The noise estimate finds the noise of the scene within 25 %, with read noise
// alone and with shot noise too (whose level varies with brightness).
void check_noise_estimate() {
  for (const NoiseModel noise : {NoiseModel{.read_sigma = 8.0F, .shot_gain = 0.0F, .seed = 3},
                                 NoiseModel{.read_sigma = 4.0F, .shot_gain = 1.0F, .seed = 4}}) {
    const LowLightScene scene = still_scene(PixelFormat::kRgb8, 640, 480);
    Denoiser denoiser(Backend::kCuda, params_of(SpatialFilter::kBilateral, false, 0.5F));
    DeviceImage in(scene.format, scene.width, scene.height);
    DeviceImage out(scene.format, scene.width, scene.height);
    test::cuda_check(render_noisy(scene, noise, 0, in.view(), nullptr), "render_noisy");
    denoiser.process(in.view(), out.view(), nullptr);
    const double truth = std::sqrt(noise_variance(noise, 45.0));
    const float estimate = denoiser.noise_sigma(nullptr);
    std::printf("  noise estimate %.2f for sigma %.2f\n", estimate, truth);
    VERIFY_MSG(std::abs(estimate - truth) < 0.25 * truth,
               "estimate " + std::to_string(estimate) + " for " + std::to_string(truth));
  }
  // A configured level is used as it is.
  DenoiseParams params = params_of(SpatialFilter::kBilateral, false, 0.5F);
  params.noise_sigma = 5.5F;
  Denoiser denoiser(Backend::kCuda, params);
  DeviceImage in(PixelFormat::kRgb8, 32, 32);
  DeviceImage out(PixelFormat::kRgb8, 32, 32);
  denoiser.process(in.view(), out.view(), nullptr);
  VERIFY(denoiser.noise_sigma(nullptr) == 5.5F);
}

// The cuda backend and VPI's TNR on the same frames. They are different
// filters, so their results differ, but both must denoise the still scene, keep
// its brightness and colour (channel means within 1 code value of each other),
// and leave no trail behind the moving disc.
void check_cuda_vpi_parity() {
  const DenoiseParams params = params_of(SpatialFilter::kNone, true, 0.5F);
  const LowLightScene scene = still_scene();
  std::vector<std::uint8_t> outputs[2];
  Quality quality[2];
  const Backend backends[2] = {Backend::kCuda, Backend::kVpi};
  for (int b = 0; b < 2; ++b) {
    run_sequence(backends[b], params, scene, 16, [&](int f, const auto& noisy, const auto& out) {
      const auto clean = render_clean(scene, f);
      quality[b] = Quality{psnr(noisy.data(), clean.data(), clean.size()),
                           psnr(out.data(), clean.data(), clean.size())};
      outputs[b] = out;
    });
  }
  std::printf("  still scene, 16 frames: cuda %.2f dB, vpi %.2f dB (input %.2f dB)\n",
              quality[0].output_psnr, quality[1].output_psnr, quality[0].input_psnr);
  VERIFY(quality[0].gain() > 5.0 && quality[1].gain() > 5.0);
  for (int c = 0; c < 3; ++c) {
    double means[2] = {};
    for (int b = 0; b < 2; ++b) {
      for (std::size_t i = c; i < outputs[b].size(); i += 3) means[b] += outputs[b][i];
      means[b] /= static_cast<double>(outputs[b].size() / 3);
    }
    std::printf("  channel %d mean: cuda %.2f, vpi %.2f\n", c, means[0], means[1]);
    VERIFY_MSG(std::abs(means[0] - means[1]) < 1.0, "the backends' brightness differs");
  }
  check_no_ghosting(Backend::kVpi, params);
}

// Where VPI runs TNR on the VIC, the vpi backend uses it, except for frames
// smaller than the VIC takes (64x16), which run on VPI's CUDA backend; asking for
// the VIC with such a frame is refused.
void check_vic_engine() {
  const DenoiseParams params = params_of(SpatialFilter::kNone, true, 0.5F);
  const auto engine_for = [&](VpiEngine engine, int width, int height) {
    Denoiser denoiser(Backend::kVpi, params, VpiOptions{.engine = engine});
    DeviceImage in(PixelFormat::kRgb8, width, height);
    DeviceImage out(PixelFormat::kRgb8, width, height);
    denoiser.process(in.view(), out.view(), nullptr);
    return std::string(denoiser.engine());
  };
  VERIFY(engine_for(VpiEngine::kAuto, 320, 240) == "vpi-vic");
  VERIFY(engine_for(VpiEngine::kVic, 64, 16) == "vpi-vic");
  VERIFY(engine_for(VpiEngine::kAuto, 48, 48) == "vpi-cuda");
  VERIFY(engine_for(VpiEngine::kCuda, 320, 240) == "vpi-cuda");
  VERIFY(test::throws<std::invalid_argument>([&] { engine_for(VpiEngine::kVic, 33, 17); }));
}

// Unusable settings and views are refused.
void check_rejections() {
  using test::throws;
  DenoiseParams p;
  p.strength = 1.5F;
  VERIFY(throws<std::invalid_argument>([&] { Denoiser d(Backend::kCuda, p); }));
  p = DenoiseParams{.spatial = SpatialFilter::kNone, .temporal = false};
  VERIFY(throws<std::invalid_argument>([&] { Denoiser d(Backend::kCuda, p); }));
  p = DenoiseParams{.spatial = SpatialFilter::kBilateral, .temporal = true};
  VERIFY(throws<std::invalid_argument>([&] { Denoiser d(Backend::kVpi, p); }));
  Denoiser d(Backend::kCuda, DenoiseParams{});
  DeviceImage rgb(PixelFormat::kRgb8, 16, 16);
  DeviceImage mono(PixelFormat::kMono8, 16, 16);
  DeviceImage small(PixelFormat::kRgb8, 8, 16);
  VERIFY(throws<std::invalid_argument>([&] { d.process(rgb.view(), mono.view(), nullptr); }));
  VERIFY(throws<std::invalid_argument>([&] { d.process(rgb.view(), small.view(), nullptr); }));
  const DeviceImage nv12(PixelFormat::kNv12, 16, 16);
  ImageView odd = nv12.view();
  odd.width = 15;
  VERIFY(throws<std::invalid_argument>([&] { d.process(odd, odd, nullptr); }));
}

}  // namespace
}  // namespace holoscan::denoise

int main() {
  using holoscan::denoise::Backend;
  using holoscan::denoise::DenoiseParams;
  using holoscan::denoise::SpatialFilter;
  namespace dn = holoscan::denoise;
  std::vector<dn::test::TestCase> cases{
      {"strength 0 is the identity", dn::check_strength_zero_is_identity},
      {"PSNR gain by method and strength", dn::check_psnr_gains},
      {"temporal convergence on a still scene", dn::check_temporal_convergence},
      {"no ghosting: temporal",
       [] {
         dn::check_no_ghosting(Backend::kCuda, dn::params_of(SpatialFilter::kNone, true, 1.0F));
       }},
      {"no ghosting: temporal + bilateral",
       [] {
         dn::check_no_ghosting(Backend::kCuda,
                               dn::params_of(SpatialFilter::kBilateral, true, 1.0F));
       }},
      {"no ghosting: temporal + nlm",
       [] {
         dn::check_no_ghosting(Backend::kCuda, dn::params_of(SpatialFilter::kNlm, true, 1.0F));
       }},
      {"first frame", dn::check_first_frame},
      {"scene cut and reset: cuda", [] { dn::check_scene_cut(Backend::kCuda); }},
      {"formats and sizes: cuda", [] { dn::check_formats_and_sizes(Backend::kCuda); }},
      {"determinism: cuda", [] { dn::check_determinism(Backend::kCuda); }},
      {"noise estimate", dn::check_noise_estimate},
      {"rejections", dn::check_rejections},
  };
  if (dn::vpi_built()) {
    cases.push_back({"cuda against vpi", dn::check_cuda_vpi_parity});
    cases.push_back({"scene cut and reset: vpi", [] { dn::check_scene_cut(Backend::kVpi); }});
    cases.push_back({"formats and sizes: vpi", [] { dn::check_formats_and_sizes(Backend::kVpi); }});
    cases.push_back({"determinism: vpi", [] { dn::check_determinism(Backend::kVpi); }});
    if (dn::probe_platform().vic_tnr) {
      cases.push_back({"vpi on the VIC", dn::check_vic_engine});
    }
  }
  return dn::test::run(cases);
}
