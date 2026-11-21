// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The accuracy cases, run against any backend that gathers FrameStats: the CPU
// reference in the SDK-free host test, the GPU analyzer (host and device input)
// in the GPU test. Every scene is synthetic and made here; each case checks the
// metrics and flags against what the degradation it applies should produce.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/image_quality/quality.hpp>
#include <holoscan/image_quality/synthetic.hpp>

#include "test_support.hpp"

namespace holoscan::image_quality::test {

// A backend under test: gathers one frame's statistics, remembering the last frame.
class StatsSource {
 public:
  virtual ~StatsSource() = default;
  [[nodiscard]] virtual FrameStats gather(const HostImage& image) = 0;
};
using SourceFactory = std::function<std::unique_ptr<StatsSource>(const StatsOptions&)>;

// A backend and an evaluator: frames in, reports out.
class Checker {
 public:
  Checker(const SourceFactory& factory, QualityThresholds thresholds = {},
          StatsOptions options = {})
      : source_(factory(options)), evaluator_(thresholds) {}
  QualityReport operator()(const HostImage& image) {
    return evaluator_.evaluate(source_->gather(image));
  }

 private:
  std::unique_ptr<StatsSource> source_;
  QualityEvaluator evaluator_;
};

inline std::string describe(const QualityReport& report) {
  return "flags " + flag_names(report.flags) + ", focus " +
         std::to_string(report.metric(Metric::kFocus)) + ", noise " +
         std::to_string(report.metric(Metric::kNoiseSigma)) + ", mean " +
         std::to_string(report.metric(Metric::kMeanLuma));
}

// Standard deviation of luma noise when each color channel gets independent
// noise of `sigma`: the luma weights' root sum of squares, times sigma.
inline double luma_noise(double sigma) {
  return sigma * std::sqrt(77.0 * 77.0 + 150.0 * 150.0 + 29.0 * 29.0) / 256.0;
}

// The metrics computed directly from the pixels, in floating point, sharing no
// code with the backends or the evaluator.
struct NaiveMetrics {
  double mean_luma{};
  double p5{};
  double p95{};
  double underexposed{};
  double overexposed{};
  double laplacian_variance{};
  double tenengrad{};
  double noise_sigma{};
  double mean_red{};
  double mean_blue{};
};

inline NaiveMetrics naive_metrics(const HostImage& image, const QualityThresholds& t,
                                  double flat_fraction) {
  const int w = image.width;
  const int h = image.height;
  const int ch = channels(image.format);
  std::vector<double> y(static_cast<std::size_t>(w) * h);
  NaiveMetrics m;
  for (std::size_t i = 0; i < y.size(); ++i) {
    const std::uint8_t* p = image.pixels.data() + i * ch;
    const bool bgr = image.format == PixelFormat::kBgr8 || image.format == PixelFormat::kBgra8;
    const double r = ch == 1 ? p[0] : p[bgr ? 2 : 0];
    const double g = ch == 1 ? p[0] : p[1];
    const double b = ch == 1 ? p[0] : p[bgr ? 0 : 2];
    // Rounded BT.601 luma, as 8-bit values.
    y[i] = ch == 1 ? r : std::floor((77.0 * r + 150.0 * g + 29.0 * b) / 256.0 + 0.5);
    m.mean_red += r;
    m.mean_blue += b;
  }
  const double n = static_cast<double>(y.size());
  m.mean_red /= n;
  m.mean_blue /= n;
  std::vector<double> sorted = y;
  std::sort(sorted.begin(), sorted.end());
  const auto rank = [&](double q) {
    return sorted[static_cast<std::size_t>(std::max(1.0, std::ceil(q * n))) - 1];
  };
  m.p5 = rank(0.05);
  m.p95 = rank(0.95);
  for (const double v : y) {
    m.mean_luma += v / n;
    m.underexposed += v <= t.dark_level ? 1.0 / n : 0.0;
    m.overexposed += v >= t.bright_level ? 1.0 / n : 0.0;
  }
  const auto at = [&](int x, int yy) { return y[static_cast<std::size_t>(yy) * w + x]; };
  double sum = 0;
  double sum_sq = 0;
  double grad = 0;
  std::vector<std::pair<double, double>> gradient_and_high_pass;
  for (int yy = 1; yy + 1 < h; ++yy) {
    for (int x = 1; x + 1 < w; ++x) {
      const double lap =
          4 * at(x, yy) - at(x - 1, yy) - at(x + 1, yy) - at(x, yy - 1) - at(x, yy + 1);
      sum += lap;
      sum_sq += lap * lap;
      double gx = 0;
      double gy = 0;
      double hp = 0;
      const double sobel[3] = {1, 2, 1};
      const double high_pass[3][3] = {{1, -2, 1}, {-2, 4, -2}, {1, -2, 1}};
      for (int k = -1; k <= 1; ++k) {
        gx += sobel[k + 1] * (at(x + 1, yy + k) - at(x - 1, yy + k));
        gy += sobel[k + 1] * (at(x + k, yy + 1) - at(x + k, yy - 1));
        for (int j = -1; j <= 1; ++j) hp += high_pass[k + 1][j + 1] * at(x + j, yy + k);
      }
      grad += (gx * gx + gy * gy) / 64.0;
      gradient_and_high_pass.emplace_back(std::abs(gx) + std::abs(gy), std::abs(hp));
    }
  }
  const double interior = static_cast<double>(gradient_and_high_pass.size());
  if (interior > 0) {
    m.laplacian_variance = sum_sq / interior - (sum / interior) * (sum / interior);
    m.tenengrad = grad / interior;
    // The flat pixels: every pixel at or below the gradient that reaches flat_fraction.
    std::vector<double> gradients;
    for (const auto& [g, hp] : gradient_and_high_pass) gradients.push_back(g);
    std::sort(gradients.begin(), gradients.end());
    const double limit =
        gradients[static_cast<std::size_t>(std::max(1.0, std::ceil(flat_fraction * interior))) - 1];
    std::vector<double> flat;
    for (const auto& [g, hp] : gradient_and_high_pass) {
      if (g <= limit) flat.push_back(hp);
    }
    std::sort(flat.begin(), flat.end());
    // The median of |H| over the flat pixels, scaled to the pixel noise; the
    // evaluator interpolates within the median's integer bin, so they agree to
    // within that bin.
    const double median = flat.size() % 2 == 1
                              ? flat[flat.size() / 2]
                              : (flat[flat.size() / 2 - 1] + flat[flat.size() / 2]) / 2.0;
    m.noise_sigma = median / 0.6744897501960817 / 6.0;
  }
  return m;
}

inline bool near(double actual, double expected, double relative, double absolute) {
  return std::abs(actual - expected) <= std::max(absolute, relative * std::abs(expected));
}

inline void check_naive_reference(const SourceFactory& factory) {
  const QualityThresholds thresholds;
  const StatsOptions options;
  struct Case {
    const char* name;
    HostImage image;
  };
  std::vector<Case> cases;
  cases.push_back({"rgb", render_scene(64, 48, 3)});
  cases.push_back({"mono", render_scene(61, 47, 5, PixelFormat::kMono8)});
  HostImage noisy = render_scene(64, 48, 1, PixelFormat::kBgra8);
  add_noise(noisy, 9.0, 3);
  cases.push_back({"noisy bgra", std::move(noisy)});
  HostImage dark = render_scene(64, 48, 2);
  gaussian_blur(dark, 1.5);
  apply_gain(dark, 0.3);
  cases.push_back({"blurred dark", std::move(dark)});
  for (const Case& c : cases) {
    Checker check(factory, thresholds, options);
    const QualityReport r = check(c.image);
    const NaiveMetrics m = naive_metrics(c.image, thresholds, options.flat_fraction);
    const auto expect = [&](Metric metric, double expected, double relative, double absolute) {
      const double actual = r.metric(metric);
      VERIFY_MSG(near(actual, expected, relative, absolute),
                 std::string(c.name) + ": " + std::string(metric_name(metric)) + " " +
                     std::to_string(actual) + ", direct computation " + std::to_string(expected));
    };
    expect(Metric::kMeanLuma, m.mean_luma, 1e-5, 1e-3);
    expect(Metric::kLumaP5, m.p5, 0, 0);
    expect(Metric::kLumaP95, m.p95, 0, 0);
    expect(Metric::kUnderexposedFraction, m.underexposed, 1e-5, 1e-6);
    expect(Metric::kOverexposedFraction, m.overexposed, 1e-5, 1e-6);
    expect(Metric::kLaplacianVariance, m.laplacian_variance, 1e-5, 1e-3);
    expect(Metric::kTenengrad, m.tenengrad, 1e-5, 1e-3);
    expect(Metric::kMeanRed, m.mean_red, 1e-5, 1e-3);
    expect(Metric::kMeanBlue, m.mean_blue, 1e-5, 1e-3);
    // Within the median's bin: one level of |H|, a 6 * 0.6745 = 4 times smaller noise step.
    expect(Metric::kNoiseSigma, m.noise_sigma, 0, 1.0 / (6.0 * 0.6744897501960817));
  }
}

// Frames of a moving clean scene: no flag, in every format and at several sizes,
// long enough for the obstruction window to fill.
inline void check_clean_scenes(const SourceFactory& factory) {
  struct Size {
    int width;
    int height;
    PixelFormat format;
  };
  const Size sizes[] = {
      {640, 360, PixelFormat::kRgb8},  {640, 360, PixelFormat::kBgr8},
      {640, 360, PixelFormat::kRgba8}, {640, 360, PixelFormat::kBgra8},
      {640, 360, PixelFormat::kMono8}, {1280, 720, PixelFormat::kRgb8},
      {333, 211, PixelFormat::kRgb8},  {320, 240, PixelFormat::kMono8},
  };
  const QualityThresholds thresholds;
  for (const Size& size : sizes) {
    Checker check(factory, thresholds);
    int flagged = 0;
    std::string first;
    for (std::uint32_t frame = 0;
         frame < static_cast<std::uint32_t>(thresholds.obstruction_frames) + 5; ++frame) {
      const QualityReport r = check(render_scene(size.width, size.height, frame, size.format));
      if (r.flags != 0U) {
        if (flagged++ == 0) first = "frame " + std::to_string(frame) + ": " + describe(r);
      }
    }
    VERIFY_MSG(flagged == 0, std::to_string(size.width) + "x" + std::to_string(size.height) +
                                 " format " + std::to_string(static_cast<int>(size.format)) + ": " +
                                 std::to_string(flagged) + " clean frames flagged, first " + first);
  }
}

// Gaussian blur of growing sigma: every sharpness score falls, and only strong
// blur flags the frame.
inline void check_blur(const SourceFactory& factory) {
  const double sigmas[] = {0.0, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0};
  float previous[3] = {INFINITY, INFINITY, INFINITY};
  for (const double sigma : sigmas) {
    Checker check(factory);
    HostImage image = render_scene(640, 360, 0);
    gaussian_blur(image, sigma);
    const QualityReport r = check(image);
    const float scores[3] = {r.metric(Metric::kLaplacianVariance), r.metric(Metric::kTenengrad),
                             r.metric(Metric::kFocus)};
    for (int s = 0; s < 3; ++s) {
      VERIFY_MSG(scores[s] < previous[s], "sigma " + std::to_string(sigma) + ": score " +
                                              std::to_string(s) + " " + std::to_string(scores[s]) +
                                              " not below " + std::to_string(previous[s]));
      previous[s] = scores[s];
    }
    // Near the threshold (sigma 1.5) the frame may go either way.
    const std::uint32_t expected = sigma >= 2.0 ? kBlurred : 0U;
    if (sigma <= 1.0 || sigma >= 2.0) {
      VERIFY_MSG(r.flags == expected, "sigma " + std::to_string(sigma) + ": " + describe(r));
    }
  }
  // The same blur flags a noisy frame too, once the noise's share is taken out.
  Checker check(factory);
  HostImage image = render_scene(640, 360, 0);
  gaussian_blur(image, 3.0);
  add_noise(image, 3.0, 11);
  const QualityReport r = check(image);
  VERIFY_MSG(r.flags == kBlurred, "blurred and noisy: " + describe(r));
}

// Pixel bounds of tile (tx, ty) of a report's grid over a width x height image.
inline PixelRect tile_rect(const QualityReport& r, int tx, int ty, int width, int height) {
  const int x0 = tile_begin(tx, r.tiles_x, width);
  const int y0 = tile_begin(ty, r.tiles_y, height);
  return PixelRect{x0, y0, tile_begin(tx + 1, r.tiles_x, width) - x0,
                   tile_begin(ty + 1, r.tiles_y, height) - y0};
}

// Whether every pixel of `rect` (when `all`), or any pixel, satisfies `in`.
template <typename In>
bool rect_pixels(const PixelRect& rect, bool all, In&& in) {
  for (int y = rect.y; y < rect.y + rect.height; ++y) {
    for (int x = rect.x; x < rect.x + rect.width; ++x) {
      if (in(x, y) != all) return !all;
    }
  }
  return all;
}

// Blur inside one region: the tiles inside it are blurred, those clear of it
// (and of the blur's reach) are not.
inline void check_local_blur(const SourceFactory& factory) {
  Checker check(factory);
  HostImage image = render_scene(640, 360, 4);
  const PixelRect region{100, 60, 160, 120};
  const double sigma = 3.0;
  gaussian_blur(image, sigma, region);
  const QualityReport r = check(image);
  VERIFY_MSG(r.flags == kLocalBlur, describe(r));
  // A tile's stencils reach one pixel beyond it; the blur, 3 sigma into the sharp area.
  const int reach = 1 + static_cast<int>(std::ceil(3 * sigma));
  int inside_tiles = 0;
  for (int ty = 0; ty < r.tiles_y; ++ty) {
    for (int tx = 0; tx < r.tiles_x; ++tx) {
      const PixelRect rect = tile_rect(r, tx, ty, image.width, image.height);
      const bool inside = rect_pixels(rect, true, [&](int x, int y) {
        return x >= region.x && x < region.x + region.width && y >= region.y &&
               y < region.y + region.height;
      });
      const bool clear = !rect_pixels(rect, false, [&](int x, int y) {
        return x >= region.x - reach && x < region.x + region.width + reach &&
               y >= region.y - reach && y < region.y + region.height + reach;
      });
      const bool blurred = (r.tile_flags(tx, ty) & kTileBlurred) != 0U;
      const std::string where = "tile (" + std::to_string(tx) + ", " + std::to_string(ty) + ")";
      inside_tiles += inside ? 1 : 0;
      if (inside) VERIFY_MSG(blurred, where + " inside the blur is not flagged");
      if (clear) VERIFY_MSG(!blurred, where + " clear of the blur is flagged");
    }
  }
  VERIFY_MSG(inside_tiles == 6, std::to_string(inside_tiles) + " tiles inside the blur");
}

inline void check_exposure(const SourceFactory& factory) {
  {
    Checker check(factory);
    HostImage image = render_scene(640, 360, 2);
    apply_gain(image, 0.25);
    const QualityReport r = check(image);
    VERIFY_MSG(r.flags == kUnderexposed, "dark: " + describe(r));
    VERIFY(r.metric(Metric::kMeanLuma) < 40.0F);
  }
  {
    Checker check(factory);
    HostImage image = render_scene(640, 360, 2);
    apply_gain(image, 2.0);
    const QualityReport r = check(image);
    VERIFY_MSG(r.flags == kOverexposed, "bright: " + describe(r));
    VERIFY(r.metric(Metric::kOverexposedFraction) > 0.25F);
    VERIFY(r.metric(Metric::kLumaP99) == 255.0F);
  }
  {
    // The clean scene's luma spans most of the range.
    Checker check(factory);
    const QualityReport r = check(render_scene(640, 360, 2));
    VERIFY(r.metric(Metric::kDynamicRange) > 100.0F);
    VERIFY(r.metric(Metric::kUnderexposedFraction) == 0.0F);
    VERIFY(r.metric(Metric::kOverexposedFraction) == 0.0F);
  }
}

// Noise of a known sigma: the estimate within 20 %, and the flag above the threshold.
inline void check_noise(const SourceFactory& factory) {
  for (const PixelFormat format : {PixelFormat::kRgb8, PixelFormat::kMono8}) {
    for (const double sigma : {2.0, 4.0, 8.0, 16.0}) {
      Checker check(factory);
      HostImage image = render_scene(640, 360, 7, format);
      add_noise(image, sigma, 42);
      const QualityReport r = check(image);
      const double expected = format == PixelFormat::kMono8 ? sigma : luma_noise(sigma);
      const double estimate = r.metric(Metric::kNoiseSigma);
      const std::string what = (format == PixelFormat::kMono8 ? "mono sigma " : "rgb sigma ") +
                               std::to_string(sigma) + ": estimate " + std::to_string(estimate) +
                               ", expected " + std::to_string(expected);
      VERIFY_MSG(std::abs(estimate - expected) <= 0.2 * expected, what);
      // Only the noise flag, and that only clearly above the threshold.
      VERIFY_MSG((r.flags & ~kNoisy) == 0U, what + ": " + describe(r));
      if (expected > 1.2 * QualityThresholds{}.max_noise_sigma) {
        VERIFY_MSG(r.flags == kNoisy, what + ": not flagged");
      }
      if (expected < 0.8 * QualityThresholds{}.max_noise_sigma) {
        VERIFY_MSG(r.flags == 0U, what + ": flagged");
      }
    }
  }
  Checker check(factory);
  const QualityReport r = check(render_scene(640, 360, 7));
  VERIFY_MSG(r.metric(Metric::kNoiseSigma) < 0.2F, "clean scene noise " + describe(r));
}

// Repeated frames are frozen; a near-identical frame (a few pixels off by one) too.
inline void check_frozen(const SourceFactory& factory) {
  Checker check(factory);
  const std::uint32_t frames[] = {0, 1, 1, 1, 2};
  const std::uint32_t expected[] = {0U, 0U, kFrozen, kFrozen, 0U};
  const float runs[] = {0, 0, 1, 2, 0};
  for (std::size_t i = 0; i < std::size(frames); ++i) {
    const QualityReport r = check(render_scene(320, 240, frames[i]));
    VERIFY_MSG(r.flags == expected[i], "frame " + std::to_string(i) + ": " + describe(r));
    VERIFY(r.metric(Metric::kFrozenFrames) == runs[i]);
    if (i == 0) {
      VERIFY(std::isnan(r.metric(Metric::kFrameDifference)));
    } else if (frames[i] == frames[i - 1]) {
      VERIFY(r.metric(Metric::kFrameDifference) == 0.0F);
    } else {
      VERIFY(r.metric(Metric::kFrameDifference) > 10.0F);
    }
  }
  HostImage nearly = render_scene(320, 240, 2);
  for (int i = 0; i < 10; ++i) nearly.pixels[i * 997 * 3] += 1;
  const QualityReport r = check(nearly);
  VERIFY_MSG(r.flags == kFrozen, "near-identical frame: " + describe(r));
  VERIFY(r.metric(Metric::kFrameDifference) > 0.0F);
  // A frozen run of two frames needed: the first repeat passes.
  QualityThresholds two;
  two.frozen_frames = 2;
  Checker check_two(factory, two);
  VERIFY(check_two(render_scene(320, 240, 5)).flags == 0U);
  VERIFY(check_two(render_scene(320, 240, 5)).flags == 0U);
  VERIFY(check_two(render_scene(320, 240, 5)).flags == kFrozen);
}

// A static smudge over a moving scene: obstructed after N frames, not before,
// on the tiles it covers only, and clear again once it is gone.
inline void check_obstruction(const SourceFactory& factory) {
  QualityThresholds thresholds;
  thresholds.obstruction_frames = 10;
  const Smudge smudge{340.0, 180.0, 130.0, 90.0};
  // Whether (x, y) is in the smudge grown by `margin` pixels on each radius.
  const auto in_smudge = [&smudge](int x, int y, double margin) {
    const double dx = (x - smudge.cx) / (smudge.rx + margin);
    const double dy = (y - smudge.cy) / (smudge.ry + margin);
    return dx * dx + dy * dy <= 1.0;
  };
  const auto smudged = [&](std::uint32_t frame) {
    HostImage image = render_scene(640, 360, frame);
    apply_smudge(image, smudge);
    return image;
  };
  for (const std::uint32_t clean_before : {0U, 5U}) {
    Checker check(factory, thresholds);
    std::uint32_t frame = 0;
    for (; frame < clean_before; ++frame) {
      VERIFY(check(render_scene(640, 360, frame)).flags == 0U);
    }
    for (int k = 0; k < thresholds.obstruction_frames; ++k, ++frame) {
      const QualityReport r = check(smudged(frame));
      const bool last = k + 1 == thresholds.obstruction_frames;
      VERIFY_MSG(r.flags == (last ? kObstructed : 0U),
                 "smudged frame " + std::to_string(k) + " after " + std::to_string(clean_before) +
                     " clean: " + describe(r));
      if (!last) continue;
      int inside_tiles = 0;
      for (int ty = 0; ty < r.tiles_y; ++ty) {
        for (int tx = 0; tx < r.tiles_x; ++tx) {
          const PixelRect rect = tile_rect(r, tx, ty, 640, 360);
          const bool inside =
              rect_pixels(rect, true, [&](int x, int y) { return in_smudge(x, y, -1.0); });
          const bool clear =
              !rect_pixels(rect, false, [&](int x, int y) { return in_smudge(x, y, 2.0); });
          inside_tiles += inside ? 1 : 0;
          const bool obstructed = (r.tile_flags(tx, ty) & kTileObstructed) != 0U;
          const std::string where = "tile (" + std::to_string(tx) + ", " + std::to_string(ty) + ")";
          if (inside) VERIFY_MSG(obstructed, where + " under the smudge is not flagged");
          if (clear) VERIFY_MSG(!obstructed, where + " clear of the smudge is flagged");
        }
      }
      VERIFY_MSG(inside_tiles >= 6, std::to_string(inside_tiles) + " tiles under the smudge");
    }
    const QualityReport r = check(render_scene(640, 360, frame));
    VERIFY_MSG(r.flags == 0U, "smudge gone: " + describe(r));
  }
}

inline void check_color_cast(const SourceFactory& factory) {
  {
    Checker check(factory);
    HostImage image = render_scene(640, 360, 9);
    apply_channel_gains(image, 1.3, 1.0, 0.75);
    const QualityReport r = check(image);
    VERIFY_MSG(r.flags == kColorCast, "tinted: " + describe(r));
    VERIFY(r.metric(Metric::kMeanRed) > r.metric(Metric::kMeanBlue) * 1.4F);
  }
  {
    Checker check(factory);
    HostImage image = render_scene(640, 360, 9, PixelFormat::kBgr8);
    apply_channel_gains(image, 0.75, 1.0, 1.3);
    const QualityReport r = check(image);
    VERIFY_MSG(r.flags == kColorCast, "tinted BGR: " + describe(r));
    VERIFY(r.metric(Metric::kMeanBlue) > r.metric(Metric::kMeanRed) * 1.4F);
  }
  {
    Checker check(factory);
    const QualityReport r = check(render_scene(640, 360, 9));
    VERIFY_MSG(r.metric(Metric::kColorCast) < 0.03F, "clean cast " + describe(r));
  }
  {
    Checker check(factory);
    const QualityReport r = check(render_scene(640, 360, 9, PixelFormat::kMono8));
    VERIFY(r.metric(Metric::kColorCast) == 0.0F);
  }
}

// Tiny and constant images: no failure, NaN for what has no pixels to compute
// it from, and only the flags that hold.
inline void check_edge_cases(const SourceFactory& factory) {
  const int sizes[][2] = {{1, 1}, {2, 2}, {3, 3}, {4, 3}, {17, 1}, {1, 17}, {5, 40}};
  for (const auto& size : sizes) {
    for (const PixelFormat format : {PixelFormat::kMono8, PixelFormat::kRgba8}) {
      Checker check(factory);
      const std::string what = std::to_string(size[0]) + "x" + std::to_string(size[1]);
      const QualityReport r = check(render_scene(size[0], size[1], 0, format));
      VERIFY_MSG(
          r.tiles_x == std::min(size[0], 16) && r.tiles_y == std::min(size[1], 9),
          what + ": tile grid " + std::to_string(r.tiles_x) + "x" + std::to_string(r.tiles_y));
      VERIFY_MSG(std::isfinite(r.metric(Metric::kMeanLuma)), what + ": mean luma");
      const bool interior = size[0] >= 3 && size[1] >= 3;
      VERIFY_MSG(
          std::isfinite(r.metric(Metric::kLaplacianVariance)) == interior,
          what + ": laplacian variance " + std::to_string(r.metric(Metric::kLaplacianVariance)));
      VERIFY_MSG(std::isfinite(r.metric(Metric::kNoiseSigma)) == interior, what + ": noise");
      VERIFY_MSG((r.flags & (kNoisy | kFrozen | kObstructed)) == 0U, what + ": " + describe(r));
      const QualityReport again = check(render_scene(size[0], size[1], 0, format));
      VERIFY_MSG((again.flags & kFrozen) != 0U, what + ": repeat not frozen");
    }
  }
  // A constant image: nothing to judge focus on, no noise, frozen when repeated,
  // and obstructed once the window is full.
  QualityThresholds thresholds;
  thresholds.obstruction_frames = 3;
  Checker check(factory, thresholds);
  HostImage flat{64, 48, PixelFormat::kRgb8,
                 std::vector<std::uint8_t>(static_cast<std::size_t>(64) * 48 * 3, 128)};
  for (int frame = 0; frame < 3; ++frame) {
    const QualityReport r = check(flat);
    const std::uint32_t expected = (frame > 0 ? kFrozen : 0U) | (frame == 2 ? kObstructed : 0U);
    VERIFY_MSG(r.flags == expected, "constant frame " + std::to_string(frame) + ": " + describe(r));
    VERIFY(std::isnan(r.metric(Metric::kFocus)));
    VERIFY(r.metric(Metric::kLaplacianVariance) == 0.0F);
    VERIFY(r.metric(Metric::kTexturedTileFraction) == 0.0F);
    VERIFY(r.metric(Metric::kNoiseSigma) < 0.1F);
    VERIFY(r.metric(Metric::kDynamicRange) == 0.0F);
  }
}

// The bits of every float, so that NaNs compare too.
inline bool same_report(const QualityReport& a, const QualityReport& b) {
  return a.flags == b.flags && a.hash == b.hash && a.tiles_x == b.tiles_x &&
         a.tiles_y == b.tiles_y && a.tiles.size() == b.tiles.size() &&
         std::memcmp(a.metrics.data(), b.metrics.data(), sizeof(a.metrics)) == 0 &&
         std::memcmp(a.tiles.data(), b.tiles.data(), a.tiles.size() * sizeof(float)) == 0;
}

// Two runs over the same frames give the same bits.
inline void check_determinism(const SourceFactory& factory) {
  std::vector<HostImage> frames;
  for (std::uint32_t i = 0; i < 4; ++i) {
    HostImage image = render_scene(500, 300, i);
    add_noise(image, 6.0, i);
    frames.push_back(std::move(image));
  }
  frames.push_back(frames.back());
  Checker first(factory);
  Checker second(factory);
  for (std::size_t i = 0; i < frames.size(); ++i) {
    VERIFY_MSG(same_report(first(frames[i]), second(frames[i])),
               "frame " + std::to_string(i) + " differs between runs");
  }
}

inline std::vector<TestCase> accuracy_cases(const std::string& prefix,
                                            const SourceFactory& factory) {
  return {
      {prefix + "naive_reference", [factory] { check_naive_reference(factory); }},
      {prefix + "clean_scenes", [factory] { check_clean_scenes(factory); }},
      {prefix + "blur", [factory] { check_blur(factory); }},
      {prefix + "local_blur", [factory] { check_local_blur(factory); }},
      {prefix + "exposure", [factory] { check_exposure(factory); }},
      {prefix + "noise", [factory] { check_noise(factory); }},
      {prefix + "frozen", [factory] { check_frozen(factory); }},
      {prefix + "obstruction", [factory] { check_obstruction(factory); }},
      {prefix + "color_cast", [factory] { check_color_cast(factory); }},
      {prefix + "edge_cases", [factory] { check_edge_cases(factory); }},
      {prefix + "determinism", [factory] { check_determinism(factory); }},
  };
}

}  // namespace holoscan::image_quality::test
