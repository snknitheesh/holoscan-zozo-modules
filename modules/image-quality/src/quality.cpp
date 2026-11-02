// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/image_quality/quality.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace holoscan::image_quality {
namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// Noise of standard deviation sigma on every pixel gives the 4-neighbor
// Laplacian a variance of 20 sigma^2 (4^2 + 4 * 1^2), the Sobel / 8 gradient a
// mean square of 0.375 sigma^2 (2 * 12 / 64), and the high-pass a standard
// deviation of 6 sigma (the square root of the sum of its squared weights, 36).
constexpr double kLaplacianNoiseGain = 20.0;
constexpr double kGradientNoiseGain = 0.375;
constexpr double kHighPassNoiseGain = 6.0;
constexpr double kRoundingVariance = 1.0 / 12.0;
// Median of |X| for a zero-mean Gaussian X, in standard deviations.
constexpr double kGaussianMedianAbs = 0.6744897501960817;

constexpr std::pair<Metric, std::string_view> kMetricNames[] = {
    {Metric::kMeanLuma, "mean_luma"},
    {Metric::kLumaP1, "luma_p1"},
    {Metric::kLumaP5, "luma_p5"},
    {Metric::kLumaP50, "luma_p50"},
    {Metric::kLumaP95, "luma_p95"},
    {Metric::kLumaP99, "luma_p99"},
    {Metric::kUnderexposedFraction, "underexposed_fraction"},
    {Metric::kOverexposedFraction, "overexposed_fraction"},
    {Metric::kDynamicRange, "dynamic_range"},
    {Metric::kLaplacianVariance, "laplacian_variance"},
    {Metric::kTenengrad, "tenengrad"},
    {Metric::kFocus, "focus"},
    {Metric::kBlurredTileFraction, "blurred_tile_fraction"},
    {Metric::kTexturedTileFraction, "textured_tile_fraction"},
    {Metric::kNoiseSigma, "noise_sigma"},
    {Metric::kFrameDifference, "frame_difference"},
    {Metric::kFrozenFrames, "frozen_frames"},
    {Metric::kObstructedTileFraction, "obstructed_tile_fraction"},
    {Metric::kMeanRed, "mean_red"},
    {Metric::kMeanGreen, "mean_green"},
    {Metric::kMeanBlue, "mean_blue"},
    {Metric::kColorCast, "color_cast"},
};
static_assert(std::size(kMetricNames) == kMetricCount);

constexpr std::pair<std::uint32_t, std::string_view> kFlagNames[] = {
    {kBlurred, "blurred"},         {kLocalBlur, "local_blur"}, {kUnderexposed, "underexposed"},
    {kOverexposed, "overexposed"}, {kNoisy, "noisy"},          {kFrozen, "frozen"},
    {kObstructed, "obstructed"},   {kColorCast, "color_cast"},
};

// Nearest-rank percentile: the smallest level whose cumulative count reaches
// `fraction` of `total`.
float percentile(const std::array<std::uint32_t, kLumaBins>& histogram, std::uint64_t total,
                 double fraction) {
  const auto rank = static_cast<std::uint64_t>(std::max(1.0, std::ceil(fraction * total)));
  std::uint64_t cumulative = 0;
  for (int level = 0; level < kLumaBins; ++level) {
    cumulative += histogram[level];
    if (cumulative >= rank) {
      return static_cast<float>(level);
    }
  }
  return 255.0F;
}

// The noise standard deviation from the flat pixels' |H| histogram: its median,
// interpolated within the bin that holds it (bin k holds values that round to k),
// scaled from |H| to a pixel's noise.
double noise_sigma(const std::vector<std::uint32_t>& histogram) {
  std::uint64_t total = 0;
  for (const std::uint32_t count : histogram) total += count;
  if (total == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double half = static_cast<double>(total) / 2.0;
  std::uint64_t before = 0;
  for (std::size_t bin = 0; bin < histogram.size(); ++bin) {
    const std::uint32_t count = histogram[bin];
    if (count != 0 && static_cast<double>(before + count) >= half) {
      const double low = bin == 0 ? 0.0 : static_cast<double>(bin) - 0.5;
      const double width = bin == 0 ? 0.5 : 1.0;
      const double median = low + width * (half - static_cast<double>(before)) / count;
      return median / kGaussianMedianAbs / kHighPassNoiseGain;
    }
    before += count;
  }
  return std::numeric_limits<double>::quiet_NaN();
}

double median(std::vector<double> values) {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return values.size() % 2 == 1 ? values[middle] : (values[middle - 1] + values[middle]) / 2.0;
}

}  // namespace

std::string_view metric_name(Metric metric) noexcept {
  for (const auto& [which, name] : kMetricNames) {
    if (which == metric) return name;
  }
  return "unknown";
}

std::string flag_names(std::uint32_t flags) {
  std::string names;
  for (const auto& [flag, name] : kFlagNames) {
    if ((flags & flag) != 0U) {
      if (!names.empty()) names += '|';
      names += name;
    }
  }
  return names.empty() ? "ok" : names;
}

void validate_stats_options(const StatsOptions& options) {
  if (options.tiles_x < 1 || options.tiles_y < 1 || options.tiles_x > 256 ||
      options.tiles_y > 256) {
    throw std::invalid_argument("tiles_x and tiles_y must be 1..256");
  }
  if (!(options.flat_fraction > 0 && options.flat_fraction <= 1)) {
    throw std::invalid_argument("flat_fraction must be in (0, 1]");
  }
}

void validate_thresholds(const QualityThresholds& t) {
  const auto finite_nonnegative = [](double value) { return std::isfinite(value) && value >= 0; };
  if (!finite_nonnegative(t.blur_focus)) {
    throw std::invalid_argument("blur_focus must be finite and non-negative");
  }
  if (!finite_nonnegative(t.min_texture)) {
    throw std::invalid_argument("min_texture must be finite and non-negative");
  }
  if (t.dark_level < 0 || t.bright_level > 255 || t.dark_level >= t.bright_level) {
    throw std::invalid_argument("dark_level and bright_level need 0 <= dark < bright <= 255");
  }
  if (!(std::isfinite(t.min_mean_luma) && std::isfinite(t.max_mean_luma) &&
        t.min_mean_luma < t.max_mean_luma)) {
    throw std::invalid_argument("min_mean_luma must be below max_mean_luma");
  }
  if (!(t.max_underexposed_fraction >= 0 && t.max_underexposed_fraction <= 1 &&
        t.max_overexposed_fraction >= 0 && t.max_overexposed_fraction <= 1)) {
    throw std::invalid_argument("the exposed fractions must be in 0..1");
  }
  if (!finite_nonnegative(t.max_noise_sigma)) {
    throw std::invalid_argument("max_noise_sigma must be finite and non-negative");
  }
  if (!finite_nonnegative(t.frozen_difference)) {
    throw std::invalid_argument("frozen_difference must be finite and non-negative");
  }
  if (t.frozen_frames < 1) {
    throw std::invalid_argument("frozen_frames must be at least 1");
  }
  if (t.obstruction_frames < 1 || t.obstruction_frames > 3600) {
    throw std::invalid_argument("obstruction_frames must be 1..3600");
  }
  if (!(t.obstruction_fraction > 0 && t.obstruction_fraction <= 1)) {
    throw std::invalid_argument("obstruction_fraction must be in (0, 1]");
  }
  if (!finite_nonnegative(t.max_color_cast)) {
    throw std::invalid_argument("max_color_cast must be finite and non-negative");
  }
}

TileStats& TileStats::operator+=(const TileStats& other) noexcept {
  hash += other.hash;
  pixels += other.pixels;
  sum_luma += other.sum_luma;
  sum_red += other.sum_red;
  sum_green += other.sum_green;
  sum_blue += other.sum_blue;
  sum_abs_diff += other.sum_abs_diff;
  interior += other.interior;
  sum_laplacian += other.sum_laplacian;
  sum_laplacian_sq += other.sum_laplacian_sq;
  sum_gradient_sq += other.sum_gradient_sq;
  return *this;
}

TileGrid tile_grid(int width, int height, int tiles_x, int tiles_y) noexcept {
  return TileGrid{std::clamp(tiles_x, 1, std::max(width, 1)),
                  std::clamp(tiles_y, 1, std::max(height, 1))};
}

std::uint32_t flat_threshold(const std::uint32_t* histogram, int bins, double fraction) noexcept {
  std::uint64_t total = 0;
  for (int bin = 0; bin < bins; ++bin) total += histogram[bin];
  const auto target = static_cast<std::uint64_t>(std::ceil(fraction * static_cast<double>(total)));
  std::uint64_t cumulative = 0;
  for (int bin = 0; bin < bins; ++bin) {
    cumulative += histogram[bin];
    if (cumulative >= target) {
      return static_cast<std::uint32_t>(bin);
    }
  }
  return static_cast<std::uint32_t>(bins - 1);
}

QualityEvaluator::QualityEvaluator(QualityThresholds thresholds)
    : thresholds_(std::move(thresholds)) {
  validate_thresholds(thresholds_);
}

void QualityEvaluator::reset() noexcept {
  width_ = height_ = tiles_x_ = tiles_y_ = 0;
  previous_hash_.reset();
  frozen_run_ = 0;
  history_.clear();
  low_counts_.clear();
  seen_ = 0;
  cursor_ = 0;
}

QualityReport QualityEvaluator::evaluate(const FrameStats& stats) {
  const QualityThresholds& t = thresholds_;
  if (stats.width != width_ || stats.height != height_ || stats.tiles_x != tiles_x_ ||
      stats.tiles_y != tiles_y_) {
    reset();
    width_ = stats.width;
    height_ = stats.height;
    tiles_x_ = stats.tiles_x;
    tiles_y_ = stats.tiles_y;
  }
  const std::size_t tile_count = stats.tiles.size();
  const auto window = static_cast<std::uint32_t>(t.obstruction_frames);
  if (low_counts_.size() != tile_count) {
    history_.assign(tile_count * window, 0U);
    low_counts_.assign(tile_count, 0U);
    seen_ = 0;
    cursor_ = 0;
  }

  QualityReport report;
  report.tiles_x = stats.tiles_x;
  report.tiles_y = stats.tiles_y;
  report.tiles.assign(tile_count * kTileChannelCount, kNaN);
  auto& m = report.metrics;
  m.fill(kNaN);
  const auto set = [&m](Metric which, double value) {
    m[static_cast<std::size_t>(which)] = static_cast<float>(value);
  };

  TileStats total;
  for (const TileStats& tile : stats.tiles) total += tile;
  report.hash = total.hash;
  const auto pixels = static_cast<std::uint64_t>(std::max<std::int64_t>(total.pixels, 0));

  // Exposure.
  if (pixels != 0) {
    const double n = static_cast<double>(pixels);
    set(Metric::kMeanLuma, static_cast<double>(total.sum_luma) / n);
    const float p1 = percentile(stats.luma_histogram, pixels, 0.01);
    const float p99 = percentile(stats.luma_histogram, pixels, 0.99);
    set(Metric::kLumaP1, p1);
    set(Metric::kLumaP5, percentile(stats.luma_histogram, pixels, 0.05));
    set(Metric::kLumaP50, percentile(stats.luma_histogram, pixels, 0.50));
    set(Metric::kLumaP95, percentile(stats.luma_histogram, pixels, 0.95));
    set(Metric::kLumaP99, p99);
    set(Metric::kDynamicRange, p99 - p1);
    std::uint64_t dark = 0;
    std::uint64_t bright = 0;
    for (int level = 0; level < kLumaBins; ++level) {
      if (level <= t.dark_level) dark += stats.luma_histogram[level];
      if (level >= t.bright_level) bright += stats.luma_histogram[level];
    }
    set(Metric::kUnderexposedFraction, static_cast<double>(dark) / n);
    set(Metric::kOverexposedFraction, static_cast<double>(bright) / n);
    const double red = static_cast<double>(total.sum_red) / n;
    const double green = static_cast<double>(total.sum_green) / n;
    const double blue = static_cast<double>(total.sum_blue) / n;
    set(Metric::kMeanRed, red);
    set(Metric::kMeanGreen, green);
    set(Metric::kMeanBlue, blue);
    const double gray = (red + green + blue) / 3.0;
    double cast = 0.0;
    if (stats.color && gray >= 1.0) {
      for (const double channel : {red, green, blue}) {
        cast = std::max(cast, std::abs(channel / gray - 1.0));
      }
    }
    set(Metric::kColorCast, cast);
  }

  // Noise, then sharpness with the noise's share taken out.
  const double sigma = noise_sigma(stats.high_pass_histogram);
  set(Metric::kNoiseSigma, sigma);
  // An 8-bit image carries at least the rounding's noise, a variance of 1/12,
  // wherever its signal varies, though its flat areas may show none.
  const double noise_var = std::max(std::isfinite(sigma) ? sigma * sigma : 0.0, kRoundingVariance);
  if (total.interior > 0) {
    const double n = static_cast<double>(total.interior);
    const double mean = static_cast<double>(total.sum_laplacian) / n;
    set(Metric::kLaplacianVariance,
        std::max(0.0, static_cast<double>(total.sum_laplacian_sq) / n - mean * mean));
    set(Metric::kTenengrad, static_cast<double>(total.sum_gradient_sq) / 64.0 / n);
  }

  std::vector<double> focus_values;
  std::size_t textured = 0;
  std::size_t blurred = 0;
  std::size_t obstructed = 0;
  const auto needed = static_cast<std::uint32_t>(
      std::ceil(t.obstruction_fraction * static_cast<double>(window) - 1e-9));
  const bool window_full = seen_ + 1 >= window;
  for (std::size_t i = 0; i < tile_count; ++i) {
    const TileStats& tile = stats.tiles[i];
    float* out = report.tiles.data() + i * kTileChannelCount;
    const auto put = [out](TileChannel channel, double value) {
      out[static_cast<std::size_t>(channel)] = static_cast<float>(value);
    };
    std::uint32_t tile_flags = 0;
    bool low_texture = false;
    if (tile.pixels > 0) {
      put(TileChannel::kMeanLuma,
          static_cast<double>(tile.sum_luma) / static_cast<double>(tile.pixels));
    }
    if (tile.interior > 0) {
      const double n = static_cast<double>(tile.interior);
      const double lap_mean = static_cast<double>(tile.sum_laplacian) / n;
      const double lap_var =
          std::max(0.0, static_cast<double>(tile.sum_laplacian_sq) / n - lap_mean * lap_mean);
      const double tenengrad = static_cast<double>(tile.sum_gradient_sq) / 64.0 / n;
      put(TileChannel::kLaplacianVariance, lap_var);
      put(TileChannel::kTenengrad, tenengrad);
      const double signal_lap = std::max(0.0, lap_var - kLaplacianNoiseGain * noise_var);
      const double signal_grad = std::max(0.0, tenengrad - kGradientNoiseGain * noise_var);
      const double mean_luma =
          static_cast<double>(tile.sum_luma) / static_cast<double>(tile.pixels);
      const double texture = std::sqrt(signal_grad) / (mean_luma + 1.0);
      put(TileChannel::kTexture, texture);
      // Judged for focus only when the tile's structure stands out from the noise:
      // its gradient energy at least the noise's, and its texture at least min_texture.
      if (texture >= t.min_texture && signal_grad > 0.0 &&
          signal_grad >= kGradientNoiseGain * noise_var) {
        tile_flags |= kTileTextured;
        ++textured;
        const double focus = signal_lap / signal_grad;
        put(TileChannel::kFocus, focus);
        focus_values.push_back(focus);
        // Blurred only when the focus stays below the threshold even with two
        // standard errors of the noise's share of the Laplacian variance added:
        // in a noisy tile, a sharp edge's Laplacian can drown in that share.
        const double noise_error = kLaplacianNoiseGain * noise_var * std::sqrt(2.0 / n);
        if ((signal_lap + 2.0 * noise_error) / signal_grad < t.blur_focus) {
          tile_flags |= kTileBlurred;
          ++blurred;
        }
      } else {
        tile_flags |= kTileLowTexture;
        low_texture = true;
      }
    }
    // The tile's low-texture history over the window.
    std::uint8_t& slot = history_[i * window + cursor_];
    low_counts_[i] -= slot;
    slot = low_texture ? 1U : 0U;
    low_counts_[i] += slot;
    put(TileChannel::kObstruction,
        static_cast<double>(low_counts_[i]) / static_cast<double>(window));
    if (window_full && low_counts_[i] >= needed) {
      tile_flags |= kTileObstructed;
      ++obstructed;
    }
    put(TileChannel::kFlags, tile_flags);
  }
  cursor_ = (cursor_ + 1) % window;
  seen_ = std::min(seen_ + 1, window);

  const double focus = median(std::move(focus_values));
  set(Metric::kFocus, focus);
  if (tile_count != 0) {
    const double n = static_cast<double>(tile_count);
    set(Metric::kBlurredTileFraction, static_cast<double>(blurred) / n);
    set(Metric::kTexturedTileFraction, static_cast<double>(textured) / n);
    set(Metric::kObstructedTileFraction, static_cast<double>(obstructed) / n);
  }

  // Frozen: the same bytes as the previous frame, or nearly the same luma.
  bool repeat = false;
  if (stats.has_previous && previous_hash_ && pixels != 0) {
    const double difference = static_cast<double>(total.sum_abs_diff) / static_cast<double>(pixels);
    set(Metric::kFrameDifference, difference);
    repeat = *previous_hash_ == total.hash || difference <= t.frozen_difference;
  }
  frozen_run_ = repeat ? frozen_run_ + 1 : 0;
  previous_hash_ = total.hash;
  set(Metric::kFrozenFrames, frozen_run_);

  std::uint32_t flags = 0;
  if (std::isfinite(focus) && focus < t.blur_focus) {
    flags |= kBlurred;
  } else if (blurred != 0) {
    flags |= kLocalBlur;
  }
  if (pixels != 0) {
    const float mean = report.metric(Metric::kMeanLuma);
    if (mean < t.min_mean_luma ||
        report.metric(Metric::kUnderexposedFraction) > t.max_underexposed_fraction) {
      flags |= kUnderexposed;
    }
    if (mean > t.max_mean_luma ||
        report.metric(Metric::kOverexposedFraction) > t.max_overexposed_fraction) {
      flags |= kOverexposed;
    }
    if (report.metric(Metric::kColorCast) > t.max_color_cast) {
      flags |= kColorCast;
    }
  }
  if (std::isfinite(sigma) && sigma > t.max_noise_sigma) {
    flags |= kNoisy;
  }
  if (frozen_run_ >= static_cast<std::uint32_t>(t.frozen_frames)) {
    flags |= kFrozen;
  }
  if (obstructed != 0) {
    flags |= kObstructed;
  }
  report.flags = flags;
  return report;
}

}  // namespace holoscan::image_quality
