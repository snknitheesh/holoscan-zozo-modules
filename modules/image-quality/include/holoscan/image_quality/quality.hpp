// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What the image-quality checks measure and decide, with no SDK, CUDA or GPU:
// the metric and flag vocabulary, the thresholds, the integer statistics a
// backend gathers from one frame (FrameStats), and the evaluator that turns a
// sequence of them into metrics and flags (QualityEvaluator). The GPU analyzer
// (analyzer.hpp) and the CPU reference (reference.hpp) gather the same
// statistics, bit for bit, so both give the same report.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace holoscan::image_quality {

// Pixel layouts the checks read: 8 bits per channel, channels interleaved.
enum class PixelFormat : std::uint8_t { kMono8, kRgb8, kBgr8, kRgba8, kBgra8 };

[[nodiscard]] constexpr int channels(PixelFormat format) noexcept {
  switch (format) {
    case PixelFormat::kMono8:
      return 1;
    case PixelFormat::kRgb8:
    case PixelFormat::kBgr8:
      return 3;
    case PixelFormat::kRgba8:
    case PixelFormat::kBgra8:
      return 4;
  }
  return 0;
}

// One image: `pitch` bytes from one row to the next, at least width * channels.
// Where `data` lives (host or device) is up to the function that takes it.
struct ImageView {
  const std::uint8_t* data{};
  int width{};
  int height{};
  std::size_t pitch{};
  PixelFormat format{PixelFormat::kRgb8};
};

// Luma of an RGB pixel, BT.601 weights in integers: the value every metric but
// color cast is computed on. A gray pixel's luma is its value.
[[nodiscard]] constexpr int luma(int r, int g, int b) noexcept {
  return (77 * r + 150 * g + 29 * b + 128) >> 8;
}

// The per-frame metrics, in the order of the metrics tensor (float32 [kMetricCount]).
// Luma is in 8-bit levels; a metric that cannot be computed is NaN.
enum class Metric : std::uint8_t {
  kMeanLuma,                // Mean luma.
  kLumaP1,                  // Luma percentiles (nearest rank): 1st, 5th, 50th, 95th, 99th.
  kLumaP5,                  //
  kLumaP50,                 //
  kLumaP95,                 //
  kLumaP99,                 //
  kUnderexposedFraction,    // Fraction of pixels with luma <= dark_level.
  kOverexposedFraction,     // Fraction of pixels with luma >= bright_level.
  kDynamicRange,            // P99 - P1.
  kLaplacianVariance,       // Variance of the 4-neighbor Laplacian over the frame.
  kTenengrad,               // Mean squared Sobel gradient (Sobel / 8: a slope-1 ramp gives 1).
  kFocus,                   // Median over textured tiles of their focus (see TileChannel::kFocus).
  kBlurredTileFraction,     // Fraction of all tiles that are textured and blurred.
  kTexturedTileFraction,    // Fraction of all tiles that are textured.
  kNoiseSigma,              // Estimated noise standard deviation, levels: MAD of a high-pass.
  kFrameDifference,         // Mean absolute luma difference to the previous frame; NaN for none.
  kFrozenFrames,            // Consecutive frames frozen so far: 0 for a frame that changed.
  kObstructedTileFraction,  // Fraction of tiles flagged as obstructed.
  kMeanRed,                 // Mean of each color channel; all equal luma for MONO8.
  kMeanGreen,               //
  kMeanBlue,                //
  kColorCast,  // Gray-world deviation: max over channels of |mean / gray - 1|; 0 for MONO8.
  kCount
};
inline constexpr std::size_t kMetricCount = static_cast<std::size_t>(Metric::kCount);

// Snake-case name of a metric, as the demo and logs print it.
[[nodiscard]] std::string_view metric_name(Metric metric) noexcept;

// The flags bitfield (uint32): a frame passes when it is zero.
enum Flag : std::uint32_t {
  kBlurred = 1U << 0U,       // Focus below blur_focus over the frame.
  kLocalBlur = 1U << 1U,     // Not kBlurred, but at least one textured tile is.
  kUnderexposed = 1U << 2U,  // Too dark: mean luma or the dark fraction.
  kOverexposed = 1U << 3U,   // Too bright: mean luma or the bright fraction.
  kNoisy = 1U << 4U,         // Noise above max_noise_sigma.
  kFrozen = 1U << 5U,        // Identical or near identical to the previous frame.
  kObstructed = 1U << 6U,    // A tile has stayed low-texture over obstruction_frames frames.
  kColorCast = 1U << 7U,     // Gray-world deviation above max_color_cast (color input only).
};
inline constexpr std::uint32_t kAllFlags = (1U << 8U) - 1U;

// "blurred|noisy" style list of the set flags; "ok" for none.
[[nodiscard]] std::string flag_names(std::uint32_t flags);

// The per-tile values, in the order of the last axis of the tile tensor
// (float32 [tiles_y, tiles_x, kTileChannelCount]).
enum class TileChannel : std::uint8_t {
  kMeanLuma,
  kLaplacianVariance,
  kTenengrad,
  // RMS Sobel gradient with the estimated noise taken out, over (mean luma + 1):
  // how much structure the tile shows, independent of exposure.
  kTexture,
  // Laplacian variance over Tenengrad, both with the estimated noise taken out:
  // about 1 / (2 sigma^2) for an edge blurred by a Gaussian of sigma pixels, and
  // independent of the edge's contrast. NaN for a tile that is not textured.
  kFocus,
  // Fraction of the last obstruction_frames frames in which the tile was low-texture.
  kObstruction,
  kFlags,  // TileFlag bits, as a float.
  kCount
};
inline constexpr std::size_t kTileChannelCount = static_cast<std::size_t>(TileChannel::kCount);

enum TileFlag : std::uint32_t {
  kTileTextured = 1U << 0U,    // texture >= min_texture: judged for focus.
  kTileBlurred = 1U << 1U,     // Textured, focus below blur_focus.
  kTileLowTexture = 1U << 2U,  // Not textured this frame.
  kTileObstructed = 1U << 3U,  // Low-texture over the obstruction window.
};

struct QualityThresholds {
  // Focus (see TileChannel::kFocus) below which a tile, or the frame's median, is blurred.
  double blur_focus{0.2};
  // Tile texture below which a tile is low-texture: not judged for focus, and counted
  // toward obstruction.
  double min_texture{0.01};
  // Luma at or below which a pixel counts as underexposed, at or above which overexposed.
  int dark_level{16};
  int bright_level{239};
  // Exposure flags: mean luma outside [min_mean_luma, max_mean_luma], or more than
  // the given fraction of under- or overexposed pixels.
  double min_mean_luma{50.0};
  double max_mean_luma{205.0};
  double max_underexposed_fraction{0.25};
  double max_overexposed_fraction{0.25};
  // Noise standard deviation, in luma levels, above which a frame is noisy.
  double max_noise_sigma{5.0};
  // Mean absolute luma difference at or below which a frame counts as a repeat of
  // the previous one (an identical frame always does), and how many consecutive
  // repeats flag it.
  double frozen_difference{0.05};
  int frozen_frames{1};
  // A tile is obstructed when it was low-texture in at least obstruction_fraction
  // of the last obstruction_frames frames, counted only once that many frames
  // have been seen.
  int obstruction_frames{30};
  double obstruction_fraction{1.0};
  // Gray-world deviation above which a color frame has a color cast.
  double max_color_cast{0.15};
};

// How a backend gathers a frame's statistics.
struct StatsOptions {
  // The tile grid; an image smaller than it gets one tile per pixel on that axis.
  int tiles_x{16};
  int tiles_y{9};
  // Fraction of the interior pixels, those with the smallest Sobel gradient, that
  // the noise estimate reads as flat.
  double flat_fraction{0.5};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_stats_options(const StatsOptions& options);

// @throws std::invalid_argument naming the first unusable field.
void validate_thresholds(const QualityThresholds& thresholds);

// Integer sums over one tile, or one part of a tile: exact, so any order of
// summation gives the same result. Stencil sums cover the pixels whose 3x3
// neighborhood is inside the image ("interior" pixels).
struct TileStats {
  std::uint64_t hash{};  // Sum of a 64-bit mix of each pixel's position and bytes.
  std::int64_t pixels{};
  std::int64_t sum_luma{};
  std::int64_t sum_red{};
  std::int64_t sum_green{};
  std::int64_t sum_blue{};
  std::int64_t sum_abs_diff{};  // Against the previous frame's luma, when there is one.
  std::int64_t interior{};
  std::int64_t sum_laplacian{};
  std::int64_t sum_laplacian_sq{};
  std::int64_t sum_gradient_sq{};  // Unscaled Sobel gx^2 + gy^2.

  TileStats& operator+=(const TileStats& other) noexcept;
  friend bool operator==(const TileStats&, const TileStats&) = default;
};

inline constexpr int kLumaBins = 256;
// |gx| + |gy| of the unscaled Sobel is at most 2040.
inline constexpr int kGradientBins = 2048;
// |H| of the 3x3 high-pass [1 -2 1; -2 4 -2; 1 -2 1] is at most 4080.
inline constexpr int kHighPassBins = 4096;

// Everything a backend gathers from one frame.
struct FrameStats {
  int width{};
  int height{};
  int tiles_x{};
  int tiles_y{};
  bool color{};
  // Whether sum_abs_diff compares with a previous frame of the same size.
  bool has_previous{};
  std::vector<TileStats> tiles;  // Row-major, tiles_y x tiles_x.
  std::array<std::uint32_t, kLumaBins> luma_histogram{};
  // Interior pixels by |gx| + |gy|; the flat pixels are those at or below
  // flat_threshold, the smallest value whose cumulative count reaches
  // flat_fraction of the interior pixels.
  std::vector<std::uint32_t> gradient_histogram;
  std::uint32_t flat_threshold{};
  // Flat pixels by |H| of the high-pass.
  std::vector<std::uint32_t> high_pass_histogram;

  friend bool operator==(const FrameStats&, const FrameStats&) = default;
};

// The tile grid of a width x height image: the requested one, with no more tiles
// than pixels on either axis.
struct TileGrid {
  int tiles_x{};
  int tiles_y{};
};
[[nodiscard]] TileGrid tile_grid(int width, int height, int tiles_x, int tiles_y) noexcept;

// Pixel bounds [begin, end) of tile `index` of `count` along an axis of `size` pixels.
[[nodiscard]] constexpr int tile_begin(int index, int count, int size) noexcept {
  return static_cast<int>(static_cast<std::int64_t>(index) * size / count);
}

// The smallest gradient value whose cumulative count reaches `fraction` of all.
[[nodiscard]] std::uint32_t flat_threshold(const std::uint32_t* histogram, int bins,
                                           double fraction) noexcept;

// The 64-bit mix the frame hash sums: SplitMix64's finalizer of position and bytes.
[[nodiscard]] constexpr std::uint64_t pixel_hash(std::uint64_t index,
                                                 std::uint32_t bytes) noexcept {
  std::uint64_t z = (index << 32U) ^ bytes;
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31U);
}

struct QualityReport {
  std::array<float, kMetricCount> metrics{};
  std::uint32_t flags{};
  int tiles_x{};
  int tiles_y{};
  // [tiles_y][tiles_x][kTileChannelCount], row-major.
  std::vector<float> tiles;
  std::uint64_t hash{};

  [[nodiscard]] float metric(Metric which) const noexcept {
    return metrics[static_cast<std::size_t>(which)];
  }
  [[nodiscard]] float tile(int tx, int ty, TileChannel channel) const noexcept {
    return tiles[(static_cast<std::size_t>(ty) * tiles_x + tx) * kTileChannelCount +
                 static_cast<std::size_t>(channel)];
  }
  [[nodiscard]] std::uint32_t tile_flags(int tx, int ty) const noexcept {
    return static_cast<std::uint32_t>(tile(tx, ty, TileChannel::kFlags));
  }
};

// Turns each frame's statistics into its report, remembering what the temporal
// checks need: the previous frame's hash, the run of frozen frames, and each
// tile's low-texture history. A change of image size or tile grid starts over.
class QualityEvaluator {
 public:
  // @throws std::invalid_argument if `thresholds` is not valid.
  explicit QualityEvaluator(QualityThresholds thresholds);

  [[nodiscard]] QualityReport evaluate(const FrameStats& stats);
  // Forgets every previous frame.
  void reset() noexcept;
  [[nodiscard]] const QualityThresholds& thresholds() const noexcept { return thresholds_; }

 private:
  QualityThresholds thresholds_;
  int width_{};
  int height_{};
  int tiles_x_{};
  int tiles_y_{};
  std::optional<std::uint64_t> previous_hash_;
  std::uint32_t frozen_run_{};
  // Per tile, a ring of the last obstruction_frames low-texture decisions.
  std::vector<std::uint8_t> history_;
  std::vector<std::uint32_t> low_counts_;
  std::uint32_t seen_{};
  std::uint32_t cursor_{};
};

}  // namespace holoscan::image_quality
