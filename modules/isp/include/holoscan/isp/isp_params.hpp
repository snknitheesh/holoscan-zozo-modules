// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The image signal processor's settings, the raw and RGB image views it works
// on, and the layout of its statistics. Plain C++: no Holoscan SDK, no CUDA,
// so that SDK-free code and tests can use it.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace holoscan::isp {

// The 2x2 colour filter order, reading left to right, top to bottom from the
// image's first pixel, as the Holoscan image schema's BAYER_* encodings name it.
enum class BayerPattern : std::uint8_t { kRGGB, kBGGR, kGRBG, kGBRG };

// How the raw samples are stored in a row.
enum class RawFormat : std::uint8_t {
  kUint8,   // One byte per sample.
  kUint16,  // One little-endian uint16 per sample, the value in its low bits.
  kRaw10,   // MIPI CSI-2 RAW10: four samples in five bytes.
  kRaw12,   // MIPI CSI-2 RAW12: two samples in three bytes.
};

enum class DemosaicMethod : std::uint8_t {
  kBilinear,  // Fast: each missing colour from its nearest neighbours of that colour.
  kMalvar,    // Malvar-He-Cutler: bilinear corrected by the gradient of the known colour.
};

enum class WhiteBalanceMode : std::uint8_t {
  kManual,      // wb_gains as configured.
  kGrayWorld,   // Gains that make the frame's mean gray.
  kWhitePatch,  // Gains that make the frame's brightest pixels white.
};

enum class ToneCurve : std::uint8_t {
  kSrgb,    // The sRGB transfer function.
  kGamma,   // out = in^(1 / gamma).
  kLinear,  // out = in.
  kLut,     // The configured table.
};

enum class OutputFormat : std::uint8_t {
  kRgb8,    // uint8 [H, W, 3].
  kRgba8,   // uint8 [H, W, 4], alpha 255.
  kRgb16,   // uint16 [H, W, 3].
  kRgb32f,  // float32 [H, W, 3], 0..1.
};

// Where the pipeline runs.
enum class Backend : std::uint8_t {
  kCuda,  // CUDA kernels on the GPU: images in device memory.
  kCpu,   // The reference implementation on the host: images in host memory.
  kAuto,  // cuda where a CUDA device is usable, else cpu; chosen once.
};

// Channels of the mosaic, in the order of every per-channel setting: red, the
// green in a red row, the green in a blue row, blue.
inline constexpr int kChannelR = 0;
inline constexpr int kChannelGr = 1;
inline constexpr int kChannelGb = 2;
inline constexpr int kChannelB = 3;

// A lens shading (vignetting) correction: gains on a grid whose corner points
// lie on the image's corner pixels, interpolated bilinearly in between. One
// gain per point for every channel, or four (R, Gr, Gb, B).
struct LensShading {
  int width{};      // Grid points across, 2..256.
  int height{};     // Grid points down, 2..256.
  int channels{1};  // 1 or 4.
  // [height][width][channels], row-major; each in (0, 16].
  std::vector<float> gains;
};

// The auto-exposure suggestion: never applied, only reported in the statistics.
struct AutoExposureParams {
  // Mean linear luminance (0..1, after white balance and colour correction)
  // that a well exposed frame has.
  float target{0.18F};
  // Share of the error, in stops, that one suggestion corrects: 0..1.
  float damping{0.5F};
  // Bounds of the suggested exposure time and analog gain.
  std::int64_t min_exposure_ns{10'000};
  std::int64_t max_exposure_ns{33'000'000};
  float min_gain{1.0F};
  float max_gain{16.0F};
};

struct IspParams {
  // Raw counts subtracted from every sample; per channel (R, Gr, Gb, B) when
  // channel_black_levels is given.
  float black_level{0.0F};
  std::optional<std::array<float, 4>> channel_black_levels;
  // Raw counts of a saturated sample; 0 takes 2^bit_depth - 1.
  float white_level{0.0F};
  std::optional<LensShading> lens_shading;
  WhiteBalanceMode white_balance{WhiteBalanceMode::kManual};
  // R, G, B gains for kManual; the auto modes start from them.
  std::array<float, 3> wb_gains{1.0F, 1.0F, 1.0F};
  // Auto modes: the share of the way from the last gains to the frame's
  // estimate, per frame (1: no smoothing). The first frame takes its estimate.
  float awb_smoothing{0.25F};
  // kWhitePatch: the percentile of each channel taken as white.
  float white_patch_percentile{0.99F};
  DemosaicMethod demosaic{DemosaicMethod::kMalvar};
  // Camera RGB to output linear RGB, row-major.
  std::array<float, 9> ccm{1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F};
  ToneCurve tone_curve{ToneCurve::kSrgb};
  float gamma{2.2F};
  // kLut: 2..65536 outputs (0..1) for inputs evenly spaced over 0..1.
  std::vector<float> lut;
  // Unsharp mask amount, 0 (off) to 4.
  float sharpen{0.0F};
  AutoExposureParams auto_exposure;
};

// @throws std::invalid_argument naming the first unusable field.
void validate_params(const IspParams& params);

// The tone curve as the pipeline evaluates it: a table interpolated linearly
// over 0..1, kToneLutSize entries for the built-in curves, `lut` for kLut.
inline constexpr int kToneLutSize = 4097;
[[nodiscard]] std::vector<float> tone_lut(const IspParams& params);

// A borrowed raw mosaic. `pitch` is the row stride in bytes.
struct RawImageView {
  const void* data{};
  int width{};   // At least 2.
  int height{};  // At least 2.
  std::size_t pitch{};
  RawFormat format{RawFormat::kUint16};
  int bit_depth{};  // Significant bits: 1..8 for kUint8, 10 for kRaw10, 12 for kRaw12, else 1..16.
  BayerPattern pattern{BayerPattern::kRGGB};
};

// A borrowed output image of the raw image's size. `pitch` is the row stride in bytes.
struct RgbImageView {
  void* data{};
  std::size_t pitch{};
  OutputFormat format{OutputFormat::kRgb8};
};

// Where the statistics go; either may be null. `histogram` holds
// kHistogramChannels x kHistogramBins counts, `stats` kStatsSize values.
struct StatsOutput {
  std::uint32_t* histogram{};
  float* stats{};
};

// The capture settings of a frame, for the auto-exposure suggestion; 0 is unknown.
struct FrameExposure {
  std::int64_t exposure_ns{};
  float gain{};
};

// Histograms: kHistogramChannels rows of kHistogramBins uint32 counts, of the
// output image's 8-bit code values (16-bit and float outputs rounded to 8 bits):
// luma (Rec. 709 weights of the coded R, G, B), red, green, blue. Each row sums
// to the pixel count.
inline constexpr int kHistogramBins = 256;
inline constexpr int kHistogramChannels = 4;
inline constexpr int kHistogramLuma = 0;
inline constexpr int kHistogramRed = 1;
inline constexpr int kHistogramGreen = 2;
inline constexpr int kHistogramBlue = 3;

// The float32 statistics vector, kStatsSize entries.
namespace stats {
inline constexpr int kMeanR = 0;  // Mean linear red, 0..1, before the tone curve.
inline constexpr int kMeanG = 1;
inline constexpr int kMeanB = 2;
inline constexpr int kMeanLuma = 3;  // Rec. 709 luminance of the three means.
// log2(target / mean luminance): stops to add to reach the target, -8..8.
inline constexpr int kAeErrorEv = 4;
// 2^(damping * error): the factor the next exposure time x gain should change by.
inline constexpr int kExposureScale = 5;
// The next exposure time and gain within the bounds, preferring exposure
// time over gain; 0 when the frame carries no exposure time.
inline constexpr int kSuggestedExposureNs = 6;
inline constexpr int kSuggestedGain = 7;
inline constexpr int kWbGainR = 8;  // The white balance gains the frame was processed with.
inline constexpr int kWbGainG = 9;
inline constexpr int kWbGainB = 10;
// Share of raw samples at or above the white level.
inline constexpr int kSaturatedFraction = 11;
inline constexpr int kStatsSize = 12;
}  // namespace stats

[[nodiscard]] std::string_view backend_name(Backend backend) noexcept;
// Parses "cuda", "cpu" or "auto"; nullopt for anything else.
[[nodiscard]] std::optional<Backend> parse_backend(std::string_view name) noexcept;
[[nodiscard]] std::string_view demosaic_name(DemosaicMethod method) noexcept;
// Parses "bilinear" or "malvar".
[[nodiscard]] std::optional<DemosaicMethod> parse_demosaic(std::string_view name) noexcept;
[[nodiscard]] std::string_view pattern_name(BayerPattern pattern) noexcept;
// Parses "RGGB", "BGGR", "GRBG" or "GBRG".
[[nodiscard]] std::optional<BayerPattern> parse_pattern(std::string_view name) noexcept;
[[nodiscard]] std::string_view white_balance_name(WhiteBalanceMode mode) noexcept;
// Parses "manual", "gray-world" or "white-patch".
[[nodiscard]] std::optional<WhiteBalanceMode> parse_white_balance(std::string_view name) noexcept;

// What Backend::kAuto's choice depends on.
struct PlatformFacts {
  bool cuda_device{};  // A CUDA device is usable.
};

// Backend::kAuto's choice: cuda where a CUDA device is usable, else cpu.
[[nodiscard]] constexpr Backend resolve_auto(const PlatformFacts& facts) noexcept {
  return facts.cuda_device ? Backend::kCuda : Backend::kCpu;
}

// Channels per pixel and bytes per channel of an output format.
[[nodiscard]] constexpr int output_channels(OutputFormat format) noexcept {
  return format == OutputFormat::kRgba8 ? 4 : 3;
}
[[nodiscard]] constexpr int output_channel_bytes(OutputFormat format) noexcept {
  return format == OutputFormat::kRgb16 ? 2 : format == OutputFormat::kRgb32f ? 4 : 1;
}

// Bytes a row of `width` samples needs in `format`.
[[nodiscard]] std::size_t raw_row_bytes(RawFormat format, int width) noexcept;

// @throws std::invalid_argument if `raw` cannot be processed, or `params`'
//         black and white levels do not fit its bit depth.
void validate_raw(const RawImageView& raw, const IspParams& params);

}  // namespace holoscan::isp
