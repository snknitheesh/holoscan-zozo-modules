// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What stitching computes, on the CPU: the remap tables of a rig and an output
// geometry, the exposure gains, and the blended image. The GPU's Stitcher
// (stitcher.hpp) computes the same with the same arithmetic; these are its
// reference. No SDK, no CUDA.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <holoscan/stitching/camera_rig.hpp>
#include <holoscan/stitching/projection.hpp>

namespace holoscan::stitching {

// The pixel formats the stitcher reads; it writes kRgb8 or kRgba8.
enum class PixelFormat : std::uint8_t { kMono8, kRgb8, kRgba8, kBgr8, kBgra8 };

// "mono8", "rgb8", "rgba8", "bgr8" or "bgra8".
[[nodiscard]] std::string_view format_name(PixelFormat format) noexcept;
[[nodiscard]] std::optional<PixelFormat> parse_format(std::string_view name) noexcept;
[[nodiscard]] int channels(PixelFormat format) noexcept;

// One camera's part of the remap: for every output pixel of a rectangle (its
// region of interest), where the camera sees it and how much it counts.
struct CameraTable {
  // The size of the camera's images.
  int image_width{};
  int image_height{};
  // The region of interest in output pixels. Its columns run from x0 for
  // `width` columns, modulo the output width when the output wraps around. An
  // empty region (width or height 0) means the camera sees none of the output.
  int x0{};
  int y0{};
  int width{};
  int height{};
  // Per pixel of the region, row by row: the camera pixel (u, v), interleaved,
  // with pixel centres at integers.
  std::vector<float> uv;
  // Per pixel: the distance in camera pixels from where the camera stops
  // seeing (its image border, or the edge of its lens' valid range); 0 where
  // it does not see the pixel. The feathering weight.
  std::vector<float> margin;
};

struct RemapTables {
  int output_width{};
  int output_height{};
  bool wraps{false};
  std::vector<CameraTable> cameras;
};

// The tables of `rig` for `geometry`, on the CPU. Validates both.
// @throws std::invalid_argument if either is unusable.
[[nodiscard]] RemapTables build_remap_tables(const CameraRig& rig, const OutputGeometry& geometry);

// How finely the cameras resolve each output pixel: for the camera that sees
// it with the largest margin, the camera pixels one output pixel spans in the
// direction it spans fewest (the smaller singular value of the Jacobian of
// (u, v) over (x, y)); 0 where no camera sees it. Below 1 the output samples
// the camera more finely than it sees: at 0.25, a feature can only be placed
// to about 4 output pixels' worth of one camera pixel. Row-major.
[[nodiscard]] std::vector<float> resolution_map(const RemapTables& tables);

enum class BlendMode : std::uint8_t {
  // Every camera that sees a pixel contributes, weighted by its margin: the
  // seams fade over the overlap (distance-weighted feathering).
  kFeather,
  // Each pixel comes from the camera that sees it with the largest margin.
  kHardSeam,
};

[[nodiscard]] std::string_view blend_mode_name(BlendMode mode) noexcept;
[[nodiscard]] std::optional<BlendMode> parse_blend_mode(std::string_view name) noexcept;

struct BlendSettings {
  BlendMode mode{BlendMode::kFeather};
  // With kFeather, caps the weight at this margin in camera pixels, as
  // OpenCV's FeatherBlender's sharpness does: a camera reaches full weight
  // this far from its edge, so the ramps at the overlap's ends are this wide
  // and between them the cameras count equally. 0 lets the weight grow over
  // the whole overlap, so the transition spans all of it.
  float feather_width{0.0F};
  // Scale each camera by a gain that matches the overlaps' mean brightness
  // (gain compensation after Brown and Lowe, 2007), computed every frame.
  bool exposure_compensation{false};
  // The noise of the overlap means (sigma_N, in 8-bit levels) and how far the
  // gains may stray from 1 (sigma_g): a smaller sigma_g holds them nearer to 1
  // and leaves more of a difference uncorrected. Brown and Lowe use 10 and
  // 0.1, which on overlaps of mean brightness near 100 leaves about a third of
  // a camera's exposure difference; 0.5 leaves about 2%.
  float exposure_sigma_noise{10.0F};
  float exposure_sigma_gain{0.5F};
  // Each frame's gains are blended with the previous frame's:
  // g = (1 - smoothing) * new + smoothing * previous; 0 uses each frame's own.
  float exposure_smoothing{0.0F};
  // The output: kRgb8 or kRgba8, whose alpha is 255 where a camera sees the
  // pixel and 0 elsewhere.
  PixelFormat output_format{PixelFormat::kRgba8};
  // The colour of pixels no camera sees.
  std::array<std::uint8_t, 3> background{0, 0, 0};
};

// @throws std::invalid_argument naming the first unusable field.
void validate(const BlendSettings& settings);

// An image the CPU can read: `pitch` bytes from one row to the next.
struct HostImage {
  const std::uint8_t* data{};
  int width{};
  int height{};
  std::size_t pitch{};
  PixelFormat format{PixelFormat::kRgb8};
};

// Every 4th pixel in each direction, from (2, 2), is sampled for the overlap
// means of the exposure gains.
inline constexpr int kExposureStride = 4;

// The overlap statistics of the exposure gains: for each ordered pair of
// cameras (i, j), at the sample pixels both see, the sum of camera i's
// luminance and the number of such pixels. Row-major, cameras x cameras.
struct OverlapStats {
  std::vector<std::uint64_t> sum;
  std::vector<std::uint32_t> count;
};
[[nodiscard]] OverlapStats reference_overlap_stats(const RemapTables& tables,
                                                   std::span<const HostImage> images);
// The gains of one frame from its overlap statistics, blended with
// `previous` (all 1 for the first frame) by the settings' smoothing.
[[nodiscard]] std::vector<float> reference_gains(const OverlapStats& stats,
                                                 const BlendSettings& settings,
                                                 std::span<const float> previous);

// Blends `images` (one per camera, in the rig's order, each of its camera's
// size) into `output`, `tables.output_width` x `tables.output_height` pixels
// of the settings' format, `output_pitch` bytes per row. `gains` holds one per
// camera; empty means all 1.
// @throws std::invalid_argument if the images do not match the tables.
void reference_stitch(const RemapTables& tables, std::span<const HostImage> images,
                      const BlendSettings& settings, std::span<const float> gains,
                      std::uint8_t* output, std::size_t output_pitch);

}  // namespace holoscan::stitching
