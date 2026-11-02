// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The denoiser's settings, its backends and image views: plain C++, with no
// SDK, CUDA or VPI, so that SDK-free code and tests can use them.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace holoscan::denoise {

// Where the denoiser runs.
enum class Backend : std::uint8_t {
  kCuda,  // This module's CUDA kernels: any spatial filter, motion-adaptive temporal filter.
  kVpi,   // VPI 4's Temporal Noise Reduction, on the VIC where there is one, else on CUDA.
  kAuto,  // One of the above, chosen once when the denoiser is built: see resolve_auto().
};

// The spatial filter, applied to each frame after the temporal filter (cuda backend).
enum class SpatialFilter : std::uint8_t {
  kNone,
  kBilateral,  // 7x7 bilateral on the colour difference.
  kNlm,        // Non-local means: 3x3 patches compared on luma over a 7x7 search window.
};

// The VPI engine that runs Temporal Noise Reduction (vpi backend).
enum class VpiEngine : std::uint8_t {
  kAuto,  // The VIC where VPI can run TNR on it (Jetson), else CUDA.
  kVic,
  kCuda,
};

// VPI's TNR scene presets (VPITNRPreset).
enum class VpiPreset : std::uint8_t {
  kDefault,
  kOutdoorLowLight,
  kOutdoorMediumLight,
  kOutdoorHighLight,
  kIndoorLowLight,
  kIndoorMediumLight,
  kIndoorHighLight,
};

struct DenoiseParams {
  SpatialFilter spatial{SpatialFilter::kBilateral};
  // Blend each frame with the previous output, per pixel by how much it moved.
  bool temporal{true};
  // 0 leaves frames as they are; 1 filters the most. Scales the spatial filter's
  // range and the frames a static pixel averages (cuda), and is VPI TNR's
  // strength (vpi).
  float strength{0.5F};
  // The noise's standard deviation in 8-bit code values; 0 estimates it from
  // each frame on the GPU (Immerkaer's method, on green or luma).
  float noise_sigma{0.0F};
  // A pixel whose 3x3 mean differs from the history by more than this many
  // standard deviations of that difference starts to count as moving; at twice
  // it, the history there is dropped. 1..20.
  float motion_threshold{3.0F};
  // Frames a static pixel averages at strength 1, 2..255.
  int max_history{32};
  // A frame in which more than this share of the pixels move is a scene cut:
  // the whole history is dropped. (0, 1].
  float scene_cut_fraction{0.5F};
};

[[nodiscard]] std::string_view backend_name(Backend backend) noexcept;
// Parses "cuda", "vpi" or "auto"; nullopt for anything else.
[[nodiscard]] std::optional<Backend> parse_backend(std::string_view name) noexcept;
// Backends compiled into this build, in a stable order; never kAuto.
[[nodiscard]] std::vector<Backend> available_backends();

[[nodiscard]] std::string_view spatial_name(SpatialFilter filter) noexcept;
// Parses "none", "bilateral" or "nlm".
[[nodiscard]] std::optional<SpatialFilter> parse_spatial(std::string_view name) noexcept;

[[nodiscard]] std::string_view vpi_engine_name(VpiEngine engine) noexcept;
// Parses "auto", "vic" or "cuda".
[[nodiscard]] std::optional<VpiEngine> parse_vpi_engine(std::string_view name) noexcept;

// @throws std::invalid_argument naming the first unusable field.
void validate_params(const DenoiseParams& params);
// Whether `backend` (not kAuto) can run `params`: VPI TNR is a temporal filter
// with its own spatial stage, so the vpi backend needs `temporal` and no
// spatial filter. Returns why not, or an empty view.
[[nodiscard]] std::string_view unsupported_reason(Backend backend,
                                                  const DenoiseParams& params) noexcept;

// What Backend::kAuto's choice depends on.
struct PlatformFacts {
  bool vpi_compiled{};  // This build has the vpi backend.
  bool vic_tnr{};       // VPI can run TNR on the VIC (Jetson Orin and Thor).
};

// Backend::kAuto's choice: vpi where it is compiled in, the VIC runs TNR and
// the settings suit it (temporal, no spatial filter); cuda otherwise. On a GPU
// without a VIC the cuda backend is the faster and the more configurable.
[[nodiscard]] constexpr Backend resolve_auto(const PlatformFacts& facts,
                                             const DenoiseParams& params) noexcept {
  if (facts.vpi_compiled && facts.vic_tnr && params.temporal &&
      params.spatial == SpatialFilter::kNone) {
    return Backend::kVpi;
  }
  return Backend::kCuda;
}

// The filters' settings for a strength and a noise level, as the CUDA kernels use them.
namespace tuning {

// Bilateral: 7x7, spatial sigma in pixels; range sigma = kBilateralRange * strength * sigma.
inline constexpr int kBilateralRadius = 3;
inline constexpr float kBilateralSpatialSigma = 2.0F;
inline constexpr float kBilateralRange = 2.5F;
// Non-local means: 3x3 patches over a 7x7 window; h = kNlmH * strength * sigma.
inline constexpr int kNlmPatchRadius = 1;
inline constexpr int kNlmSearchRadius = 3;
inline constexpr float kNlmH = 1.0F;
// Below this noise sigma (code values) the denoiser assumes 0.5.
inline constexpr float kMinSigma = 0.5F;

// Frames a static pixel averages: 1 at strength 0, max_history at 1.
[[nodiscard]] constexpr int history_frames(const DenoiseParams& params) noexcept {
  const float frames = 1.0F + params.strength * static_cast<float>(params.max_history - 1);
  const int rounded = static_cast<int>(frames + 0.5F);
  return rounded < 1 ? 1 : rounded > params.max_history ? params.max_history : rounded;
}

// The share of the noise variance left in a pixel that has averaged `frames`
// frames, at most `limit` (history_frames()); with a limit, the recursive filter
// settles at weight 1/limit, whose variance is (1/limit) / (2 - 1/limit).
[[nodiscard]] constexpr float residual_variance(int frames, int limit) noexcept {
  const float running = 1.0F / static_cast<float>(frames < 1 ? 1 : frames);
  const float weight = 1.0F / static_cast<float>(limit < 1 ? 1 : limit);
  const float settled = weight / (2.0F - weight);
  return running > settled ? running : settled;
}

}  // namespace tuning

// Pixel layouts the denoiser takes. Each is 8 bits per sample.
enum class PixelFormat : std::uint8_t {
  kMono8,
  kRgb8,
  kRgba8,  // Alpha passes through unchanged.
  kNv12,   // Full-range Y plane and interleaved UV plane at half resolution; even sizes.
};

[[nodiscard]] std::string_view format_name(PixelFormat format) noexcept;
// Bytes of one pixel of the first (only, for all but NV12) plane.
[[nodiscard]] constexpr int pixel_bytes(PixelFormat format) noexcept {
  return format == PixelFormat::kRgb8 ? 3 : format == PixelFormat::kRgba8 ? 4 : 1;
}

// One plane of an image: its first byte and the bytes between rows.
struct Plane {
  std::uint8_t* data{};
  std::size_t pitch{};
};

// A borrowed 8-bit image in CUDA device memory. NV12 uses both planes, every
// other format the first. An input view's pixels are only read.
struct ImageView {
  PixelFormat format{PixelFormat::kRgb8};
  int width{};
  int height{};
  Plane planes[2]{};
};

}  // namespace holoscan::denoise
