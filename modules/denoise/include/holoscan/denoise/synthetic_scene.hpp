// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Synthetic scenes with ground truth, for the tests, the benchmark and the demo:
//
// - A low-light scene (dark gradients, coloured patches, fine stripes and a
//   checkerboard, and a bright disc that moves), rendered clean on the host and,
//   with Poisson shot noise and Gaussian read noise of known levels, on the GPU.
//   From `cut_frame` on it shows a different layout: a scene cut.
// - An HDR scene (a textured shadow, mid-tones and a bright textured window,
//   about 13 stops apart), in linear radiance, and its 8-bit exposure brackets
//   through the sRGB curve.
//
// Everything is a function of the frame number, so a checker can recompute any
// frame's truth.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include <holoscan/denoise/denoise_params.hpp>

namespace holoscan::denoise {

struct LowLightScene {
  int width{640};
  int height{480};
  // MONO8 (the luma (R + 2G + B) / 4 of the colour scene), RGB8, or RGBA8 (alpha 255).
  PixelFormat format{PixelFormat::kRgb8};
  float speed{6.0F};      // The disc's pixels per frame; 0 keeps it still.
  int object_radius{40};  // 0: no disc.
  int cut_frame{-1};      // From this frame on, the second layout; negative: never.
};

struct NoiseModel {
  // Gaussian read noise, code values.
  float read_sigma{6.0F};
  // Shot noise: a value v is Poisson(v / shot_gain) * shot_gain, of variance
  // shot_gain * v. 0: none.
  float shot_gain{1.0F};
  std::uint32_t seed{1};
};

// The noise variance at clean value `v`: shot_gain * v + read_sigma^2 (before clipping).
[[nodiscard]] double noise_variance(const NoiseModel& noise, double value) noexcept;

// The disc's centre in `frame`, or (-1, -1) without a disc.
struct Point {
  int x{-1};
  int y{-1};
};
[[nodiscard]] Point object_center(const LowLightScene& scene, int frame) noexcept;
// Whether the disc covers (x, y) in `frame`.
[[nodiscard]] bool object_covers(const LowLightScene& scene, int frame, int x, int y) noexcept;

// The clean frame, packed (width * bytes per pixel per row), on the host.
[[nodiscard]] std::vector<std::uint8_t> render_clean(const LowLightScene& scene, int frame);
// The noisy frame into `output` (device memory, the scene's format and size) on `stream`.
[[nodiscard]] cudaError_t render_noisy(const LowLightScene& scene, const NoiseModel& noise,
                                       int frame, const ImageView& output, cudaStream_t stream);
// The noisy frame, packed, on the host (rendered on the GPU; synchronizes).
// @throws std::runtime_error on a CUDA failure.
[[nodiscard]] std::vector<std::uint8_t> render_noisy(const LowLightScene& scene,
                                                     const NoiseModel& noise, int frame);

// PSNR in dB of `test` against `truth`, over `count` bytes; 99 when identical.
[[nodiscard]] double psnr(const std::uint8_t* test, const std::uint8_t* truth,
                          std::size_t count) noexcept;

struct HdrScene {
  int width{640};
  int height{480};
  int channels{3};  // 1 (gray) or 3.
  // Pixels the textures move per frame; 0 keeps the scene still.
  int drift{0};
};

// Where an HDR scene pixel lies.
enum class HdrRegion : std::uint8_t {
  kShadow,    // Left third: radiance about 0.002 to 0.006, striped.
  kMidtones,  // Middle third: a 0.02 to 0.5 ramp with patches.
  kWindow,    // Right third: radiance about 1.5 to 6, striped.
};
[[nodiscard]] HdrRegion hdr_region(const HdrScene& scene, int x) noexcept;

// Linear radiance, packed, `channels` floats per pixel.
[[nodiscard]] std::vector<float> render_radiance(const HdrScene& scene, int frame);
// The 8-bit bracket of exposure time `exposure` (radiance x exposure, clipped
// to 1, through the sRGB curve), packed: MONO8 or RGB8 as the scene has channels.
[[nodiscard]] std::vector<std::uint8_t> render_bracket(const HdrScene& scene, int frame,
                                                       float exposure);
// The same on the GPU, into `output` (MONO8, RGB8 or RGBA8 of the scene's size).
[[nodiscard]] cudaError_t render_bracket(const HdrScene& scene, int frame, float exposure,
                                         const ImageView& output, cudaStream_t stream);

// The brackets the tests and the demo use: an eighth, one, and 32 times the exposure.
inline constexpr float kBracketExposures[3] = {0.125F, 1.0F, 32.0F};

}  // namespace holoscan::denoise
