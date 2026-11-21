// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/isp/synthetic_raw.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace holoscan::isp {
namespace {

constexpr std::array<std::array<int, 4>, 4> kCfa{{
    {kChannelR, kChannelGr, kChannelGb, kChannelB},  // RGGB
    {kChannelB, kChannelGb, kChannelGr, kChannelR},  // BGGR
    {kChannelGr, kChannelR, kChannelB, kChannelGb},  // GRBG
    {kChannelGb, kChannelB, kChannelR, kChannelGr},  // GBRG
}};

// The inverse of a row-major 3x3 matrix, in double.
std::array<float, 9> inverse(const std::array<float, 9>& m) {
  const double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7],
               i = m[8];
  const double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (!(std::abs(det) > 1e-6)) {
    throw std::invalid_argument("synthetic ccm must be invertible");
  }
  const double inv[9] = {(e * i - f * h),  -(b * i - c * h), (b * f - c * e),
                         -(d * i - f * g), (a * i - c * g),  -(a * f - c * d),
                         (d * h - e * g),  -(a * h - b * g), (a * e - b * d)};
  std::array<float, 9> result{};
  for (int k = 0; k < 9; ++k)
    result[static_cast<std::size_t>(k)] = static_cast<float>(inv[k] / det);
  return result;
}

int max_bits(RawFormat format) {
  return format == RawFormat::kUint8   ? 8
         : format == RawFormat::kRaw10 ? 10
         : format == RawFormat::kRaw12 ? 12
                                       : 16;
}

}  // namespace

SyntheticRawScene make_synthetic_scene(const SyntheticRawConfig& config) {
  if (config.width < 16 || config.height < 16 || config.width > 16384 || config.height > 16384) {
    throw std::invalid_argument("synthetic frames must be 16x16 to 16384x16384");
  }
  const bool packed = config.format == RawFormat::kRaw10 || config.format == RawFormat::kRaw12;
  if (config.bit_depth < 2 || config.bit_depth > max_bits(config.format) ||
      (packed && config.bit_depth != max_bits(config.format))) {
    throw std::invalid_argument("synthetic bit depth does not fit its sample format");
  }
  if (!(config.vignetting >= 0.0F && config.vignetting <= 0.9F)) {
    throw std::invalid_argument("synthetic vignetting must be 0..0.9");
  }
  if (!(std::isfinite(config.exposure) && config.exposure > 0.0F && config.exposure <= 64.0F)) {
    throw std::invalid_argument("synthetic exposure must be in (0, 64]");
  }
  SyntheticRawScene s;
  s.width = config.width;
  s.height = config.height;
  s.kind = static_cast<int>(config.kind);
  s.format = static_cast<int>(config.format);
  const auto& cfa = kCfa[static_cast<std::size_t>(config.pattern)];
  for (int k = 0; k < 4; ++k) s.cfa[k] = cfa[static_cast<std::size_t>(k)];
  s.white = static_cast<float>((1U << static_cast<unsigned>(config.bit_depth)) - 1U);
  for (int k = 0; k < 4; ++k) {
    const float black = config.black_levels[static_cast<std::size_t>(k)];
    if (!(black >= 0.0F && black < s.white / 2.0F)) {
      throw std::invalid_argument("synthetic black levels must be below half the white level");
    }
    s.black[k] = black;
  }
  for (int k = 0; k < 3; ++k) {
    const float gain = config.wb_gains[static_cast<std::size_t>(k)];
    if (!(std::isfinite(gain) && gain >= 0.25F && gain <= 8.0F)) {
      throw std::invalid_argument("synthetic wb_gains must be 0.25..8");
    }
    s.wb[k] = gain;
    s.inv_wb[k] = 1.0F / gain;
  }
  const std::array<float, 9> inv = inverse(config.ccm);
  for (int k = 0; k < 9; ++k) {
    s.ccm[k] = config.ccm[static_cast<std::size_t>(k)];
    s.inv_ccm[k] = inv[static_cast<std::size_t>(k)];
  }
  s.vignetting = config.vignetting;
  s.exposure = config.exposure;
  s.cx = static_cast<float>(config.width - 1) / 2.0F;
  s.cy = static_cast<float>(config.height - 1) / 2.0F;
  s.inv_cx2 = 1.0F / (s.cx * s.cx);
  s.inv_cy2 = 1.0F / (s.cy * s.cy);
  // Back and forth by two pixels a frame, over a sixteenth of the width.
  const auto span = static_cast<std::uint32_t>(config.width / 32 > 1 ? config.width / 32 : 1);
  const std::uint32_t phase = config.frame % (2U * span);
  s.shift = 2 * static_cast<int>(phase < span ? phase : 2U * span - phase);
  return s;
}

IspParams synthetic_isp_params(const SyntheticRawConfig& config, int grid) {
  const SyntheticRawScene s = make_synthetic_scene(config);
  IspParams params;
  params.channel_black_levels = config.black_levels;
  params.white_level = s.white;
  params.wb_gains = config.wb_gains;
  params.ccm = config.ccm;
  if (config.vignetting > 0.0F) {
    if (grid < 2 || grid > 256) {
      throw std::invalid_argument("lens shading grid must be 2..256 points");
    }
    LensShading shading;
    shading.width = grid;
    shading.height = grid;
    shading.channels = 1;
    shading.gains.resize(static_cast<std::size_t>(grid) * grid);
    for (int gy = 0; gy < grid; ++gy) {
      for (int gx = 0; gx < grid; ++gx) {
        // The grid point's position in pixels, where the falloff is evaluated.
        const double x = static_cast<double>(gx) * (config.width - 1) / (grid - 1);
        const double y = static_cast<double>(gy) * (config.height - 1) / (grid - 1);
        const double dx = x - s.cx;
        const double dy = y - s.cy;
        const double falloff =
            1.0 - config.vignetting * 0.5 * (dx * dx / (s.cx * s.cx) + dy * dy / (s.cy * s.cy));
        shading.gains[static_cast<std::size_t>(gy) * grid + gx] = static_cast<float>(1.0 / falloff);
      }
    }
    params.lens_shading = std::move(shading);
  }
  return params;
}

RawImageView synthetic_raw_view(const SyntheticRawConfig& config, const void* data,
                                std::size_t pitch) {
  return RawImageView{.data = data,
                      .width = config.width,
                      .height = config.height,
                      .pitch = pitch,
                      .format = config.format,
                      .bit_depth = config.bit_depth,
                      .pattern = config.pattern};
}

void render_synthetic_raw(const SyntheticRawScene& scene, void* data, std::size_t pitch) {
  auto* bytes = static_cast<unsigned char*>(data);
  const int units = synthetic_units(scene);
  for (int y = 0; y < scene.height; ++y) {
    for (int unit = 0; unit < units; ++unit) {
      synthetic_write_unit(scene, unit, y, bytes + static_cast<std::size_t>(y) * pitch);
    }
  }
}

}  // namespace holoscan::isp
