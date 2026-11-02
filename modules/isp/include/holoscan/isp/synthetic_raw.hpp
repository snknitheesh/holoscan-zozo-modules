// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A synthetic raw camera with known ground truth, for tests and demos: a
// colour checker, a gray ramp, a hue ramp, bars, a slanted edge and a disc
// (or, for white balance, a scene of grays only), moving from frame to frame,
// softened by a lens blur of about two pixels and seen through a colour
// matrix, white balance gains and lens vignetting that
// the ISP is told to undo, then mosaicked, offset by black levels, quantized
// and packed as a sensor would.
//
// synthetic_truth() is what an ideal ISP outputs before its tone curve. The
// same functions run on the host and, through synthetic_raw_cuda.hpp, on the
// GPU, bit for bit: the module compiles them without fused multiply-adds, and
// host code evaluating them must not contract either (-ffp-contract=off).
// Plain C++: no SDK, no CUDA.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <holoscan/isp/isp_params.hpp>

#if defined(__CUDACC__)
#define HOLOSCAN_ISP_SYNTHETIC_HD __host__ __device__
#else
#define HOLOSCAN_ISP_SYNTHETIC_HD
#endif

namespace holoscan::isp {

enum class SceneKind : std::uint8_t {
  kChart,  // Colour checker, ramps, bars, edge and disc.
  kGray,   // Grays only: a ramp and patches, gray on average in every channel.
};

struct SyntheticRawConfig {
  int width{640};   // 16..16384
  int height{480};  // 16..16384
  BayerPattern pattern{BayerPattern::kRGGB};
  RawFormat format{RawFormat::kUint16};
  int bit_depth{12};
  // Raw counts of black, per channel (R, Gr, Gb, B).
  std::array<float, 4> black_levels{256.0F, 256.0F, 256.0F, 256.0F};
  // The sensor's response to each colour relative to green: the white balance
  // gains that undo it are these.
  std::array<float, 3> wb_gains{1.9F, 1.0F, 1.6F};
  // The colour correction the ISP applies; the camera sees through its inverse.
  std::array<float, 9> ccm{1.6F, -0.4F, -0.2F, -0.3F, 1.5F, -0.2F, -0.1F, -0.5F, 1.6F};
  // Lens falloff: the corners get (1 - vignetting) of the centre's light, 0..0.9.
  float vignetting{0.0F};
  SceneKind kind{SceneKind::kChart};
  // Multiplies the scene's light, before the sensor clips.
  float exposure{1.0F};
  std::uint32_t frame{0};  // Moves the patches.
};

// What the per-pixel functions read, derived from a SyntheticRawConfig.
struct SyntheticRawScene {
  int width{};
  int height{};
  int kind{};
  int format{};  // RawFormat.
  int cfa[4]{};  // The channel (kChannelR...) at quad position (y & 1) * 2 + (x & 1).
  float black[4]{};
  float white{};  // Raw counts of a saturated sample: 2^bit_depth - 1.
  float wb[3]{};
  float inv_wb[3]{};
  float ccm[9]{};
  float inv_ccm[9]{};
  float vignetting{};
  float exposure{};
  float cx{};  // Image centre.
  float cy{};
  float inv_cx2{};
  float inv_cy2{};
  int shift{};  // The patches' horizontal offset in pixels.
};

// @throws std::invalid_argument naming the first unusable field.
[[nodiscard]] SyntheticRawScene make_synthetic_scene(const SyntheticRawConfig& config);

// The ISP settings that undo the synthetic camera: its black levels, white
// level, white balance gains, colour matrix and, with vignetting, a lens
// shading grid of grid x grid points; the rest at their defaults.
[[nodiscard]] IspParams synthetic_isp_params(const SyntheticRawConfig& config, int grid = 17);

// The raw view of a frame rendered into `data` with row stride `pitch`.
[[nodiscard]] RawImageView synthetic_raw_view(const SyntheticRawConfig& config, const void* data,
                                              std::size_t pitch);

// Renders the raw frame into host memory: rows of raw_row_bytes(format, width)
// bytes at stride `pitch`.
void render_synthetic_raw(const SyntheticRawScene& scene, void* data, std::size_t pitch);

namespace synthetic_detail {

// The 24 colour checker patches, linear sRGB.
HOLOSCAN_ISP_SYNTHETIC_HD inline void checker_color(int index, float rgb[3]) {
  constexpr float kPatches[24][3] = {
      {0.1714F, 0.0844F, 0.0578F}, {0.5395F, 0.3050F, 0.2232F}, {0.1221F, 0.1946F, 0.3372F},
      {0.0953F, 0.1500F, 0.0561F}, {0.2346F, 0.2159F, 0.4397F}, {0.1356F, 0.5089F, 0.4020F},
      {0.6724F, 0.2086F, 0.0252F}, {0.0802F, 0.1046F, 0.3813F}, {0.5333F, 0.1022F, 0.1248F},
      {0.1119F, 0.0452F, 0.1500F}, {0.3372F, 0.5029F, 0.0513F}, {0.7454F, 0.3663F, 0.0273F},
      {0.0395F, 0.0467F, 0.3050F}, {0.0612F, 0.2961F, 0.0666F}, {0.4287F, 0.0369F, 0.0452F},
      {0.7991F, 0.5711F, 0.0137F}, {0.4969F, 0.0931F, 0.3005F}, {0.0024F, 0.2346F, 0.3564F},
      {0.8963F, 0.8963F, 0.8879F}, {0.5776F, 0.5776F, 0.5776F}, {0.3515F, 0.3515F, 0.3515F},
      {0.1946F, 0.1946F, 0.1912F}, {0.0908F, 0.0908F, 0.0908F}, {0.0343F, 0.0343F, 0.0343F},
  };
  rgb[0] = kPatches[index][0];
  rgb[1] = kPatches[index][1];
  rgb[2] = kPatches[index][2];
}

HOLOSCAN_ISP_SYNTHETIC_HD inline void set_rgb(float rgb[3], float r, float g, float b) {
  rgb[0] = r;
  rgb[1] = g;
  rgb[2] = b;
}

}  // namespace synthetic_detail

// The colour checker's patch grid in a chart frame: 6 x 4 patches of
// patch_width x patch_height pixels at a pitch of cell_width x cell_height,
// the first at (left + scene.shift, top).
struct ChartLayout {
  int left{};
  int top{};
  int cell_width{};
  int cell_height{};
  int patch_width{};
  int patch_height{};
};

HOLOSCAN_ISP_SYNTHETIC_HD inline ChartLayout chart_layout(const SyntheticRawScene& s) {
  ChartLayout layout;
  const int bottom = s.height * 58 / 100;
  layout.left = s.width / 24;
  layout.top = s.height / 30;
  layout.cell_width = (s.width - 2 * layout.left) / 6;
  layout.cell_height = (bottom - 2 * layout.top) / 4;
  layout.patch_width = layout.cell_width - layout.cell_width / 6;
  layout.patch_height = layout.cell_height - layout.cell_height / 6;
  return layout;
}

// The scene's linear RGB at the point (fx, fy), in pixels, before the lens
// blurs it and without the exposure.
HOLOSCAN_ISP_SYNTHETIC_HD inline void synthetic_scene_point(const SyntheticRawScene& s, float fx,
                                                            float fy, float rgb[3]) {
  using synthetic_detail::set_rgb;
  const int w = s.width;
  const int h = s.height;
  if (s.kind == static_cast<int>(SceneKind::kGray)) {
    float v;
    if (fx < static_cast<float>(w / 2)) {
      v = 0.03F + 0.8F * fy / static_cast<float>(h - 1);
    } else {
      const float cell = static_cast<float>(h / 4 > 1 ? h / 4 : 1);
      const int col =
          static_cast<int>(floorf((fx - static_cast<float>(w / 2 - s.shift)) / cell)) & 3;
      const int row = static_cast<int>(floorf(fy / cell)) & 3;
      v = 0.05F + 0.8F * static_cast<float>((row * 4 + col) * 7 % 16) / 15.0F;
    }
    set_rgb(rgb, v, v, v);
  } else if (fy < static_cast<float>(h * 58 / 100)) {
    const ChartLayout c = chart_layout(s);
    set_rgb(rgb, 0.05F, 0.05F, 0.05F);
    const float px = fx - static_cast<float>(c.left + s.shift);
    const float py = fy - static_cast<float>(c.top);
    if (px >= 0.0F && py >= 0.0F) {
      const int col = static_cast<int>(px) / c.cell_width;
      const int row = static_cast<int>(py) / c.cell_height;
      if (col < 6 && row < 4 &&
          px - static_cast<float>(col * c.cell_width) < static_cast<float>(c.patch_width) &&
          py - static_cast<float>(row * c.cell_height) < static_cast<float>(c.patch_height)) {
        synthetic_detail::checker_color(row * 6 + col, rgb);
      }
    }
  } else if (fy < static_cast<float>(h * 70 / 100)) {
    const float v = 0.01F + 0.89F * fx / static_cast<float>(w - 1);
    set_rgb(rgb, v, v, v);
  } else if (fy < static_cast<float>(h * 82 / 100)) {
    // Hue ramp at full saturation, linear 0.05..0.7.
    float hue = 6.0F * fx / static_cast<float>(w);
    hue = hue < 0.0F ? 0.0F : hue;
    const int sector = static_cast<int>(hue) % 6;
    const float f = hue - static_cast<float>(static_cast<int>(hue));
    const float lo = 0.05F;
    const float hi = 0.7F;
    const float up = lo + (hi - lo) * f;
    const float down = hi - (hi - lo) * f;
    switch (sector) {
      case 0:
        set_rgb(rgb, hi, up, lo);
        break;
      case 1:
        set_rgb(rgb, down, hi, lo);
        break;
      case 2:
        set_rgb(rgb, lo, hi, up);
        break;
      case 3:
        set_rgb(rgb, lo, down, hi);
        break;
      case 4:
        set_rgb(rgb, up, lo, hi);
        break;
      default:
        set_rgb(rgb, hi, lo, down);
        break;
    }
  } else if (fx < static_cast<float>(w / 2)) {
    // Black and white bars 4, 8 and 16 pixels wide.
    const float third = static_cast<float>(w / 2 > 2 ? w / 2 : 2) / 3.0F;
    const int region = fx < third ? 0 : fx < 2.0F * third ? 1 : 2;
    const float width = static_cast<float>(4 << region);
    const float v = (static_cast<int>(floorf(fx / width)) & 1) != 0 ? 0.85F : 0.04F;
    set_rgb(rgb, v, v, v);
  } else {
    // A slanted edge between two grays, and a red disc over it.
    const int top = h * 82 / 100;
    const float dx = fx - static_cast<float>(3 * w / 4 + s.shift);
    const float dy = fy - static_cast<float>((top + h) / 2);
    const float radius = static_cast<float>((h - top) / 3);
    if (dx * dx + dy * dy < radius * radius) {
      set_rgb(rgb, 0.6F, 0.08F, 0.06F);
    } else {
      const float v = 5.0F * dx + dy > 0.0F ? 0.75F : 0.12F;
      set_rgb(rgb, v, v, v);
    }
  }
}

// The scene's linear RGB at pixel (x, y) as the lens delivers it: the mean of
// 4 x 4 points over the 2 x 2 pixels around it (a soft optical blur, so that
// edges are not sharper than a real lens draws them), exposure included.
HOLOSCAN_ISP_SYNTHETIC_HD inline void synthetic_scene_rgb(const SyntheticRawScene& s, int x, int y,
                                                          float rgb[3]) {
  float sum[3] = {0.0F, 0.0F, 0.0F};
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      float point[3];
      synthetic_scene_point(s, static_cast<float>(x) + 0.5F * static_cast<float>(i) - 0.75F,
                            static_cast<float>(y) + 0.5F * static_cast<float>(j) - 0.75F, point);
      for (int k = 0; k < 3; ++k) sum[k] = sum[k] + point[k];
    }
  }
  for (int k = 0; k < 3; ++k) rgb[k] = sum[k] * 0.0625F * s.exposure;
}

// The lens falloff at (x, y): 1 at the centre, 1 - vignetting at the corners.
HOLOSCAN_ISP_SYNTHETIC_HD inline float synthetic_falloff(const SyntheticRawScene& s, int x, int y) {
  const float dx = static_cast<float>(x) - s.cx;
  const float dy = static_cast<float>(y) - s.cy;
  return 1.0F - s.vignetting * ((dx * dx * s.inv_cx2 + dy * dy * s.inv_cy2) * 0.5F);
}

// The camera's linear RGB at (x, y): the scene through the inverse colour
// matrix, negative values cut to 0.
HOLOSCAN_ISP_SYNTHETIC_HD inline void synthetic_camera_rgb(const SyntheticRawScene& s, int x, int y,
                                                           float cam[3]) {
  float rgb[3];
  synthetic_scene_rgb(s, x, y, rgb);
  for (int k = 0; k < 3; ++k) {
    const float v =
        s.inv_ccm[3 * k] * rgb[0] + s.inv_ccm[3 * k + 1] * rgb[1] + s.inv_ccm[3 * k + 2] * rgb[2];
    cam[k] = v > 0.0F ? v : 0.0F;
  }
}

// The raw sample at (x, y), in counts.
HOLOSCAN_ISP_SYNTHETIC_HD inline int synthetic_raw_sample(const SyntheticRawScene& s, int x,
                                                          int y) {
  float cam[3];
  synthetic_camera_rgb(s, x, y, cam);
  const int channel = s.cfa[((y & 1) << 1) | (x & 1)];
  const int color = channel == kChannelR ? 0 : channel == kChannelB ? 2 : 1;
  float signal = cam[color] * synthetic_falloff(s, x, y) * s.inv_wb[color];
  signal = signal < 1.0F ? signal : 1.0F;
  const float counts = s.black[channel] + signal * (s.white - s.black[channel]);
  const int value = static_cast<int>(counts + 0.5F);
  const int white = static_cast<int>(s.white);
  return value < 0 ? 0 : value > white ? white : value;
}

// What an ideal ISP outputs at (x, y) before its tone curve: linear RGB 0..1,
// with the sensor's clipping and the ISP's own clipping accounted for.
HOLOSCAN_ISP_SYNTHETIC_HD inline void synthetic_truth(const SyntheticRawScene& s, int x, int y,
                                                      float linear[3]) {
  float cam[3];
  synthetic_camera_rgb(s, x, y, cam);
  const float falloff = synthetic_falloff(s, x, y);
  float balanced[3];
  for (int k = 0; k < 3; ++k) {
    float signal = cam[k] * falloff * s.inv_wb[k];
    signal = signal < 1.0F ? signal : 1.0F;
    const float v = signal / falloff * s.wb[k];
    balanced[k] = v < 1.0F ? v : 1.0F;
  }
  for (int k = 0; k < 3; ++k) {
    const float v = s.ccm[3 * k] * balanced[0] + s.ccm[3 * k + 1] * balanced[1] +
                    s.ccm[3 * k + 2] * balanced[2];
    linear[k] = v < 0.0F ? 0.0F : v > 1.0F ? 1.0F : v;
  }
}

// Writes the samples of packing unit `unit` of row y into `row`: one sample
// for kUint8 and kUint16, four for kRaw10, two for kRaw12. Samples past the
// image's width are 0.
HOLOSCAN_ISP_SYNTHETIC_HD inline void synthetic_write_unit(const SyntheticRawScene& s, int unit,
                                                           int y, unsigned char* row) {
  switch (s.format) {
    case static_cast<int>(RawFormat::kUint8):
      row[unit] = static_cast<unsigned char>(synthetic_raw_sample(s, unit, y));
      break;
    case static_cast<int>(RawFormat::kUint16):
      reinterpret_cast<std::uint16_t*>(row)[unit] =
          static_cast<std::uint16_t>(synthetic_raw_sample(s, unit, y));
      break;
    case static_cast<int>(RawFormat::kRaw10): {
      unsigned char* group = row + 5 * unit;
      group[4] = 0;
      for (int k = 0; k < 4; ++k) {
        const int x = 4 * unit + k;
        const int v = x < s.width ? synthetic_raw_sample(s, x, y) : 0;
        group[k] = static_cast<unsigned char>(v >> 2);
        group[4] = static_cast<unsigned char>(group[4] | ((v & 3) << (2 * k)));
      }
      break;
    }
    default: {
      unsigned char* group = row + 3 * unit;
      group[2] = 0;
      for (int k = 0; k < 2; ++k) {
        const int x = 2 * unit + k;
        const int v = x < s.width ? synthetic_raw_sample(s, x, y) : 0;
        group[k] = static_cast<unsigned char>(v >> 4);
        group[2] = static_cast<unsigned char>(group[2] | ((v & 15) << (4 * k)));
      }
      break;
    }
  }
}

// Packing units in a row of the frame.
HOLOSCAN_ISP_SYNTHETIC_HD inline int synthetic_units(const SyntheticRawScene& s) {
  return s.format == static_cast<int>(RawFormat::kRaw10)   ? (s.width + 3) / 4
         : s.format == static_cast<int>(RawFormat::kRaw12) ? (s.width + 1) / 2
                                                           : s.width;
}

}  // namespace holoscan::isp
