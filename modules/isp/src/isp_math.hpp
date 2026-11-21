// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The pipeline's arithmetic, one pixel at a time, shared by the CUDA kernels
// and the CPU backend so that both compute the same values: the library is
// compiled without fused multiply-adds (--fmad=false, -ffp-contract=off), and
// every sum the statistics take is in integers, so the two backends agree
// bit for bit. Plain C++ outside nvcc.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "holoscan/isp/isp_params.hpp"

#if defined(__CUDACC__)
#define HOLOSCAN_ISP_HD __host__ __device__ __forceinline__
#else
#define HOLOSCAN_ISP_HD inline
#endif

namespace holoscan::isp::detail {

// The type CUDA's 64-bit atomicAdd takes, which std::uint64_t is not on every platform.
using AtomicU64 = unsigned long long;  // NOLINT(runtime/int)

// Linear values in the statistics' integer sums are in units of 2^-20.
inline constexpr float kFixedOne = 1048576.0F;
// Bins of the white-patch histograms, over 0..1 before white balance.
inline constexpr int kAwbBins = 1024;

// Everything a frame's per-pixel stages read, passed by value to the kernels.
struct FrameParams {
  int width{};
  int height{};
  const unsigned char* raw{};
  std::size_t raw_pitch{};
  int raw_format{};  // RawFormat.
  int cfa[4]{};      // The channel (kChannelR...) at quad position (y & 1) * 2 + (x & 1).
  float black[4]{};
  float inv_range[4]{};  // 1 / (white - black), per channel.
  int white{};           // Raw counts at or above which a sample is saturated.
  float wb[3]{};         // R, G, B gains, unless `awb` points at the estimated ones.
  const float* awb{};
  const float* lsc{};  // Lens shading grid, or none.
  int lsc_width{};
  int lsc_height{};
  int lsc_channels{};
  float lsc_sx{};  // Grid points per pixel across and down.
  float lsc_sy{};
  float ccm[9]{};
  const float* lut{};
  int lut_size{};
  float sharpen{};
  unsigned char* out{};
  std::size_t out_pitch{};
  int out_format{};  // OutputFormat.
};

// The white balance statistics of one frame, taken over complete 2x2 quads
// without a saturated sample.
struct AwbAccum {
  AtomicU64 sum[3];  // R, G (both greens), B before white balance, units of 2^-20.
  AtomicU64 quads;
  unsigned int hist[3][kAwbBins];  // White patch: R, G (both greens), B.
};

// The estimated gains, carried from frame to frame.
struct AwbState {
  float gains[3];
  int valid;  // An estimate has been made.
};

// The main pass's sums for the statistics.
struct StatsAccum {
  AtomicU64 sum[3];  // Linear R, G, B after the colour correction, units of 2^-20.
  AtomicU64 saturated;
};

HOLOSCAN_ISP_HD int mirror(int v, int n) {
  // Reflection about the first and last pixel (..., 2, 1, 0, 1, 2, ...): it
  // keeps a coordinate's parity, and so its colour in the mosaic.
  const int period = 2 * (n - 1);
  v %= period;
  if (v < 0) {
    v += period;
  }
  return v < n ? v : period - v;
}

HOLOSCAN_ISP_HD int color_of(int channel) {
  return channel == kChannelR ? 0 : channel == kChannelB ? 2 : 1;
}

HOLOSCAN_ISP_HD int channel_at(const FrameParams& p, int x, int y) {
  return p.cfa[((y & 1) << 1) | (x & 1)];
}

HOLOSCAN_ISP_HD int load_raw(const FrameParams& p, int x, int y) {
  const unsigned char* row = p.raw + static_cast<std::size_t>(y) * p.raw_pitch;
  switch (p.raw_format) {
    case static_cast<int>(RawFormat::kUint8):
      return row[x];
    case static_cast<int>(RawFormat::kUint16):
      return reinterpret_cast<const std::uint16_t*>(row)[x];
    case static_cast<int>(RawFormat::kRaw10): {
      const unsigned char* group = row + 5 * (x >> 2);
      const int k = x & 3;
      return (static_cast<int>(group[k]) << 2) | ((group[4] >> (2 * k)) & 3);
    }
    default: {
      const unsigned char* group = row + 3 * (x >> 1);
      const int k = x & 1;
      return (static_cast<int>(group[k]) << 4) | ((group[2] >> (4 * k)) & 15);
    }
  }
}

HOLOSCAN_ISP_HD float lens_shading_gain(const FrameParams& p, int x, int y, int channel) {
  if (p.lsc == nullptr) {
    return 1.0F;
  }
  const float fx = static_cast<float>(x) * p.lsc_sx;
  const float fy = static_cast<float>(y) * p.lsc_sy;
  int ix = static_cast<int>(fx);
  int iy = static_cast<int>(fy);
  ix = ix > p.lsc_width - 2 ? p.lsc_width - 2 : ix;
  iy = iy > p.lsc_height - 2 ? p.lsc_height - 2 : iy;
  const float tx = fx - static_cast<float>(ix);
  const float ty = fy - static_cast<float>(iy);
  const int step = p.lsc_channels;
  const int row = p.lsc_width * step;
  const float* g =
      p.lsc + (static_cast<std::size_t>(iy) * p.lsc_width + ix) * step + (step == 4 ? channel : 0);
  const float top = g[0] + (g[step] - g[0]) * tx;
  const float bottom = g[row] + (g[row + step] - g[row]) * tx;
  return top + (bottom - top) * ty;
}

// A sample after black level and lens shading, before white balance: 0 at
// black, 1 at white (more where the lens shading gain lifts it).
HOLOSCAN_ISP_HD float linearize(const FrameParams& p, int raw, int x, int y, int channel) {
  float v = (static_cast<float>(raw) - p.black[channel]) * p.inv_range[channel];
  v = v > 0.0F ? v : 0.0F;
  return v * lens_shading_gain(p, x, y, channel);
}

// A sample ready to demosaic: white balanced and clipped to 0..1.
HOLOSCAN_ISP_HD float preprocess(const FrameParams& p, const float gains[3], int x, int y) {
  const int channel = channel_at(p, x, y);
  const float v = linearize(p, load_raw(p, x, y), x, y, channel) * gains[color_of(channel)];
  return v < 1.0F ? v : 1.0F;
}

HOLOSCAN_ISP_HD float clamp01(float v) { return v < 0.0F ? 0.0F : v > 1.0F ? 1.0F : v; }

// RGB at the centre of a 5x5 neighbourhood n[dy + 2][dx + 2] of preprocessed
// samples whose centre is of channel `channel`. Bilinear reads the inner 3x3.
template <bool kMalvar>
HOLOSCAN_ISP_HD void demosaic(const float (&n)[5][5], int channel, float rgb[3]) {
  const float c = n[2][2];
  float at_green;  // Green at a red or blue sample.
  float diagonal;  // Red at blue, blue at red.
  float across;    // The colour left and right of a green sample.
  float down;      // The colour above and below a green sample.
  if constexpr (!kMalvar) {
    at_green = (n[1][2] + n[3][2] + n[2][1] + n[2][3]) * 0.25F;
    diagonal = (n[1][1] + n[1][3] + n[3][1] + n[3][3]) * 0.25F;
    across = (n[2][1] + n[2][3]) * 0.5F;
    down = (n[1][2] + n[3][2]) * 0.5F;
  } else {
    // Malvar, He and Cutler, "High-quality linear interpolation for
    // demosaicing of Bayer-patterned color images", ICASSP 2004: bilinear
    // plus the Laplacian of the colour known at the centre.
    const float axial1 = n[1][2] + n[3][2] + n[2][1] + n[2][3];
    const float axial2 = n[0][2] + n[4][2] + n[2][0] + n[2][4];
    const float diagonal1 = n[1][1] + n[1][3] + n[3][1] + n[3][3];
    const float horizontal2 = n[2][0] + n[2][4];
    const float vertical2 = n[0][2] + n[4][2];
    at_green = (4.0F * c + 2.0F * axial1 - axial2) * 0.125F;
    diagonal = (6.0F * c + 2.0F * diagonal1 - 1.5F * axial2) * 0.125F;
    across = (5.0F * c + 4.0F * (n[2][1] + n[2][3]) - horizontal2 - diagonal1 + 0.5F * vertical2) *
             0.125F;
    down = (5.0F * c + 4.0F * (n[1][2] + n[3][2]) - vertical2 - diagonal1 + 0.5F * horizontal2) *
           0.125F;
    at_green = clamp01(at_green);
    diagonal = clamp01(diagonal);
    across = clamp01(across);
    down = clamp01(down);
  }
  switch (channel) {
    case kChannelR:
      rgb[0] = c;
      rgb[1] = at_green;
      rgb[2] = diagonal;
      break;
    case kChannelB:
      rgb[0] = diagonal;
      rgb[1] = at_green;
      rgb[2] = c;
      break;
    case kChannelGr:  // Red left and right, blue above and below.
      rgb[0] = across;
      rgb[1] = c;
      rgb[2] = down;
      break;
    default:  // kChannelGb: blue left and right, red above and below.
      rgb[0] = down;
      rgb[1] = c;
      rgb[2] = across;
      break;
  }
}

// Unsharp mask of the centre of a 3x3 neighbourhood m[dy + 1][dx + 1] of RGB,
// against a 1-2-1 binomial blur.
HOLOSCAN_ISP_HD void sharpen(const float (&m)[3][3][3], float amount, float rgb[3]) {
  for (int k = 0; k < 3; ++k) {
    const float c = m[1][1][k];
    const float blur = (4.0F * c + 2.0F * (m[0][1][k] + m[2][1][k] + m[1][0][k] + m[1][2][k]) +
                        (m[0][0][k] + m[0][2][k] + m[2][0][k] + m[2][2][k])) *
                       0.0625F;
    rgb[k] = clamp01(c + amount * (c - blur));
  }
}

HOLOSCAN_ISP_HD void correct_color(const FrameParams& p, const float in[3], float out[3]) {
  for (int k = 0; k < 3; ++k) {
    out[k] = clamp01(p.ccm[3 * k] * in[0] + p.ccm[3 * k + 1] * in[1] + p.ccm[3 * k + 2] * in[2]);
  }
}

HOLOSCAN_ISP_HD float tone(const FrameParams& p, float x) {
  const float f = x * static_cast<float>(p.lut_size - 1);
  const int i = static_cast<int>(f);
  if (i >= p.lut_size - 1) {
    return p.lut[p.lut_size - 1];
  }
  const float t = f - static_cast<float>(i);
  return p.lut[i] + t * (p.lut[i + 1] - p.lut[i]);
}

HOLOSCAN_ISP_HD unsigned int to_fixed(float v) {
  return static_cast<unsigned int>(v * kFixedOne + 0.5F);
}

// Writes tone-mapped RGB (0..1) at (x, y) of the output.
HOLOSCAN_ISP_HD void store(const FrameParams& p, int x, int y, const float t[3]) {
  unsigned char* row = p.out + static_cast<std::size_t>(y) * p.out_pitch;
  switch (p.out_format) {
    case static_cast<int>(OutputFormat::kRgb8):
      for (int k = 0; k < 3; ++k) {
        row[3 * x + k] = static_cast<unsigned char>(t[k] * 255.0F + 0.5F);
      }
      break;
    case static_cast<int>(OutputFormat::kRgba8):
      for (int k = 0; k < 3; ++k) {
        row[4 * x + k] = static_cast<unsigned char>(t[k] * 255.0F + 0.5F);
      }
      row[4 * x + 3] = 255;
      break;
    case static_cast<int>(OutputFormat::kRgb16):
      for (int k = 0; k < 3; ++k) {
        reinterpret_cast<std::uint16_t*>(row)[3 * x + k] =
            static_cast<std::uint16_t>(t[k] * 65535.0F + 0.5F);
      }
      break;
    default:
      for (int k = 0; k < 3; ++k) {
        reinterpret_cast<float*>(row)[3 * x + k] = t[k];
      }
      break;
  }
}

// The 8-bit codes of the output at (x, y), for the histograms.
HOLOSCAN_ISP_HD void output_codes(const FrameParams& p, int x, int y, int q[3]) {
  const unsigned char* row = p.out + static_cast<std::size_t>(y) * p.out_pitch;
  switch (p.out_format) {
    case static_cast<int>(OutputFormat::kRgb8):
      for (int k = 0; k < 3; ++k) q[k] = row[3 * x + k];
      break;
    case static_cast<int>(OutputFormat::kRgba8):
      for (int k = 0; k < 3; ++k) q[k] = row[4 * x + k];
      break;
    case static_cast<int>(OutputFormat::kRgb16):
      for (int k = 0; k < 3; ++k) {
        q[k] = static_cast<int>(
            (reinterpret_cast<const std::uint16_t*>(row)[3 * x + k] * 255U + 32767U) / 65535U);
      }
      break;
    default:
      for (int k = 0; k < 3; ++k) {
        q[k] = static_cast<int>(clamp01(reinterpret_cast<const float*>(row)[3 * x + k]) * 255.0F +
                                0.5F);
      }
      break;
  }
}

HOLOSCAN_ISP_HD int luma_code(const int q[3]) {
  return (2126 * q[0] + 7152 * q[1] + 722 * q[2] + 5000) / 10000;
}

// One complete 2x2 quad of the mosaic for the white balance statistics, its
// top-left sample at (2 qx, 2 qy): false when a sample is saturated, else each
// sample's colour (0..2), linear value before white balance in units of 2^-20
// (capped at 4) and white-patch bin.
HOLOSCAN_ISP_HD bool white_balance_quad(const FrameParams& p, int qx, int qy, int color[4],
                                        unsigned int fixed[4], int bin[4]) {
  int raw[4];
  for (int k = 0; k < 4; ++k) {
    raw[k] = load_raw(p, 2 * qx + (k & 1), 2 * qy + (k >> 1));
    if (raw[k] >= p.white) {
      return false;
    }
  }
  for (int k = 0; k < 4; ++k) {
    const int x = 2 * qx + (k & 1);
    const int y = 2 * qy + (k >> 1);
    const int channel = channel_at(p, x, y);
    float v = linearize(p, raw[k], x, y, channel);
    v = v < 4.0F ? v : 4.0F;
    color[k] = color_of(channel);
    fixed[k] = to_fixed(v);
    const int b = static_cast<int>(v * static_cast<float>(kAwbBins));
    bin[k] = b < kAwbBins - 1 ? b : kAwbBins - 1;
  }
  return true;
}

// The value at `percentile` of a white-patch histogram, 0..1, interpolated
// within its bin; 0 for an empty histogram.
HOLOSCAN_ISP_HD float histogram_percentile(const unsigned int* hist, float percentile) {
  AtomicU64 total = 0;
  for (int b = 0; b < kAwbBins; ++b) total += hist[b];
  if (total == 0) {
    return 0.0F;
  }
  const double target = static_cast<double>(percentile) * static_cast<double>(total);
  AtomicU64 below = 0;
  for (int b = 0; b < kAwbBins; ++b) {
    if (hist[b] != 0U && static_cast<double>(below + hist[b]) >= target) {
      const double within = (target - static_cast<double>(below)) / static_cast<double>(hist[b]);
      return static_cast<float>((static_cast<double>(b) + within) / kAwbBins);
    }
    below += hist[b];
  }
  return 1.0F;
}

// Updates the white balance estimate from a frame's statistics: gray world
// equalises the channel means, white patch the channels' `percentile`, green
// staying at 1. A frame without usable quads keeps the last gains.
HOLOSCAN_ISP_HD void finish_white_balance(const AwbAccum& a, int mode, float percentile,
                                          float smoothing, AwbState& state) {
  if (a.quads == 0) {
    return;
  }
  float level[3];
  if (mode == static_cast<int>(WhiteBalanceMode::kGrayWorld)) {
    const double quads = static_cast<double>(a.quads);
    level[0] = static_cast<float>(static_cast<double>(a.sum[0]) / quads);
    level[1] = static_cast<float>(static_cast<double>(a.sum[1]) / (2.0 * quads));
    level[2] = static_cast<float>(static_cast<double>(a.sum[2]) / quads);
  } else {
    for (int k = 0; k < 3; ++k) level[k] = histogram_percentile(a.hist[k], percentile);
  }
  if (!(level[0] > 0.0F && level[1] > 0.0F && level[2] > 0.0F)) {
    return;
  }
  float target[3] = {level[1] / level[0], 1.0F, level[1] / level[2]};
  for (int k = 0; k < 3; ++k) {
    target[k] = target[k] < 0.0625F ? 0.0625F : target[k] > 16.0F ? 16.0F : target[k];
  }
  for (int k = 0; k < 3; ++k) {
    state.gains[k] = state.valid != 0 && smoothing < 1.0F
                         ? state.gains[k] + smoothing * (target[k] - state.gains[k])
                         : target[k];
  }
  state.valid = 1;
}

// The statistics vector from the main pass's sums.
HOLOSCAN_ISP_HD void finish_stats(const StatsAccum& a, std::int64_t pixels, const float gains[3],
                                  const AutoExposureParams& ae, const FrameExposure& exposure,
                                  float* out) {
  const double n = static_cast<double>(pixels) * static_cast<double>(kFixedOne);
  const double mean[3] = {static_cast<double>(a.sum[0]) / n, static_cast<double>(a.sum[1]) / n,
                          static_cast<double>(a.sum[2]) / n};
  const double luma = 0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2];
  double error = luma > 0.0 ? log2(static_cast<double>(ae.target) / luma) : 8.0;
  error = error < -8.0 ? -8.0 : error > 8.0 ? 8.0 : error;
  const double scale = exp2(static_cast<double>(ae.damping) * error);
  double next_exposure = 0.0;
  double next_gain = 0.0;
  if (exposure.exposure_ns > 0) {
    const double gain = exposure.gain > 0.0F ? exposure.gain : ae.min_gain;
    const double wanted = static_cast<double>(exposure.exposure_ns) * gain * scale;
    // Exposure time first, at the lowest gain; gain only for what it cannot reach.
    next_exposure = wanted / ae.min_gain;
    const double low = static_cast<double>(ae.min_exposure_ns);
    const double high = static_cast<double>(ae.max_exposure_ns);
    next_exposure = next_exposure < low ? low : next_exposure > high ? high : next_exposure;
    next_gain = wanted / next_exposure;
    next_gain = next_gain < ae.min_gain   ? ae.min_gain
                : next_gain > ae.max_gain ? ae.max_gain
                                          : next_gain;
  }
  out[stats::kMeanR] = static_cast<float>(mean[0]);
  out[stats::kMeanG] = static_cast<float>(mean[1]);
  out[stats::kMeanB] = static_cast<float>(mean[2]);
  out[stats::kMeanLuma] = static_cast<float>(luma);
  out[stats::kAeErrorEv] = static_cast<float>(error);
  out[stats::kExposureScale] = static_cast<float>(scale);
  out[stats::kSuggestedExposureNs] = static_cast<float>(next_exposure);
  out[stats::kSuggestedGain] = static_cast<float>(next_gain);
  out[stats::kWbGainR] = gains[0];
  out[stats::kWbGainG] = gains[1];
  out[stats::kWbGainB] = gains[2];
  out[stats::kSaturatedFraction] =
      static_cast<float>(static_cast<double>(a.saturated) / static_cast<double>(pixels));
}

}  // namespace holoscan::isp::detail
