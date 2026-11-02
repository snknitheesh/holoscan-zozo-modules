// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "cpu_pipeline.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

namespace holoscan::isp::detail {
namespace {

constexpr std::array<std::array<int, 4>, 4> kCfa{{
    {kChannelR, kChannelGr, kChannelGb, kChannelB},  // RGGB
    {kChannelB, kChannelGb, kChannelGr, kChannelR},  // BGGR
    {kChannelGr, kChannelR, kChannelB, kChannelGb},  // GRBG
    {kChannelGb, kChannelB, kChannelR, kChannelGr},  // GBRG
}};

template <bool kMalvar>
void demosaic_frame(const FrameParams& p, const std::vector<float>& samples, int border,
                    std::vector<float>& rgb) {
  // RGB at every pixel and, with a border, at the virtual pixels around the
  // image, whose neighbours are reflected as the kernels' tiles reflect them.
  const int w = p.width + 2 * border;
  const int h = p.height + 2 * border;
  rgb.resize(static_cast<std::size_t>(w) * h * 3U);
  for (int ty = 0; ty < h; ++ty) {
    const int vy = ty - border;
    for (int tx = 0; tx < w; ++tx) {
      const int vx = tx - border;
      float n[5][5] = {};
      for (int dy = -2; dy <= 2; ++dy) {
        const int y = mirror(vy + dy, p.height);
        for (int dx = -2; dx <= 2; ++dx) {
          n[dy + 2][dx + 2] =
              samples[static_cast<std::size_t>(y) * p.width + mirror(vx + dx, p.width)];
        }
      }
      demosaic<kMalvar>(n, channel_at(p, vx, vy),
                        &rgb[(static_cast<std::size_t>(ty) * w + tx) * 3U]);
    }
  }
}

}  // namespace

FrameParams make_frame_params(const IspParams& params, const RawImageView& raw,
                              const RgbImageView& out, const float* lut, int lut_size,
                              const float* lsc) {
  FrameParams p;
  p.width = raw.width;
  p.height = raw.height;
  p.raw = static_cast<const unsigned char*>(raw.data);
  p.raw_pitch = raw.pitch;
  p.raw_format = static_cast<int>(raw.format);
  const auto& cfa = kCfa[static_cast<std::size_t>(raw.pattern)];
  for (int k = 0; k < 4; ++k) p.cfa[k] = cfa[static_cast<std::size_t>(k)];
  const float white = params.white_level > 0.0F
                          ? params.white_level
                          : static_cast<float>((1U << static_cast<unsigned>(raw.bit_depth)) - 1U);
  for (int k = 0; k < 4; ++k) {
    p.black[k] =
        params.channel_black_levels ? (*params.channel_black_levels)[k] : params.black_level;
    p.inv_range[k] = 1.0F / (white - p.black[k]);
  }
  p.white = static_cast<int>(std::ceil(white));
  for (int k = 0; k < 3; ++k) p.wb[k] = params.wb_gains[static_cast<std::size_t>(k)];
  if (params.lens_shading) {
    const LensShading& shading = *params.lens_shading;
    p.lsc = lsc;
    p.lsc_width = shading.width;
    p.lsc_height = shading.height;
    p.lsc_channels = shading.channels;
    p.lsc_sx = static_cast<float>(shading.width - 1) / static_cast<float>(raw.width - 1);
    p.lsc_sy = static_cast<float>(shading.height - 1) / static_cast<float>(raw.height - 1);
  }
  for (int k = 0; k < 9; ++k) p.ccm[k] = params.ccm[static_cast<std::size_t>(k)];
  p.lut = lut;
  p.lut_size = lut_size;
  p.sharpen = params.sharpen;
  p.out = static_cast<unsigned char*>(out.data);
  p.out_pitch = out.pitch;
  p.out_format = static_cast<int>(out.format);
  return p;
}

AwbState initial_white_balance(const IspParams& params) {
  AwbState state{};
  for (int k = 0; k < 3; ++k) state.gains[k] = params.wb_gains[static_cast<std::size_t>(k)];
  state.valid = 0;
  return state;
}

void cpu_process(const FrameParams& p, const IspParams& params, const StatsOutput& stats,
                 const FrameExposure& exposure, AwbState& awb, CpuScratch& scratch) {
  // White balance: this frame's estimate, from its own statistics.
  float gains[3] = {p.wb[0], p.wb[1], p.wb[2]};
  if (params.white_balance != WhiteBalanceMode::kManual) {
    scratch.awb.resize(1);
    AwbAccum& a = scratch.awb[0];
    std::memset(&a, 0, sizeof(a));
    for (int qy = 0; qy < p.height / 2; ++qy) {
      for (int qx = 0; qx < p.width / 2; ++qx) {
        int color[4];
        unsigned int fixed[4];
        int bin[4];
        if (!white_balance_quad(p, qx, qy, color, fixed, bin)) {
          continue;
        }
        for (int k = 0; k < 4; ++k) {
          a.sum[color[k]] += fixed[k];
          ++a.hist[color[k]][bin[k]];
        }
        ++a.quads;
      }
    }
    finish_white_balance(a, static_cast<int>(params.white_balance), params.white_patch_percentile,
                         params.awb_smoothing, awb);
    for (int k = 0; k < 3; ++k) gains[k] = awb.gains[k];
  }

  // Black level, lens shading, white balance.
  const std::size_t pixels = static_cast<std::size_t>(p.width) * p.height;
  scratch.samples.resize(pixels);
  for (int y = 0; y < p.height; ++y) {
    for (int x = 0; x < p.width; ++x) {
      scratch.samples[static_cast<std::size_t>(y) * p.width + x] = preprocess(p, gains, x, y);
    }
  }

  // Demosaic, with a border for the sharpening's neighbours.
  const int border = params.sharpen > 0.0F ? 1 : 0;
  if (params.demosaic == DemosaicMethod::kMalvar) {
    demosaic_frame<true>(p, scratch.samples, border, scratch.rgb);
  } else {
    demosaic_frame<false>(p, scratch.samples, border, scratch.rgb);
  }

  // Sharpening, colour correction, tone curve, and the sums for the statistics.
  StatsAccum sums{};
  const int w = p.width + 2 * border;
  for (int y = 0; y < p.height; ++y) {
    for (int x = 0; x < p.width; ++x) {
      float rgb[3];
      const float* centre =
          &scratch.rgb[(static_cast<std::size_t>(y + border) * w + x + border) * 3U];
      if (border != 0) {
        float m[3][3][3];
        for (int dy = 0; dy < 3; ++dy) {
          for (int dx = 0; dx < 3; ++dx) {
            const float* v = &scratch.rgb[(static_cast<std::size_t>(y + dy) * w + x + dx) * 3U];
            for (int k = 0; k < 3; ++k) m[dy][dx][k] = v[k];
          }
        }
        sharpen(m, p.sharpen, rgb);
      } else {
        for (int k = 0; k < 3; ++k) rgb[k] = centre[k];
      }
      float linear[3];
      correct_color(p, rgb, linear);
      if (stats.stats != nullptr) {
        for (int k = 0; k < 3; ++k) sums.sum[k] += to_fixed(linear[k]);
        sums.saturated += load_raw(p, x, y) >= p.white ? 1U : 0U;
      }
      float toned[3];
      for (int k = 0; k < 3; ++k) toned[k] = tone(p, linear[k]);
      store(p, x, y, toned);
    }
  }

  if (stats.histogram != nullptr) {
    std::memset(stats.histogram, 0, sizeof(std::uint32_t) * kHistogramChannels * kHistogramBins);
    for (int y = 0; y < p.height; ++y) {
      for (int x = 0; x < p.width; ++x) {
        int q[3];
        output_codes(p, x, y, q);
        ++stats.histogram[kHistogramLuma * kHistogramBins + luma_code(q)];
        ++stats.histogram[kHistogramRed * kHistogramBins + q[0]];
        ++stats.histogram[kHistogramGreen * kHistogramBins + q[1]];
        ++stats.histogram[kHistogramBlue * kHistogramBins + q[2]];
      }
    }
  }
  if (stats.stats != nullptr) {
    finish_stats(sums, static_cast<std::int64_t>(pixels), gains, params.auto_exposure, exposure,
                 stats.stats);
  }
}

}  // namespace holoscan::isp::detail
