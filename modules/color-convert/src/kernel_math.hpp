// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The arithmetic of one destination pixel, shared by the CUDA kernel and the
// CPU reference so that both produce the same bytes: decoding source samples,
// resampling, the colour conversion and encoding. Both are compiled without
// floating-point contraction (CMakeLists.txt), so every operation rounds the
// same way on either side. conversion.hpp states what this computes.

#pragma once

#include <cstddef>
#include <cstdint>

#include <holoscan/color_convert/conversion.hpp>
#include <holoscan/color_convert/pixel_format.hpp>

#if defined(__CUDACC__)
#define CC_HD __host__ __device__ __forceinline__
#else
#define CC_HD inline
#endif

namespace holoscan::color_convert::detail {

struct Sample4 {
  float c0;  // Y, R or luma.
  float c1;  // Cb or G; 0 for luma.
  float c2;  // Cr or B; 0 for luma.
  float a;   // Alpha, 255 for a format without it.
};

// c = m * s + o on the three colour samples; alpha passes through.
struct Affine {
  float m[9];
  float o[3];
  bool identity;
};

struct ImageDesc {
  PixelFormat format;
  int width;
  int height;
  std::uint8_t* data[3];
  std::size_t pitch[3];
};

// One conversion, ready to run: everything a pixel needs, in plain values that
// a kernel takes by value.
struct Plan {
  ImageDesc src;
  ImageDesc dst;
  ResizeMode mode;
  // The source rectangle, as inclusive pixel bounds, and its edges.
  int src_x0, src_y0, src_x1, src_y1;
  // Source pixels per destination pixel along each axis.
  float scale_x, scale_y;
  // The content rectangle in the destination, [x0, x1) x [y0, y1).
  int content_x0, content_y0, content_x1, content_y1;
  Affine affine;
  Sample4 fill;  // Already in the destination's colour model.
  // Pixels whose chroma one destination chroma sample holds, as log2.
  int block_shift_x, block_shift_y;
};

// Builds the plan of converting `src` into `dst` with `params`. Validated
// first: see validate_conversion(). Host only.
Plan make_plan(const ImageView& src, const ImageView& dst, const ConversionParams& params);

CC_HD float floor_f(float v) {
#if defined(__CUDA_ARCH__)
  return floorf(v);
#else
  return __builtin_floorf(v);
#endif
}

CC_HD int clamp_i(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

CC_HD std::uint8_t to_u8(float v) {
  const float r = floor_f(v + 0.5F);
  return static_cast<std::uint8_t>(r <= 0.0F ? 0.0F : (r >= 255.0F ? 255.0F : r));
}

CC_HD std::uint16_t to_u16(float v) {
  const float r = floor_f(v * 257.0F + 0.5F);
  return static_cast<std::uint16_t>(r <= 0.0F ? 0.0F : (r >= 65535.0F ? 65535.0F : r));
}

CC_HD float u16_sample(const std::uint8_t* p) {
  std::uint16_t v;
  // Byte-wise, as rows of 16-bit images need not be 2-byte aligned in a byte buffer.
  v = static_cast<std::uint16_t>(p[0] | (p[1] << 8));
  return static_cast<float>(v) / 257.0F;
}

CC_HD void store_u16(std::uint8_t* p, float v) {
  const std::uint16_t w = to_u16(v);
  p[0] = static_cast<std::uint8_t>(w & 0xFFU);
  p[1] = static_cast<std::uint8_t>(w >> 8);
}

// Function arguments convert bytes to float, which braces would refuse.
CC_HD Sample4 make_sample(float c0, float c1, float c2, float a) { return {c0, c1, c2, a}; }

// Pixel (x, y) of `img` in its own samples.
CC_HD Sample4 load(const ImageDesc& img, int x, int y) {
  const std::uint8_t* row = img.data[0] + static_cast<std::size_t>(y) * img.pitch[0];
  switch (img.format) {
    case PixelFormat::kNv12:
    case PixelFormat::kNv16: {
      const int cy = img.format == PixelFormat::kNv12 ? (y >> 1) : y;
      const std::uint8_t* uv =
          img.data[1] + static_cast<std::size_t>(cy) * img.pitch[1] + 2 * (x >> 1);
      return make_sample(row[x], uv[0], uv[1], 255.0F);
    }
    case PixelFormat::kI420: {
      const std::size_t u = static_cast<std::size_t>(y >> 1) * img.pitch[1] + (x >> 1);
      const std::size_t v = static_cast<std::size_t>(y >> 1) * img.pitch[2] + (x >> 1);
      return make_sample(row[x], img.data[1][u], img.data[2][v], 255.0F);
    }
    case PixelFormat::kYuyv: {
      const std::uint8_t* pair = row + 4 * (x >> 1);
      return make_sample(pair[2 * (x & 1)], pair[1], pair[3], 255.0F);
    }
    case PixelFormat::kUyvy: {
      const std::uint8_t* pair = row + 4 * (x >> 1);
      return make_sample(pair[1 + 2 * (x & 1)], pair[0], pair[2], 255.0F);
    }
    case PixelFormat::kRgb8: {
      const std::uint8_t* p = row + 3 * x;
      return make_sample(p[0], p[1], p[2], 255.0F);
    }
    case PixelFormat::kBgr8: {
      const std::uint8_t* p = row + 3 * x;
      return make_sample(p[2], p[1], p[0], 255.0F);
    }
    case PixelFormat::kRgba8: {
      const std::uint8_t* p = row + 4 * x;
      return make_sample(p[0], p[1], p[2], p[3]);
    }
    case PixelFormat::kBgra8: {
      const std::uint8_t* p = row + 4 * x;
      return make_sample(p[2], p[1], p[0], p[3]);
    }
    case PixelFormat::kMono8:
      return make_sample(row[x], 0.0F, 0.0F, 255.0F);
    case PixelFormat::kMono16:
      return make_sample(u16_sample(row + 2 * x), 0.0F, 0.0F, 255.0F);
    case PixelFormat::kRgb16: {
      const std::uint8_t* p = row + 6 * x;
      return make_sample(u16_sample(p), u16_sample(p + 2), u16_sample(p + 4), 255.0F);
    }
  }
  return make_sample(0.0F, 0.0F, 0.0F, 255.0F);
}

CC_HD Sample4 add_weighted(Sample4 sum, Sample4 s, float w) {
  return {sum.c0 + s.c0 * w, sum.c1 + s.c1 * w, sum.c2 + s.c2 * w, sum.a + s.a * w};
}

// Weight of pixel [i, i + 1) in the footprint [lo, hi).
CC_HD float coverage(int i, float lo, float hi) {
  const float a = static_cast<float>(i) > lo ? static_cast<float>(i) : lo;
  const float b = static_cast<float>(i + 1) < hi ? static_cast<float>(i + 1) : hi;
  return b > a ? b - a : 0.0F;
}

// The source sampled for content pixel (dx, dy), in the source's samples.
CC_HD Sample4 sample(const Plan& p, int dx, int dy) {
  const float rx = static_cast<float>(dx - p.content_x0);
  const float ry = static_cast<float>(dy - p.content_y0);
  const auto x0 = static_cast<float>(p.src_x0);
  const auto y0 = static_cast<float>(p.src_y0);
  if (p.mode == ResizeMode::kNearest) {
    const int sx =
        clamp_i(static_cast<int>(floor_f(x0 + (rx + 0.5F) * p.scale_x)), p.src_x0, p.src_x1);
    const int sy =
        clamp_i(static_cast<int>(floor_f(y0 + (ry + 0.5F) * p.scale_y)), p.src_y0, p.src_y1);
    return load(p.src, sx, sy);
  }
  if (p.mode == ResizeMode::kBilinear) {
    const float fx = x0 + (rx + 0.5F) * p.scale_x - 0.5F;
    const float fy = y0 + (ry + 0.5F) * p.scale_y - 0.5F;
    const float bx = floor_f(fx);
    const float by = floor_f(fy);
    const float ax = fx - bx;
    const float ay = fy - by;
    const int ix = static_cast<int>(bx);
    const int iy = static_cast<int>(by);
    const int xa = clamp_i(ix, p.src_x0, p.src_x1);
    const int xb = clamp_i(ix + 1, p.src_x0, p.src_x1);
    const int ya = clamp_i(iy, p.src_y0, p.src_y1);
    const int yb = clamp_i(iy + 1, p.src_y0, p.src_y1);
    // Taps of zero weight are skipped: an unscaled image reads one pixel.
    Sample4 s{0.0F, 0.0F, 0.0F, 0.0F};
    const float w00 = (1.0F - ax) * (1.0F - ay);
    const float w10 = ax * (1.0F - ay);
    const float w01 = (1.0F - ax) * ay;
    const float w11 = ax * ay;
    if (w00 != 0.0F) s = add_weighted(s, load(p.src, xa, ya), w00);
    if (w10 != 0.0F) s = add_weighted(s, load(p.src, xb, ya), w10);
    if (w01 != 0.0F) s = add_weighted(s, load(p.src, xa, yb), w01);
    if (w11 != 0.0F) s = add_weighted(s, load(p.src, xb, yb), w11);
    return s;
  }
  // Area: the footprint [lo, hi) on each axis, inside the source rectangle.
  const float lx = x0 + rx * p.scale_x;
  const float ly = y0 + ry * p.scale_y;
  const float right = static_cast<float>(p.src_x1 + 1);
  const float bottom = static_cast<float>(p.src_y1 + 1);
  const float hx = lx + p.scale_x < right ? lx + p.scale_x : right;
  const float hy = ly + p.scale_y < bottom ? ly + p.scale_y : bottom;
  const int ix0 = clamp_i(static_cast<int>(floor_f(lx)), p.src_x0, p.src_x1);
  const int iy0 = clamp_i(static_cast<int>(floor_f(ly)), p.src_y0, p.src_y1);
  Sample4 s{0.0F, 0.0F, 0.0F, 0.0F};
  float total = 0.0F;
  for (int y = iy0; y <= p.src_y1 && static_cast<float>(y) < hy; ++y) {
    const float wy = coverage(y, ly, hy);
    for (int x = ix0; x <= p.src_x1 && static_cast<float>(x) < hx; ++x) {
      const float w = coverage(x, lx, hx) * wy;
      if (w != 0.0F) {
        s = add_weighted(s, load(p.src, x, y), w);
        total += w;
      }
    }
  }
  if (total <= 0.0F) {
    return load(p.src, ix0, iy0);
  }
  return {s.c0 / total, s.c1 / total, s.c2 / total, s.a / total};
}

CC_HD Sample4 apply(const Affine& t, Sample4 s) {
  if (t.identity) {
    return s;
  }
  return {t.m[0] * s.c0 + t.m[1] * s.c1 + t.m[2] * s.c2 + t.o[0],
          t.m[3] * s.c0 + t.m[4] * s.c1 + t.m[5] * s.c2 + t.o[1],
          t.m[6] * s.c0 + t.m[7] * s.c1 + t.m[8] * s.c2 + t.o[2], s.a};
}

// Destination pixel (dx, dy) in the destination's colour model.
CC_HD Sample4 pixel(const Plan& p, int dx, int dy) {
  if (dx < p.content_x0 || dx >= p.content_x1 || dy < p.content_y0 || dy >= p.content_y1) {
    return p.fill;
  }
  return apply(p.affine, sample(p, dx, dy));
}

// Writes everything but chroma for pixel (x, y) of a non-YCbCr destination,
// or the luma of a YCbCr one.
CC_HD void store(const ImageDesc& img, int x, int y, Sample4 v) {
  std::uint8_t* row = img.data[0] + static_cast<std::size_t>(y) * img.pitch[0];
  switch (img.format) {
    case PixelFormat::kNv12:
    case PixelFormat::kNv16:
    case PixelFormat::kI420:
      row[x] = to_u8(v.c0);
      return;
    case PixelFormat::kYuyv:
      row[4 * (x >> 1) + 2 * (x & 1)] = to_u8(v.c0);
      return;
    case PixelFormat::kUyvy:
      row[4 * (x >> 1) + 1 + 2 * (x & 1)] = to_u8(v.c0);
      return;
    case PixelFormat::kRgb8:
    case PixelFormat::kBgr8: {
      std::uint8_t* q = row + 3 * x;
      const bool bgr = img.format == PixelFormat::kBgr8;
      q[0] = to_u8(bgr ? v.c2 : v.c0);
      q[1] = to_u8(v.c1);
      q[2] = to_u8(bgr ? v.c0 : v.c2);
      return;
    }
    case PixelFormat::kRgba8:
    case PixelFormat::kBgra8: {
      std::uint8_t* q = row + 4 * x;
      const bool bgr = img.format == PixelFormat::kBgra8;
      q[0] = to_u8(bgr ? v.c2 : v.c0);
      q[1] = to_u8(v.c1);
      q[2] = to_u8(bgr ? v.c0 : v.c2);
      q[3] = to_u8(v.a);
      return;
    }
    case PixelFormat::kMono8:
      row[x] = to_u8(v.c0);
      return;
    case PixelFormat::kMono16:
      store_u16(row + 2 * x, v.c0);
      return;
    case PixelFormat::kRgb16: {
      std::uint8_t* q = row + 6 * x;
      store_u16(q, v.c0);
      store_u16(q + 2, v.c1);
      store_u16(q + 4, v.c2);
      return;
    }
  }
}

// Writes the chroma sample (cx, cy) of a YCbCr destination.
CC_HD void store_chroma(const ImageDesc& img, int cx, int cy, float cb, float cr) {
  switch (img.format) {
    case PixelFormat::kNv12:
    case PixelFormat::kNv16: {
      std::uint8_t* q = img.data[1] + static_cast<std::size_t>(cy) * img.pitch[1] + 2 * cx;
      q[0] = to_u8(cb);
      q[1] = to_u8(cr);
      return;
    }
    case PixelFormat::kI420:
      img.data[1][static_cast<std::size_t>(cy) * img.pitch[1] + cx] = to_u8(cb);
      img.data[2][static_cast<std::size_t>(cy) * img.pitch[2] + cx] = to_u8(cr);
      return;
    case PixelFormat::kYuyv: {
      std::uint8_t* q = img.data[0] + static_cast<std::size_t>(cy) * img.pitch[0] + 4 * cx;
      q[1] = to_u8(cb);
      q[3] = to_u8(cr);
      return;
    }
    case PixelFormat::kUyvy: {
      std::uint8_t* q = img.data[0] + static_cast<std::size_t>(cy) * img.pitch[0] + 4 * cx;
      q[0] = to_u8(cb);
      q[2] = to_u8(cr);
      return;
    }
    default:
      return;
  }
}

// Converts block (bx, by) of the destination: one pixel, or for a YCbCr
// destination the pixels sharing chroma sample (bx, by).
CC_HD void convert_block(const Plan& p, int bx, int by) {
  if (!is_yuv(p.dst.format)) {
    store(p.dst, bx, by, pixel(p, bx, by));
    return;
  }
  const int x0 = bx << p.block_shift_x;
  const int y0 = by << p.block_shift_y;
  const int x1 =
      x0 + (1 << p.block_shift_x) < p.dst.width ? x0 + (1 << p.block_shift_x) : p.dst.width;
  const int y1 =
      y0 + (1 << p.block_shift_y) < p.dst.height ? y0 + (1 << p.block_shift_y) : p.dst.height;
  float cb = 0.0F;
  float cr = 0.0F;
  int n = 0;
  for (int y = y0; y < y1; ++y) {
    for (int x = x0; x < x1; ++x) {
      const Sample4 v = pixel(p, x, y);
      store(p.dst, x, y, v);
      cb += v.c1;
      cr += v.c2;
      ++n;
    }
  }
  const auto count = static_cast<float>(n);
  store_chroma(p.dst, bx, by, cb / count, cr / count);
}

// Blocks across and down the destination.
inline int blocks_x(const Plan& p) {
  return (p.dst.width + (1 << p.block_shift_x) - 1) >> p.block_shift_x;
}
inline int blocks_y(const Plan& p) {
  return (p.dst.height + (1 << p.block_shift_y) - 1) >> p.block_shift_y;
}

}  // namespace holoscan::color_convert::detail
