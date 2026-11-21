// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The tests' own account of the colour science, in double precision and
// written apart from the module's code: the BT.601, BT.709 and BT.2020
// matrices from their luma weights, every format's byte layout, and resizing.
// The module's reference and kernel are checked against it. Needs no SDK,
// CUDA or GPU.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <holoscan/color_convert/conversion.hpp>
#include <holoscan/color_convert/image_view.hpp>
#include <holoscan/color_convert/pixel_format.hpp>

namespace holoscan::color_convert::test {

// A pixel in its colour model's samples on the 8-bit scale: Y, Cb, Cr; R, G,
// B; or luma in c[0]. Alpha is 255 where a format has none.
struct Px {
  std::array<double, 3> c{};
  double a{255.0};
};

struct Weights {
  double kr;
  double kb;
};

inline Weights weights(ColorMatrix matrix) {
  // ITU-R BT.601-7, BT.709-6 and BT.2020-2 (non-constant luminance).
  switch (matrix) {
    case ColorMatrix::kBt601:
      return {0.299, 0.114};
    case ColorMatrix::kBt709:
      return {0.2126, 0.0722};
    case ColorMatrix::kBt2020:
      return {0.2627, 0.0593};
  }
  return {0.299, 0.114};
}

// R, G, B on 0..255 to Y, Cb, Cr codes.
inline Px rgb_to_ycc(const Px& rgb, ColorMatrix matrix, ColorRange range) {
  const auto [kr, kb] = weights(matrix);
  const double kg = 1.0 - kr - kb;
  const double r = rgb.c[0] / 255.0;
  const double g = rgb.c[1] / 255.0;
  const double b = rgb.c[2] / 255.0;
  const double y = kr * r + kg * g + kb * b;
  const double cb = (b - y) / (2.0 * (1.0 - kb));
  const double cr = (r - y) / (2.0 * (1.0 - kr));
  if (range == ColorRange::kLimited) {
    return {{16.0 + 219.0 * y, 128.0 + 224.0 * cb, 128.0 + 224.0 * cr}, rgb.a};
  }
  return {{255.0 * y, 128.0 + 255.0 * cb, 128.0 + 255.0 * cr}, rgb.a};
}

// Y, Cb, Cr codes to R, G, B on 0..255, by inverting rgb_to_ycc.
inline Px ycc_to_rgb(const Px& ycc, ColorMatrix matrix, ColorRange range) {
  const auto [kr, kb] = weights(matrix);
  const double kg = 1.0 - kr - kb;
  const bool limited = range == ColorRange::kLimited;
  const double y = limited ? (ycc.c[0] - 16.0) / 219.0 : ycc.c[0] / 255.0;
  const double cb = (ycc.c[1] - 128.0) / (limited ? 224.0 : 255.0);
  const double cr = (ycc.c[2] - 128.0) / (limited ? 224.0 : 255.0);
  const double r = y + 2.0 * (1.0 - kr) * cr;
  const double b = y + 2.0 * (1.0 - kb) * cb;
  const double g = (y - kr * r - kb * b) / kg;
  return {{255.0 * r, 255.0 * g, 255.0 * b}, ycc.a};
}

// `v` in model `from` converted to model `to`.
inline Px convert_px(const Px& v, ColorModel from, ColorModel to, ColorMatrix matrix,
                     ColorRange range) {
  if (from == to) {
    return v;
  }
  Px rgb = v;
  if (from == ColorModel::kYuv) {
    rgb = ycc_to_rgb(v, matrix, range);
  } else if (from == ColorModel::kGray) {
    rgb = {{v.c[0], v.c[0], v.c[0]}, v.a};
  }
  if (to == ColorModel::kRgb) {
    return rgb;
  }
  if (to == ColorModel::kYuv) {
    return rgb_to_ycc(rgb, matrix, range);
  }
  const auto [kr, kb] = weights(matrix);
  return {{kr * rgb.c[0] + (1.0 - kr - kb) * rgb.c[1] + kb * rgb.c[2], 0.0, 0.0}, rgb.a};
}

// A packed host image whose bytes the test owns.
struct Image {
  PixelFormat format{PixelFormat::kRgb8};
  int width{};
  int height{};
  std::vector<std::uint8_t> bytes;

  Image() = default;
  Image(PixelFormat f, int w, int h)
      : format(f), width(w), height(h), bytes(packed_size(f, w, h), 0) {}

  [[nodiscard]] ImageView view() {
    return packed_view(bytes.data(), format, width, height, MemoryLocation::kHost);
  }
  // Offset of plane `plane` and its packed row.
  [[nodiscard]] std::size_t plane_offset(int plane) const {
    std::size_t offset = 0;
    for (int p = 0; p < plane; ++p) {
      const PlaneSize size = plane_size(format, width, height, p);
      offset += size.row_bytes * static_cast<std::size_t>(size.rows);
    }
    return offset;
  }
  [[nodiscard]] std::size_t row_bytes(int plane) const {
    return plane_size(format, width, height, plane).row_bytes;
  }
};

inline std::uint8_t* at(Image& img, int plane, int row, std::size_t byte) {
  return img.bytes.data() + img.plane_offset(plane) +
         static_cast<std::size_t>(row) * img.row_bytes(plane) + byte;
}
inline const std::uint8_t* at(const Image& img, int plane, int row, std::size_t byte) {
  return img.bytes.data() + img.plane_offset(plane) +
         static_cast<std::size_t>(row) * img.row_bytes(plane) + byte;
}

inline double read16(const std::uint8_t* p) {
  return static_cast<double>(p[0] | (p[1] << 8)) / 257.0;
}
inline std::uint16_t raw16(const std::uint8_t* p) {
  return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

// Pixel (x, y) in its format's samples: the chroma is that of the sample
// covering it.
inline Px read(const Image& img, int x, int y) {
  const auto b = [](std::uint8_t v) { return static_cast<double>(v); };
  switch (img.format) {
    case PixelFormat::kNv12:
    case PixelFormat::kNv16: {
      const int cy = img.format == PixelFormat::kNv12 ? y / 2 : y;
      const std::uint8_t* uv = at(img, 1, cy, 2 * static_cast<std::size_t>(x / 2));
      return {{b(*at(img, 0, y, x)), b(uv[0]), b(uv[1])}};
    }
    case PixelFormat::kI420:
      return {{b(*at(img, 0, y, x)), b(*at(img, 1, y / 2, x / 2)), b(*at(img, 2, y / 2, x / 2))}};
    case PixelFormat::kYuyv: {
      const std::uint8_t* q = at(img, 0, y, 4 * static_cast<std::size_t>(x / 2));
      return {{b(q[x % 2 == 0 ? 0 : 2]), b(q[1]), b(q[3])}};
    }
    case PixelFormat::kUyvy: {
      const std::uint8_t* q = at(img, 0, y, 4 * static_cast<std::size_t>(x / 2));
      return {{b(q[x % 2 == 0 ? 1 : 3]), b(q[0]), b(q[2])}};
    }
    case PixelFormat::kRgb8: {
      const std::uint8_t* q = at(img, 0, y, 3 * static_cast<std::size_t>(x));
      return {{b(q[0]), b(q[1]), b(q[2])}};
    }
    case PixelFormat::kBgr8: {
      const std::uint8_t* q = at(img, 0, y, 3 * static_cast<std::size_t>(x));
      return {{b(q[2]), b(q[1]), b(q[0])}};
    }
    case PixelFormat::kRgba8: {
      const std::uint8_t* q = at(img, 0, y, 4 * static_cast<std::size_t>(x));
      return {{b(q[0]), b(q[1]), b(q[2])}, b(q[3])};
    }
    case PixelFormat::kBgra8: {
      const std::uint8_t* q = at(img, 0, y, 4 * static_cast<std::size_t>(x));
      return {{b(q[2]), b(q[1]), b(q[0])}, b(q[3])};
    }
    case PixelFormat::kMono8:
      return {{b(*at(img, 0, y, x)), 0.0, 0.0}};
    case PixelFormat::kMono16:
      return {{read16(at(img, 0, y, 2 * static_cast<std::size_t>(x))), 0.0, 0.0}};
    case PixelFormat::kRgb16: {
      const std::uint8_t* q = at(img, 0, y, 6 * static_cast<std::size_t>(x));
      return {{read16(q), read16(q + 2), read16(q + 4)}};
    }
  }
  return {};
}

// The stored codes of pixel (x, y), as read() orders them, unscaled: 16-bit
// formats give their words.
inline std::array<int, 4> codes(const Image& img, int x, int y) {
  if (img.format == PixelFormat::kMono16) {
    return {raw16(at(img, 0, y, 2 * static_cast<std::size_t>(x))), 0, 0, 255};
  }
  if (img.format == PixelFormat::kRgb16) {
    const std::uint8_t* q = at(img, 0, y, 6 * static_cast<std::size_t>(x));
    return {raw16(q), raw16(q + 2), raw16(q + 4), 255};
  }
  const Px p = read(img, x, y);
  return {static_cast<int>(p.c[0]), static_cast<int>(p.c[1]), static_cast<int>(p.c[2]),
          static_cast<int>(p.a)};
}

inline std::uint8_t q8(double v) { return static_cast<std::uint8_t>(std::clamp(v, 0.0, 255.0)); }
inline void write16(std::uint8_t* p, double v8) {
  const auto w = static_cast<std::uint16_t>(std::clamp(std::floor(v8 * 257.0 + 0.5), 0.0, 65535.0));
  p[0] = static_cast<std::uint8_t>(w & 0xFFU);
  p[1] = static_cast<std::uint8_t>(w >> 8);
}

// Writes pixel (x, y) from whole sample values; chroma goes to the sample
// covering the pixel, so a pixel written later in its block wins.
inline void write(Image& img, int x, int y, const Px& v) {
  const auto u = [](double s) { return q8(s); };
  switch (img.format) {
    case PixelFormat::kNv12:
    case PixelFormat::kNv16: {
      *at(img, 0, y, x) = u(v.c[0]);
      const int cy = img.format == PixelFormat::kNv12 ? y / 2 : y;
      std::uint8_t* uv = at(img, 1, cy, 2 * static_cast<std::size_t>(x / 2));
      uv[0] = u(v.c[1]);
      uv[1] = u(v.c[2]);
      return;
    }
    case PixelFormat::kI420:
      *at(img, 0, y, x) = u(v.c[0]);
      *at(img, 1, y / 2, x / 2) = u(v.c[1]);
      *at(img, 2, y / 2, x / 2) = u(v.c[2]);
      return;
    case PixelFormat::kYuyv: {
      std::uint8_t* q = at(img, 0, y, 4 * static_cast<std::size_t>(x / 2));
      q[x % 2 == 0 ? 0 : 2] = u(v.c[0]);
      q[1] = u(v.c[1]);
      q[3] = u(v.c[2]);
      return;
    }
    case PixelFormat::kUyvy: {
      std::uint8_t* q = at(img, 0, y, 4 * static_cast<std::size_t>(x / 2));
      q[x % 2 == 0 ? 1 : 3] = u(v.c[0]);
      q[0] = u(v.c[1]);
      q[2] = u(v.c[2]);
      return;
    }
    case PixelFormat::kRgb8:
    case PixelFormat::kBgr8: {
      std::uint8_t* q = at(img, 0, y, 3 * static_cast<std::size_t>(x));
      const bool bgr = img.format == PixelFormat::kBgr8;
      q[0] = u(v.c[bgr ? 2 : 0]);
      q[1] = u(v.c[1]);
      q[2] = u(v.c[bgr ? 0 : 2]);
      return;
    }
    case PixelFormat::kRgba8:
    case PixelFormat::kBgra8: {
      std::uint8_t* q = at(img, 0, y, 4 * static_cast<std::size_t>(x));
      const bool bgr = img.format == PixelFormat::kBgra8;
      q[0] = u(v.c[bgr ? 2 : 0]);
      q[1] = u(v.c[1]);
      q[2] = u(v.c[bgr ? 0 : 2]);
      q[3] = u(v.a);
      return;
    }
    case PixelFormat::kMono8:
      *at(img, 0, y, x) = u(v.c[0]);
      return;
    case PixelFormat::kMono16:
      write16(at(img, 0, y, 2 * static_cast<std::size_t>(x)), v.c[0]);
      return;
    case PixelFormat::kRgb16: {
      std::uint8_t* q = at(img, 0, y, 6 * static_cast<std::size_t>(x));
      write16(q, v.c[0]);
      write16(q + 2, v.c[1]);
      write16(q + 4, v.c[2]);
      return;
    }
  }
}

// A random image whose pixels are constant over each 2x2 block, so that
// chroma subsampling loses nothing: a conversion's every pixel can be
// checked on its own. YCbCr samples stay inside their range's legal codes,
// whose colours RGB can hold, so nothing clips.
inline Image block_image(PixelFormat format, int width, int height, ColorMatrix matrix,
                         ColorRange range, std::uint32_t seed) {
  Image img(format, width, height);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> byte(0, 255);
  const ColorModel model = format_info(format).model;
  for (int by = 0; by < (height + 1) / 2; ++by) {
    for (int bx = 0; bx < (width + 1) / 2; ++bx) {
      Px v;
      const Px rgb{{static_cast<double>(byte(rng)), static_cast<double>(byte(rng)),
                    static_cast<double>(byte(rng))},
                   has_alpha(format) ? static_cast<double>(byte(rng)) : 255.0};
      if (model == ColorModel::kYuv) {
        const Px ycc = rgb_to_ycc(rgb, matrix, range);
        v = {{std::floor(ycc.c[0] + 0.5), std::floor(ycc.c[1] + 0.5), std::floor(ycc.c[2] + 0.5)},
             255.0};
      } else if (model == ColorModel::kGray) {
        v = {{rgb.c[0], 0.0, 0.0}, 255.0};
      } else {
        v = rgb;
      }
      for (int y = 2 * by; y < std::min(2 * by + 2, height); ++y) {
        for (int x = 2 * bx; x < std::min(2 * bx + 2, width); ++x) {
          write(img, x, y, v);
        }
      }
    }
  }
  return img;
}

// The codes a conversion of `v`, in model `from`, into `format` should store.
inline std::array<double, 4> expected_codes(const Px& v, ColorModel from, PixelFormat format,
                                            ColorMatrix matrix, ColorRange range) {
  const Px out = convert_px(v, from, format_info(format).model, matrix, range);
  const double scale = format_info(format).bytes_per_sample == 2 ? 257.0 : 1.0;
  const double top = scale * 255.0;
  std::array<double, 4> result{};
  for (int i = 0; i < 3; ++i) {
    result[i] = std::clamp(std::floor(out.c[i] * scale + 0.5), 0.0, top);
  }
  result[3] = has_alpha(format) ? out.a : 255.0;
  return result;
}

// Resizing as conversion.hpp defines it, in double precision, for an RGB8
// image: the colour at destination pixel (dx, dy) of the content rectangle.
inline std::array<double, 3> resized_rgb(const Image& src, const Mapping& m, ResizeMode mode,
                                         int dx, int dy) {
  const double sx = static_cast<double>(m.source.width) / m.content.width;
  const double sy = static_cast<double>(m.source.height) / m.content.height;
  const double rx = dx - m.content.x;
  const double ry = dy - m.content.y;
  const int x0 = m.source.x;
  const int y0 = m.source.y;
  const int x1 = m.source.x + m.source.width - 1;
  const int y1 = m.source.y + m.source.height - 1;
  const auto pick = [&](int x, int y) {
    const Px p = read(src, std::clamp(x, x0, x1), std::clamp(y, y0, y1));
    return p.c;
  };
  if (mode == ResizeMode::kNearest) {
    return pick(static_cast<int>(std::floor(x0 + (rx + 0.5) * sx)),
                static_cast<int>(std::floor(y0 + (ry + 0.5) * sy)));
  }
  std::array<double, 3> sum{};
  if (mode == ResizeMode::kBilinear) {
    const double fx = x0 + (rx + 0.5) * sx - 0.5;
    const double fy = y0 + (ry + 0.5) * sy - 0.5;
    const int ix = static_cast<int>(std::floor(fx));
    const int iy = static_cast<int>(std::floor(fy));
    const double ax = fx - ix;
    const double ay = fy - iy;
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 2; ++i) {
        const double w = (i == 0 ? 1.0 - ax : ax) * (j == 0 ? 1.0 - ay : ay);
        const auto c = pick(ix + i, iy + j);
        for (int k = 0; k < 3; ++k) sum[k] += w * c[k];
      }
    }
    return sum;
  }
  const double lx = x0 + rx * sx;
  const double ly = y0 + ry * sy;
  const double hx = std::min(lx + sx, static_cast<double>(x1 + 1));
  const double hy = std::min(ly + sy, static_cast<double>(y1 + 1));
  double total = 0.0;
  for (int y = static_cast<int>(std::floor(ly)); y < hy; ++y) {
    const double wy = std::min<double>(y + 1, hy) - std::max<double>(y, ly);
    for (int x = static_cast<int>(std::floor(lx)); x < hx; ++x) {
      const double w = (std::min<double>(x + 1, hx) - std::max<double>(x, lx)) * wy;
      if (w <= 0.0) continue;
      const auto c = pick(x, y);
      for (int k = 0; k < 3; ++k) sum[k] += w * c[k];
      total += w;
    }
  }
  for (double& v : sum) v /= total;
  return sum;
}

}  // namespace holoscan::color_convert::test
