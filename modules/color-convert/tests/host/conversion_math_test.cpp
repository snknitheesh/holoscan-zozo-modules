// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The CPU reference against the colour science (textbook.hpp) and published
// colour bars, without the SDK or a GPU: every format pair, round trips,
// colour bars per matrix and range, odd sizes and chroma-subsampled edges,
// resizing per mode, crops and letterbox geometry. The CUDA kernel computes
// the same bytes as this reference, which the GPU tests check.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include <holoscan/color_convert/conversion.hpp>
#include <holoscan/color_convert/reference.hpp>

#include "test_support.hpp"
#include "textbook.hpp"

namespace holoscan::color_convert {
namespace {

using test::Image;
using test::Px;

constexpr std::array kMatrices{ColorMatrix::kBt601, ColorMatrix::kBt709, ColorMatrix::kBt2020};
constexpr ColorRange kRanges[] = {ColorRange::kLimited, ColorRange::kFull};

std::string describe(PixelFormat from, PixelFormat to, ColorMatrix matrix, ColorRange range) {
  return std::string(format_name(from)) + " -> " + std::string(format_name(to)) + " (" +
         std::string(matrix_name(matrix)) + ", " + std::string(range_name(range)) + ")";
}

// Formats that store the same colour model, where a conversion only moves
// samples and must be exact.
bool same_model(PixelFormat a, PixelFormat b) {
  return format_info(a).model == format_info(b).model;
}

// The largest difference between `dst`'s codes and what converting `src`
// pixel by pixel gives, both block-constant (so subsampling is lossless).
double worst_error(const Image& src, const Image& dst, ColorMatrix matrix, ColorRange range) {
  double worst = 0.0;
  const ColorModel from = format_info(src.format).model;
  for (int y = 0; y < src.height; ++y) {
    for (int x = 0; x < src.width; ++x) {
      const auto want =
          test::expected_codes(test::read(src, x, y), from, dst.format, matrix, range);
      const auto got = test::codes(dst, x, y);
      const int channels = format_info(dst.format).model == ColorModel::kGray ? 1 : 3;
      for (int i = 0; i < channels; ++i) {
        worst = std::max(worst, std::abs(want[i] - got[i]));
      }
      if (has_alpha(dst.format)) {
        worst = std::max(worst, std::abs(want[3] - got[3]));
      }
    }
  }
  return worst;
}

// Every format into every other, for every matrix and range: within one code
// of the colour science (two words for 16-bit formats, whose single-precision
// arithmetic resolves a 257th of a code), and exact between formats of one
// colour model.
void check_every_pair() {
  for (const ColorMatrix matrix : kMatrices) {
    for (const ColorRange range : kRanges) {
      for (const PixelFormat from : kAllFormats) {
        const Image src = test::block_image(from, 34, 18, matrix, range, 7U);
        for (const PixelFormat to : kAllFormats) {
          Image dst(to, src.width, src.height);
          Image in = src;
          reference_convert(in.view(), dst.view(),
                            ConversionParams{.matrix = matrix, .range = range});
          const double tolerance =
              same_model(from, to) ? 0.0 : (format_info(to).bytes_per_sample == 2 ? 2.0 : 1.0);
          const double worst = worst_error(src, dst, matrix, range);
          VERIFY_MSG(worst <= tolerance, describe(from, to, matrix, range) + ": off by " +
                                             std::to_string(worst) + " codes");
        }
      }
    }
  }
}

// Converting to another format and back gives the image back within two
// codes (YCbCr and RGB quantize differently), exactly within a colour model.
// MONO keeps only luma, so it is checked only from MONO and RGB-gray images.
void check_round_trips() {
  for (const ColorMatrix matrix : kMatrices) {
    for (const ColorRange range : kRanges) {
      for (const PixelFormat from : kAllFormats) {
        for (const PixelFormat via : kAllFormats) {
          const bool lossy_gray = format_info(via).model == ColorModel::kGray &&
                                  format_info(from).model != ColorModel::kGray;
          // RGB and BGR drop alpha, and 8-bit formats round 16-bit ones.
          const bool drops_alpha = has_alpha(from) && !has_alpha(via);
          const bool narrows =
              format_info(from).bytes_per_sample == 2 && format_info(via).bytes_per_sample == 1;
          if (lossy_gray || drops_alpha || narrows) {
            continue;
          }
          Image src = test::block_image(from, 18, 10, matrix, range, 11U);
          Image mid(via, src.width, src.height);
          Image back(from, src.width, src.height);
          const ConversionParams params{.matrix = matrix, .range = range};
          reference_convert(src.view(), mid.view(), params);
          reference_convert(mid.view(), back.view(), params);
          int worst = 0;
          for (int y = 0; y < src.height; ++y) {
            for (int x = 0; x < src.width; ++x) {
              const auto a = test::codes(src, x, y);
              const auto b = test::codes(back, x, y);
              for (int i = 0; i < 4; ++i) {
                worst = std::max(worst, std::abs(a[i] - b[i]));
              }
            }
          }
          // 16-bit images go only through 16-bit ones (narrows), within two words.
          const int tolerance = same_model(from, via) ? 0 : 2;
          VERIFY_MSG(worst <= tolerance, describe(from, via, matrix, range) + " and back: off by " +
                                             std::to_string(worst));
        }
      }
    }
  }
}

using Bars = std::array<std::array<int, 3>, 8>;

// 100% colour bars (white, yellow, cyan, green, magenta, red, blue, black).
const Bars& bars_rgb_values() {
  static constexpr Bars kBars{{{255, 255, 255},
                               {255, 255, 0},
                               {0, 255, 255},
                               {0, 255, 0},
                               {255, 0, 255},
                               {255, 0, 0},
                               {0, 0, 255},
                               {0, 0, 0}}};
  return kBars;
}

// Y, Cb, Cr of the 100% bars as published: ITU-R BT.801 for BT.601, SMPTE RP
// 219 for BT.709, and JPEG (JFIF) for BT.601 full range.
const Bars& bt601_limited_bars() {
  static constexpr Bars kBars{{{235, 128, 128},
                               {210, 16, 146},
                               {170, 166, 16},
                               {145, 54, 34},
                               {106, 202, 222},
                               {81, 90, 240},
                               {41, 240, 110},
                               {16, 128, 128}}};
  return kBars;
}

const Bars& bt709_limited_bars() {
  static constexpr Bars kBars{{{235, 128, 128},
                               {219, 16, 138},
                               {188, 154, 16},
                               {173, 42, 26},
                               {78, 214, 230},
                               {63, 102, 240},
                               {32, 240, 118},
                               {16, 128, 128}}};
  return kBars;
}

const Bars& bt601_full_bars() {
  static constexpr Bars kBars{{{255, 128, 128},
                               {226, 0, 149},
                               {179, 171, 0},
                               {150, 44, 21},
                               {105, 212, 235},
                               {76, 85, 255},
                               {29, 255, 107},
                               {0, 128, 128}}};
  return kBars;
}

Px px(const std::array<int, 3>& c) {
  return Px{{static_cast<double>(c[0]), static_cast<double>(c[1]), static_cast<double>(c[2])}};
}

// Bars two pixels wide and two high, so 4:2:0 keeps each bar's chroma.
Image bars_rgb() {
  Image img(PixelFormat::kRgb8, 16, 2);
  for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 16; ++x) {
      test::write(img, x, y, px(bars_rgb_values()[x / 2]));
    }
  }
  return img;
}

void check_bars_into(PixelFormat format, ColorMatrix matrix, ColorRange range,
                     const Bars* published) {
  Image rgb = bars_rgb();
  Image yuv(format, rgb.width, rgb.height);
  reference_convert(rgb.view(), yuv.view(), ConversionParams{.matrix = matrix, .range = range});
  for (int bar = 0; bar < 8; ++bar) {
    const Px got = test::read(yuv, 2 * bar, 1);
    const Px want = test::rgb_to_ycc(px(bars_rgb_values()[bar]), matrix, range);
    for (int i = 0; i < 3; ++i) {
      const double rounded = std::clamp(std::floor(want.c[i] + 0.5), 0.0, 255.0);
      // A sample exactly half way between two codes (full-range yellow's Cb and
      // cyan's Cr are 0.5) may round either way in single precision; JPEG
      // publishes 0 for both.
      const bool tie = std::abs(want.c[i] - std::floor(want.c[i]) - 0.5) < 1e-6;
      const double slack = tie ? 1.0 : 0.0;
      VERIFY_MSG(std::abs(got.c[i] - rounded) <= slack,
                 describe(PixelFormat::kRgb8, format, matrix, range) + " bar " +
                     std::to_string(bar) + " sample " + std::to_string(i) + " is " +
                     std::to_string(got.c[i]) + ", the matrix gives " + std::to_string(rounded));
      if (published != nullptr) {
        VERIFY_MSG(std::abs(got.c[i] - (*published)[bar][i]) <= slack,
                   describe(PixelFormat::kRgb8, format, matrix, range) + " bar " +
                       std::to_string(bar) + " sample " + std::to_string(i) + " is " +
                       std::to_string(got.c[i]) + ", published " +
                       std::to_string((*published)[bar][i]));
      }
    }
  }
  // And back: each bar's YCbCr gives its RGB within one code.
  Image back(PixelFormat::kRgb8, rgb.width, rgb.height);
  reference_convert(yuv.view(), back.view(), ConversionParams{.matrix = matrix, .range = range});
  for (int bar = 0; bar < 8; ++bar) {
    const Px got = test::read(back, 2 * bar, 0);
    for (int i = 0; i < 3; ++i) {
      VERIFY_MSG(std::abs(got.c[i] - bars_rgb_values()[bar][i]) <= 1,
                 describe(format, PixelFormat::kRgb8, matrix, range) + " bar " +
                     std::to_string(bar) + " comes back as " + std::to_string(got.c[i]));
    }
  }
}

void check_colour_bars() {
  for (const PixelFormat format : {PixelFormat::kNv12, PixelFormat::kNv16, PixelFormat::kI420,
                                   PixelFormat::kYuyv, PixelFormat::kUyvy}) {
    check_bars_into(format, ColorMatrix::kBt601, ColorRange::kLimited, &bt601_limited_bars());
    check_bars_into(format, ColorMatrix::kBt709, ColorRange::kLimited, &bt709_limited_bars());
    check_bars_into(format, ColorMatrix::kBt601, ColorRange::kFull, &bt601_full_bars());
    check_bars_into(format, ColorMatrix::kBt709, ColorRange::kFull, nullptr);
    check_bars_into(format, ColorMatrix::kBt2020, ColorRange::kLimited, nullptr);
    check_bars_into(format, ColorMatrix::kBt2020, ColorRange::kFull, nullptr);
  }
}

// Odd widths and heights: the chroma planes round up, and the last column's
// and row's chroma sample belongs to them alone.
void check_odd_sizes() {
  VERIFY(packed_size(PixelFormat::kNv12, 5, 3) == 5 * 3 + 3 * 2 * 2);
  VERIFY(packed_size(PixelFormat::kI420, 5, 3) == 5 * 3 + 2 * 3 * 2);
  VERIFY(packed_size(PixelFormat::kNv16, 5, 3) == 5 * 3 + 3 * 2 * 3);
  VERIFY(plane_size(PixelFormat::kNv12, 1919, 1079, 1).row_bytes == 960 * 2);
  VERIFY(plane_size(PixelFormat::kNv12, 1919, 1079, 1).rows == 540);
  for (const ColorMatrix matrix : kMatrices) {
    for (const PixelFormat from : kAllFormats) {
      for (const PixelFormat to : kAllFormats) {
        const int width = needs_even_width(from) || needs_even_width(to) ? 6 : 7;
        for (const int height : {1, 5}) {
          const Image src =
              test::block_image(from, width, height, matrix, ColorRange::kLimited, 3U);
          Image in = src;
          Image dst(to, width, height);
          reference_convert(in.view(), dst.view(),
                            ConversionParams{.matrix = matrix, .range = ColorRange::kLimited});
          const double tolerance =
              same_model(from, to) ? 0.0 : (format_info(to).bytes_per_sample == 2 ? 2.0 : 1.0);
          const double worst = worst_error(src, dst, matrix, ColorRange::kLimited);
          VERIFY_MSG(worst <= tolerance, describe(from, to, matrix, ColorRange::kLimited) + " " +
                                             std::to_string(width) + "x" + std::to_string(height) +
                                             ": off by " + std::to_string(worst));
        }
      }
    }
  }
  // A subsampled destination averages its chroma over the pixels it covers:
  // a 2x2 block of two colours stores their mean, a partial block its own.
  Image rgb(PixelFormat::kRgb8, 3, 1);
  test::write(rgb, 0, 0, Px{{255, 0, 0}});
  test::write(rgb, 1, 0, Px{{0, 0, 255}});
  test::write(rgb, 2, 0, Px{{0, 255, 0}});
  Image nv12(PixelFormat::kNv12, 3, 1);
  reference_convert(rgb.view(), nv12.view(), ConversionParams{});
  const Px red = test::rgb_to_ycc(Px{{255, 0, 0}}, ColorMatrix::kBt601, ColorRange::kLimited);
  const Px blue = test::rgb_to_ycc(Px{{0, 0, 255}}, ColorMatrix::kBt601, ColorRange::kLimited);
  const Px green = test::rgb_to_ycc(Px{{0, 255, 0}}, ColorMatrix::kBt601, ColorRange::kLimited);
  const Px first = test::read(nv12, 0, 0);
  const Px last = test::read(nv12, 2, 0);
  VERIFY(std::abs(first.c[1] - (red.c[1] + blue.c[1]) / 2.0) <= 0.5 + 1e-9);
  VERIFY(std::abs(first.c[2] - (red.c[2] + blue.c[2]) / 2.0) <= 0.5 + 1e-9);
  VERIFY(std::abs(last.c[1] - green.c[1]) <= 0.5 + 1e-9);
  VERIFY(std::abs(last.c[2] - green.c[2]) <= 0.5 + 1e-9);
}

// A smooth RGB8 test image with detail at every scale.
Image smooth_rgb(int width, int height) {
  Image img(PixelFormat::kRgb8, width, height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const double r = 127.5 + 127.5 * std::sin(0.11 * x + 0.05 * y);
      const double g = 255.0 * x / std::max(1, width - 1);
      const double b = ((x / 3 + y / 5) % 2 == 0) ? 40.0 : 210.0;
      test::write(img, x, y, Px{{std::floor(r), std::floor(g), b}});
    }
  }
  return img;
}

// Whether nearest's footprint center lies within `epsilon` of a pixel edge,
// where single and double precision may pick neighbouring pixels.
bool on_edge(double u, double epsilon = 1e-4) { return std::abs(u - std::round(u)) < epsilon; }

// Every mode, up and down, with and without crop and letterbox, against the
// double-precision definition: nearest exactly (barring footprint centers on
// a pixel edge), bilinear and area within one code.
void check_resize() {
  const Image src = smooth_rgb(53, 41);
  struct Case {
    int width;
    int height;
    std::optional<Rect> crop;
    bool letterbox;
  };
  const Case cases[] = {
      {53, 41, std::nullopt, false},        {26, 20, std::nullopt, false},
      {17, 13, std::nullopt, false},        {120, 77, std::nullopt, false},
      {40, 40, Rect{5, 3, 30, 21}, false},  {64, 32, std::nullopt, true},
      {31, 60, Rect{10, 10, 33, 20}, true}, {7, 5, std::nullopt, false},
  };
  for (const ResizeMode mode : {ResizeMode::kNearest, ResizeMode::kBilinear, ResizeMode::kArea}) {
    for (const Case& c : cases) {
      const ConversionParams params{
          .mode = mode, .crop = c.crop, .letterbox = c.letterbox, .fill_rgb = {10, 20, 30}};
      Image in = src;
      Image dst(PixelFormat::kRgb8, c.width, c.height);
      reference_convert(in.view(), dst.view(), params);
      const Mapping m =
          compute_mapping(src.width, src.height, c.width, c.height, PixelFormat::kRgb8, params);
      int wrong = 0;
      for (int y = 0; y < c.height; ++y) {
        for (int x = 0; x < c.width; ++x) {
          const Px got = test::read(dst, x, y);
          const bool inside = x >= m.content.x && x < m.content.x + m.content.width &&
                              y >= m.content.y && y < m.content.y + m.content.height;
          if (!inside) {
            wrong += got.c[0] != 10 || got.c[1] != 20 || got.c[2] != 30;
            continue;
          }
          const auto want = test::resized_rgb(src, m, mode, x, y);
          if (mode == ResizeMode::kNearest) {
            const double u = m.source.x + (x - m.content.x + 0.5) * m.source.width /
                                              static_cast<double>(m.content.width);
            const double v = m.source.y + (y - m.content.y + 0.5) * m.source.height /
                                              static_cast<double>(m.content.height);
            if (on_edge(u) || on_edge(v)) continue;
          }
          const double tolerance = mode == ResizeMode::kNearest ? 0.0 : 1.0;
          for (int i = 0; i < 3; ++i) {
            if (std::abs(got.c[i] - std::floor(want[i] + 0.5)) > tolerance) {
              ++wrong;
              break;
            }
          }
        }
      }
      VERIFY_MSG(wrong == 0, std::string(resize_mode_name(mode)) + " to " +
                                 std::to_string(c.width) + "x" + std::to_string(c.height) + ": " +
                                 std::to_string(wrong) + " pixels differ from the definition");
    }
  }
}

// Resizing a YCbCr image, and resizing into one, follow the same definition
// in its own samples: compare against the RGB8 path through exact conversions.
void check_fused_matches_two_passes() {
  for (const ResizeMode mode : {ResizeMode::kNearest, ResizeMode::kBilinear, ResizeMode::kArea}) {
    for (const PixelFormat format :
         {PixelFormat::kRgba8, PixelFormat::kBgr8, PixelFormat::kBgra8}) {
      // Fused RGB8 -> format with a resize equals a resize to RGB8, then an exact
      // conversion to format within the RGB model.
      Image src = smooth_rgb(40, 30);
      const ConversionParams params{.mode = mode};
      Image fused(format, 24, 18);
      reference_convert(src.view(), fused.view(), params);
      Image resized(PixelFormat::kRgb8, 24, 18);
      reference_convert(src.view(), resized.view(), params);
      Image converted(format, 24, 18);
      reference_convert(resized.view(), converted.view(), ConversionParams{});
      VERIFY_MSG(fused.bytes == converted.bytes, std::string(resize_mode_name(mode)) + " into " +
                                                     std::string(format_name(format)) +
                                                     " differs from two passes");
    }
  }
}

// Letterbox geometry: the content keeps the source's aspect ratio, is
// centered, and is aligned to chroma samples where it leaves a border.
void check_letterbox_geometry() {
  const ConversionParams box{.letterbox = true};
  Mapping m = compute_mapping(1920, 1080, 640, 640, PixelFormat::kRgb8, box);
  VERIFY(m.content == (Rect{0, 140, 640, 360}));
  VERIFY(m.source == (Rect{0, 0, 1920, 1080}));
  m = compute_mapping(1080, 1920, 640, 640, PixelFormat::kRgb8, box);
  VERIFY(m.content == (Rect{140, 0, 360, 640}));
  // 1000x600 into 333x333: content 333x200, offset (0, 66); NV12 makes the
  // bordered axis even: 200 high at row 66.
  m = compute_mapping(1000, 600, 333, 333, PixelFormat::kRgb8, box);
  VERIFY(m.content == (Rect{0, 66, 333, 200}));
  m = compute_mapping(1000, 600, 333, 333, PixelFormat::kNv12, box);
  VERIFY(m.content == (Rect{0, 66, 333, 200}));
  m = compute_mapping(1000, 601, 334, 335, PixelFormat::kNv12, box);
  VERIFY(m.content.width == 334 && m.content.x == 0);
  VERIFY(m.content.height % 2 == 0 && m.content.y % 2 == 0);
  // YUYV subsamples only horizontally.
  m = compute_mapping(601, 1000, 335, 334, PixelFormat::kYuyv, box);
  VERIFY(m.content.width % 2 == 0 && m.content.x % 2 == 0);
  VERIFY(m.content.height == 334);
  // Same aspect ratio: no border.
  m = compute_mapping(1280, 720, 640, 360, PixelFormat::kNv12, box);
  VERIFY(m.content == (Rect{0, 0, 640, 360}));
  // Crop then letterbox.
  m = compute_mapping(1920, 1080, 300, 300, PixelFormat::kRgb8,
                      ConversionParams{.crop = Rect{100, 100, 400, 200}, .letterbox = true});
  VERIFY(m.source == (Rect{100, 100, 400, 200}));
  VERIFY(m.content == (Rect{0, 75, 300, 150}));
  // Without letterbox the content is everything.
  m = compute_mapping(1920, 1080, 640, 640, PixelFormat::kRgb8, ConversionParams{});
  VERIFY(m.content == (Rect{0, 0, 640, 640}));
  // A crop outside the source is refused.
  VERIFY(test::throws<std::invalid_argument>([] {
    (void)compute_mapping(100, 100, 10, 10, PixelFormat::kRgb8,
                          ConversionParams{.crop = Rect{90, 0, 20, 10}});
  }));
  VERIFY(test::throws<std::invalid_argument>([] {
    (void)compute_mapping(100, 100, 10, 10, PixelFormat::kRgb8,
                          ConversionParams{.crop = Rect{0, 0, 0, 10}});
  }));
  // A letterboxed NV12 image fills its border with the fill colour's YCbCr.
  Image src = smooth_rgb(64, 32);
  Image dst(PixelFormat::kNv12, 32, 32);
  reference_convert(src.view(), dst.view(),
                    ConversionParams{.letterbox = true, .fill_rgb = {0, 0, 255}});
  const Px fill = test::read(dst, 0, 0);
  const Px blue = test::rgb_to_ycc(Px{{0, 0, 255}}, ColorMatrix::kBt601, ColorRange::kLimited);
  for (int i = 0; i < 3; ++i) {
    VERIFY(fill.c[i] == std::floor(blue.c[i] + 0.5));
  }
}

// Unusable images are refused before anything is written.
void check_rejects() {
  Image yuyv(PixelFormat::kYuyv, 4, 2);
  ImageView odd = yuyv.view();
  odd.width = 3;
  Image rgb(PixelFormat::kRgb8, 3, 2);
  VERIFY(test::throws<std::invalid_argument>(
      [&] { reference_convert(odd, rgb.view(), ConversionParams{}); }));
  ImageView short_pitch = rgb.view();
  short_pitch.planes[0].pitch = 8;
  VERIFY(test::throws<std::invalid_argument>(
      [&] { reference_convert(yuyv.view(), short_pitch, ConversionParams{}); }));
  ImageView device = rgb.view();
  device.location = MemoryLocation::kDevice;
  VERIFY(test::throws<std::invalid_argument>(
      [&] { reference_convert(yuyv.view(), device, ConversionParams{}); }));
}

// The same input gives the same bytes every time.
void check_determinism() {
  Image src = smooth_rgb(37, 29);
  Image a(PixelFormat::kNv12, 20, 16);
  Image b(PixelFormat::kNv12, 20, 16);
  const ConversionParams params{.matrix = ColorMatrix::kBt709, .mode = ResizeMode::kArea};
  reference_convert(src.view(), a.view(), params);
  reference_convert(src.view(), b.view(), params);
  VERIFY(a.bytes == b.bytes);
}

// Pitched rows are read and written where their pitch says.
void check_pitch() {
  Image packed =
      test::block_image(PixelFormat::kNv12, 10, 6, ColorMatrix::kBt601, ColorRange::kLimited, 5U);
  // The same image with 16 padding bytes after each row of each plane.
  std::vector<std::uint8_t> padded(packed.bytes.size() + 16 * (6 + 3), 0xEE);
  ImageView view{.format = PixelFormat::kNv12, .width = 10, .height = 6};
  view.location = MemoryLocation::kHost;
  view.planes[0] = {padded.data(), 26};
  view.planes[1] = {padded.data() + 26 * 6, 26};
  for (int y = 0; y < 6; ++y) {
    std::copy_n(test::at(packed, 0, y, 0), 10, view.planes[0].data + y * 26);
  }
  for (int y = 0; y < 3; ++y) {
    std::copy_n(test::at(packed, 1, y, 0), 10, view.planes[1].data + y * 26);
  }
  Image from_packed(PixelFormat::kRgb8, 10, 6);
  Image from_padded(PixelFormat::kRgb8, 10, 6);
  reference_convert(packed.view(), from_packed.view(), ConversionParams{});
  reference_convert(view, from_padded.view(), ConversionParams{});
  VERIFY(from_packed.bytes == from_padded.bytes);
}

void check_names() {
  for (const PixelFormat format : kAllFormats) {
    VERIFY(parse_format(format_name(format)) == format);
  }
  for (const ColorMatrix matrix : kMatrices) {
    VERIFY(parse_matrix(matrix_name(matrix)) == matrix);
  }
  for (const ColorRange range : kRanges) {
    VERIFY(parse_range(range_name(range)) == range);
  }
  for (const ResizeMode mode : {ResizeMode::kNearest, ResizeMode::kBilinear, ResizeMode::kArea}) {
    VERIFY(parse_resize_mode(resize_mode_name(mode)) == mode);
  }
  VERIFY(!parse_format("nv21").has_value());
  VERIFY(!parse_matrix("bt2100").has_value());
}

}  // namespace
}  // namespace holoscan::color_convert

int main() {
  namespace cc = holoscan::color_convert;
  return cc::test::run({
      {"every format pair against the colour science", cc::check_every_pair},
      {"round trips", cc::check_round_trips},
      {"colour bars per matrix and range", cc::check_colour_bars},
      {"odd sizes and chroma-subsampled edges", cc::check_odd_sizes},
      {"resize against the definition per mode", cc::check_resize},
      {"fused conversion equals two passes", cc::check_fused_matches_two_passes},
      {"letterbox geometry", cc::check_letterbox_geometry},
      {"unusable images refused", cc::check_rejects},
      {"determinism", cc::check_determinism},
      {"pitched rows", cc::check_pitch},
      {"names", cc::check_names},
  });
}
