// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Format and setting names, the geometry of a conversion and its plan: what
// every backend and the reference share. Needs no SDK, CUDA or GPU.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/color_convert/backend.hpp>

#include "kernel_math.hpp"

namespace holoscan::color_convert {
namespace {

constexpr std::array<std::pair<PixelFormat, std::string_view>, 12> kFormatNames{{
    {PixelFormat::kNv12, "nv12"},
    {PixelFormat::kNv16, "nv16"},
    {PixelFormat::kI420, "i420"},
    {PixelFormat::kYuyv, "yuyv"},
    {PixelFormat::kUyvy, "uyvy"},
    {PixelFormat::kRgb8, "rgb8"},
    {PixelFormat::kBgr8, "bgr8"},
    {PixelFormat::kRgba8, "rgba8"},
    {PixelFormat::kBgra8, "bgra8"},
    {PixelFormat::kMono8, "mono8"},
    {PixelFormat::kMono16, "mono16"},
    {PixelFormat::kRgb16, "rgb16"},
}};

constexpr std::array<std::pair<ColorMatrix, std::string_view>, 3> kMatrixNames{{
    {ColorMatrix::kBt601, "bt601"},
    {ColorMatrix::kBt709, "bt709"},
    {ColorMatrix::kBt2020, "bt2020"},
}};

constexpr std::array<std::pair<ColorRange, std::string_view>, 2> kRangeNames{{
    {ColorRange::kLimited, "limited"},
    {ColorRange::kFull, "full"},
}};

constexpr std::array<std::pair<Backend, std::string_view>, 3> kBackendNames{{
    {Backend::kCuda, "cuda"},
    {Backend::kVic, "vic"},
    {Backend::kAuto, "auto"},
}};

constexpr std::array<std::pair<ResizeMode, std::string_view>, 3> kModeNames{{
    {ResizeMode::kNearest, "nearest"},
    {ResizeMode::kBilinear, "bilinear"},
    {ResizeMode::kArea, "area"},
}};

template <typename Enum, std::size_t N>
std::string_view name_of(const std::array<std::pair<Enum, std::string_view>, N>& names,
                         Enum value) noexcept {
  for (const auto& [kind, name] : names) {
    if (kind == value) {
      return name;
    }
  }
  return "unknown";
}

template <typename Enum, std::size_t N>
std::optional<Enum> parse_name(const std::array<std::pair<Enum, std::string_view>, N>& names,
                               std::string_view text) noexcept {
  for (const auto& [kind, name] : names) {
    if (name == text) {
      return kind;
    }
  }
  return std::nullopt;
}

// A 3x4 affine map on colour samples, in double precision.
struct Map34 {
  double m[3][4];
};

Map34 identity() { return {{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}}; }

Map34 compose(const Map34& outer, const Map34& inner) {
  Map34 r{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      double v = j == 3 ? outer.m[i][3] : 0.0;
      for (int k = 0; k < 3; ++k) {
        v += outer.m[i][k] * inner.m[k][j];
      }
      r.m[i][j] = v;
    }
  }
  return r;
}

struct Luma {
  double kr;
  double kb;
};

Luma luma_weights(ColorMatrix matrix) {
  switch (matrix) {
    case ColorMatrix::kBt709:
      return {0.2126, 0.0722};
    case ColorMatrix::kBt2020:
      return {0.2627, 0.0593};
    case ColorMatrix::kBt601:
      break;
  }
  return {0.299, 0.114};
}

// Y, Cb, Cr codes per normalized Y' (0..1) and chroma (-0.5..0.5).
struct Quantization {
  double y_scale;
  double y_offset;
  double c_scale;
};

Quantization quantization(ColorRange range) {
  return range == ColorRange::kLimited ? Quantization{219.0, 16.0, 224.0}
                                       : Quantization{255.0, 0.0, 255.0};
}

// From the samples of `model` to R, G, B on 0..255.
Map34 to_rgb(ColorModel model, ColorMatrix matrix, ColorRange range) {
  if (model == ColorModel::kRgb) {
    return identity();
  }
  if (model == ColorModel::kGray) {
    return {{{1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}}};
  }
  const auto [kr, kb] = luma_weights(matrix);
  const double kg = 1.0 - kr - kb;
  const Quantization q = quantization(range);
  // Normalized: y = (Y - yo) / ys, cb = (Cb - 128) / cs, cr likewise.
  // R = y + 2 (1 - kr) cr; B = y + 2 (1 - kb) cb; G = (y - kr R - kb B) / kg.
  const double ry = 1.0 / q.y_scale;
  const double rc = 1.0 / q.c_scale;
  const double r_cr = 2.0 * (1.0 - kr);
  const double b_cb = 2.0 * (1.0 - kb);
  const double g_cb = -kb * b_cb / kg;
  const double g_cr = -kr * r_cr / kg;
  Map34 n{};  // R, G, B (0..1) from Y, Cb, Cr codes.
  const double y_off = -q.y_offset * ry;
  n.m[0][0] = ry;
  n.m[0][1] = 0.0;
  n.m[0][2] = r_cr * rc;
  n.m[0][3] = y_off - r_cr * rc * 128.0;
  n.m[1][0] = ry;
  n.m[1][1] = g_cb * rc;
  n.m[1][2] = g_cr * rc;
  n.m[1][3] = y_off - (g_cb + g_cr) * rc * 128.0;
  n.m[2][0] = ry;
  n.m[2][1] = b_cb * rc;
  n.m[2][2] = 0.0;
  n.m[2][3] = y_off - b_cb * rc * 128.0;
  for (auto& row : n.m) {
    for (double& v : row) {
      v *= 255.0;
    }
  }
  return n;
}

// From R, G, B on 0..255 to the samples of `model`.
Map34 from_rgb(ColorModel model, ColorMatrix matrix, ColorRange range) {
  if (model == ColorModel::kRgb) {
    return identity();
  }
  const auto [kr, kb] = luma_weights(matrix);
  const double kg = 1.0 - kr - kb;
  if (model == ColorModel::kGray) {
    return {{{kr, kg, kb, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}};
  }
  const Quantization q = quantization(range);
  // y = kr r + kg g + kb b; cb = (b - y) / (2 (1 - kb)); cr = (r - y) / (2 (1 - kr)).
  const double cb_den = 2.0 * (1.0 - kb);
  const double cr_den = 2.0 * (1.0 - kr);
  const double s = q.y_scale / 255.0;
  const double c = q.c_scale / 255.0;
  return {{
      {kr * s, kg * s, kb * s, q.y_offset},
      {-kr / cb_den * c, -kg / cb_den * c, (1.0 - kb) / cb_den * c, 128.0},
      {(1.0 - kr) / cr_den * c, -kg / cr_den * c, -kb / cr_den * c, 128.0},
  }};
}

detail::ImageDesc describe(const ImageView& view) {
  detail::ImageDesc desc{.format = view.format, .width = view.width, .height = view.height};
  for (int i = 0; i < 3; ++i) {
    desc.data[i] = view.planes[i].data;
    desc.pitch[i] = view.planes[i].pitch;
  }
  return desc;
}

// One axis of a letterboxed content rectangle: its size and offset.
std::pair<int, int> fit_axis(double scaled, int size, bool even) {
  int content = std::clamp(static_cast<int>(std::lround(scaled)), 1, size);
  if (content == size) {
    return {size, 0};  // No border on this axis.
  }
  if (even) {
    content = std::max(2, content & ~1);
  }
  int offset = (size - content) / 2;
  if (even) {
    offset &= ~1;
  }
  return {content, offset};
}

void check_view(const ImageView& view, const char* which) {
  const std::string name(which);
  if (view.width <= 0 || view.height <= 0) {
    throw std::invalid_argument(name + " image size must be positive");
  }
  if (needs_even_width(view.format) && view.width % 2 != 0) {
    throw std::invalid_argument(name + " image: " + std::string(format_name(view.format)) +
                                " needs an even width, not " + std::to_string(view.width));
  }
  for (int plane = 0; plane < format_info(view.format).planes; ++plane) {
    const PlaneSize size = plane_size(view.format, view.width, view.height, plane);
    if (view.planes[plane].data == nullptr) {
      throw std::invalid_argument(name + " image plane " + std::to_string(plane) + " has no data");
    }
    if (view.planes[plane].pitch < size.row_bytes) {
      throw std::invalid_argument(name + " image plane " + std::to_string(plane) + " pitch " +
                                  std::to_string(view.planes[plane].pitch) +
                                  " is shorter than a row of " + std::to_string(size.row_bytes) +
                                  " bytes");
    }
  }
}

}  // namespace

std::string_view format_name(PixelFormat format) noexcept { return name_of(kFormatNames, format); }
std::optional<PixelFormat> parse_format(std::string_view name) noexcept {
  return parse_name(kFormatNames, name);
}
std::string_view matrix_name(ColorMatrix matrix) noexcept { return name_of(kMatrixNames, matrix); }
std::optional<ColorMatrix> parse_matrix(std::string_view name) noexcept {
  return parse_name(kMatrixNames, name);
}
std::string_view range_name(ColorRange range) noexcept { return name_of(kRangeNames, range); }
std::optional<ColorRange> parse_range(std::string_view name) noexcept {
  return parse_name(kRangeNames, name);
}
std::string_view resize_mode_name(ResizeMode mode) noexcept { return name_of(kModeNames, mode); }
std::optional<ResizeMode> parse_resize_mode(std::string_view name) noexcept {
  return parse_name(kModeNames, name);
}

std::string_view backend_name(Backend backend) noexcept { return name_of(kBackendNames, backend); }
std::optional<Backend> parse_backend(std::string_view name) noexcept {
  return parse_name(kBackendNames, name);
}

namespace {

bool vic_format(PixelFormat format) {
  switch (format) {
    case PixelFormat::kNv12:
    case PixelFormat::kYuyv:
    case PixelFormat::kUyvy:
    case PixelFormat::kRgba8:
    case PixelFormat::kBgra8:
    case PixelFormat::kMono8:
      return true;
    default:
      return false;
  }
}

// Why the VIC cannot read or write `rect` of a `format` image, or empty.
std::string vic_rect_problem(PixelFormat format, const Rect& rect, const char* which) {
  constexpr int kMinSide = 16;
  constexpr int kMaxSide = 16384;
  if (rect.width < kMinSide || rect.height < kMinSide || rect.width > kMaxSide ||
      rect.height > kMaxSide) {
    return std::string(which) + " is " + std::to_string(rect.width) + "x" +
           std::to_string(rect.height) + "; the VIC needs sides of 16 to 16384 pixels";
  }
  const FormatInfo info = format_info(format);
  const bool odd_x = info.chroma_shift_x != 0 && (rect.x % 2 != 0 || rect.width % 2 != 0);
  const bool odd_y = info.chroma_shift_y != 0 && (rect.y % 2 != 0 || rect.height % 2 != 0);
  if (odd_x || odd_y) {
    return std::string(which) + " has an odd offset or size in a chroma-subsampled " +
           std::string(format_name(format)) + " image, which the VIC cannot address";
  }
  return {};
}

}  // namespace

std::optional<std::string> vic_limitation(PixelFormat source, int source_width, int source_height,
                                          PixelFormat destination, int width, int height,
                                          const ConversionParams& params) {
  for (const PixelFormat format : {source, destination}) {
    if (!vic_format(format)) {
      return "the VIC does not handle " + std::string(format_name(format)) +
             "; it handles nv12, yuyv, uyvy, rgba8, bgra8 and mono8";
    }
  }
  if (params.mode == ResizeMode::kArea) {
    return std::string("the VIC has no area sampling; use nearest or bilinear");
  }
  // Where a matrix applies, only BT.709 gives the same colours on the VIC of
  // every Jetson: Thor's applies BT.709 where BT.601 is asked, and VPI's
  // BT.2020 also converts the primaries. Luma computed on the VIC (MONO8 from
  // colour) differs on Thor by tens of codes, so it is left to CUDA as well.
  const ColorModel from = format_info(source).model;
  const ColorModel to = format_info(destination).model;
  if (from != to) {
    if (to == ColorModel::kGray) {
      return std::string("the VIC's luma differs between Jetsons; MONO8 from colour runs on CUDA");
    }
    if (params.matrix != ColorMatrix::kBt709) {
      return "the VIC converts between YUV and RGB the same way on every Jetson only with "
             "BT.709, not " +
             std::string(matrix_name(params.matrix));
    }
  }
  Mapping mapping;
  try {
    mapping = compute_mapping(source_width, source_height, width, height, destination, params);
  } catch (const std::invalid_argument& error) {
    return std::string(error.what());
  }
  if (std::string problem = vic_rect_problem(source, mapping.source, "the source rectangle");
      !problem.empty()) {
    return problem;
  }
  if (std::string problem = vic_rect_problem(destination, mapping.content, "the content");
      !problem.empty()) {
    return problem;
  }
  for (const auto [from, to] : {std::pair{mapping.source.width, mapping.content.width},
                                std::pair{mapping.source.height, mapping.content.height}}) {
    if (to * 16 < from || from * 16 < to) {
      return std::string("the VIC scales by at most 16 times up or down");
    }
  }
  return std::nullopt;
}

Mapping compute_mapping(int source_width, int source_height, int width, int height,
                        PixelFormat destination, const ConversionParams& params) {
  if (source_width <= 0 || source_height <= 0 || width <= 0 || height <= 0) {
    throw std::invalid_argument("image sizes must be positive");
  }
  Rect source{0, 0, source_width, source_height};
  if (params.crop) {
    const Rect& c = *params.crop;
    if (c.width <= 0 || c.height <= 0 || c.x < 0 || c.y < 0 || c.x > source_width - c.width ||
        c.y > source_height - c.height) {
      throw std::invalid_argument(
          "crop " + std::to_string(c.width) + "x" + std::to_string(c.height) + "+" +
          std::to_string(c.x) + "+" + std::to_string(c.y) + " is not inside the " +
          std::to_string(source_width) + "x" + std::to_string(source_height) + " source");
    }
    source = c;
  }
  Rect content{0, 0, width, height};
  if (params.letterbox) {
    const double scale = std::min(static_cast<double>(width) / source.width,
                                  static_cast<double>(height) / source.height);
    const FormatInfo info = format_info(destination);
    const auto [cw, ox] = fit_axis(source.width * scale, width, info.chroma_shift_x != 0);
    const auto [ch, oy] = fit_axis(source.height * scale, height, info.chroma_shift_y != 0);
    content = {ox, oy, cw, ch};
  }
  return {source, content};
}

void validate_conversion(const ImageView& source, const ImageView& destination,
                         const ConversionParams& params) {
  check_view(source, "source");
  check_view(destination, "destination");
  (void)compute_mapping(source.width, source.height, destination.width, destination.height,
                        destination.format, params);
}

namespace detail {

Plan make_plan(const ImageView& src, const ImageView& dst, const ConversionParams& params) {
  validate_conversion(src, dst, params);
  const Mapping mapping =
      compute_mapping(src.width, src.height, dst.width, dst.height, dst.format, params);
  Plan plan{};
  plan.src = describe(src);
  plan.dst = describe(dst);
  plan.mode = params.mode;
  plan.src_x0 = mapping.source.x;
  plan.src_y0 = mapping.source.y;
  plan.src_x1 = mapping.source.x + mapping.source.width - 1;
  plan.src_y1 = mapping.source.y + mapping.source.height - 1;
  plan.scale_x = static_cast<float>(static_cast<double>(mapping.source.width) /
                                    static_cast<double>(mapping.content.width));
  plan.scale_y = static_cast<float>(static_cast<double>(mapping.source.height) /
                                    static_cast<double>(mapping.content.height));
  plan.content_x0 = mapping.content.x;
  plan.content_y0 = mapping.content.y;
  plan.content_x1 = mapping.content.x + mapping.content.width;
  plan.content_y1 = mapping.content.y + mapping.content.height;

  const ColorModel from = format_info(src.format).model;
  const ColorModel to = format_info(dst.format).model;
  plan.affine.identity = from == to;
  const Map34 map =
      compose(from_rgb(to, params.matrix, params.range), to_rgb(from, params.matrix, params.range));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      plan.affine.m[i * 3 + j] = static_cast<float>(map.m[i][j]);
    }
    plan.affine.o[i] = static_cast<float>(map.m[i][3]);
  }

  const Map34 fill = from_rgb(to, params.matrix, params.range);
  float fill_samples[3];
  for (int i = 0; i < 3; ++i) {
    double v = fill.m[i][3];
    for (int k = 0; k < 3; ++k) {
      v += fill.m[i][k] * params.fill_rgb[k];
    }
    fill_samples[i] = static_cast<float>(v);
  }
  plan.fill = {fill_samples[0], fill_samples[1], fill_samples[2], 255.0F};

  const FormatInfo info = format_info(dst.format);
  plan.block_shift_x = info.model == ColorModel::kYuv ? info.chroma_shift_x : 0;
  plan.block_shift_y = info.model == ColorModel::kYuv ? info.chroma_shift_y : 0;
  return plan;
}

}  // namespace detail
}  // namespace holoscan::color_convert
