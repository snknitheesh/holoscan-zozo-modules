// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/stitching/remap.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "blend_setup.hpp"
#include "stitch_math.hpp"

namespace holoscan::stitching {

std::string_view format_name(PixelFormat format) noexcept {
  switch (format) {
    case PixelFormat::kMono8:
      return "mono8";
    case PixelFormat::kRgb8:
      return "rgb8";
    case PixelFormat::kRgba8:
      return "rgba8";
    case PixelFormat::kBgr8:
      return "bgr8";
    case PixelFormat::kBgra8:
      return "bgra8";
  }
  return "unknown";
}

std::optional<PixelFormat> parse_format(std::string_view name) noexcept {
  for (const PixelFormat format : {PixelFormat::kMono8, PixelFormat::kRgb8, PixelFormat::kRgba8,
                                   PixelFormat::kBgr8, PixelFormat::kBgra8}) {
    if (name == format_name(format)) return format;
  }
  return std::nullopt;
}

int channels(PixelFormat format) noexcept {
  switch (format) {
    case PixelFormat::kMono8:
      return 1;
    case PixelFormat::kRgb8:
    case PixelFormat::kBgr8:
      return 3;
    case PixelFormat::kRgba8:
    case PixelFormat::kBgra8:
      return 4;
  }
  return 0;
}

std::string_view blend_mode_name(BlendMode mode) noexcept {
  return mode == BlendMode::kFeather ? "feather" : "hard_seam";
}

std::optional<BlendMode> parse_blend_mode(std::string_view name) noexcept {
  if (name == "feather") return BlendMode::kFeather;
  if (name == "hard_seam") return BlendMode::kHardSeam;
  return std::nullopt;
}

void validate(const BlendSettings& settings) {
  if (settings.mode != BlendMode::kFeather && settings.mode != BlendMode::kHardSeam) {
    throw std::invalid_argument("blend: unknown mode");
  }
  if (!(std::isfinite(settings.feather_width) && settings.feather_width >= 0.0F)) {
    throw std::invalid_argument("blend: feather_width must be 0 or positive");
  }
  if (!(std::isfinite(settings.exposure_sigma_noise) && settings.exposure_sigma_noise > 0.0F &&
        std::isfinite(settings.exposure_sigma_gain) && settings.exposure_sigma_gain > 0.0F)) {
    throw std::invalid_argument(
        "blend: exposure_sigma_noise and exposure_sigma_gain must be finite and positive");
  }
  if (!(settings.exposure_smoothing >= 0.0F && settings.exposure_smoothing < 1.0F)) {
    throw std::invalid_argument("blend: exposure_smoothing must be in [0, 1)");
  }
  if (settings.output_format != PixelFormat::kRgb8 &&
      settings.output_format != PixelFormat::kRgba8) {
    throw std::invalid_argument("blend: output_format must be rgb8 or rgba8");
  }
}

RemapTables build_remap_tables(const CameraRig& rig, const OutputGeometry& geometry) {
  validate(rig);
  validate(geometry);
  const detail::GeometryParams g = detail::geometry_params(geometry);
  const int width = geometry.width;
  const int height = geometry.height;
  RemapTables tables;
  tables.output_width = width;
  tables.output_height = height;
  tables.wraps = wraps(geometry);
  const std::size_t pixels = static_cast<std::size_t>(width) * height;
  std::vector<float> uv(2 * pixels);
  std::vector<float> margin(pixels);
  for (const Camera& camera : rig.cameras) {
    const detail::CameraParams c = detail::camera_params(camera);
    std::vector<std::uint8_t> columns(static_cast<std::size_t>(width), 0U);
    int y_first = height;
    int y_last = -1;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const std::size_t i = static_cast<std::size_t>(y) * width + x;
        float u = 0.0F;
        float v = 0.0F;
        float m = 0.0F;
        if (detail::remap_entry(g, c, x, y, u, v, m)) {
          uv[2 * i] = u;
          uv[2 * i + 1] = v;
          margin[i] = m;
          columns[x] = 1U;
          y_first = std::min(y_first, y);
          y_last = std::max(y_last, y);
        } else {
          margin[i] = 0.0F;
        }
      }
    }
    const detail::Roi roi = detail::region_of_interest(columns, y_first, y_last, tables.wraps);
    CameraTable table;
    table.image_width = camera.width;
    table.image_height = camera.height;
    table.x0 = roi.x0;
    table.y0 = roi.y0;
    table.width = roi.width;
    table.height = roi.height;
    const std::size_t entries = static_cast<std::size_t>(roi.width) * roi.height;
    table.uv.assign(2 * entries, 0.0F);
    table.margin.assign(entries, 0.0F);
    for (int dy = 0; dy < roi.height; ++dy) {
      for (int dx = 0; dx < roi.width; ++dx) {
        const int x = (roi.x0 + dx) % width;
        const std::size_t from = static_cast<std::size_t>(roi.y0 + dy) * width + x;
        const std::size_t to = static_cast<std::size_t>(dy) * roi.width + dx;
        if (margin[from] > 0.0F) {
          table.uv[2 * to] = uv[2 * from];
          table.uv[2 * to + 1] = uv[2 * from + 1];
          table.margin[to] = margin[from];
        }
      }
    }
    tables.cameras.push_back(std::move(table));
  }
  return tables;
}

std::vector<float> resolution_map(const RemapTables& tables) {
  const int w = tables.output_width;
  const int h = tables.output_height;
  const std::size_t pixels = static_cast<std::size_t>(w) * h;
  std::vector<int> owner(pixels, -1);
  std::vector<float> best(pixels, 0.0F);
  // The table entry of camera c at output pixel (x, y), or nullptr.
  const auto entry = [&](std::size_t c, int x, int y) -> std::ptrdiff_t {
    const CameraTable& t = tables.cameras[c];
    if (y < 0 || y >= h || x < 0 || x >= w) return -1;
    int dx = x - t.x0;
    if (tables.wraps && dx < 0) dx += w;
    const int dy = y - t.y0;
    if (dx < 0 || dx >= t.width || dy < 0 || dy >= t.height) return -1;
    const std::ptrdiff_t i = static_cast<std::ptrdiff_t>(dy) * t.width + dx;
    return t.margin[i] > 0.0F ? i : -1;
  };
  for (std::size_t c = 0; c < tables.cameras.size(); ++c) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const std::ptrdiff_t i = entry(c, x, y);
        const std::size_t o = static_cast<std::size_t>(y) * w + x;
        if (i >= 0 && tables.cameras[c].margin[i] > best[o]) {
          best[o] = tables.cameras[c].margin[i];
          owner[o] = static_cast<int>(c);
        }
      }
    }
  }
  std::vector<float> result(pixels, 0.0F);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t o = static_cast<std::size_t>(y) * w + x;
      if (owner[o] < 0) continue;
      const auto c = static_cast<std::size_t>(owner[o]);
      const std::vector<float>& uv = tables.cameras[c].uv;
      // One-sided differences where a neighbour is not seen by this camera.
      double jacobian[2][2];
      bool ok = true;
      for (int axis = 0; axis < 2 && ok; ++axis) {
        const int ax = axis == 0 ? 1 : 0;
        const int ay = axis == 0 ? 0 : 1;
        int xa = x + ax;
        int ya = y + ay;
        int xb = x - ax;
        int yb = y - ay;
        if (tables.wraps && axis == 0) {
          xa = (xa + w) % w;
          xb = (xb + w) % w;
        }
        std::ptrdiff_t ia = entry(c, xa, ya);
        std::ptrdiff_t ib = entry(c, xb, yb);
        double span = 2.0;
        const std::ptrdiff_t here = entry(c, x, y);
        if (ia < 0) {
          ia = here;
          span = 1.0;
        }
        if (ib < 0) {
          ib = here;
          span -= 1.0;
        }
        if (span <= 0.0) {
          ok = false;
          break;
        }
        jacobian[0][axis] = (uv[2 * ia] - uv[2 * ib]) / span;
        jacobian[1][axis] = (uv[2 * ia + 1] - uv[2 * ib + 1]) / span;
      }
      if (!ok) continue;
      // The smaller singular value of the 2x2 Jacobian.
      const double a = jacobian[0][0];
      const double b = jacobian[0][1];
      const double c2 = jacobian[1][0];
      const double d = jacobian[1][1];
      const double s1 = a * a + b * b + c2 * c2 + d * d;
      const double det = a * d - b * c2;
      const double root = std::sqrt(std::max(0.0, s1 * s1 - 4.0 * det * det));
      result[o] = static_cast<float>(std::sqrt(std::max(0.0, 0.5 * (s1 - root))));
    }
  }
  return result;
}

namespace {

detail::BlendParams host_params(const RemapTables& tables, std::span<const HostImage> images,
                                const BlendSettings& settings) {
  validate(settings);
  if (images.size() != tables.cameras.size()) {
    throw std::invalid_argument("stitch: " + std::to_string(images.size()) + " images for " +
                                std::to_string(tables.cameras.size()) + " cameras");
  }
  detail::BlendParams params =
      detail::blend_params(settings, tables.output_width, tables.output_height, tables.wraps);
  for (std::size_t c = 0; c < images.size(); ++c) {
    const HostImage& image = images[c];
    const CameraTable& table = tables.cameras[c];
    if (image.data == nullptr || image.width != table.image_width ||
        image.height != table.image_height ||
        image.pitch < static_cast<std::size_t>(image.width) * channels(image.format)) {
      throw std::invalid_argument("stitch: image " + std::to_string(c) +
                                  " does not match its camera");
    }
    detail::set_camera(params, static_cast<int>(c), image.data, image.pitch, image.width,
                       image.height, image.format,
                       detail::Roi{table.x0, table.y0, table.width, table.height}, table.uv.data(),
                       table.margin.data());
  }
  return params;
}

}  // namespace

OverlapStats reference_overlap_stats(const RemapTables& tables, std::span<const HostImage> images) {
  const detail::BlendParams params = host_params(tables, images, BlendSettings{});
  const int n = params.cameras;
  OverlapStats stats;
  stats.sum.assign(static_cast<std::size_t>(n) * n, 0U);
  stats.count.assign(static_cast<std::size_t>(n) * n, 0U);
  const int half = kExposureStride / 2;
  for (int y = half; y < tables.output_height; y += kExposureStride) {
    for (int x = half; x < tables.output_width; x += kExposureStride) {
      int luminance[detail::kMaxCameras];
      detail::sample_luminance(params, x, y, luminance);
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          if (i != j && luminance[i] >= 0 && luminance[j] >= 0) {
            stats.sum[i * n + j] += static_cast<std::uint64_t>(luminance[i]);
            ++stats.count[i * n + j];
          }
        }
      }
    }
  }
  return stats;
}

std::vector<float> reference_gains(const OverlapStats& stats, const BlendSettings& settings,
                                   std::span<const float> previous) {
  validate(settings);
  const auto n = static_cast<int>(std::lround(std::sqrt(static_cast<double>(stats.sum.size()))));
  if (n < 1 || n > static_cast<int>(kMaxCameras) ||
      stats.sum.size() != static_cast<std::size_t>(n) * n ||
      stats.count.size() != stats.sum.size()) {
    throw std::invalid_argument("gains: the statistics are not of 1..8 cameras");
  }
  std::vector<float> before(previous.begin(), previous.end());
  if (before.empty()) before.assign(n, 1.0F);
  if (before.size() != static_cast<std::size_t>(n)) {
    throw std::invalid_argument("gains: previous holds one gain per camera");
  }
  std::vector<unsigned long long> sum(stats.sum.begin(), stats.sum.end());  // NOLINT(runtime/int)
  std::vector<float> gains(n);
  detail::solve_gains(n, sum.data(), stats.count.data(), detail::gain_alpha(settings),
                      detail::gain_beta(settings), settings.exposure_smoothing, before.data(),
                      gains.data());
  return gains;
}

void reference_stitch(const RemapTables& tables, std::span<const HostImage> images,
                      const BlendSettings& settings, std::span<const float> gains,
                      std::uint8_t* output, std::size_t output_pitch) {
  const detail::BlendParams params = host_params(tables, images, settings);
  std::vector<float> g(gains.begin(), gains.end());
  if (g.empty()) g.assign(images.size(), 1.0F);
  if (g.size() != images.size()) {
    throw std::invalid_argument("stitch: gains holds one gain per camera");
  }
  const int out_channels = channels(settings.output_format);
  if (output == nullptr ||
      output_pitch < static_cast<std::size_t>(tables.output_width) * out_channels) {
    throw std::invalid_argument("stitch: the output's pitch is shorter than a row");
  }
  for (int y = 0; y < tables.output_height; ++y) {
    std::uint8_t* row = output + static_cast<std::size_t>(y) * output_pitch;
    for (int x = 0; x < tables.output_width; ++x) {
      detail::blend_pixel(params, g.data(), x, y, row + static_cast<std::size_t>(x) * out_channels);
    }
  }
}

namespace detail {

Roi region_of_interest(const std::vector<std::uint8_t>& columns, int y_first, int y_last,
                       bool wraps) {
  const int width = static_cast<int>(columns.size());
  if (y_last < y_first) {
    return {};
  }
  int first = -1;
  int last = -1;
  for (int x = 0; x < width; ++x) {
    if (columns[x] != 0U) {
      if (first < 0) first = x;
      last = x;
    }
  }
  if (first < 0) {
    return {};
  }
  Roi roi{first, y_first, last - first + 1, y_last - y_first + 1};
  if (!wraps) {
    return roi;
  }
  // The widest run of empty columns, around the edge too: the region is the
  // rest, starting after it.
  int best_start = -1;
  int best_length = 0;
  for (int x = 0; x < width; ++x) {
    if (columns[x] != 0U || columns[(x + width - 1) % width] == 0U) {
      continue;  // Not the start of an empty run.
    }
    int length = 0;
    while (length < width && columns[(x + length) % width] == 0U) ++length;
    if (length > best_length) {
      best_length = length;
      best_start = x;
    }
  }
  if (best_start < 0) {
    return Roi{0, y_first, width, roi.height};  // Every column is covered.
  }
  roi.x0 = (best_start + best_length) % width;
  roi.width = width - best_length;
  return roi;
}

BlendParams blend_params(const BlendSettings& settings, int output_width, int output_height,
                         bool wraps) {
  BlendParams p{};
  p.cameras = 0;
  p.output_width = output_width;
  p.output_height = output_height;
  p.wraps = wraps;
  p.hard_seam = settings.mode == BlendMode::kHardSeam;
  p.feather_width = settings.feather_width;
  p.output_channels = channels(settings.output_format);
  for (int i = 0; i < 3; ++i) p.background[i] = settings.background[i];
  return p;
}

void set_camera(BlendParams& params, int index, const std::uint8_t* image, std::size_t pitch,
                int width, int height, PixelFormat format, const Roi& roi, const float* uv,
                const float* margin) {
  CameraSampler& s = params.camera[index];
  s.image = image;
  s.pitch = pitch;
  s.width = width;
  s.height = height;
  s.channels = channels(format);
  s.bgr = format == PixelFormat::kBgr8 || format == PixelFormat::kBgra8;
  s.x0 = roi.x0;
  s.y0 = roi.y0;
  s.roi_width = roi.width;
  s.roi_height = roi.height;
  s.uv = uv;
  s.margin = margin;
  params.cameras = std::max(params.cameras, index + 1);
}

double gain_alpha(const BlendSettings& settings) {
  const double sigma = settings.exposure_sigma_noise;
  return 1.0 / (sigma * sigma);
}

double gain_beta(const BlendSettings& settings) {
  const double sigma = settings.exposure_sigma_gain;
  return 1.0 / (sigma * sigma);
}

}  // namespace detail
}  // namespace holoscan::stitching
