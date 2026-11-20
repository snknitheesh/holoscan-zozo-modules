// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/color_convert/camera_info.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace holoscan::color_convert {
namespace {

std::string size_text(std::uint32_t width, std::uint32_t height) {
  return std::to_string(width) + "x" + std::to_string(height);
}

// `pixels` image pixels of `per_pixel` sensor pixels each, in whole sensor
// pixels.
std::uint32_t sensor_pixels(int pixels, double per_pixel, const char* what) {
  const double value = std::round(static_cast<double>(pixels) * per_pixel);
  if (!(value >= 0.0) || value > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(std::string("the crop's ") + what +
                                " in sensor pixels does not fit the region of interest");
  }
  return static_cast<std::uint32_t>(value);
}

}  // namespace

std::optional<std::string> camera_info_problem(const messages::CameraInfo& info, int width,
                                               int height) {
  if (width < 1 || height < 1 || info.width != static_cast<std::uint32_t>(width) ||
      info.height != static_cast<std::uint32_t>(height)) {
    return "the camera info is for " + size_text(info.width, info.height) +
           " images, and the image is " +
           size_text(static_cast<std::uint32_t>(std::max(width, 0)),
                     static_cast<std::uint32_t>(std::max(height, 0)));
  }
  return std::nullopt;
}

messages::CameraInfo resized_camera_info(const messages::CameraInfo& info, int source_width,
                                         int source_height, int width, int height,
                                         PixelFormat destination, const ConversionParams& params) {
  if (auto problem = camera_info_problem(info, source_width, source_height)) {
    throw std::invalid_argument(*problem);
  }
  const Mapping mapping =
      compute_mapping(source_width, source_height, width, height, destination, params);
  const Rect& s = mapping.source;
  const Rect& c = mapping.content;
  const double sx = static_cast<double>(c.width) / s.width;
  const double sy = static_cast<double>(c.height) / s.height;

  messages::CameraInfo out = info;
  out.width = static_cast<std::uint32_t>(width);
  out.height = static_cast<std::uint32_t>(height);
  out.fx = info.fx * sx;
  out.fy = info.fy * sy;
  out.skew = info.skew * sx;
  out.cx = (info.cx + 0.5 - s.x) * sx - 0.5 + c.x;
  out.cy = (info.cy + 0.5 - s.y) * sy - 0.5 + c.y;
  if (params.crop) {
    double per_x = info.binning_x;
    double per_y = info.binning_y;
    std::uint32_t origin_x = 0;
    std::uint32_t origin_y = 0;
    if (info.roi) {
      per_x = static_cast<double>(info.roi->width) / source_width;
      per_y = static_cast<double>(info.roi->height) / source_height;
      origin_x = info.roi->x;
      origin_y = info.roi->y;
    }
    const std::uint32_t x = sensor_pixels(s.x, per_x, "offset");
    const std::uint32_t y = sensor_pixels(s.y, per_y, "offset");
    const std::uint32_t w = std::max(1U, sensor_pixels(s.width, per_x, "width"));
    const std::uint32_t h = std::max(1U, sensor_pixels(s.height, per_y, "height"));
    if (x > std::numeric_limits<std::uint32_t>::max() - origin_x ||
        y > std::numeric_limits<std::uint32_t>::max() - origin_y) {
      throw std::invalid_argument("the crop's offset in sensor pixels does not fit");
    }
    out.roi = messages::RegionOfInterest{origin_x + x, origin_y + y, w, h};
  }
  messages::validate(out, "resized_camera_info");
  return out;
}

}  // namespace holoscan::color_convert
