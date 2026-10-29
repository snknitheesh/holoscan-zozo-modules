// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/stereo_camera/image.hpp"

#include <cstring>

namespace holoscan::stereo_camera {

std::string_view pixel_format_name(PixelFormat format) noexcept {
  switch (format) {
    case PixelFormat::kMono8:
      return "mono8";
    case PixelFormat::kMono16:
      return "mono16";
    case PixelFormat::kRgb8:
      return "rgb8";
    case PixelFormat::kBgr8:
      return "bgr8";
  }
  return "unknown";
}

int bytes_per_pixel(PixelFormat format) noexcept {
  switch (format) {
    case PixelFormat::kMono8:
      return 1;
    case PixelFormat::kMono16:
      return 2;
    case PixelFormat::kRgb8:
    case PixelFormat::kBgr8:
      return 3;
  }
  return 1;
}

int channels(PixelFormat format) noexcept {
  return format == PixelFormat::kRgb8 || format == PixelFormat::kBgr8 ? 3 : 1;
}

std::size_t ImageView::packed_bytes() const noexcept {
  return static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
         static_cast<std::size_t>(bytes_per_pixel(format));
}

void Image::resize(int new_width, int new_height, PixelFormat new_format) {
  width = new_width;
  height = new_height;
  format = new_format;
  pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                static_cast<std::size_t>(bytes_per_pixel(format)));
}

std::size_t Image::pitch() const noexcept {
  return static_cast<std::size_t>(width) * static_cast<std::size_t>(bytes_per_pixel(format));
}

ImageView Image::view() const noexcept {
  return ImageView{
      .data = pixels.data(), .width = width, .height = height, .pitch = pitch(), .format = format};
}

void copy_image(const ImageView& view, Image& image) {
  image.resize(view.width, view.height, view.format);
  const std::size_t row = image.pitch();
  for (int y = 0; y < view.height; ++y) {
    std::memcpy(image.pixels.data() + static_cast<std::size_t>(y) * row,
                view.data + static_cast<std::size_t>(y) * view.pitch, row);
  }
}

bool same_pixels(const ImageView& a, const ImageView& b) noexcept {
  if (a.width != b.width || a.height != b.height || a.format != b.format) {
    return false;
  }
  const std::size_t row = static_cast<std::size_t>(a.width) * bytes_per_pixel(a.format);
  for (int y = 0; y < a.height; ++y) {
    if (std::memcmp(a.data + static_cast<std::size_t>(y) * a.pitch,
                    b.data + static_cast<std::size_t>(y) * b.pitch, row) != 0) {
      return false;
    }
  }
  return true;
}

}  // namespace holoscan::stereo_camera
