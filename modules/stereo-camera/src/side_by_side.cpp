// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/stereo_camera/side_by_side.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace holoscan::stereo_camera {
namespace {

std::uint8_t clamp_byte(int value) noexcept {
  return static_cast<std::uint8_t>(std::clamp(value, 0, 255));
}

// One eye's row of `width` pixels, starting at `in`, into `out`.
void convert_row(const std::uint8_t* in, int width, SideBySideFormat input, PixelFormat output,
                 std::uint8_t* out) {
  if (input == SideBySideFormat::kGray) {
    if (output == PixelFormat::kMono8) {
      std::memcpy(out, in, static_cast<std::size_t>(width));
    } else {
      for (int x = 0; x < width; ++x) {
        out[3 * x] = out[3 * x + 1] = out[3 * x + 2] = in[x];
      }
    }
    return;
  }
  // YUYV: Y0 U Y1 V for each pair of pixels.
  if (output == PixelFormat::kMono8) {
    for (int x = 0; x < width; ++x) {
      out[x] = in[2 * x];
    }
    return;
  }
  for (int x = 0; x < width; x += 2) {
    const std::uint8_t* pair = in + 2 * x;
    yuv_to_rgb(pair[0], pair[1], pair[3], out + 3 * x);
    yuv_to_rgb(pair[2], pair[1], pair[3], out + 3 * (x + 1));
  }
}

}  // namespace

std::string_view side_by_side_format_name(SideBySideFormat format) noexcept {
  return format == SideBySideFormat::kYuyv ? "yuyv" : "gray";
}

std::optional<SideBySideFormat> parse_side_by_side_format(std::string_view name) noexcept {
  if (name == "yuyv") return SideBySideFormat::kYuyv;
  if (name == "gray") return SideBySideFormat::kGray;
  return std::nullopt;
}

void yuv_to_rgb(int y, int u, int v, std::uint8_t* rgb) noexcept {
  // BT.601, limited range, in fixed point: R = 1.164 (Y - 16) + 1.596 (V - 128), ...
  const int c = 298 * (y - 16);
  const int d = u - 128;
  const int e = v - 128;
  rgb[0] = clamp_byte((c + 409 * e + 128) >> 8);
  rgb[1] = clamp_byte((c - 100 * d - 208 * e + 128) >> 8);
  rgb[2] = clamp_byte((c + 516 * d + 128) >> 8);
}

void split_side_by_side(const std::uint8_t* frame, int full_width, int height, std::size_t pitch,
                        SideBySideFormat input, PixelFormat output, Image& left, Image& right) {
  if (full_width < 2 || full_width % 2 != 0 || height < 1) {
    throw std::invalid_argument(
        "a side-by-side frame needs an even width and a height of 1 or more");
  }
  const int eye = full_width / 2;
  if (input == SideBySideFormat::kYuyv && eye % 2 != 0) {
    throw std::invalid_argument("a YUYV side-by-side frame needs an even width for each eye, not " +
                                std::to_string(eye));
  }
  const std::size_t in_bytes = input == SideBySideFormat::kYuyv ? 2U : 1U;
  if (pitch < static_cast<std::size_t>(full_width) * in_bytes) {
    throw std::invalid_argument("the frame's pitch is shorter than a row");
  }
  if (output != PixelFormat::kMono8 && output != PixelFormat::kRgb8) {
    throw std::invalid_argument("side-by-side eyes are mono8 or rgb8");
  }
  left.resize(eye, height, output);
  right.resize(eye, height, output);
  const std::size_t out_pitch = left.pitch();
  for (int y = 0; y < height; ++y) {
    const std::uint8_t* row = frame + static_cast<std::size_t>(y) * pitch;
    convert_row(row, eye, input, output, left.pixels.data() + y * out_pitch);
    convert_row(row + static_cast<std::size_t>(eye) * in_bytes, eye, input, output,
                right.pixels.data() + y * out_pitch);
  }
}

}  // namespace holoscan::stereo_camera
