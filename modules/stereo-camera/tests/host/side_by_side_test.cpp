// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Splitting side-by-side frames: gray and YUYV, into gray and RGB eyes, with
// padded rows; the BT.601 conversion at known colors; and frames that cannot
// be split.

#include <cstdint>
#include <string>
#include <vector>

#include <holoscan/stereo_camera/side_by_side.hpp>

#include "host_support.hpp"
#include "test_support.hpp"

namespace holoscan::stereo_camera {
namespace {

using test::contains;
using test::invalid_message;

// A frame where each byte names its row and column, with `padding` bytes after each row.
std::vector<std::uint8_t> numbered(int row_bytes, int height, int padding) {
  std::vector<std::uint8_t> frame(static_cast<std::size_t>(row_bytes + padding) * height, 0xEE);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < row_bytes; ++x) {
      frame[y * (row_bytes + padding) + x] = static_cast<std::uint8_t>(16 * y + x);
    }
  }
  return frame;
}

void check_gray() {
  constexpr int kFull = 8;
  constexpr int kHeight = 3;
  const std::vector<std::uint8_t> frame = numbered(kFull, kHeight, 4);
  Image left;
  Image right;
  split_side_by_side(frame.data(), kFull, kHeight, kFull + 4, SideBySideFormat::kGray,
                     PixelFormat::kMono8, left, right);
  VERIFY(left.width == 4 && left.height == 3 && left.format == PixelFormat::kMono8);
  bool ok = true;
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < 4; ++x) {
      ok &= left.pixels[y * 4 + x] == 16 * y + x;
      ok &= right.pixels[y * 4 + x] == 16 * y + 4 + x;
    }
  }
  VERIFY(ok);
  split_side_by_side(frame.data(), kFull, kHeight, kFull + 4, SideBySideFormat::kGray,
                     PixelFormat::kRgb8, left, right);
  VERIFY(left.format == PixelFormat::kRgb8 && right.pixels[3 * 5 + 0] == 16 + 4 + 1 &&
         right.pixels[3 * 5 + 2] == 16 + 4 + 1);
}

void check_yuyv() {
  // Two eyes of 2x1 pixels: each one pair Y0 U Y1 V.
  const std::vector<std::uint8_t> frame{// Left: white then black, no chroma.
                                        235, 128, 16, 128,
                                        // Right: pure red in BT.601 limited range, twice.
                                        81, 90, 81, 240,
                                        // Padding.
                                        0xEE, 0xEE};
  Image left;
  Image right;
  split_side_by_side(frame.data(), 4, 1, frame.size(), SideBySideFormat::kYuyv, PixelFormat::kMono8,
                     left, right);
  VERIFY(left.width == 2 && left.pixels == (std::vector<std::uint8_t>{235, 16}));
  VERIFY(right.pixels == (std::vector<std::uint8_t>{81, 81}));
  split_side_by_side(frame.data(), 4, 1, frame.size(), SideBySideFormat::kYuyv, PixelFormat::kRgb8,
                     left, right);
  VERIFY(left.pixels == (std::vector<std::uint8_t>{255, 255, 255, 0, 0, 0}));
  // Red, to within the rounding of the fixed-point coefficients.
  VERIFY(right.pixels[0] >= 254 && right.pixels[1] <= 1 && right.pixels[2] <= 1);
  VERIFY(right.pixels[3] == right.pixels[0] && right.pixels[4] == right.pixels[1]);
}

void check_rejections() {
  std::vector<std::uint8_t> frame(64);
  Image left;
  Image right;
  const auto split = [&](int full, int height, std::size_t pitch, SideBySideFormat input,
                         PixelFormat output) {
    return invalid_message(
        [&] { split_side_by_side(frame.data(), full, height, pitch, input, output, left, right); });
  };
  VERIFY(contains(split(7, 1, 7, SideBySideFormat::kGray, PixelFormat::kMono8), "even width"));
  VERIFY(contains(split(6, 1, 12, SideBySideFormat::kYuyv, PixelFormat::kMono8),
                  "even width for each eye, not 3"));
  VERIFY(contains(split(8, 1, 15, SideBySideFormat::kYuyv, PixelFormat::kMono8),
                  "pitch is shorter than a row"));
  VERIFY(contains(split(8, 1, 8, SideBySideFormat::kGray, PixelFormat::kMono16), "mono8 or rgb8"));
  VERIFY(parse_side_by_side_format("yuyv") == SideBySideFormat::kYuyv);
  VERIFY(parse_side_by_side_format(side_by_side_format_name(SideBySideFormat::kGray)) ==
         SideBySideFormat::kGray);
  VERIFY(!parse_side_by_side_format("mjpeg"));
}

}  // namespace
}  // namespace holoscan::stereo_camera

int main() {
  namespace sc = holoscan::stereo_camera;
  return sc::test::run({
      {"gray side by side", sc::check_gray},
      {"YUYV side by side", sc::check_yuyv},
      {"frames that cannot be split", sc::check_rejections},
  });
}
