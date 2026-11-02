// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A synthetic scene a real COCO model recognizes, with its silhouette known:
// a stop sign (COCO class 11), a red octagon with a white rim and "STOP" in
// white block letters, on a pole, against sky and grass. The octagon, rim
// included, is the silhouette its mask must match. Ultralytics' YOLO26n-seg
// finds such signs with scores of 0.93 to 0.95 and masks that overlap the
// octagon by an IoU of 0.98. For the real-model tests; host code only.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

namespace holoscan::yolo26::test {

inline constexpr std::int32_t kStopSignClass = 11;

struct StopSign {
  int width{};
  int height{};
  // The octagon's center and circumradius, in pixels.
  float center_x{};
  float center_y{};
  float radius{};
};

// Whether pixel (x, y)'s center is inside the regular octagon of
// circumradius `radius` around (cx, cy), with flat top and sides.
inline bool in_octagon(int x, int y, float cx, float cy, float radius) {
  const float apothem = radius * std::cos(std::numbers::pi_v<float> / 8.0F);
  const float dx = std::fabs(static_cast<float>(x) + 0.5F - cx);
  const float dy = std::fabs(static_cast<float>(y) + 0.5F - cy);
  return dx <= apothem && dy <= apothem && dx + dy <= apothem * std::numbers::sqrt2_v<float>;
}

// The sign's silhouette: 1 inside the octagon, rim included.
inline std::vector<std::uint8_t> stop_sign_silhouette(const StopSign& sign) {
  std::vector<std::uint8_t> mask(static_cast<std::size_t>(sign.width) * sign.height);
  for (int y = 0; y < sign.height; ++y) {
    for (int x = 0; x < sign.width; ++x) {
      mask[static_cast<std::size_t>(y) * sign.width + x] =
          in_octagon(x, y, sign.center_x, sign.center_y, sign.radius) ? 1 : 0;
    }
  }
  return mask;
}

// The scene as packed RGB8.
inline std::vector<std::uint8_t> render_stop_sign(const StopSign& sign) {
  const int w = sign.width;
  const int h = sign.height;
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(w) * h * 3);
  const auto put = [&](int x, int y, std::array<std::uint8_t, 3> color) {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    std::uint8_t* p = rgb.data() + (static_cast<std::size_t>(y) * w + x) * 3;
    p[0] = color[0];
    p[1] = color[1];
    p[2] = color[2];
  };
  // Sky, darker toward the horizon, then grass.
  for (int y = 0; y < h; ++y) {
    const auto blue = static_cast<std::uint8_t>(200 - y * 60 / h);
    for (int x = 0; x < w; ++x) {
      put(x, y,
          y >= h * 3 / 4 ? std::array<std::uint8_t, 3>{90, 110, 70}
                         : std::array<std::uint8_t, 3>{110, 150, blue});
    }
  }
  // The pole, below the center.
  const float r = sign.radius;
  for (int y = static_cast<int>(sign.center_y); y < h; ++y) {
    for (int x = static_cast<int>(sign.center_x - r * 0.06F);
         x < static_cast<int>(sign.center_x + r * 0.06F); ++x) {
      put(x, y, {120, 120, 120});
    }
  }
  // The rim, then the red face.
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (in_octagon(x, y, sign.center_x, sign.center_y, r * 0.9F)) {
        put(x, y, {200, 20, 30});
      } else if (in_octagon(x, y, sign.center_x, sign.center_y, r)) {
        put(x, y, {245, 245, 245});
      }
    }
  }
  // "STOP" on a 5 x 7 grid per letter, one cell between letters.
  static constexpr std::array<std::array<const char*, 7>, 4> kGlyphs{{
      {"01111", "10000", "10000", "01110", "00001", "00001", "11110"},
      {"11111", "00100", "00100", "00100", "00100", "00100", "00100"},
      {"01110", "10001", "10001", "10001", "10001", "10001", "01110"},
      {"11110", "10001", "10001", "11110", "10000", "10000", "10000"},
  }};
  const float cell = r * 0.065F;
  const float left = sign.center_x - (4.0F * 5.0F + 3.0F) * cell / 2.0F;
  const float top = sign.center_y - 3.5F * cell;
  for (std::size_t letter = 0; letter < kGlyphs.size(); ++letter) {
    for (int gy = 0; gy < 7; ++gy) {
      for (int gx = 0; gx < 5; ++gx) {
        if (kGlyphs[letter][static_cast<std::size_t>(gy)][gx] != '1') continue;
        const int x0 = static_cast<int>(
            std::lround(left + static_cast<float>(letter) * 6.0F * cell + gx * cell));
        const int y0 = static_cast<int>(std::lround(top + gy * cell));
        const int side = static_cast<int>(std::lround(cell));
        for (int y = y0; y < y0 + side; ++y) {
          for (int x = x0; x < x0 + side; ++x) put(x, y, {250, 250, 250});
        }
      }
    }
  }
  return rgb;
}

}  // namespace holoscan::yolo26::test
