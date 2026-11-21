// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The offscreen render check, without a GPU: renders composed on the host as
// HoloViz composes them, a synthetic frame with the boxes (and masks) of a
// frame drawn over it, must pass when the video and the boxes are of the same
// frame, and fail when the video is one frame behind or ahead of its boxes.
//
// Usage: yolo26_render_check_test

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <holoscan/yolo26/letterbox.hpp>
#include <holoscan/yolo26/synthetic_scene.hpp>

#include "render_check_op.hpp"
#include "test_support.hpp"

namespace yolo26_app {
namespace {

namespace y26 = holoscan::yolo26;

constexpr int kWidth = 1280;
constexpr int kHeight = 720;
constexpr std::uint8_t kMaskAlpha = 110;
// The overlays' colors of classes 0 and 1.
constexpr std::array<std::array<std::uint8_t, 3>, 2> kClassColors{{{230, 25, 75}, {60, 180, 75}}};

// Synthetic frame `video_frame` with the class 0 and 1 rectangles of frame
// `box_frame` drawn over it as the segmentation overlay draws them: each
// rectangle filled in its class's color at `mask_alpha` when `masks`, then
// its 2 px box outline, opaque, inside its edges. Frames are numbered from 1.
std::vector<std::uint8_t> compose(std::uint32_t video_frame, std::uint32_t box_frame, bool masks) {
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(kWidth) * kHeight * 3U);
  y26::render_scene(y26::make_scene(kWidth, kHeight, video_frame - 1U), y26::PixelFormat::kRgb8,
                    rgb.data(), static_cast<std::size_t>(kWidth) * 3U);
  std::vector<std::uint8_t> rgba(static_cast<std::size_t>(kWidth) * kHeight * 4U);
  for (std::size_t p = 0; p < static_cast<std::size_t>(kWidth) * kHeight; ++p) {
    for (std::size_t k = 0; k < 3; ++k) {
      rgba[4 * p + k] = rgb[3 * p + k];
    }
    rgba[4 * p + 3] = 255;
  }
  const auto paint = [&](int x, int y, const std::array<std::uint8_t, 3>& color, double alpha) {
    std::uint8_t* pixel = rgba.data() + (static_cast<std::size_t>(y) * kWidth + x) * 4U;
    for (std::size_t k = 0; k < 3; ++k) {
      pixel[k] =
          static_cast<std::uint8_t>(std::lround(pixel[k] * (1.0 - alpha) + color[k] * alpha));
    }
  };
  const y26::SyntheticScene boxes = y26::make_scene(kWidth, kHeight, box_frame - 1U);
  for (std::size_t c = 0; c < kClassColors.size(); ++c) {
    const y26::SceneRect& rect = boxes.rects[c];
    for (int y = rect.y0; y < rect.y1; ++y) {
      for (int x = rect.x0; x < rect.x1; ++x) {
        const bool line =
            x < rect.x0 + 2 || x >= rect.x1 - 2 || y < rect.y0 + 2 || y >= rect.y1 - 2;
        if (line) {
          paint(x, y, kClassColors[c], 1.0);
        } else if (masks) {
          paint(x, y, kClassColors[c], kMaskAlpha / 255.0);
        }
      }
    }
  }
  return rgba;
}

void check(bool masks) {
  // Frames moving either way, and at the turns of the rectangles' path.
  for (const std::uint32_t frame : {2U, 10U, 25U, 26U, 27U, 40U, 50U, 51U}) {
    const std::string what = "frame " + std::to_string(frame);
    VERIFY_MSG(
        render_shows(compose(frame, frame, masks), kWidth, kHeight, frame, masks, kMaskAlpha),
        what + ": its own video and boxes do not pass");
    for (const std::uint32_t video : {frame - 1U, frame + 1U}) {
      VERIFY_MSG(
          !render_shows(compose(video, frame, masks), kWidth, kHeight, frame, masks, kMaskAlpha),
          what + ": boxes over frame " + std::to_string(video) + "'s video pass");
    }
  }
}

}  // namespace
}  // namespace yolo26_app

int main() {
  return yolo26_app::test::run({{"detection renders", [] { yolo26_app::check(false); }},
                                {"segmentation renders", [] { yolo26_app::check(true); }}});
}
