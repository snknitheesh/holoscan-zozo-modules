// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The overlay's drawing: lines clipped to the canvas, a marker's outline and first corner, a
// chessboard corner's cross, and a target's axes projected in the convention's directions.

#include <array>
#include <vector>

#include <holoscan/fiducials/overlay_drawing.hpp>

#include "test_support.hpp"

namespace holoscan::fiducials {
namespace {

constexpr int kWidth = 200;
constexpr int kHeight = 100;

struct Canvas {
  std::vector<std::uint8_t> pixels = std::vector<std::uint8_t>(kWidth * kHeight * 4, 0);
  [[nodiscard]] RgbaCanvas canvas() { return {pixels.data(), kWidth, kHeight}; }
  [[nodiscard]] Rgba at(int x, int y) const {
    const std::uint8_t* p = pixels.data() + (y * kWidth + x) * 4;
    return {p[0], p[1], p[2], p[3]};
  }
};

constexpr Rgba kClear{0, 0, 0, 0};

void check_lines() {
  Canvas c;
  draw_line(c.canvas(), {10, 10}, {50, 10}, kGreen);
  VERIFY(c.at(30, 10) == kGreen && c.at(30, 20) == kClear);
  // Far outside, and through: only what is on the canvas is drawn, without overrunning it.
  draw_line(c.canvas(), {-1e9, 50}, {1e9, 50}, kBlue);
  draw_line(c.canvas(), {NAN, 0}, {10, 10}, kBlue);
  VERIFY(c.at(100, 50) == kBlue);
}

void check_marker_and_corner() {
  Canvas c;
  draw_marker(c.canvas(), {{{40, 20}, {80, 20}, {80, 60}, {40, 60}}});
  VERIFY(c.at(60, 60) == kYellow && c.at(80, 40) == kYellow);
  VERIFY(c.at(43, 23) == kRed);       // The first corner's mark.
  VERIFY(c.at(60, 40) == kClear);     // Inside stays clear.
  draw_corner(c.canvas(), {150, 50});
  VERIFY(c.at(146, 50) == kMagenta && c.at(150, 46) == kMagenta && c.at(147, 47) == kClear);
}

void check_axes() {
  const CameraIntrinsics camera{400, 400, 100, 50, {}};
  const Pose facing{.rotation = {1, 0, 0, 0, 1, 0, 0, 0, 1}, .translation = {0, 0, 1}};
  const auto axes = target_axes(facing, camera, 0.1);
  VERIFY(axes.has_value());
  if (!axes) return;
  VERIFY(axes->origin == (Point2{100, 50}));
  // x to the right, y down, z into the target: away from the camera, toward the center.
  VERIFY((*axes->tips[0])[0] > 100 && std::abs((*axes->tips[0])[1] - 50) < 1e-9);
  VERIFY((*axes->tips[1])[1] > 50 && std::abs((*axes->tips[1])[0] - 100) < 1e-9);
  VERIFY(axes->tips[2] && std::abs((*axes->tips[2])[0] - 100) < 1e-9);
  Canvas c;
  draw_axes(c.canvas(), *axes);
  VERIFY(c.at(120, 50) == kRed && c.at(100, 70) == kGreen);
  // Behind the camera: nothing.
  const Pose behind{.rotation = facing.rotation, .translation = {0, 0, -1}};
  VERIFY(!target_axes(behind, camera, 0.1));
}

}  // namespace
}  // namespace holoscan::fiducials

int main() {
  return holoscan::fiducials::test::run({
      {"lines", holoscan::fiducials::check_lines},
      {"marker outline and chessboard corner", holoscan::fiducials::check_marker_and_corner},
      {"target axes", holoscan::fiducials::check_axes},
  });
}
