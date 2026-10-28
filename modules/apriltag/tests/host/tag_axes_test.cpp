// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The overlay's axis projection and drawing, on known poses and pixels. Needs no
// SDK or GPU, so it runs on hosted CI.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <holoscan/apriltag/tag_axes.hpp>

#include "test_support.hpp"

namespace holoscan::apriltag {
namespace {

constexpr int kWidth = 200;
constexpr int kHeight = 100;
constexpr int kGuardBytes = 64;
constexpr std::uint8_t kGuard = 0xA5;

bool at(const std::optional<Pixel>& p, double x, double y) {
  return p && std::abs((*p)[0] - x) < 1e-9 && std::abs((*p)[1] - y) < 1e-9;
}

// Identity rotation, the tag 2 m ahead: each axis tip is 0.1 m along it.
Detection posed_tag() {
  Detection detection;
  detection.pose =
      TagPose{.rotation = {1, 0, 0, 0, 1, 0, 0, 0, 1}, .translation = {0.25, -0.5, 2.0}};
  return detection;
}

void check_tips() {
  const CameraIntrinsics k{400, 400, 50, 150};
  const auto axes = tag_axes(posed_tag(), k, 0.2);
  VERIFY(axes && at(axes->origin, 100, 50));
  VERIFY(at(axes->tips[0], 120, 50) && at(axes->tips[1], 100, 70));
  VERIFY(at(axes->tips[2], 400 * 0.25 / 2.1 + 50, 400 * -0.5 / 2.1 + 150));  // z into the tag.
  Detection behind = posed_tag();
  behind.pose->translation[2] = -1.0;
  VERIFY(!tag_axes(behind, k, 0.2));
  // Without a pose: X toward the middle of corners 1-2, Y of corners 0-1, no Z.
  Detection flat;
  flat.center = {50, 50};
  flat.corners = {{{20, 80}, {80, 80}, {80, 20}, {20, 20}}};
  const auto plane = tag_axes(flat, std::nullopt, 0.0);
  VERIFY(plane && at(plane->origin, 50, 50) && at(plane->tips[0], 80, 50));
  VERIFY(at(plane->tips[1], 50, 80) && !plane->tips[2]);
}

void check_drawing() {
  // A transparent canvas between guard bytes that catch writes outside it.
  std::vector<std::uint8_t> bytes(kWidth * kHeight * 4 + 2 * kGuardBytes, 0);
  std::fill_n(bytes.begin(), kGuardBytes, kGuard);
  std::fill_n(bytes.end() - kGuardBytes, kGuardBytes, kGuard);
  const RgbaCanvas canvas{bytes.data() + kGuardBytes, kWidth, kHeight};
  const auto pixel = [&](int x, int y) {
    const std::uint8_t* p = canvas.pixels + (static_cast<std::size_t>(y) * kWidth + x) * 4;
    return Rgba{p[0], p[1], p[2], p[3]};
  };
  draw_axes(canvas, *tag_axes(posed_tag(), CameraIntrinsics{400, 400, 50, 150}, 0.2));
  VERIFY(pixel(110, 50) == kRed && pixel(100, 62) == kGreen && pixel(97, 54) == kBlue);
  VERIFY(pixel(116, 47) == kRed || pixel(116, 52) == kRed);  // The X arrowhead.
  VERIFY((pixel(10, 10) == Rgba{0, 0, 0, 0}));
  // Lines leaving the canvas, and one with a NaN end, write nothing outside it.
  draw_line(canvas, {-50, 10}, {kWidth + 50, 10}, kRed);
  draw_line(canvas, {-1e6, -1e6}, {1e6, 1e6}, kGreen);
  draw_line(canvas, {NAN, 0}, {150, 90}, kBlue);
  VERIFY(pixel(0, 10) == kRed && pixel(kWidth - 1, 10) == kRed);
  VERIFY(pixel(150, 90) != kBlue);
  const auto guard = [](std::uint8_t b) { return b == kGuard; };
  VERIFY(std::all_of(bytes.begin(), bytes.begin() + kGuardBytes, guard));
  VERIFY(std::all_of(bytes.end() - kGuardBytes, bytes.end(), guard));
}

}  // namespace
}  // namespace holoscan::apriltag

int main() {
  return holoscan::apriltag::test::run({
      {"axis tips at the projected points", holoscan::apriltag::check_tips},
      {"axes drawn in their colors, inside the image", holoscan::apriltag::check_drawing},
  });
}
