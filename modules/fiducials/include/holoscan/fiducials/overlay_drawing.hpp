// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What FiducialOverlayOp draws into its RGBA8 layer: markers' outlines, chessboard corners and
// targets' axes. Free of the SDK, so its test runs anywhere.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include <holoscan/fiducials/pose.hpp>
#include <holoscan/fiducials/types.hpp>

namespace holoscan::fiducials {

using Rgba = std::array<std::uint8_t, 4>;

inline constexpr Rgba kRed{255, 0, 0, 255};
inline constexpr Rgba kGreen{0, 255, 0, 255};
inline constexpr Rgba kBlue{0, 0, 255, 255};
inline constexpr Rgba kYellow{255, 255, 0, 255};
inline constexpr Rgba kMagenta{255, 0, 255, 255};
inline constexpr std::array<Rgba, 3> kAxisColors{kRed, kGreen, kBlue};
inline constexpr double kArrowheadPx = 6.0;

// An RGBA8 image, rows packed.
struct RgbaCanvas {
  std::uint8_t* pixels;
  int width;
  int height;
};

// A 2 px line from a to b, clipped to the canvas.
inline void draw_line(const RgbaCanvas& canvas, Point2 a, Point2 b, const Rgba& color) {
  const double dx = b[0] - a[0];
  const double dy = b[1] - a[1];
  if (!std::isfinite(dx) || !std::isfinite(dy)) return;
  // Liang-Barsky: the part of a + t (b - a), 0 <= t <= 1, within a pixel of the canvas.
  double t0 = 0.0;
  double t1 = 1.0;
  const auto clip = [&](double p, double q) {
    if (p == 0.0) return q >= 0.0;
    const double t = q / p;
    if (p < 0.0) {
      t0 = std::max(t0, t);
    } else {
      t1 = std::min(t1, t);
    }
    return t0 <= t1;
  };
  if (!clip(-dx, a[0] + 1.0) || !clip(dx, canvas.width - a[0]) || !clip(-dy, a[1] + 1.0) ||
      !clip(dy, canvas.height - a[1])) {
    return;
  }
  const Point2 from{a[0] + t0 * dx, a[1] + t0 * dy};
  const double span_x = (t1 - t0) * dx;
  const double span_y = (t1 - t0) * dy;
  // At most a pixel per step, setting the 2 x 2 pixels around each point.
  const int steps = static_cast<int>(std::ceil(std::max(std::abs(span_x), std::abs(span_y))));
  for (int s = 0; s <= steps; ++s) {
    const double t = steps == 0 ? 0.0 : static_cast<double>(s) / steps;
    const int left = static_cast<int>(std::floor(from[0] + t * span_x - 0.5));
    const int top = static_cast<int>(std::floor(from[1] + t * span_y - 0.5));
    for (int y = top; y <= top + 1; ++y) {
      for (int x = left; x <= left + 1; ++x) {
        if (x >= 0 && y >= 0 && x < canvas.width && y < canvas.height) {
          std::copy(color.begin(), color.end(),
                    canvas.pixels + (static_cast<std::size_t>(y) * canvas.width + x) * 4);
        }
      }
    }
  }
}

// A marker's outline, with its first (top-left) corner marked by a small square.
inline void draw_marker(const RgbaCanvas& canvas, const std::array<Point2, 4>& corners) {
  for (int i = 0; i < 4; ++i) draw_line(canvas, corners[i], corners[(i + 1) % 4], kYellow);
  const Point2& c = corners[0];
  const std::array<Point2, 4> box{{{c[0] - 3, c[1] - 3}, {c[0] + 3, c[1] - 3},
                                   {c[0] + 3, c[1] + 3}, {c[0] - 3, c[1] + 3}}};
  for (int i = 0; i < 4; ++i) draw_line(canvas, box[i], box[(i + 1) % 4], kRed);
}

// A chessboard corner as a small cross.
inline void draw_corner(const RgbaCanvas& canvas, const Point2& p) {
  draw_line(canvas, {p[0] - 4, p[1]}, {p[0] + 4, p[1]}, kMagenta);
  draw_line(canvas, {p[0], p[1] - 4}, {p[0], p[1] + 4}, kMagenta);
}

// Where a target's origin and its X, Y and Z tips appear; a missing tip is not drawn.
struct TargetAxes {
  Point2 origin;
  std::array<std::optional<Point2>, 3> tips;
};

// The origin of a target at `pose` and points `length` along its x, y and z (z into the
// target), projected; nullopt when the origin is behind the camera. A tip behind the camera is
// left out.
inline std::optional<TargetAxes> target_axes(const Pose& pose, const CameraIntrinsics& intrinsics,
                                             double length) {
  const auto seen = [&](const Point3& p) -> std::optional<Point2> {
    const auto& r = pose.rotation;
    const auto& t = pose.translation;
    const double z = r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2];
    if (!(z > 1e-6)) return std::nullopt;
    return project(intrinsics, pose, p);
  };
  const auto origin = seen({0, 0, 0});
  if (!origin) return std::nullopt;
  return TargetAxes{*origin, {seen({length, 0, 0}), seen({0, length, 0}), seen({0, 0, length})}};
}

// Each axis as a line from the origin to its tip with a small arrowhead, red, green, blue.
inline void draw_axes(const RgbaCanvas& canvas, const TargetAxes& axes) {
  for (std::size_t i = 0; i < axes.tips.size(); ++i) {
    if (!axes.tips[i]) continue;
    const Point2 tip = *axes.tips[i];
    draw_line(canvas, axes.origin, tip, kAxisColors[i]);
    const double length = std::hypot(tip[0] - axes.origin[0], tip[1] - axes.origin[1]);
    if (!(length > 1.0)) continue;
    // Two strokes back from the tip, 25 degrees either side of the shaft.
    const double ux = (axes.origin[0] - tip[0]) / length;
    const double uy = (axes.origin[1] - tip[1]) / length;
    for (const double angle : {0.436, -0.436}) {
      const double c = std::cos(angle) * kArrowheadPx;
      const double s = std::sin(angle) * kArrowheadPx;
      draw_line(canvas, tip, {tip[0] + c * ux - s * uy, tip[1] + s * ux + c * uy}, kAxisColors[i]);
    }
  }
}

}  // namespace holoscan::fiducials
