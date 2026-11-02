// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

#include <holoscan/sam3/synthetic_scene.hpp>

namespace holoscan::sam3 {
namespace {

constexpr std::array<std::uint8_t, 3> kBackground{128, 128, 128};
constexpr std::array<std::uint8_t, 3> kRed{220, 30, 30};
constexpr std::array<std::uint8_t, 3> kBlue{30, 60, 220};
constexpr std::array<std::uint8_t, 3> kYellow{240, 220, 30};

// Which side of the edge from a to b the point p is on.
float edge(const std::array<float, 2>& a, const std::array<float, 2>& b, float px, float py) {
  return (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0]);
}

}  // namespace

ShapeScene make_shape_scene(int width, int height, std::uint32_t frame) {
  if (width < kMinSceneSide || height < kMinSceneSide || width > kMaxSceneSide ||
      height > kMaxSceneSide) {
    throw std::invalid_argument("synthetic images are " + std::to_string(kMinSceneSide) + ".." +
                                std::to_string(kMaxSceneSide) + " pixels on a side");
  }
  const auto w = static_cast<float>(width);
  const auto h = static_cast<float>(height);
  // A slow drift, a few pixels per frame at 640 x 480, that repeats.
  const float phase = static_cast<float>(frame % 240U) * (2.0F * 3.14159265F / 240.0F);
  const float dx = 0.03F * std::sin(phase);
  const float dy = 0.03F * std::cos(phase);
  const float side = std::min(w, h);
  ShapeScene scene;
  scene.width = width;
  scene.height = height;
  scene.circle_center = {(0.22F + dx) * w, (0.33F + dy) * h};
  scene.circle_radius = 0.17F * side;
  scene.rectangle = {(0.55F - dx) * w, (0.14F + dy) * h, (0.86F - dx) * w, (0.42F + dy) * h};
  scene.triangle = {{{(0.32F + dx) * w, (0.91F - dy) * h},
                     {(0.50F + dx) * w, (0.58F - dy) * h},
                     {(0.68F + dx) * w, (0.91F - dy) * h}}};
  return scene;
}

SceneShape shape_at(const ShapeScene& scene, int x, int y) {
  const float px = static_cast<float>(x) + 0.5F;
  const float py = static_cast<float>(y) + 0.5F;
  // Drawn in this order, the last on top.
  const auto& t = scene.triangle;
  const float e0 = edge(t[0], t[1], px, py);
  const float e1 = edge(t[1], t[2], px, py);
  const float e2 = edge(t[2], t[0], px, py);
  if ((e0 >= 0.0F && e1 >= 0.0F && e2 >= 0.0F) || (e0 <= 0.0F && e1 <= 0.0F && e2 <= 0.0F)) {
    return SceneShape::kTriangle;
  }
  const auto& r = scene.rectangle;
  if (px >= r[0] && px < r[2] && py >= r[1] && py < r[3]) {
    return SceneShape::kRectangle;
  }
  const float cx = px - scene.circle_center[0];
  const float cy = py - scene.circle_center[1];
  if (cx * cx + cy * cy <= scene.circle_radius * scene.circle_radius) {
    return SceneShape::kCircle;
  }
  return SceneShape::kBackground;
}

void render_shape_scene(const ShapeScene& scene, PixelFormat format, std::uint8_t* pixels,
                        std::size_t pitch, SceneShape* shapes) {
  const int n = channels(format);
  const bool bgr = blue_first(format);
  for (int y = 0; y < scene.height; ++y) {
    std::uint8_t* row = pixels + static_cast<std::size_t>(y) * pitch;
    for (int x = 0; x < scene.width; ++x) {
      const SceneShape shape = shape_at(scene, x, y);
      const std::array<std::uint8_t, 3>* color = &kBackground;
      if (shape == SceneShape::kCircle) {
        color = &kRed;
      } else if (shape == SceneShape::kRectangle) {
        color = &kBlue;
      } else if (shape == SceneShape::kTriangle) {
        color = &kYellow;
      }
      std::uint8_t* pixel = row + static_cast<std::size_t>(x) * n;
      pixel[0] = (*color)[bgr ? 2 : 0];
      pixel[1] = (*color)[1];
      pixel[2] = (*color)[bgr ? 0 : 2];
      if (n == 4) {
        pixel[3] = 255;
      }
      if (shapes != nullptr) {
        shapes[static_cast<std::size_t>(y) * scene.width + x] = shape;
      }
    }
  }
}

std::array<float, 4> shape_box(const ShapeScene& scene, SceneShape shape) {
  const auto w = static_cast<float>(scene.width);
  const auto h = static_cast<float>(scene.height);
  std::array<float, 4> box{};
  if (shape == SceneShape::kCircle) {
    const float r = scene.circle_radius;
    box = {scene.circle_center[0] - r, scene.circle_center[1] - r, scene.circle_center[0] + r,
           scene.circle_center[1] + r};
  } else if (shape == SceneShape::kRectangle) {
    box = scene.rectangle;
  } else if (shape == SceneShape::kTriangle) {
    const auto& t = scene.triangle;
    box = {std::min({t[0][0], t[1][0], t[2][0]}), std::min({t[0][1], t[1][1], t[2][1]}),
           std::max({t[0][0], t[1][0], t[2][0]}), std::max({t[0][1], t[1][1], t[2][1]})};
  } else {
    box = {0.0F, 0.0F, w, h};
  }
  return {std::clamp(box[0] / w, 0.0F, 1.0F), std::clamp(box[1] / h, 0.0F, 1.0F),
          std::clamp(box[2] / w, 0.0F, 1.0F), std::clamp(box[3] / h, 0.0F, 1.0F)};
}

double mask_iou(std::span<const std::uint8_t> mask, std::span<const SceneShape> shapes,
                SceneShape shape) {
  if (mask.size() != shapes.size()) {
    throw std::invalid_argument("mask_iou: the mask and the shapes differ in size");
  }
  std::size_t intersection = 0;
  std::size_t either = 0;
  for (std::size_t i = 0; i < mask.size(); ++i) {
    const bool in_mask = mask[i] != 0;
    const bool in_shape = shapes[i] == shape;
    intersection += in_mask && in_shape ? 1U : 0U;
    either += in_mask || in_shape ? 1U : 0U;
  }
  return either == 0 ? 0.0 : static_cast<double>(intersection) / static_cast<double>(either);
}

}  // namespace holoscan::sam3
