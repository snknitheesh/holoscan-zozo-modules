// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A synthetic scene with known instances, for tests and the demo: a red
// circle, a blue rectangle and a yellow triangle on a mid-gray background,
// drifting slowly from frame to frame. SAM 3 finds them by name ("circle",
// "blue rectangle", "triangle") or by an exemplar box around one. Rendered
// with the shape of every pixel, its true instance; and the intersection over
// union of a mask with a shape. No SDK, no CUDA.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include <holoscan/sam3/pixel_math.hpp>

namespace holoscan::sam3 {

// The smallest and largest synthetic images.
inline constexpr int kMinSceneSide = 32;
inline constexpr int kMaxSceneSide = 4096;

// What a pixel shows.
enum class SceneShape : std::uint8_t { kBackground, kCircle, kRectangle, kTriangle };

// The scene in pixels.
struct ShapeScene {
  int width{};
  int height{};
  std::array<float, 2> circle_center{};
  float circle_radius{};
  // x1, y1, x2, y2.
  std::array<float, 4> rectangle{};
  // Three vertices, x and y.
  std::array<std::array<float, 2>, 3> triangle{};
};

// Frame `frame` of a width x height scene.
// @throws std::invalid_argument for a size outside the synthetic range.
[[nodiscard]] ShapeScene make_shape_scene(int width, int height, std::uint32_t frame);

// What pixel (x, y) shows; pixels are sampled at their centers.
[[nodiscard]] SceneShape shape_at(const ShapeScene& scene, int x, int y);

// Draws `scene` into `pixels`, rows of `pitch` bytes in `format`, alpha 255.
// `shapes`, when not null, gets width x height values, row by row.
void render_shape_scene(const ShapeScene& scene, PixelFormat format, std::uint8_t* pixels,
                        std::size_t pitch, SceneShape* shapes = nullptr);

// The box around `shape` in fractions of the image: x1, y1, x2, y2.
[[nodiscard]] std::array<float, 4> shape_box(const ShapeScene& scene, SceneShape shape);

// The intersection over union of `mask` (nonzero inside) with the pixels of
// `shape`, both width x height row by row; 0 when both are empty.
// @throws std::invalid_argument if the sizes differ.
[[nodiscard]] double mask_iou(std::span<const std::uint8_t> mask,
                              std::span<const SceneShape> shapes, SceneShape shape);

}  // namespace holoscan::sam3
