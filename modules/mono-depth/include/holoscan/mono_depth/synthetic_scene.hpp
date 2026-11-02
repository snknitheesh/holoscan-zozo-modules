// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A synthetic 3D scene with known depth, for tests and the demo: a pinhole
// camera 1.2 m above a checkered floor that recedes to a far wall, a textured
// panel standing near the camera on the left, and a shaded sphere resting on
// the floor in between, which moves from frame to frame. Rendered by ray
// casting, with its true depth and the region of every pixel. And the check
// of a predicted relative inverse depth against it: a monocular model cannot
// know the scale, but it must rank the pixels by nearness as the truth does
// (Spearman's rank correlation), and the regions in their order: the panel
// nearer than the sphere, the sphere nearer than the wall. No SDK, no CUDA.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <holoscan/mono_depth/image_fit.hpp>

namespace holoscan::mono_depth {

// The smallest and largest synthetic images.
inline constexpr int kMinSceneWidth = 32;
inline constexpr int kMinSceneHeight = 32;
inline constexpr int kMaxSceneSide = 4096;

// What a pixel shows.
enum class SceneRegion : std::uint8_t { kFloor, kWall, kPanel, kSphere };

// The scene's geometry in camera coordinates (x right, y down, z forward, in
// meters) and the camera's intrinsics in pixels.
struct DepthScene {
  int width{};
  int height{};
  float focal{};
  float cx{};
  float cy{};
  // The floor is the plane y = floor_y, the wall the plane z = wall_z.
  float floor_y{};
  float wall_z{};
  // The panel: x in [panel_left, panel_right], y in [panel_top, floor_y], at
  // z = panel_z.
  float panel_left{};
  float panel_right{};
  float panel_top{};
  float panel_z{};
  std::array<float, 3> sphere_center{};
  float sphere_radius{};
};

// Frame `frame` of a width x height scene.
// @throws std::invalid_argument for a size outside the synthetic range.
[[nodiscard]] DepthScene make_depth_scene(int width, int height, std::uint32_t frame);

// Draws `scene` into `pixels`, rows of `pitch` bytes in `format`, alpha 255.
// `depth` (meters along z) and `regions`, when not null, get width x height
// values each, row by row.
void render_depth_scene(const DepthScene& scene, PixelFormat format, std::uint8_t* pixels,
                        std::size_t pitch, float* depth = nullptr, SceneRegion* regions = nullptr);

// Spearman's rank correlation of two equally long series, ties ranked by
// their average; 0 when either has fewer than two distinct values.
[[nodiscard]] double spearman(std::span<const float> a, std::span<const float> b);

struct OrderingResult {
  // Spearman's correlation of the predicted inverse depth with the true one,
  // over every `stride`-th pixel along each axis.
  double spearman{};
  // The median predicted inverse depth in each region, by SceneRegion.
  std::array<double, 4> median{};
  // Whether the panel's median exceeds the sphere's, and the sphere's the
  // wall's: nearer regions are predicted nearer.
  bool ordered{};
};

// Checks a predicted inverse depth, width x height values row by row (larger
// is nearer, any scale and shift), against `scene`.
// @throws std::invalid_argument if `inverse_depth` does not hold the image.
[[nodiscard]] OrderingResult check_ordering(const DepthScene& scene,
                                            std::span<const float> inverse_depth, int stride = 4);

}  // namespace holoscan::mono_depth
