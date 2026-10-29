// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A synthetic, rectified stereo pair with known disparity, for tests and
// demos: textured rectangles facing the camera at known depths in front of a
// textured background, seen by a left camera and by a right one `baseline`
// meters to its right. The scene is analytic, so the GPU renders it and the
// host computes its truth from the same functions.

#pragma once

#include <cuda_runtime_api.h>
#include <math.h>

#include <cstdint>

#include <holoscan/stereo_disparity/stereo_geometry.hpp>

namespace holoscan::stereo_disparity {

// One plane at depth `z`: the rectangle [x0, x1] x [y0, y1] in meters, or the
// whole view for the background.
struct ScenePlane {
  float z{};
  float x0{};
  float x1{};
  float y0{};
  float y1{};
  bool background{};
  std::uint32_t seed{};
  float tint[3]{};  // Per-channel gain of the plane's gray texture.
};

inline constexpr int kScenePlanes = 4;

struct SyntheticScene {
  int width{};
  int height{};
  float focal{};  // Pixels.
  float cx{};
  float cy{};
  float baseline{};                 // Meters.
  ScenePlane planes[kScenePlanes];  // Nearest first; the last is the background.
};

// Smallest and largest synthetic image sides.
inline constexpr int kMinSceneWidth = 160;
inline constexpr int kMinSceneHeight = 120;
inline constexpr int kMaxSceneSide = 4096;

// Frame `frame` of the scene at `width` x `height`: its rectangles drift a
// little from frame to frame. The focal length is 0.8 image widths and the
// baseline 10 cm, so disparities reach 0.0533 image widths (68 px at 1280).
// @throws std::invalid_argument outside kMinScene* .. kMaxSceneSide.
[[nodiscard]] SyntheticScene make_scene(std::uint32_t frame, int width, int height);

// The largest disparity of any frame of a scene `width` pixels wide.
[[nodiscard]] float max_scene_disparity(int width);

// The scene's camera pair.
[[nodiscard]] StereoGeometry scene_geometry(const SyntheticScene& scene);

namespace detail {

HOLOSCAN_STEREO_DISPARITY_HD inline float hash01(std::int32_t x, std::int32_t y,
                                                 std::uint32_t seed) {
  std::uint32_t h = static_cast<std::uint32_t>(x) * 0x8da6b343U ^
                    static_cast<std::uint32_t>(y) * 0xd8163841U ^ seed * 0xcb1ab31fU;
  h ^= h >> 13U;
  h *= 0x5bd1e995U;
  h ^= h >> 15U;
  return static_cast<float>(h & 0xffffU) / 65535.0F;
}

// Smoothly interpolated lattice noise in [0, 1], one lattice cell per unit.
HOLOSCAN_STEREO_DISPARITY_HD inline float value_noise(float x, float y, std::uint32_t seed) {
  const float fx = floorf(x);
  const float fy = floorf(y);
  const auto ix = static_cast<std::int32_t>(fx);
  const auto iy = static_cast<std::int32_t>(fy);
  float tx = x - fx;
  float ty = y - fy;
  tx = tx * tx * (3.0F - 2.0F * tx);
  ty = ty * ty * (3.0F - 2.0F * ty);
  const float a = hash01(ix, iy, seed);
  const float b = hash01(ix + 1, iy, seed);
  const float c = hash01(ix, iy + 1, seed);
  const float d = hash01(ix + 1, iy + 1, seed);
  const float top = a + (b - a) * tx;
  const float bottom = c + (d - c) * tx;
  return top + (bottom - top) * ty;
}

}  // namespace detail

// The plane the camera `camera_x` meters right of the left camera sees at
// pixel (u, v), and where on it: an index into scene.planes.
HOLOSCAN_STEREO_DISPARITY_HD inline int visible_plane(const SyntheticScene& scene, float camera_x,
                                                      float u, float v, float* x, float* y) {
  for (int i = 0; i < kScenePlanes; ++i) {
    const ScenePlane& plane = scene.planes[i];
    const float px = (u - scene.cx) * plane.z / scene.focal + camera_x;
    const float py = (v - scene.cy) * plane.z / scene.focal;
    if (plane.background ||
        (px >= plane.x0 && px <= plane.x1 && py >= plane.y0 && py <= plane.y1)) {
      *x = px;
      *y = py;
      return i;
    }
  }
  return kScenePlanes - 1;  // Not reached: the last plane is the background.
}

// The RGB the camera `camera_x` meters right of the left camera sees at pixel
// (u, v): two octaves of noise, about 3 and 12 pixels across at the plane's
// depth.
HOLOSCAN_STEREO_DISPARITY_HD inline void scene_color(const SyntheticScene& scene, float camera_x,
                                                     float u, float v, std::uint8_t* rgb) {
  float x = 0.0F;
  float y = 0.0F;
  const ScenePlane& plane = scene.planes[visible_plane(scene, camera_x, u, v, &x, &y)];
  const float cell = 3.0F * plane.z / scene.focal;
  const float fine = detail::value_noise(x / cell, y / cell, plane.seed);
  const float coarse = detail::value_noise(x / (4.0F * cell), y / (4.0F * cell), plane.seed + 1U);
  const float gray = 30.0F + 200.0F * (0.6F * fine + 0.4F * coarse);
  for (int c = 0; c < 3; ++c) {
    rgb[c] = static_cast<std::uint8_t>(fminf(255.0F, gray * plane.tint[c]) + 0.5F);
  }
}

// The true disparity of left pixel (u, v), or a negative value where the right
// camera does not see the same point: occluded, or outside its view.
HOLOSCAN_STEREO_DISPARITY_HD inline float true_disparity(const SyntheticScene& scene, float u,
                                                         float v) {
  float x = 0.0F;
  float y = 0.0F;
  const int plane = visible_plane(scene, 0.0F, u, v, &x, &y);
  const float disparity = scene.focal * scene.baseline / scene.planes[plane].z;
  const float right_u = u - disparity;
  if (right_u < 0.0F || visible_plane(scene, scene.baseline, right_u, v, &x, &y) != plane) {
    return -1.0F;
  }
  return disparity;
}

// Renders the scene's left and right views into device memory as packed rows
// of `channels` bytes a pixel: 3 for RGB8, 1 for MONO8 (the green channel).
// Queued on `stream`.
[[nodiscard]] cudaError_t render_scene(const SyntheticScene& scene, int channels,
                                       std::uint8_t* left, std::uint8_t* right,
                                       cudaStream_t stream);

}  // namespace holoscan::stereo_disparity
