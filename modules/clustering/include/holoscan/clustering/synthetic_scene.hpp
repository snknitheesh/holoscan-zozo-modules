// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A synthetic depth camera over objects on a removed ground, for tests,
// benchmarks and demos. A camera above flat ground, pitched down, sees boxes
// and spheres of a range of sizes standing on it: a small box resting on a
// larger one (touching: one object), a sphere 5 cm from a box and two spheres
// 4 cm apart (nearly touching: separate objects), a pole, a wall, a box too
// small to count and a sphere that moves from frame to frame. The ground
// itself has no depth, as if a ground filter had removed it. Depth has noise,
// some pixels have none, and some are flying pixels at a random depth along
// their ray.
//
// Each pixel knows which object it shows, so clusters can be checked against
// the truth. The same functions run on the GPU to render depth images and on
// the host to make the clouds; with nothing contracted (the module renders
// with --fmad=false; host code must use -ffp-contract=off) both give the same
// bits.

#pragma once

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

#if defined(__CUDACC__)
#define HOLOSCAN_CLUSTERING_HOST_DEVICE __host__ __device__
#else
#define HOLOSCAN_CLUSTERING_HOST_DEVICE
#endif

namespace holoscan::clustering {

// What a pixel shows, other than an object.
inline constexpr std::int32_t kSceneNoDepth = -2;  // The ground, a missing pixel, or too far.
inline constexpr std::int32_t kSceneFlying = -1;   // A flying pixel.
// Objects, by group: touching objects are one group. Group kSceneTinyGroup
// is a box of a few hundred pixels at most, below a sensible minimum size.
inline constexpr int kSceneGroups = 9;
inline constexpr int kSceneTinyGroup = 7;

// One frame of the scene. Make it with synthetic_scene(), which computes the
// values derived on the host only, then change the settings if needed.
struct SyntheticScene {
  int width{};
  int height{};
  // Pinhole intrinsics in pixels, pixel centers at integer coordinates.
  float fx{};
  float fy{};
  float cx{};
  float cy{};
  // The world is x forward, y left, z up, with the ground at z = 0; the
  // camera is camera_height above the world origin, pitched down.
  float camera_height{1.0F};
  float pitch_sin{};
  float pitch_cos{};
  // The moving sphere's y.
  float moving_y{};
  std::uint32_t frame{};
  std::uint32_t seed{1U};
  // Zero-mean depth noise of this standard deviation in meters (the sum of
  // four uniform draws, close to Gaussian).
  float noise_sigma{0.002F};
  // Fractions of pixels that are flying pixels and that have no depth.
  float flying_rate{0.003F};
  float invalid_rate{0.01F};
  // Surfaces farther than this (along the ray, in meters) are not seen.
  float max_range{6.0F};
};

inline SyntheticScene synthetic_scene(int width, int height, std::uint32_t frame,
                                      float pitch_deg = 30.0F) {
  SyntheticScene scene;
  scene.width = width;
  scene.height = height;
  scene.fx = 0.75F * static_cast<float>(width);
  scene.fy = scene.fx;
  scene.cx = static_cast<float>(width - 1) / 2.0F;
  scene.cy = static_cast<float>(height - 1) / 2.0F;
  const double pitch = static_cast<double>(pitch_deg) * std::numbers::pi / 180.0;
  scene.pitch_sin = static_cast<float>(std::sin(pitch));
  scene.pitch_cos = static_cast<float>(std::cos(pitch));
  scene.moving_y = static_cast<float>(0.2 * std::sin(0.07 * frame) - 0.3);
  scene.frame = frame;
  return scene;
}

struct ScenePixel {
  // Camera z in meters, as a depth camera reports it; 0 without depth.
  float depth{};
  // The object group, kSceneFlying or kSceneNoDepth.
  std::int32_t group{kSceneNoDepth};
};

namespace detail {

HOLOSCAN_CLUSTERING_HOST_DEVICE inline std::uint32_t scene_hash(std::uint32_t value) {
  value ^= value >> 16U;
  value *= 0x7FEB352DU;
  value ^= value >> 15U;
  value *= 0x846CA68BU;
  value ^= value >> 16U;
  return value;
}

// A uniform draw in [0, 1) for pixel `pixel` of the frame, stream `which`.
HOLOSCAN_CLUSTERING_HOST_DEVICE inline float scene_uniform(const SyntheticScene& s,
                                                           std::uint32_t pixel,
                                                           std::uint32_t which) {
  const std::uint32_t h =
      scene_hash(scene_hash(scene_hash(s.seed + 0x9E3779B9U * which) ^ s.frame) ^ pixel);
  return static_cast<float>(h >> 8U) * (1.0F / 16777216.0F);
}

// A box standing upright, turned by `yaw` about the vertical: its center,
// half extents, and the yaw's cosine and sine.
struct SceneBox {
  float center[3];
  float half[3];
  float cos_yaw;
  float sin_yaw;
  int group;
};

// Where the ray o + t d enters the box, or a negative t.
HOLOSCAN_CLUSTERING_HOST_DEVICE inline float scene_box(const float o[3], const float d[3],
                                                       const SceneBox& box) {
  // The ray in the box's frame.
  const float rx = o[0] - box.center[0];
  const float ry = o[1] - box.center[1];
  const float lo[3] = {box.cos_yaw * rx + box.sin_yaw * ry, -box.sin_yaw * rx + box.cos_yaw * ry,
                       o[2] - box.center[2]};
  const float ld[3] = {box.cos_yaw * d[0] + box.sin_yaw * d[1],
                       -box.sin_yaw * d[0] + box.cos_yaw * d[1], d[2]};
  float near = -1.0e30F;
  float far = 1.0e30F;
  for (int axis = 0; axis < 3; ++axis) {
    if (ld[axis] == 0.0F) {
      if (lo[axis] < -box.half[axis] || lo[axis] > box.half[axis]) {
        return -1.0F;
      }
      continue;
    }
    float t0 = (-box.half[axis] - lo[axis]) / ld[axis];
    float t1 = (box.half[axis] - lo[axis]) / ld[axis];
    if (t0 > t1) {
      const float swap = t0;
      t0 = t1;
      t1 = swap;
    }
    near = t0 > near ? t0 : near;
    far = t1 < far ? t1 : far;
  }
  return near <= far && near > 0.0F ? near : -1.0F;
}

// Where the ray o + t d enters the sphere, or a negative t.
HOLOSCAN_CLUSTERING_HOST_DEVICE inline float scene_sphere(const float o[3], const float d[3],
                                                          float cx, float cy, float cz,
                                                          float radius) {
  const float oc[3] = {o[0] - cx, o[1] - cy, o[2] - cz};
  const float a = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  const float b = 2.0F * (d[0] * oc[0] + d[1] * oc[1] + d[2] * oc[2]);
  const float c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - radius * radius;
  const float discriminant = b * b - 4.0F * a * c;
  if (discriminant < 0.0F) {
    return -1.0F;
  }
  return (-b - sqrtf(discriminant)) / (2.0F * a);
}

}  // namespace detail

// What pixel (u, v) of the scene shows.
HOLOSCAN_CLUSTERING_HOST_DEVICE inline ScenePixel synthetic_scene_pixel(const SyntheticScene& s,
                                                                        int u, int v) {
  ScenePixel pixel;
  const auto index = static_cast<std::uint32_t>(v) * static_cast<std::uint32_t>(s.width) +
                     static_cast<std::uint32_t>(u);
  if (detail::scene_uniform(s, index, 0U) < s.invalid_rate) {
    return pixel;
  }
  // The ray through the pixel, in the world, scaled so that t is camera z.
  const float dx = (static_cast<float>(u) - s.cx) / s.fx;
  const float dy = (static_cast<float>(v) - s.cy) / s.fy;
  // Camera x is world -y; camera y is (-sin, 0, -cos); camera z (cos, 0, -sin).
  const float d[3] = {-dy * s.pitch_sin + s.pitch_cos, -dx, -dy * s.pitch_cos - s.pitch_sin};
  const float o[3] = {0.0F, 0.0F, s.camera_height};
  const float ray_length = sqrtf(dx * dx + dy * dy + 1.0F);
  if (detail::scene_uniform(s, index, 1U) < s.flying_rate) {
    pixel.depth = 0.5F + 5.0F * detail::scene_uniform(s, index, 2U);
    pixel.group = kSceneFlying;
    return pixel;
  }
  // cos and sin of 45, -10, 30 and 0 degrees.
  constexpr float kC45 = 0.70710678F;
  constexpr float kC10 = 0.98480775F;
  constexpr float kS10 = 0.17364818F;
  constexpr float kC30 = 0.86602540F;
  const detail::SceneBox boxes[6] = {
      // A box, and a small box resting on it: touching, one object.
      {{1.8F, 0.55F, 0.2F}, {0.2F, 0.15F, 0.2F}, kC45, kC45, 0},
      {{1.8F, 0.55F, 0.48F}, {0.08F, 0.08F, 0.08F}, kC10, -kS10, 0},
      // A tall box 5 cm from the sphere of group 1, its top above the camera.
      {{2.6F, -0.85F, 0.55F}, {0.15F, 0.15F, 0.55F}, kC30, 0.5F, 2},
      // A pole and a wall.
      {{2.2F, -1.2F, 0.5F}, {0.05F, 0.05F, 0.5F}, 1.0F, 0.0F, 5},
      {{4.8F, -0.2F, 0.4F}, {0.05F, 0.9F, 0.4F}, 1.0F, 0.0F, 6},
      // A box too small to count.
      {{2.0F, -0.4F, 0.012F}, {0.012F, 0.012F, 0.012F}, 1.0F, 0.0F, kSceneTinyGroup},
  };
  // Spheres resting on the ground: x, y, radius, group. The last one moves.
  const float spheres[4][4] = {{2.6F, -0.35F, 0.25F, 1.0F},
                               {3.4F, 0.6F, 0.18F, 3.0F},
                               {3.4F, 1.0F, 0.18F, 4.0F},
                               {1.4F, s.moving_y, 0.12F, 8.0F}};
  float best = 1.0e30F;
  int hit = kSceneNoDepth;
  if (d[2] < 0.0F) {
    best = -s.camera_height / d[2];  // The ground, which has no depth.
  }
  for (const detail::SceneBox& box : boxes) {
    const float t = detail::scene_box(o, d, box);
    if (t > 0.0F && t < best) {
      best = t;
      hit = box.group;
    }
  }
  for (const auto& sphere : spheres) {
    const float t = detail::scene_sphere(o, d, sphere[0], sphere[1], sphere[2], sphere[2]);
    if (t > 0.0F && t < best) {
      best = t;
      hit = static_cast<int>(sphere[3]);
    }
  }
  if (hit < 0 || best * ray_length > s.max_range) {
    return pixel;
  }
  // Noise: four uniform draws summed have variance 1/3.
  float sum = 0.0F;
  for (std::uint32_t k = 0; k < 4U; ++k) {
    sum += detail::scene_uniform(s, index, 3U + k);
  }
  pixel.depth = best + s.noise_sigma * 1.7320508F * (sum - 2.0F);
  pixel.group = hit;
  return pixel;
}

// The camera-frame point of a pixel at depth z, as DepthToPointCloudOp
// deprojects it.
HOLOSCAN_CLUSTERING_HOST_DEVICE inline void synthetic_scene_point(const SyntheticScene& s, int u,
                                                                  int v, float z, float xyz[3]) {
  xyz[0] = (static_cast<float>(u) - s.cx) * z / s.fx;
  xyz[1] = (static_cast<float>(v) - s.cy) * z / s.fy;
  xyz[2] = z;
}

// The organized cloud of the scene on the host: [height, width, 3] float32,
// NaN where there is no depth, and each pixel's group.
void synthetic_scene_cloud(const SyntheticScene& scene, std::vector<float>& xyz,
                           std::vector<std::int32_t>& groups);

// Renders the scene on the GPU, into device memory on `stream`: `depth`
// float32 meters [height, width], 0 without depth; `groups` int32 when given;
// `xyz` the organized cloud, NaN without depth, when given.
cudaError_t render_synthetic_scene(const SyntheticScene& scene, float* depth, std::int32_t* groups,
                                   float* xyz, cudaStream_t stream);

}  // namespace holoscan::clustering
