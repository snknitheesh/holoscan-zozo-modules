// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A synthetic scene with known surfaces, rendered on the GPU into the depth
// and color images of a pinhole camera at a known pose, with optional depth
// noise; camera paths around it; and the scene's exact distance and depth
// functions, on the host, to check a reconstruction against. For tests,
// benchmarks and demos; nothing here needs the Holoscan SDK.
//
// The world is z up. The scenes:
//   kTabletop  a floor (the plane z = 0, solid below), a box standing on it
//              and a sphere floating above it.
//   kSphere    the sphere alone: a closed surface, for cameras all round it.
//
// The functions marked HOLOSCAN_TSDF_HOST_DEVICE run on the host and the GPU;
// built without contracting floating-point arithmetic (--fmad=false,
// -ffp-contract=off) they give the same results on both.

#pragma once

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>

#include <holoscan/tsdf/tsdf_volume.hpp>

#if defined(__CUDACC__)
#define HOLOSCAN_TSDF_HOST_DEVICE __host__ __device__
#else
#define HOLOSCAN_TSDF_HOST_DEVICE
#endif

namespace holoscan::tsdf {

enum class SceneKind : std::uint8_t { kTabletop, kSphere };

struct SyntheticScene {
  bool floor{true};  // The plane z = 0, solid below.
  bool box{true};
  float box_min[3]{-0.30F, -0.22F, 0.0F};
  float box_max[3]{-0.06F, 0.06F, 0.22F};
  float sphere_center[3]{0.16F, 0.06F, 0.26F};
  float sphere_radius{0.15F};
};

[[nodiscard]] SyntheticScene synthetic_scene(SceneKind kind);

// What a ray hits.
enum class SceneSurface : std::uint8_t { kNone, kFloor, kBox, kSphere };

// The first surface along the ray origin + t * direction, t > 0, and its t;
// kNone (t 0) when nothing lies within t_max. `direction` need not be a unit
// vector: with direction's camera z component 1, t is the camera depth.
HOLOSCAN_TSDF_HOST_DEVICE inline SceneSurface scene_ray(const SyntheticScene& s,
                                                        const float origin[3],
                                                        const float direction[3], float t_max,
                                                        float* t_hit) {
  SceneSurface hit = SceneSurface::kNone;
  float best = t_max;
  if (s.floor && origin[2] > 0.0F && direction[2] < 0.0F) {
    const float t = -origin[2] / direction[2];
    if (t > 0.0F && t < best) {
      best = t;
      hit = SceneSurface::kFloor;
    }
  }
  if (s.box) {
    float t_near = 0.0F;
    float t_far = best;
    bool missed = false;
    for (int axis = 0; axis < 3; ++axis) {
      const float o = origin[axis];
      const float d = direction[axis];
      if (d == 0.0F) {
        if (o < s.box_min[axis] || o > s.box_max[axis]) {
          missed = true;
        }
        continue;
      }
      float t0 = (s.box_min[axis] - o) / d;
      float t1 = (s.box_max[axis] - o) / d;
      if (t0 > t1) {
        const float swap = t0;
        t0 = t1;
        t1 = swap;
      }
      t_near = t0 > t_near ? t0 : t_near;
      t_far = t1 < t_far ? t1 : t_far;
    }
    if (!missed && t_near > 0.0F && t_near <= t_far && t_near < best) {
      best = t_near;
      hit = SceneSurface::kBox;
    }
  }
  {
    const float oc[3] = {origin[0] - s.sphere_center[0], origin[1] - s.sphere_center[1],
                         origin[2] - s.sphere_center[2]};
    const float a =
        direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2];
    const float b = oc[0] * direction[0] + oc[1] * direction[1] + oc[2] * direction[2];
    const float c =
        oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - s.sphere_radius * s.sphere_radius;
    const float disc = b * b - a * c;
    if (disc >= 0.0F) {
      const float root = sqrtf(disc);
      float t = (-b - root) / a;
      if (t <= 0.0F) {
        t = (-b + root) / a;  // From inside the sphere.
      }
      if (t > 0.0F && t < best) {
        best = t;
        hit = SceneSurface::kSphere;
      }
    }
  }
  *t_hit = hit == SceneSurface::kNone ? 0.0F : best;
  return hit;
}

// The camera depth of pixel (u, v) of a camera at `world_from_camera`: the z
// of the first surface on the pixel's ray, 0 where nothing lies within
// max_depth.
HOLOSCAN_TSDF_HOST_DEVICE inline float scene_depth(const SyntheticScene& s, const float* pose,
                                                   const CameraIntrinsics& k, float u, float v,
                                                   float max_depth, SceneSurface* surface) {
  const float ray[3] = {(u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0F};
  float direction[3];
  for (int row = 0; row < 3; ++row) {
    direction[row] =
        pose[4 * row] * ray[0] + pose[4 * row + 1] * ray[1] + pose[4 * row + 2] * ray[2];
  }
  const float origin[3] = {pose[3], pose[7], pose[11]};
  float t = 0.0F;
  const SceneSurface hit = scene_ray(s, origin, direction, max_depth, &t);
  if (surface != nullptr) {
    *surface = hit;
  }
  return t;
}

// The color of a surface point.
HOLOSCAN_TSDF_HOST_DEVICE inline void scene_color(SceneSurface surface, const float p[3],
                                                  std::uint8_t rgb[3]) {
  switch (surface) {
    case SceneSurface::kFloor: {
      // 10 cm checkers.
      const int cx = static_cast<int>(floorf(p[0] * 10.0F));
      const int cy = static_cast<int>(floorf(p[1] * 10.0F));
      const bool light = ((cx + cy) & 1) == 0;
      rgb[0] = light ? 200U : 90U;
      rgb[1] = light ? 200U : 90U;
      rgb[2] = light ? 190U : 100U;
      return;
    }
    case SceneSurface::kBox:
      rgb[0] = 210U;
      rgb[1] = 80U;
      rgb[2] = 50U;
      return;
    case SceneSurface::kSphere: {
      // Bands 4 cm high.
      const bool band = (static_cast<int>(floorf(p[2] * 25.0F)) & 1) == 0;
      rgb[0] = 40U;
      rgb[1] = band ? 170U : 110U;
      rgb[2] = band ? 220U : 160U;
      return;
    }
    case SceneSurface::kNone:
    default:
      rgb[0] = 0U;
      rgb[1] = 0U;
      rgb[2] = 0U;
      return;
  }
}

// The signed distance from `p` to the scene's surfaces: negative inside a
// solid. Exact outside the solids and on the surfaces.
[[nodiscard]] float scene_distance(const SyntheticScene& s, const float p[3]);

// A camera of `width` x `height` pixels with a horizontal field of view of
// about 64 degrees, square pixels, centered.
[[nodiscard]] CameraIntrinsics synthetic_intrinsics(int width, int height);

// A camera at `eye` looking at `target`, the image's up toward world z.
[[nodiscard]] Matrix4 look_at(const float eye[3], const float target[3]);

// A camera circling the scene, looking at its middle: pose `frame` of
// `frames`. kTabletop: once round, 1.0 m out, 20 to 50 degrees above the
// floor. kSphere: three times round, 0.9 m out, swinging from 70 degrees
// below the sphere's middle to 70 degrees above and back, so every side is
// seen.
[[nodiscard]] Matrix4 orbit_pose(SceneKind kind, std::uint32_t frame, std::uint32_t frames);

// A volume config covering the scene, `dims` voxels on a side: kTabletop
// x, y in [-0.5, 0.5] and z from -0.1 m; kSphere a cube 0.5 m on a side around
// the sphere.
[[nodiscard]] TsdfVolumeConfig scene_volume(SceneKind kind, int dims, bool color);

// Depth noise: each depth z gets noise of standard deviation
// sigma + sigma_per_m2 * z^2, from a sum of four uniform draws seeded by
// `seed` and the pixel, so the same seed gives the same noise on any GPU and
// the noise is never beyond 3.5 standard deviations.
struct DepthNoise {
  float sigma{};
  float sigma_per_m2{};
  std::uint32_t seed{};
};

// Renders the scene seen from `world_from_camera` into a width x height depth
// image in device memory: float32 meters, or uint16 millimeters (rounded),
// 0 where nothing lies within max_depth. `rgb`, when not null, gets the RGB8
// colors (black where nothing is seen). Ordered on `stream`.
[[nodiscard]] cudaError_t render_scene(const SyntheticScene& scene,
                                       const Matrix4& world_from_camera,
                                       const CameraIntrinsics& intrinsics, int width, int height,
                                       float max_depth, const DepthNoise& noise, DepthType type,
                                       void* depth, std::uint8_t* rgb, cudaStream_t stream);

}  // namespace holoscan::tsdf
