// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A synthetic depth camera in a known room, for tests, benchmarks and demos:
// a floor, a back wall and a side wall, a table with a box on it, a box on
// the floor and a sphere that moves from frame to frame, with depth noise and
// pixels without depth. Every pixel knows which surface it shows and that
// surface's exact normal, and every flat surface's plane is known, so normals
// and planes can be checked against the truth.
//
// The same functions run on the GPU to render depth images and on the host to
// make the clouds and the truth. They use only arithmetic and square roots,
// so with nothing contracted (the module renders with --fmad=false; host code
// must use -ffp-contract=off) both give the same bits.

#pragma once

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

#if defined(__CUDACC__)
#define HOLOSCAN_NORMALS_HOST_DEVICE __host__ __device__
#else
#define HOLOSCAN_NORMALS_HOST_DEVICE
#endif

namespace holoscan::normals {

// What a pixel shows. Each face of each box is its own surface,
// kSurfaceBoxes + 6 * box + face, the faces in the order -x, +x, -y, +y, -z,
// +z of the world.
enum SceneSurface : std::uint8_t {
  kSurfaceNone = 0,  // No depth: a missing pixel, or nothing within range.
  kSurfaceFloor = 1,
  kSurfaceBackWall = 2,
  kSurfaceSideWall = 3,
  kSurfaceSphere = 4,
  kSurfaceBoxes = 8,
  // The table's top: the +z face of box 0, its slab.
  kSurfaceTableTop = kSurfaceBoxes + 5,
};

// The boxes: the table's top slab, its four legs, a box on the floor and one
// on the table.
inline constexpr int kSceneBoxes = 7;
inline constexpr int kSceneTable = 0;
inline constexpr int kSceneFloorBox = 5;
inline constexpr int kSceneTableBox = 6;

HOLOSCAN_NORMALS_HOST_DEVICE constexpr std::uint8_t scene_box_face(int box, int face) {
  return static_cast<std::uint8_t>(kSurfaceBoxes + 6 * box + face);
}

static_assert(kSurfaceTableTop == scene_box_face(kSceneTable, 5));

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
  // The world is x forward, y left, z up, with the floor at z = 0; the camera
  // is camera_height above the world origin, pitched down.
  float camera_height{1.3F};
  float pitch_sin{};
  float pitch_cos{};
  // The sphere's y, which moves with the frame.
  float sphere_y{};
  std::uint32_t frame{};
  std::uint32_t seed{1U};
  // Zero-mean depth noise of this standard deviation in meters (the sum of
  // four uniform draws, close to Gaussian).
  float noise_sigma{0.002F};
  // Fraction of pixels without depth.
  float invalid_rate{0.01F};
  // Surfaces farther than this (along the ray, in meters) are not seen.
  float max_range{8.0F};
};

inline SyntheticScene synthetic_scene(int width, int height, std::uint32_t frame,
                                      float pitch_deg = 20.0F) {
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
  scene.sphere_y = static_cast<float>(0.9 + 0.3 * std::sin(0.05 * frame));
  scene.frame = frame;
  return scene;
}

struct ScenePixel {
  // Camera z in meters, as a depth camera reports it; 0 without depth.
  float depth{};
  // The surface's unit normal in the camera optical frame (x right, y down,
  // z forward), toward the camera; zero without depth.
  float normal[3]{};
  std::uint8_t surface{kSurfaceNone};
  std::uint8_t rgb[3]{};
};

namespace detail {

HOLOSCAN_NORMALS_HOST_DEVICE inline std::uint32_t scene_hash(std::uint32_t value) {
  value ^= value >> 16U;
  value *= 0x7FEB352DU;
  value ^= value >> 15U;
  value *= 0x846CA68BU;
  value ^= value >> 16U;
  return value;
}

// A uniform draw in [0, 1) for pixel `pixel` of the frame, stream `which`.
HOLOSCAN_NORMALS_HOST_DEVICE inline float scene_uniform(const SyntheticScene& s,
                                                        std::uint32_t pixel, std::uint32_t which) {
  const std::uint32_t h =
      scene_hash(scene_hash(scene_hash(s.seed + 0x9E3779B9U * which) ^ s.frame) ^ pixel);
  return static_cast<float>(h >> 8U) * (1.0F / 16777216.0F);
}

// Box `box`'s corners in the world, lo then hi.
HOLOSCAN_NORMALS_HOST_DEVICE inline void scene_box_bounds(int box, float lo[3], float hi[3]) {
  // Table slab, legs, a box on the floor, a box on the table.
  const float boxes[kSceneBoxes][6] = {
      {1.5F, -0.9F, 0.70F, 2.3F, 0.1F, 0.75F},   {1.55F, -0.85F, 0.0F, 1.6F, -0.8F, 0.70F},
      {2.2F, -0.85F, 0.0F, 2.25F, -0.8F, 0.70F}, {1.55F, 0.0F, 0.0F, 1.6F, 0.05F, 0.70F},
      {2.2F, 0.0F, 0.0F, 2.25F, 0.05F, 0.70F},   {2.9F, -1.7F, 0.0F, 3.4F, -1.1F, 0.55F},
      {1.7F, -0.6F, 0.75F, 2.0F, -0.3F, 1.0F},
  };
  for (int axis = 0; axis < 3; ++axis) {
    lo[axis] = boxes[box][axis];
    hi[axis] = boxes[box][3 + axis];
  }
}

// Where the ray o + t d enters the box [lo, hi], or a negative t; `face` is
// the face it enters through.
HOLOSCAN_NORMALS_HOST_DEVICE inline float scene_box(const float o[3], const float d[3],
                                                    const float lo[3], const float hi[3],
                                                    int& face) {
  float near = -1.0e30F;
  float far = 1.0e30F;
  face = -1;
  for (int axis = 0; axis < 3; ++axis) {
    if (d[axis] == 0.0F) {
      if (o[axis] < lo[axis] || o[axis] > hi[axis]) {
        return -1.0F;
      }
      continue;
    }
    float t0 = (lo[axis] - o[axis]) / d[axis];
    float t1 = (hi[axis] - o[axis]) / d[axis];
    // Entering through the low face when the ray goes up the axis.
    int entry = 2 * axis;
    if (t0 > t1) {
      const float swap = t0;
      t0 = t1;
      t1 = swap;
      entry = 2 * axis + 1;
    }
    if (t0 > near) {
      near = t0;
      face = entry;
    }
    far = t1 < far ? t1 : far;
  }
  return near <= far && near > 0.0F && face >= 0 ? near : -1.0F;
}

// A world vector in the camera optical frame. Camera x is world -y; camera y
// is (-sin, 0, -cos); camera z (cos, 0, -sin).
HOLOSCAN_NORMALS_HOST_DEVICE inline void scene_to_camera(const SyntheticScene& s, const float w[3],
                                                         float c[3]) {
  c[0] = -w[1];
  c[1] = -s.pitch_sin * w[0] - s.pitch_cos * w[2];
  c[2] = s.pitch_cos * w[0] - s.pitch_sin * w[2];
}

}  // namespace detail

// The sphere: its center in the world and its radius.
HOLOSCAN_NORMALS_HOST_DEVICE inline void synthetic_sphere(const SyntheticScene& s, float center[3],
                                                          float& radius) {
  center[0] = 2.6F;
  center[1] = s.sphere_y;
  center[2] = 0.3F;
  radius = 0.3F;
}

// The plane of a flat surface in the camera optical frame: (a, b, c, d) with
// a x + b y + c z + d = 0 and (a, b, c) the unit normal toward the camera, so
// d > 0. False for the sphere and kSurfaceNone.
HOLOSCAN_NORMALS_HOST_DEVICE inline bool synthetic_surface_plane(const SyntheticScene& s,
                                                                 std::uint8_t surface,
                                                                 float plane[4]) {
  float n[3] = {0.0F, 0.0F, 0.0F};
  float d = 0.0F;  // In the world: n . X + d = 0.
  if (surface == kSurfaceFloor) {
    n[2] = 1.0F;
  } else if (surface == kSurfaceBackWall) {
    n[0] = -1.0F;
    d = 4.5F;
  } else if (surface == kSurfaceSideWall) {
    n[1] = -1.0F;
    d = 1.8F;
  } else if (surface >= kSurfaceBoxes && surface < kSurfaceBoxes + 6 * kSceneBoxes) {
    const int box = (surface - kSurfaceBoxes) / 6;
    const int face = (surface - kSurfaceBoxes) % 6;
    float lo[3];
    float hi[3];
    detail::scene_box_bounds(box, lo, hi);
    const int axis = face / 2;
    const float sign = face % 2 == 0 ? -1.0F : 1.0F;
    n[axis] = sign;
    d = -sign * (face % 2 == 0 ? lo[axis] : hi[axis]);
  } else {
    return false;
  }
  detail::scene_to_camera(s, n, plane);
  // The camera sits at (0, 0, camera_height) in the world.
  plane[3] = d + n[2] * s.camera_height;
  return true;
}

// What pixel (u, v) of the scene shows.
HOLOSCAN_NORMALS_HOST_DEVICE inline ScenePixel synthetic_scene_pixel(const SyntheticScene& s, int u,
                                                                     int v) {
  ScenePixel pixel;
  const auto index = static_cast<std::uint32_t>(v) * static_cast<std::uint32_t>(s.width) +
                     static_cast<std::uint32_t>(u);
  if (detail::scene_uniform(s, index, 0U) < s.invalid_rate) {
    return pixel;
  }
  // The ray through the pixel, in the world, scaled so that t is camera z.
  const float dx = (static_cast<float>(u) - s.cx) / s.fx;
  const float dy = (static_cast<float>(v) - s.cy) / s.fy;
  const float d[3] = {-dy * s.pitch_sin + s.pitch_cos, -dx, -dy * s.pitch_cos - s.pitch_sin};
  const float o[3] = {0.0F, 0.0F, s.camera_height};
  const float ray_length = sqrtf(dx * dx + dy * dy + 1.0F);

  float best = 1.0e30F;
  std::uint8_t surface = kSurfaceNone;
  float world_normal[3] = {0.0F, 0.0F, 0.0F};
  if (d[2] < 0.0F) {
    best = -s.camera_height / d[2];
    surface = kSurfaceFloor;
    world_normal[2] = 1.0F;
  }
  // The walls are 3 m high.
  if (d[0] > 0.0F) {
    const float t = 4.5F / d[0];
    const float z = o[2] + t * d[2];
    if (t < best && z >= 0.0F && z <= 3.0F) {
      best = t;
      surface = kSurfaceBackWall;
      world_normal[0] = -1.0F;
      world_normal[1] = 0.0F;
      world_normal[2] = 0.0F;
    }
  }
  if (d[1] > 0.0F) {
    const float t = 1.8F / d[1];
    const float x = o[0] + t * d[0];
    const float z = o[2] + t * d[2];
    if (t < best && x <= 4.5F && z >= 0.0F && z <= 3.0F) {
      best = t;
      surface = kSurfaceSideWall;
      world_normal[0] = 0.0F;
      world_normal[1] = -1.0F;
      world_normal[2] = 0.0F;
    }
  }
  for (int box = 0; box < kSceneBoxes; ++box) {
    float lo[3];
    float hi[3];
    detail::scene_box_bounds(box, lo, hi);
    int face = -1;
    const float t = detail::scene_box(o, d, lo, hi, face);
    if (t > 0.0F && t < best) {
      best = t;
      surface = scene_box_face(box, face);
      world_normal[0] = 0.0F;
      world_normal[1] = 0.0F;
      world_normal[2] = 0.0F;
      world_normal[face / 2] = face % 2 == 0 ? -1.0F : 1.0F;
    }
  }
  float center[3];
  float radius = 0.0F;
  synthetic_sphere(s, center, radius);
  const float oc[3] = {o[0] - center[0], o[1] - center[1], o[2] - center[2]};
  const float a = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  const float b = 2.0F * (d[0] * oc[0] + d[1] * oc[1] + d[2] * oc[2]);
  const float c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - radius * radius;
  const float discriminant = b * b - 4.0F * a * c;
  if (discriminant >= 0.0F) {
    const float t = (-b - sqrtf(discriminant)) / (2.0F * a);
    if (t > 0.0F && t < best) {
      best = t;
      surface = kSurfaceSphere;
      for (int axis = 0; axis < 3; ++axis) {
        world_normal[axis] = (o[axis] + t * d[axis] - center[axis]) / radius;
      }
    }
  }
  if (surface == kSurfaceNone || best * ray_length > s.max_range) {
    return pixel;
  }
  // Noise: four uniform draws summed have variance 1/3.
  float sum = 0.0F;
  for (std::uint32_t k = 0; k < 4U; ++k) {
    sum += detail::scene_uniform(s, index, 1U + k);
  }
  pixel.depth = best + s.noise_sigma * 1.7320508F * (sum - 2.0F);
  pixel.surface = surface;
  detail::scene_to_camera(s, world_normal, pixel.normal);
  const float length = sqrtf(pixel.normal[0] * pixel.normal[0] + pixel.normal[1] * pixel.normal[1] +
                             pixel.normal[2] * pixel.normal[2]);
  // Toward the camera: against the ray, whose camera-frame direction is (dx, dy, 1).
  const float sign =
      pixel.normal[0] * dx + pixel.normal[1] * dy + pixel.normal[2] > 0.0F ? -1.0F : 1.0F;
  for (float& value : pixel.normal) {
    value = sign * value / length;
  }
  // Gray floor and walls, colored objects.
  const std::uint8_t colors[5][3] = {
      {150U, 150U, 140U}, {200U, 190U, 170U}, {170U, 180U, 200U}, {60U, 180U, 80U}, {0U, 0U, 0U}};
  const int kind = surface < kSurfaceBoxes ? surface - 1 : 4;
  for (int channel = 0; channel < 3; ++channel) {
    pixel.rgb[channel] = colors[kind][channel];
  }
  if (kind == 4) {
    const int box = (surface - kSurfaceBoxes) / 6;
    pixel.rgb[0] = box == kSceneFloorBox ? 60U : (box == kSceneTableBox ? 220U : 160U);
    pixel.rgb[1] = box == kSceneFloorBox ? 90U : (box == kSceneTableBox ? 200U : 110U);
    pixel.rgb[2] = box == kSceneFloorBox ? 200U : (box == kSceneTableBox ? 40U : 60U);
  }
  return pixel;
}

// The camera-frame point of a pixel at depth z, as DepthToPointCloudOp
// deprojects it.
HOLOSCAN_NORMALS_HOST_DEVICE inline void synthetic_scene_point(const SyntheticScene& s, int u,
                                                               int v, float z, float xyz[3]) {
  xyz[0] = (static_cast<float>(u) - s.cx) * z / s.fx;
  xyz[1] = (static_cast<float>(v) - s.cy) * z / s.fy;
  xyz[2] = z;
}

// The organized cloud of the scene on the host: [height, width, 3] float32,
// NaN where there is no depth, its RGB colors, each point's surface and its
// true normal (zero without depth).
void synthetic_scene_cloud(const SyntheticScene& scene, std::vector<float>& xyz,
                           std::vector<std::uint8_t>& rgb, std::vector<std::uint8_t>& surfaces,
                           std::vector<float>& normals);

// Renders the scene on the GPU: `depth` float32 meters [height, width], 0
// without depth; `rgb` RGB8 when given; `xyz` the organized cloud, NaN without
// depth, when given. Device memory, ordered on `stream`.
cudaError_t render_synthetic_scene(const SyntheticScene& scene, float* depth, std::uint8_t* rgb,
                                   float* xyz, cudaStream_t stream);

}  // namespace holoscan::normals
