// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A synthetic depth camera over a known scene, for tests, benchmarks and
// demos: a camera above flat ground, pitched down, looking at three boxes and
// a sphere that moves from frame to frame, with depth noise, flying pixels
// (outliers at a random depth along their ray) and pixels without depth. Each
// pixel knows what it shows, and the ground plane is known exactly, so
// filters can be checked against the truth.
//
// The same functions run on the GPU to render depth images and on the host to
// make the clouds and labels. They use only arithmetic and square roots, so
// with nothing contracted (the module renders with --fmad=false; host code
// must use -ffp-contract=off) both give the same bits.

#pragma once

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

#if defined(__CUDACC__)
#define HOLOSCAN_PCL_FILTERS_HOST_DEVICE __host__ __device__
#else
#define HOLOSCAN_PCL_FILTERS_HOST_DEVICE
#endif

namespace holoscan::pcl_filters {

enum class SceneLabel : std::uint8_t {
  kNone = 0,     // No depth: a missing pixel, or nothing within range.
  kGround = 1,   // The ground plane.
  kObject = 2,   // A box or the sphere.
  kOutlier = 3,  // A flying pixel.
};

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
  float camera_height{1.2F};
  float pitch_sin{};
  float pitch_cos{};
  // The sphere's y, which moves with the frame.
  float sphere_y{};
  std::uint32_t frame{};
  std::uint32_t seed{1U};
  // Zero-mean depth noise of this standard deviation in meters (the sum of
  // four uniform draws, close to Gaussian).
  float noise_sigma{0.004F};
  // Fractions of pixels that are flying pixels and that have no depth.
  float outlier_rate{0.01F};
  float invalid_rate{0.02F};
  // Surfaces farther than this (along the ray, in meters) are not seen.
  float max_range{8.0F};
};

inline SyntheticScene synthetic_scene(int width, int height, std::uint32_t frame,
                                      float pitch_deg = 25.0F) {
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
  scene.sphere_y = static_cast<float>(0.4 * std::sin(0.05 * frame) - 0.1);
  scene.frame = frame;
  return scene;
}

// The ground plane in the camera optical frame (x right, y down, z forward):
// (a, b, c, d) with a x + b y + c z + d = 0 and (a, b, c) the unit up.
inline void synthetic_ground_plane(const SyntheticScene& scene, float plane[4]) {
  plane[0] = 0.0F;
  plane[1] = -scene.pitch_cos;
  plane[2] = -scene.pitch_sin;
  plane[3] = scene.camera_height;
}

struct ScenePixel {
  // Camera z in meters, as a depth camera reports it; 0 without depth.
  float depth{};
  SceneLabel label{SceneLabel::kNone};
  std::uint8_t rgb[3]{};
};

namespace detail {

HOLOSCAN_PCL_FILTERS_HOST_DEVICE inline std::uint32_t scene_hash(std::uint32_t value) {
  value ^= value >> 16U;
  value *= 0x7FEB352DU;
  value ^= value >> 15U;
  value *= 0x846CA68BU;
  value ^= value >> 16U;
  return value;
}

// A uniform draw in [0, 1) for pixel `pixel` of the frame, stream `which`.
HOLOSCAN_PCL_FILTERS_HOST_DEVICE inline float scene_uniform(const SyntheticScene& s,
                                                            std::uint32_t pixel,
                                                            std::uint32_t which) {
  const std::uint32_t h =
      scene_hash(scene_hash(scene_hash(s.seed + 0x9E3779B9U * which) ^ s.frame) ^ pixel);
  return static_cast<float>(h >> 8U) * (1.0F / 16777216.0F);
}

// Where the ray o + t d enters the box [lo, hi], or a negative t.
HOLOSCAN_PCL_FILTERS_HOST_DEVICE inline float scene_box(const float o[3], const float d[3],
                                                        const float lo[3], const float hi[3]) {
  float near = -1.0e30F;
  float far = 1.0e30F;
  for (int axis = 0; axis < 3; ++axis) {
    if (d[axis] == 0.0F) {
      if (o[axis] < lo[axis] || o[axis] > hi[axis]) {
        return -1.0F;
      }
      continue;
    }
    float t0 = (lo[axis] - o[axis]) / d[axis];
    float t1 = (hi[axis] - o[axis]) / d[axis];
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

}  // namespace detail

// What pixel (u, v) of the scene shows.
HOLOSCAN_PCL_FILTERS_HOST_DEVICE inline ScenePixel synthetic_scene_pixel(const SyntheticScene& s,
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
  if (detail::scene_uniform(s, index, 1U) < s.outlier_rate) {
    pixel.depth = 0.5F + 4.5F * detail::scene_uniform(s, index, 2U);
    pixel.label = SceneLabel::kOutlier;
    pixel.rgb[0] = 255U;
    pixel.rgb[1] = 0U;
    pixel.rgb[2] = 255U;
    return pixel;
  }
  float best = 1.0e30F;
  int hit = -1;  // 0 ground, 1..3 boxes, 4 sphere.
  if (d[2] < 0.0F) {
    best = -s.camera_height / d[2];
    hit = 0;
  }
  const float boxes[3][2][3] = {{{1.8F, 0.5F, 0.0F}, {2.4F, 1.1F, 0.5F}},
                                {{2.8F, -1.3F, 0.0F}, {3.3F, -0.6F, 0.9F}},
                                {{1.4F, -0.45F, 0.0F}, {1.55F, -0.3F, 1.0F}}};
  for (int box = 0; box < 3; ++box) {
    const float t = detail::scene_box(o, d, boxes[box][0], boxes[box][1]);
    if (t > 0.0F && t < best) {
      best = t;
      hit = 1 + box;
    }
  }
  // The sphere of radius 0.3 resting on the ground.
  const float center[3] = {2.4F, s.sphere_y, 0.3F};
  const float oc[3] = {o[0] - center[0], o[1] - center[1], o[2] - center[2]};
  const float a = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  const float b = 2.0F * (d[0] * oc[0] + d[1] * oc[1] + d[2] * oc[2]);
  const float c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - 0.09F;
  const float discriminant = b * b - 4.0F * a * c;
  if (discriminant >= 0.0F) {
    const float t = (-b - sqrtf(discriminant)) / (2.0F * a);
    if (t > 0.0F && t < best) {
      best = t;
      hit = 4;
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
  pixel.label = hit == 0 ? SceneLabel::kGround : SceneLabel::kObject;
  if (hit == 0) {
    // A checkerboard of half-meter squares.
    const float x = o[0] + best * d[0];
    const float y = o[1] + best * d[1];
    const bool dark =
        (static_cast<int>(floorf(2.0F * x)) + static_cast<int>(floorf(2.0F * y))) % 2 != 0;
    pixel.rgb[0] = dark ? 90U : 150U;
    pixel.rgb[1] = dark ? 100U : 160U;
    pixel.rgb[2] = dark ? 90U : 140U;
  } else {
    const std::uint8_t colors[4][3] = {
        {200U, 60U, 50U}, {60U, 90U, 200U}, {220U, 200U, 40U}, {60U, 180U, 80U}};
    for (int channel = 0; channel < 3; ++channel) {
      pixel.rgb[channel] = colors[hit - 1][channel];
    }
  }
  return pixel;
}

// The camera-frame point of a pixel at depth z, as DepthToPointCloudOp
// deprojects it.
HOLOSCAN_PCL_FILTERS_HOST_DEVICE inline void synthetic_scene_point(const SyntheticScene& s, int u,
                                                                   int v, float z, float xyz[3]) {
  xyz[0] = (static_cast<float>(u) - s.cx) * z / s.fx;
  xyz[1] = (static_cast<float>(v) - s.cy) * z / s.fy;
  xyz[2] = z;
}

// The organized cloud of the scene on the host: [height, width, 3] float32,
// NaN where there is no depth, its RGB colors and each point's label.
void synthetic_scene_cloud(const SyntheticScene& scene, std::vector<float>& xyz,
                           std::vector<std::uint8_t>& rgb, std::vector<SceneLabel>& labels);

// Renders the scene on the GPU: `depth` float32 meters [height, width], 0
// without depth; `rgb` RGB8 when given; `xyz` the organized cloud, NaN without
// depth, when given. Device memory, ordered on `stream`.
cudaError_t render_synthetic_scene(const SyntheticScene& scene, float* depth, std::uint8_t* rgb,
                                   float* xyz, cudaStream_t stream);

}  // namespace holoscan::pcl_filters
