// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>

#include <holoscan/tsdf/synthetic_scene.hpp>

namespace holoscan::tsdf {
namespace {

// A well-mixed 32-bit hash (lowbias32).
__device__ std::uint32_t mix(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352DU;
  x ^= x >> 15;
  x *= 0x846CA68BU;
  x ^= x >> 16;
  return x;
}

// Unit-variance noise from four uniform draws, within +-2 * sqrt(3).
__device__ float noise(std::uint32_t seed, std::uint32_t pixel) {
  std::uint32_t state = mix(seed * 0x9E3779B9U ^ mix(pixel + 0x632BE5ABU));
  float sum = 0.0F;
  for (int i = 0; i < 4; ++i) {
    state = mix(state + 0x68E31DA4U);
    sum += static_cast<float>(state >> 8) * (1.0F / 16777216.0F);
  }
  return (sum - 2.0F) * 1.7320508F;
}

struct RenderArgs {
  SyntheticScene scene;
  float pose[16]{};
  CameraIntrinsics k;
  int width{};
  int height{};
  float max_depth{};
  DepthNoise noise;
  DepthType type{};
  void* depth{};
  std::uint8_t* rgb{};
};

__global__ void render_kernel(RenderArgs a) {
  const int u = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int v = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (u >= a.width || v >= a.height) {
    return;
  }
  const std::size_t pixel =
      static_cast<std::size_t>(v) * static_cast<std::size_t>(a.width) + static_cast<std::size_t>(u);
  SceneSurface surface = SceneSurface::kNone;
  float z = scene_depth(a.scene, a.pose, a.k, static_cast<float>(u), static_cast<float>(v),
                        a.max_depth, &surface);
  if (a.rgb != nullptr) {
    float p[3] = {0.0F, 0.0F, 0.0F};
    if (surface != SceneSurface::kNone) {
      const float ray[3] = {(static_cast<float>(u) - a.k.cx) / a.k.fx * z,
                            (static_cast<float>(v) - a.k.cy) / a.k.fy * z, z};
      for (int r = 0; r < 3; ++r) {
        p[r] = a.pose[4 * r] * ray[0] + a.pose[4 * r + 1] * ray[1] + a.pose[4 * r + 2] * ray[2] +
               a.pose[4 * r + 3];
      }
    }
    scene_color(surface, p, a.rgb + 3 * pixel);
  }
  if (z > 0.0F && (a.noise.sigma > 0.0F || a.noise.sigma_per_m2 > 0.0F)) {
    const float sigma = a.noise.sigma + a.noise.sigma_per_m2 * z * z;
    z = fmaxf(z + sigma * noise(a.noise.seed, static_cast<std::uint32_t>(pixel)), 0.0F);
  }
  if (a.type == DepthType::kUint16) {
    const float mm = floorf(z * 1000.0F + 0.5F);
    static_cast<std::uint16_t*>(a.depth)[pixel] =
        static_cast<std::uint16_t>(fminf(fmaxf(mm, 0.0F), 65535.0F));
  } else {
    static_cast<float*>(a.depth)[pixel] = z;
  }
}

float length3(const float v[3]) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

}  // namespace

SyntheticScene synthetic_scene(SceneKind kind) {
  SyntheticScene scene;
  if (kind == SceneKind::kSphere) {
    scene.floor = false;
    scene.box = false;
  }
  return scene;
}

float scene_distance(const SyntheticScene& s, const float p[3]) {
  float best = 1e30F;
  if (s.floor) {
    best = std::min(best, p[2]);
  }
  if (s.box) {
    float outside[3];
    float inside = -1e30F;
    for (int r = 0; r < 3; ++r) {
      const float center = 0.5F * (s.box_min[r] + s.box_max[r]);
      const float half = 0.5F * (s.box_max[r] - s.box_min[r]);
      const float q = std::abs(p[r] - center) - half;
      outside[r] = std::max(q, 0.0F);
      inside = std::max(inside, q);
    }
    best = std::min(best, length3(outside) + std::min(inside, 0.0F));
  }
  const float d[3] = {p[0] - s.sphere_center[0], p[1] - s.sphere_center[1],
                      p[2] - s.sphere_center[2]};
  return std::min(best, length3(d) - s.sphere_radius);
}

CameraIntrinsics synthetic_intrinsics(int width, int height) {
  const float f = 0.8F * static_cast<float>(width);
  return CameraIntrinsics{f, f, 0.5F * static_cast<float>(width - 1),
                          0.5F * static_cast<float>(height - 1)};
}

Matrix4 look_at(const float eye[3], const float target[3]) {
  // Columns of the rotation: the camera's x (right), y (down) and z
  // (forward) in the world.
  double forward[3];
  for (int r = 0; r < 3; ++r) {
    forward[r] = static_cast<double>(target[r]) - eye[r];
  }
  const auto normalize = [](double v[3]) {
    const double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    for (int r = 0; r < 3; ++r) {
      v[r] /= n;
    }
  };
  normalize(forward);
  double right[3] = {forward[1], -forward[0], 0.0};  // forward x up, up = +z.
  normalize(right);
  const double down[3] = {forward[1] * right[2] - forward[2] * right[1],
                          forward[2] * right[0] - forward[0] * right[2],
                          forward[0] * right[1] - forward[1] * right[0]};
  Matrix4 pose = identity_pose();
  for (int r = 0; r < 3; ++r) {
    pose[4 * r] = static_cast<float>(right[r]);
    pose[4 * r + 1] = static_cast<float>(down[r]);
    pose[4 * r + 2] = static_cast<float>(forward[r]);
    pose[4 * r + 3] = eye[r];
  }
  return pose;
}

Matrix4 orbit_pose(SceneKind kind, std::uint32_t frame, std::uint32_t frames) {
  const double s = frames == 0 ? 0.0 : static_cast<double>(frame) / static_cast<double>(frames);
  const double pi = std::numbers::pi;
  double azimuth = 0.0;
  double elevation = 0.0;
  double radius = 0.0;
  float target[3];
  if (kind == SceneKind::kSphere) {
    const SyntheticScene scene = synthetic_scene(kind);
    for (int r = 0; r < 3; ++r) {
      target[r] = scene.sphere_center[r];
    }
    azimuth = 2.0 * pi * 3.0 * s;
    elevation = 70.0 * pi / 180.0 * std::sin(2.0 * pi * s - pi / 2.0);
    radius = 0.9;
  } else {
    target[0] = -0.04F;
    target[1] = -0.04F;
    target[2] = 0.12F;
    azimuth = 2.0 * pi * s;
    elevation = (35.0 + 15.0 * std::sin(4.0 * pi * s)) * pi / 180.0;
    radius = 1.0;
  }
  const float eye[3] = {
      target[0] + static_cast<float>(radius * std::cos(elevation) * std::cos(azimuth)),
      target[1] + static_cast<float>(radius * std::cos(elevation) * std::sin(azimuth)),
      target[2] + static_cast<float>(radius * std::sin(elevation))};
  return look_at(eye, target);
}

TsdfVolumeConfig scene_volume(SceneKind kind, int dims, bool color) {
  TsdfVolumeConfig config;
  config.dims[0] = config.dims[1] = config.dims[2] = dims;
  config.color = color;
  if (kind == SceneKind::kSphere) {
    const SyntheticScene scene = synthetic_scene(kind);
    config.voxel_size = 0.5F / static_cast<float>(dims - 1);
    for (int r = 0; r < 3; ++r) {
      config.origin[r] = scene.sphere_center[r] - 0.25F;
    }
  } else {
    config.voxel_size = 1.0F / static_cast<float>(dims - 1);
    config.origin[0] = -0.5F;
    config.origin[1] = -0.5F;
    config.origin[2] = -0.1F;
  }
  // Four voxels, and at least 1.5 cm for the depth noise.
  config.truncation = std::max(4.0F * config.voxel_size, 0.015F);
  config.weight_cap = 64.0F;
  return config;
}

cudaError_t render_scene(const SyntheticScene& scene, const Matrix4& world_from_camera,
                         const CameraIntrinsics& intrinsics, int width, int height, float max_depth,
                         const DepthNoise& noise, DepthType type, void* depth, std::uint8_t* rgb,
                         cudaStream_t stream) {
  if (width < 1 || height < 1 || depth == nullptr || !(max_depth > 0.0F)) {
    return cudaErrorInvalidValue;
  }
  RenderArgs a;
  a.scene = scene;
  std::copy(world_from_camera.begin(), world_from_camera.end(), a.pose);
  a.k = intrinsics;
  a.width = width;
  a.height = height;
  a.max_depth = max_depth;
  a.noise = noise;
  a.type = type;
  a.depth = depth;
  a.rgb = rgb;
  const dim3 block(16, 16);
  const dim3 grid((width + 15) / 16, (height + 15) / 16);
  render_kernel<<<grid, block, 0, stream>>>(a);
  return cudaGetLastError();
}

}  // namespace holoscan::tsdf
