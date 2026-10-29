// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>

#include "holoscan/icp/synthetic_scene.hpp"

namespace holoscan::icp {
namespace {

constexpr double kPi = 3.14159265358979323846;

struct RenderParams {
  SyntheticScene scene;
  float r[9]{};  // world_from_camera rotation, row-major.
  float t[3]{};
  PinholeCamera camera;
  float max_depth{};
  DepthNoise noise;
};

__device__ inline std::uint32_t hash(std::uint32_t x) {
  // A 32-bit integer mix (lowbias32).
  x ^= x >> 16;
  x *= 0x7FEB352DU;
  x ^= x >> 15;
  x *= 0x846CA68BU;
  x ^= x >> 16;
  return x;
}

// A standard normal value from the seed and pixel, by Box-Muller.
__device__ inline float gaussian(std::uint32_t seed, std::uint32_t pixel) {
  const std::uint32_t a = hash(pixel * 2U + hash(seed));
  const std::uint32_t b = hash(pixel * 2U + 1U + hash(seed ^ 0x9E3779B9U));
  const float u1 = (static_cast<float>(a >> 8) + 1.0F) * (1.0F / 16777217.0F);  // (0, 1]
  const float u2 = static_cast<float>(b >> 8) * (1.0F / 16777216.0F);
  return sqrtf(-2.0F * logf(u1)) * cospif(2.0F * u2);
}

// The nearest hit along o + s * d with s > 0, or +inf.
__device__ inline float cast(const SyntheticScene& scene, float3 o, float3 d) {
  float best = __int_as_float(0x7F800000);
  constexpr float kEpsilon = 1e-6F;
  for (int i = 0; i < scene.plane_count; ++i) {
    const ScenePlane& p = scene.planes[i];
    const float denominator = p.normal[0] * d.x + p.normal[1] * d.y + p.normal[2] * d.z;
    if (fabsf(denominator) < 1e-9F) continue;
    const float s =
        (p.offset - (p.normal[0] * o.x + p.normal[1] * o.y + p.normal[2] * o.z)) / denominator;
    if (s > kEpsilon && s < best) best = s;
  }
  for (int i = 0; i < scene.box_count; ++i) {
    const SceneBox& b = scene.boxes[i];
    float near = -__int_as_float(0x7F800000);
    float far = __int_as_float(0x7F800000);
    const float origin[3] = {o.x, o.y, o.z};
    const float direction[3] = {d.x, d.y, d.z};
    bool miss = false;
    for (int axis = 0; axis < 3; ++axis) {
      if (fabsf(direction[axis]) < 1e-12F) {
        if (origin[axis] < b.min[axis] || origin[axis] > b.max[axis]) miss = true;
        continue;
      }
      float s0 = (b.min[axis] - origin[axis]) / direction[axis];
      float s1 = (b.max[axis] - origin[axis]) / direction[axis];
      if (s0 > s1) {
        const float swap = s0;
        s0 = s1;
        s1 = swap;
      }
      near = fmaxf(near, s0);
      far = fminf(far, s1);
    }
    if (miss || near > far) continue;
    const float s = near > kEpsilon ? near : far;
    if (s > kEpsilon && s < best) best = s;
  }
  for (int i = 0; i < scene.sphere_count; ++i) {
    const SceneSphere& sp = scene.spheres[i];
    const float3 oc = make_float3(o.x - sp.center[0], o.y - sp.center[1], o.z - sp.center[2]);
    const float b = oc.x * d.x + oc.y * d.y + oc.z * d.z;
    const float dd = d.x * d.x + d.y * d.y + d.z * d.z;
    const float c = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z - sp.radius * sp.radius;
    const float discriminant = b * b - dd * c;
    if (discriminant < 0.0F) continue;
    const float root = sqrtf(discriminant);
    float s = (-b - root) / dd;
    if (s <= kEpsilon) s = (-b + root) / dd;
    if (s > kEpsilon && s < best) best = s;
  }
  return best;
}

// The depth at pixel (u, v), 0 without one.
__device__ inline float pixel_depth(const RenderParams& p, int u, int v) {
  const PinholeCamera& k = p.camera;
  // The ray through the pixel, with camera z = 1, so the hit's s is its depth.
  const float3 dc = make_float3((static_cast<float>(u) - k.cx) / k.fx,
                                (static_cast<float>(v) - k.cy) / k.fy, 1.0F);
  const float3 dw = make_float3(p.r[0] * dc.x + p.r[1] * dc.y + p.r[2] * dc.z,
                                p.r[3] * dc.x + p.r[4] * dc.y + p.r[5] * dc.z,
                                p.r[6] * dc.x + p.r[7] * dc.y + p.r[8] * dc.z);
  const float z = cast(p.scene, make_float3(p.t[0], p.t[1], p.t[2]), dw);
  if (!(z <= p.max_depth)) {
    return 0.0F;
  }
  const float sigma = p.noise.sigma + p.noise.sigma_per_m2 * z * z;
  if (sigma <= 0.0F) {
    return z;
  }
  const auto pixel = static_cast<std::uint32_t>(v * k.width + u);
  const float noisy = z + sigma * gaussian(p.noise.seed, pixel);
  return noisy > 0.0F ? noisy : 0.0F;
}

__global__ void depth_kernel(RenderParams p, float* depth) {
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  const int v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= p.camera.width || v >= p.camera.height) return;
  depth[v * p.camera.width + u] = pixel_depth(p, u, v);
}

__global__ void cloud_kernel(RenderParams p, float* points) {
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  const int v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= p.camera.width || v >= p.camera.height) return;
  const float z = pixel_depth(p, u, v);
  float* out = points + 3 * (v * p.camera.width + u);
  if (z > 0.0F) {
    out[0] = (static_cast<float>(u) - p.camera.cx) / p.camera.fx * z;
    out[1] = (static_cast<float>(v) - p.camera.cy) / p.camera.fy * z;
    out[2] = z;
  } else {
    const float nan = __int_as_float(0x7FC00000);
    out[0] = nan;
    out[1] = nan;
    out[2] = nan;
  }
}

bool valid(const PinholeCamera& k) {
  return k.width > 0 && k.height > 0 && k.width <= 16384 && k.height <= 16384 &&
         std::isfinite(k.fx) && k.fx > 0.0F && std::isfinite(k.fy) && k.fy > 0.0F &&
         std::isfinite(k.cx) && std::isfinite(k.cy);
}

RenderParams params_for(const SyntheticScene& scene, const Matrix4& pose,
                        const PinholeCamera& camera, float max_depth, const DepthNoise& noise) {
  RenderParams p;
  p.scene = scene;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      p.r[3 * row + col] = static_cast<float>(pose[4 * row + col]);
    }
    p.t[row] = static_cast<float>(pose[4 * row + 3]);
  }
  p.camera = camera;
  p.max_depth = max_depth;
  p.noise = noise;
  return p;
}

ScenePlane plane(float nx, float ny, float nz, float offset) {
  return ScenePlane{{nx, ny, nz}, offset};
}

SceneBox box(float x0, float y0, float z0, float x1, float y1, float z1) {
  return SceneBox{{x0, y0, z0}, {x1, y1, z1}};
}

SceneSphere sphere(float x, float y, float z, float radius) {
  return SceneSphere{{x, y, z}, radius};
}

}  // namespace

PinholeCamera synthetic_camera(int width, int height) {
  const float focal = 0.8F * static_cast<float>(width);
  return PinholeCamera{width,
                       height,
                       focal,
                       focal,
                       static_cast<float>(width - 1) / 2.0F,
                       static_cast<float>(height - 1) / 2.0F};
}

SyntheticScene synthetic_scene(SceneKind kind) {
  SyntheticScene s;
  const auto add_plane = [&s](ScenePlane p) { s.planes[s.plane_count++] = p; };
  const auto add_box = [&s](SceneBox b) { s.boxes[s.box_count++] = b; };
  const auto add_sphere = [&s](SceneSphere sp) { s.spheres[s.sphere_count++] = sp; };
  // The floor is 1.1 m below the camera (y down).
  switch (kind) {
    case SceneKind::kRoom:
      add_plane(plane(0.0F, 1.0F, 0.0F, 1.1F));   // Floor.
      add_plane(plane(0.0F, 1.0F, 0.0F, -1.5F));  // Ceiling.
      add_plane(plane(1.0F, 0.0F, 0.0F, -3.0F));  // Left wall.
      add_plane(plane(1.0F, 0.0F, 0.0F, 3.0F));   // Right wall.
      add_plane(plane(0.0F, 0.0F, 1.0F, 6.0F));   // Far wall.
      add_plane(plane(0.0F, 0.0F, 1.0F, -2.0F));  // Wall behind.
      add_box(box(-1.6F, 0.3F, 2.6F, -0.8F, 1.1F, 3.3F));
      add_box(box(0.7F, -0.2F, 3.6F, 1.6F, 1.1F, 4.4F));
      add_box(box(-2.6F, -0.6F, 4.6F, -1.8F, 1.1F, 5.4F));
      add_box(box(1.8F, 0.6F, 1.6F, 2.4F, 1.1F, 2.2F));
      add_sphere(sphere(0.1F, 0.7F, 2.6F, 0.4F));
      add_sphere(sphere(-0.5F, 0.85F, 4.6F, 0.25F));
      add_sphere(sphere(2.3F, -0.4F, 5.2F, 0.5F));
      break;
    case SceneKind::kObjects:
      add_plane(plane(0.0F, 1.0F, 0.0F, 1.1F));  // Floor.
      add_box(box(-1.2F, 0.2F, 2.2F, -0.4F, 1.1F, 3.0F));
      add_box(box(0.3F, -0.4F, 3.0F, 1.1F, 1.1F, 3.6F));
      add_box(box(-0.2F, 0.5F, 4.0F, 0.6F, 1.1F, 4.8F));
      add_sphere(sphere(0.8F, 0.6F, 2.0F, 0.5F));
      add_sphere(sphere(-1.0F, 0.4F, 4.2F, 0.7F));
      add_sphere(sphere(1.6F, 0.2F, 4.5F, 0.9F));
      break;
    case SceneKind::kCorner:
      add_plane(plane(0.0F, 1.0F, 0.0F, 1.1F));   // Floor.
      add_plane(plane(0.0F, 0.0F, 1.0F, 4.0F));   // Back wall.
      add_plane(plane(1.0F, 0.0F, 0.0F, -1.8F));  // Side wall.
      add_sphere(sphere(-0.9F, 0.5F, 3.0F, 0.6F));
      add_sphere(sphere(0.6F, 0.75F, 2.6F, 0.35F));
      add_sphere(sphere(0.2F, -0.2F, 3.5F, 0.3F));
      break;
  }
  return s;
}

cudaError_t render_scene_depth(const SyntheticScene& scene, const Matrix4& world_from_camera,
                               const PinholeCamera& camera, float max_depth,
                               const DepthNoise& noise, float* depth, cudaStream_t stream) {
  if (depth == nullptr || !valid(camera) || !(max_depth > 0.0F) ||
      !is_rigid_shaped(world_from_camera)) {
    return cudaErrorInvalidValue;
  }
  const dim3 block(32, 8);
  const dim3 grid((camera.width + 31) / 32, (camera.height + 7) / 8);
  depth_kernel<<<grid, block, 0, stream>>>(
      params_for(scene, world_from_camera, camera, max_depth, noise), depth);
  return cudaGetLastError();
}

cudaError_t render_scene_cloud(const SyntheticScene& scene, const Matrix4& world_from_camera,
                               const PinholeCamera& camera, float max_depth,
                               const DepthNoise& noise, float* points, cudaStream_t stream) {
  if (points == nullptr || !valid(camera) || !(max_depth > 0.0F) ||
      !is_rigid_shaped(world_from_camera)) {
    return cudaErrorInvalidValue;
  }
  const dim3 block(32, 8);
  const dim3 grid((camera.width + 31) / 32, (camera.height + 7) / 8);
  cloud_kernel<<<grid, block, 0, stream>>>(
      params_for(scene, world_from_camera, camera, max_depth, noise), points);
  return cudaGetLastError();
}

Matrix4 synthetic_trajectory(std::uint32_t frame, std::uint32_t frames) {
  const double s = frames == 0U ? 0.0 : static_cast<double>(frame) / static_cast<double>(frames);
  const double a = 2.0 * kPi * s;
  // Rotation: yaw about y, pitch about x, roll about z.
  const double yaw = 0.30 * std::sin(a);
  const double pitch = 0.08 * std::sin(2.0 * a);
  const double roll = 0.05 * std::sin(a);
  const Matrix4 r_yaw = transform_from({0.0, yaw, 0.0}, {0.0, 0.0, 0.0});
  const Matrix4 r_pitch = transform_from({pitch, 0.0, 0.0}, {0.0, 0.0, 0.0});
  const Matrix4 r_roll = transform_from({0.0, 0.0, roll}, {0.0, 0.0, 0.0});
  Matrix4 pose = compose(r_yaw, compose(r_pitch, r_roll));
  pose[3] = 0.9 * std::sin(a);
  pose[7] = 0.15 * std::sin(2.0 * a);
  pose[11] = 0.6 * (1.0 - std::cos(a));
  return pose;
}

}  // namespace holoscan::icp
