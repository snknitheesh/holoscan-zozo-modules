// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>

#include <holoscan/height_map/synthetic_terrain.hpp>

namespace holoscan::height_map {
namespace {

struct RenderParams {
  Terrain terrain;
  int width;
  int height;
  float fx;
  float fy;
  float cx;
  float cy;
  float r[9];  // Optical frame to map, row-major.
  float t[3];  // The camera's position in the map frame.
  float max_depth;
  float noise_base;
  float noise_quadratic;
  std::uint32_t seed;
};

__device__ inline std::uint32_t hash(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

// A uniform number in (0, 1).
__device__ inline float uniform(std::uint32_t bits) {
  return (static_cast<float>(bits >> 8) + 0.5F) * (1.0F / 16777216.0F);
}

__device__ inline bool below(const RenderParams& p, float s, const float d[3]) {
  const float x = p.t[0] + s * d[0];
  const float y = p.t[1] + s * d[1];
  const float z = p.t[2] + s * d[2];
  return z <= terrain_height(p.terrain, x, y);
}

__global__ void render_kernel(RenderParams p, float* __restrict__ depth, float* __restrict__ xyz) {
  const int u = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int v = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (u >= p.width || v >= p.height) {
    return;
  }
  const std::size_t i = static_cast<std::size_t>(v) * p.width + u;
  // The ray in the optical frame, z = 1, so that its parameter is the depth.
  const float rx = (static_cast<float>(u) - p.cx) / p.fx;
  const float ry = (static_cast<float>(v) - p.cy) / p.fy;
  const float d[3] = {p.r[0] * rx + p.r[1] * ry + p.r[2], p.r[3] * rx + p.r[4] * ry + p.r[5],
                      p.r[6] * rx + p.r[7] * ry + p.r[8]};
  const float length = sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  float hit = 0.0F;
  float s = 0.0F;
  if (!below(p, s, d)) {
    while (s < p.max_depth) {
      const float distance = s * length;
      const float step = (0.01F + 0.005F * distance) / length;
      const float next = fminf(s + step, p.max_depth);
      if (below(p, next, d)) {
        float lo = s;
        float hi = next;
        for (int k = 0; k < 30; ++k) {
          const float mid = 0.5F * (lo + hi);
          if (below(p, mid, d)) {
            hi = mid;
          } else {
            lo = mid;
          }
        }
        hit = hi;
        break;
      }
      s = next;
    }
  }
  if (hit > 0.0F && (p.noise_base > 0.0F || p.noise_quadratic > 0.0F)) {
    // Box-Muller from two hashed uniforms.
    const std::uint32_t key = hash(p.seed * 0x9e3779b9U ^ static_cast<std::uint32_t>(i));
    const float u1 = uniform(hash(key));
    const float u2 = uniform(hash(key ^ 0x68bc21ebU));
    const float gauss = sqrtf(-2.0F * logf(u1)) * cospif(2.0F * u2);
    hit += gauss * (p.noise_base + p.noise_quadratic * (hit * hit));
  }
  depth[i] = hit;
  if (xyz != nullptr) {
    const float nan = __int_as_float(0x7fc00000);
    const bool valid = hit > 0.0F;
    xyz[3 * i] = valid ? rx * hit : nan;
    xyz[3 * i + 1] = valid ? ry * hit : nan;
    xyz[3 * i + 2] = valid ? hit : nan;
  }
}

void check_rectangle(float x0, float x1, float y0, float y1) {
  if (!(std::isfinite(x0) && std::isfinite(x1) && std::isfinite(y0) && std::isfinite(y1)) ||
      !(x0 < x1) || !(y0 < y1)) {
    throw std::invalid_argument("a feature needs a finite rectangle with x0 < x1 and y0 < y1");
  }
}

TerrainFeature feature(FeatureKind kind, float x0, float x1, float y0, float y1, float value) {
  check_rectangle(x0, x1, y0, y1);
  if (!std::isfinite(value)) {
    throw std::invalid_argument("a feature's value must be finite");
  }
  return TerrainFeature{.kind = kind, .x0 = x0, .x1 = x1, .y0 = y0, .y1 = y1, .value = value};
}

}  // namespace

TerrainFeature ramp(float x0, float x1, float y0, float y1, float slope_deg) {
  if (!(slope_deg > -89.0F && slope_deg < 89.0F)) {
    throw std::invalid_argument("a ramp's slope must be within +-89 degrees");
  }
  return feature(
      FeatureKind::kRamp, x0, x1, y0, y1,
      static_cast<float>(std::tan(static_cast<double>(slope_deg) * std::numbers::pi / 180.0)));
}

TerrainFeature steps(float x0, float x1, float y0, float y1, float height, float length) {
  if (!(std::isfinite(length) && length > 0.0F)) {
    throw std::invalid_argument("a step's length must be positive");
  }
  TerrainFeature f = feature(FeatureKind::kSteps, x0, x1, y0, y1, height);
  f.step_length = length;
  return f;
}

TerrainFeature box(float x0, float x1, float y0, float y1, float height) {
  return feature(FeatureKind::kBox, x0, x1, y0, y1, height);
}

TerrainFeature pit(float x0, float x1, float y0, float y1, float depth) {
  return feature(FeatureKind::kPit, x0, x1, y0, y1, depth);
}

void add_feature(Terrain& terrain, const TerrainFeature& feature) {
  if (terrain.feature_count >= kMaxFeatures) {
    throw std::invalid_argument("a terrain holds at most 8 features");
  }
  terrain.features[terrain.feature_count++] = feature;
}

double terrain_slope_deg(const Terrain& terrain, double x, double y) {
  for (int k = 0; k < terrain.feature_count; ++k) {
    const TerrainFeature& f = terrain.features[k];
    if (f.kind == FeatureKind::kRamp && x >= f.x0 && x < f.x1 && y >= f.y0 && y < f.y1) {
      return std::atan(static_cast<double>(f.value)) * 180.0 / std::numbers::pi;
    }
  }
  return 0.0;
}

double terrain_edge_distance(const Terrain& terrain, double x, double y) {
  double nearest = std::numeric_limits<double>::infinity();
  // The distance to the segment from (ax, ay) to (bx, by), axis-aligned.
  const auto segment = [x, y](double ax, double ay, double bx, double by) {
    const double cx = std::clamp(x, std::min(ax, bx), std::max(ax, bx));
    const double cy = std::clamp(y, std::min(ay, by), std::max(ay, by));
    return std::hypot(x - cx, y - cy);
  };
  for (int k = 0; k < terrain.feature_count; ++k) {
    const TerrainFeature& f = terrain.features[k];
    nearest = std::min({nearest, segment(f.x0, f.y0, f.x1, f.y0), segment(f.x0, f.y1, f.x1, f.y1),
                        segment(f.x0, f.y0, f.x0, f.y1), segment(f.x1, f.y0, f.x1, f.y1)});
    if (f.kind == FeatureKind::kSteps) {
      for (double e = f.x0 + f.step_length; e < f.x1; e += f.step_length) {
        nearest = std::min(nearest, segment(e, f.y0, e, f.y1));
      }
    }
  }
  return nearest;
}

Terrain course_terrain() {
  Terrain terrain;
  add_feature(terrain, ramp(1.5F, 3.0F, 0.3F, 1.1F, 15.0F));
  add_feature(terrain, steps(1.5F, 2.7F, -0.9F, -0.3F, 0.08F, 0.3F));
  add_feature(terrain, box(3.4F, 3.8F, -1.9F, -1.3F, 0.3F));
  add_feature(terrain, pit(3.4F, 4.2F, -0.4F, 0.4F, 0.2F));
  return terrain;
}

Transform camera_pose(double x, double y, double z, double yaw_rad, double pitch_down_rad) {
  // In the camera's level heading frame (x forward, y left, z up): optical
  // z (forward) = (cos p, 0, -sin p), optical x (right) = (0, -1, 0) and
  // optical y (down) = z cross x = (-sin p, 0, -cos p). Then yaw about z.
  const double cp = std::cos(pitch_down_rad);
  const double sp = std::sin(pitch_down_rad);
  const double cyaw = std::cos(yaw_rad);
  const double syaw = std::sin(yaw_rad);
  const double axes[3][3] = {{0.0, -1.0, 0.0}, {-sp, 0.0, -cp}, {cp, 0.0, -sp}};  // x, y, z.
  Transform pose;
  for (int column = 0; column < 3; ++column) {
    const double* a = axes[column];
    pose.m[column] = cyaw * a[0] - syaw * a[1];
    pose.m[4 + column] = syaw * a[0] + cyaw * a[1];
    pose.m[8 + column] = a[2];
  }
  pose.m[3] = x;
  pose.m[7] = y;
  pose.m[11] = z;
  return pose;
}

cudaError_t render_terrain_depth(const Terrain& terrain, const TerrainCamera& camera, float* depth,
                                 float* xyz, cudaStream_t stream) {
  if (camera.width < 1 || camera.height < 1 || camera.width > 16384 || camera.height > 16384 ||
      !(camera.fx > 0.0F) || !(camera.fy > 0.0F) || !(camera.max_depth > 0.0F) ||
      !std::isfinite(camera.max_depth) || !(camera.noise_base >= 0.0F) ||
      !(camera.noise_quadratic >= 0.0F) || depth == nullptr || terrain.feature_count < 0 ||
      terrain.feature_count > kMaxFeatures) {
    return cudaErrorInvalidValue;
  }
  RenderParams p{};
  p.terrain = terrain;
  p.width = camera.width;
  p.height = camera.height;
  p.fx = camera.fx;
  p.fy = camera.fy;
  p.cx = camera.cx;
  p.cy = camera.cy;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      p.r[3 * row + column] = static_cast<float>(camera.pose.m[4 * row + column]);
    }
    p.t[row] = static_cast<float>(camera.pose.m[4 * row + 3]);
  }
  p.max_depth = camera.max_depth;
  p.noise_base = camera.noise_base;
  p.noise_quadratic = camera.noise_quadratic;
  p.seed = camera.seed;
  const dim3 block(16, 16);
  const dim3 grid((static_cast<unsigned int>(camera.width) + 15U) / 16U,
                  (static_cast<unsigned int>(camera.height) + 15U) / 16U);
  render_kernel<<<grid, block, 0, stream>>>(p, depth, xyz);
  return cudaGetLastError();
}

}  // namespace holoscan::height_map
