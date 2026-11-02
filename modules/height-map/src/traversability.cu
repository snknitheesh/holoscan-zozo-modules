// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <holoscan/height_map/grid.hpp>
#include <holoscan/height_map/traversability.hpp>

namespace holoscan::height_map {
namespace {

constexpr float kDegreesPerRadian = 57.29577951308232F;

struct Params {
  int rows;
  int cols;
  float resolution;
  int window_radius;
  int min_valid;
  int step_radius;
  float max_slope_deg;
  float max_step;
  float max_roughness;
  float slope_weight;
  float step_weight;
  float roughness_weight;
};

__global__ void traversability_kernel(const float* __restrict__ elevation, Params p,
                                      TraversabilityLayers out) {
  const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int r = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (c >= p.cols || r >= p.rows) {
    return;
  }
  const std::size_t i = static_cast<std::size_t>(r) * p.cols + c;
  const float nan = __int_as_float(0x7fc00000);
  const float center = elevation[i];
  const int r0 = max(r - p.window_radius, 0);
  const int r1 = min(r + p.window_radius, p.rows - 1);
  const int c0 = max(c - p.window_radius, 0);
  const int c1 = min(c + p.window_radius, p.cols - 1);

  // The plane fit, about the window's mean, in meters relative to the cell.
  int n = 0;
  float sx = 0.0F;
  float sy = 0.0F;
  float sz = 0.0F;
  if (isfinite(center)) {
    for (int v = r0; v <= r1; ++v) {
      for (int u = c0; u <= c1; ++u) {
        const float h = elevation[static_cast<std::size_t>(v) * p.cols + u];
        if (isfinite(h)) {
          ++n;
          sx += static_cast<float>(u - c) * p.resolution;
          sy += static_cast<float>(v - r) * p.resolution;
          sz += h - center;
        }
      }
    }
  }
  bool known = isfinite(center) && n >= p.min_valid && n >= 3;
  float a = 0.0F;
  float b = 0.0F;
  float mx = 0.0F;
  float my = 0.0F;
  float mz = 0.0F;
  if (known) {
    const float count = static_cast<float>(n);
    mx = sx / count;
    my = sy / count;
    mz = sz / count;
    float cxx = 0.0F;
    float cxy = 0.0F;
    float cyy = 0.0F;
    float cxz = 0.0F;
    float cyz = 0.0F;
    for (int v = r0; v <= r1; ++v) {
      for (int u = c0; u <= c1; ++u) {
        const float h = elevation[static_cast<std::size_t>(v) * p.cols + u];
        if (isfinite(h)) {
          const float x = static_cast<float>(u - c) * p.resolution - mx;
          const float y = static_cast<float>(v - r) * p.resolution - my;
          const float z = (h - center) - mz;
          cxx += x * x;
          cxy += x * y;
          cyy += y * y;
          cxz += x * z;
          cyz += y * z;
        }
      }
    }
    const float det = cxx * cyy - cxy * cxy;
    const float scale = cxx + cyy;
    // Collinear cells fit no plane.
    known = det > 1e-6F * (scale * scale);
    if (known) {
      a = (cxz * cyy - cyz * cxy) / det;
      b = (cyz * cxx - cxz * cxy) / det;
    }
  }
  if (!known) {
    if (out.cost != nullptr) {
      out.cost[i] = nan;
    }
    if (out.unknown != nullptr) {
      out.unknown[i] = 1U;
    }
    if (out.slope != nullptr) {
      out.slope[i] = nan;
    }
    if (out.step_height != nullptr) {
      out.step_height[i] = nan;
    }
    if (out.roughness != nullptr) {
      out.roughness[i] = nan;
    }
    return;
  }

  float squares = 0.0F;
  for (int v = r0; v <= r1; ++v) {
    for (int u = c0; u <= c1; ++u) {
      const float h = elevation[static_cast<std::size_t>(v) * p.cols + u];
      if (isfinite(h)) {
        const float x = static_cast<float>(u - c) * p.resolution - mx;
        const float y = static_cast<float>(v - r) * p.resolution - my;
        const float residual = ((h - center) - mz) - (a * x + b * y);
        squares += residual * residual;
      }
    }
  }
  const float roughness = sqrtf(squares / static_cast<float>(n));
  const float slope = atanf(sqrtf(a * a + b * b)) * kDegreesPerRadian;

  float lowest = center;
  float highest = center;
  const int s0 = max(r - p.step_radius, 0);
  const int s1 = min(r + p.step_radius, p.rows - 1);
  const int t0 = max(c - p.step_radius, 0);
  const int t1 = min(c + p.step_radius, p.cols - 1);
  for (int v = s0; v <= s1; ++v) {
    for (int u = t0; u <= t1; ++u) {
      const float h = elevation[static_cast<std::size_t>(v) * p.cols + u];
      if (isfinite(h)) {
        lowest = fminf(lowest, h);
        highest = fmaxf(highest, h);
      }
    }
  }
  const float step = highest - lowest;

  const float slope_term = slope / p.max_slope_deg;
  const float step_term = step / p.max_step;
  const float roughness_term = roughness / p.max_roughness;
  float cost = 1.0F;
  if (slope_term < 1.0F && step_term < 1.0F && roughness_term < 1.0F) {
    cost = (p.slope_weight * slope_term + p.step_weight * step_term +
            p.roughness_weight * roughness_term) /
           (p.slope_weight + p.step_weight + p.roughness_weight);
    cost = fminf(fmaxf(cost, 0.0F), 1.0F);
  }
  if (out.cost != nullptr) {
    out.cost[i] = cost;
  }
  if (out.unknown != nullptr) {
    out.unknown[i] = 0U;
  }
  if (out.slope != nullptr) {
    out.slope[i] = slope;
  }
  if (out.step_height != nullptr) {
    out.step_height[i] = step;
  }
  if (out.roughness != nullptr) {
    out.roughness[i] = roughness;
  }
}

bool positive(float value) { return std::isfinite(value) && value > 0.0F; }
bool not_negative(float value) { return std::isfinite(value) && value >= 0.0F; }

}  // namespace

bool valid_params(const TraversabilityParams& p, const char** what) {
  const auto fail = [what](const char* message) {
    if (what != nullptr) {
      *what = message;
    }
    return false;
  };
  if (p.window_radius < 1 || p.window_radius > 10) {
    return fail("window_radius must be 1..10");
  }
  if (!(p.min_valid_fraction >= 0.0F && p.min_valid_fraction <= 1.0F)) {
    return fail("min_valid_fraction must be 0..1");
  }
  if (p.step_radius < 1 || p.step_radius > 10) {
    return fail("step_radius must be 1..10");
  }
  if (!positive(p.max_slope_deg) || !positive(p.max_step) || !positive(p.max_roughness)) {
    return fail("max_slope_deg, max_step and max_roughness must be positive");
  }
  if (!not_negative(p.slope_weight) || !not_negative(p.step_weight) ||
      !not_negative(p.roughness_weight) ||
      !(p.slope_weight + p.step_weight + p.roughness_weight > 0.0F)) {
    return fail("the weights must not be negative, nor all 0");
  }
  return true;
}

cudaError_t compute_traversability(const float* elevation, int rows, int cols, float resolution,
                                   const TraversabilityParams& params,
                                   const TraversabilityLayers& layers, cudaStream_t stream) {
  if (!valid_params(params) || rows < 0 || cols < 0 || rows > kMaxGridSide || cols > kMaxGridSide ||
      !positive(resolution)) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0 || cols == 0) {
    return cudaSuccess;
  }
  if (elevation == nullptr) {
    return cudaErrorInvalidValue;
  }
  const int side = 2 * params.window_radius + 1;
  const Params p{
      .rows = rows,
      .cols = cols,
      .resolution = resolution,
      .window_radius = params.window_radius,
      .min_valid =
          static_cast<int>(std::ceil(params.min_valid_fraction * static_cast<float>(side * side))),
      .step_radius = params.step_radius,
      .max_slope_deg = params.max_slope_deg,
      .max_step = params.max_step,
      .max_roughness = params.max_roughness,
      .slope_weight = params.slope_weight,
      .step_weight = params.step_weight,
      .roughness_weight = params.roughness_weight,
  };
  const dim3 block(16, 16);
  const dim3 grid((static_cast<unsigned int>(cols) + 15U) / 16U,
                  (static_cast<unsigned int>(rows) + 15U) / 16U);
  traversability_kernel<<<grid, block, 0, stream>>>(elevation, p, layers);
  return cudaGetLastError();
}

}  // namespace holoscan::height_map
