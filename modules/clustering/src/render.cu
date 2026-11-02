// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <holoscan/clustering/clustering.hpp>
#include <holoscan/clustering/render.hpp>

#include "kernel_common.cuh"

namespace holoscan::clustering {
namespace {

using detail::blocks_for;
using detail::kBlock;

// Edges are drawn from this depth on.
constexpr float kNear = 0.05F;
// The longest edge drawn, in pixels.
constexpr int kMaxEdgePixels = 8192;

__global__ void colorize_kernel(const std::int32_t* __restrict__ labels, std::uint32_t count,
                                std::uint8_t* __restrict__ rgb) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    label_color(labels[i], rgb + 3 * static_cast<std::size_t>(i));
  }
}

// Corner c of cluster row's box: bit 0 picks the high side of the first axis,
// bit 1 of the second, bit 2 of the third.
__device__ inline void corner(const float* row, bool oriented, int c, float p[3]) {
  for (int k = 0; k < 3; ++k) {
    if (!oriented) {
      p[k] = (c >> k) & 1 ? row[8 + k] : row[5 + k];
    } else {
      p[k] = row[11 + k];
    }
  }
  if (oriented) {
    for (int axis = 0; axis < 3; ++axis) {
      const float s = ((c >> axis) & 1 ? 1.0F : -1.0F) * row[23 + axis];
      for (int k = 0; k < 3; ++k) {
        p[k] += s * row[14 + 3 * axis + k];
      }
    }
  }
}

// Block b draws edge b % 12 of cluster b / 12, one thread per pixel step.
__global__ void boxes_kernel(const float* __restrict__ summaries, float fx, float fy, float cx,
                             float cy, int width, int height, std::uint8_t* __restrict__ rgb) {
  const std::uint32_t cluster = blockIdx.x / 12U;
  const int edge = static_cast<int>(blockIdx.x % 12U);
  const float* row = summaries + static_cast<std::size_t>(cluster) * kSummaryFields;
  const bool oriented = isfinite(row[23]) && isfinite(row[11]);
  // Edge e runs along axis e / 4 from the corner whose other two bits are e % 4.
  const int axis = edge / 4;
  const int rest = edge % 4;
  const int low_bits = rest & 1;
  const int high_bits = (rest >> 1) & 1;
  int base = 0;
  int bit = 0;
  for (int k = 0; k < 3; ++k) {
    if (k != axis) {
      base |= (bit == 0 ? low_bits : high_bits) << k;
      ++bit;
    }
  }
  float a[3];
  float b[3];
  corner(row, oriented, base, a);
  corner(row, oriented, base | (1 << axis), b);
  // Clip to the near plane.
  if (a[2] < kNear && b[2] < kNear) {
    return;
  }
  if (a[2] < kNear || b[2] < kNear) {
    float* behind = a[2] < kNear ? a : b;
    const float* front = a[2] < kNear ? b : a;
    const float t = (kNear - behind[2]) / (front[2] - behind[2]);
    for (int k = 0; k < 3; ++k) {
      behind[k] += t * (front[k] - behind[k]);
    }
  }
  const float u0 = fx * a[0] / a[2] + cx;
  const float v0 = fy * a[1] / a[2] + cy;
  const float u1 = fx * b[0] / b[2] + cx;
  const float v1 = fy * b[1] / b[2] + cy;
  const float span = fmaxf(fabsf(u1 - u0), fabsf(v1 - v0));
  if (!isfinite(span)) {
    return;
  }
  const int steps = static_cast<int>(fminf(ceilf(span), static_cast<float>(kMaxEdgePixels)));
  std::uint8_t color[3];
  label_color(static_cast<std::int32_t>(cluster), color);
  for (int s = static_cast<int>(threadIdx.x); s <= steps; s += static_cast<int>(blockDim.x)) {
    const float t = steps == 0 ? 0.0F : static_cast<float>(s) / static_cast<float>(steps);
    const int u = __float2int_rn(u0 + t * (u1 - u0));
    const int v = __float2int_rn(v0 + t * (v1 - v0));
    // Two pixels wide, so that the edge shows over the clusters' colors.
    for (int du = 0; du < 2; ++du) {
      if (u + du >= 0 && u + du < width && v >= 0 && v < height) {
        std::uint8_t* pixel = rgb + 3 * (static_cast<std::size_t>(v) * width + u + du);
        pixel[0] = 255U - (255U - color[0]) / 3U;
        pixel[1] = 255U - (255U - color[1]) / 3U;
        pixel[2] = 255U - (255U - color[2]) / 3U;
      }
    }
  }
}

}  // namespace

cudaError_t colorize_labels(const std::int32_t* labels, std::uint32_t count, std::uint8_t* rgb,
                            cudaStream_t stream) {
  if (count == 0U) {
    return cudaSuccess;
  }
  colorize_kernel<<<blocks_for(count), kBlock, 0, stream>>>(labels, count, rgb);
  return cudaGetLastError();
}

cudaError_t draw_cluster_boxes(const float* summaries, std::uint32_t clusters, float fx, float fy,
                               float cx, float cy, int width, int height, std::uint8_t* rgb,
                               cudaStream_t stream) {
  if (width < 1 || height < 1 || !(fx > 0.0F) || !(fy > 0.0F)) {
    return cudaErrorInvalidValue;
  }
  if (clusters == 0U) {
    return cudaSuccess;
  }
  boxes_kernel<<<clusters * 12U, 128, 0, stream>>>(summaries, fx, fy, cx, cy, width, height, rgb);
  return cudaGetLastError();
}

}  // namespace holoscan::clustering
