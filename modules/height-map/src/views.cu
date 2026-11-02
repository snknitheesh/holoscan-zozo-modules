// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <holoscan/height_map/grid.hpp>
#include <holoscan/height_map/views.hpp>

namespace holoscan::height_map {
namespace {

// Float bits ordered as integers, for atomicMin and atomicMax.
__device__ inline int ordered(float value) {
  const int bits = __float_as_int(value);
  return bits >= 0 ? bits : bits ^ 0x7fffffff;
}

__device__ inline float unordered(int bits) {
  return __int_as_float(bits >= 0 ? bits : bits ^ 0x7fffffff);
}

__global__ void reset_range_kernel(int* range) {
  range[0] = 0x7fffffff;                    // Lowest so far: +max.
  range[1] = static_cast<int>(0x80000000);  // Highest so far: -max.
}

__global__ void range_kernel(const float* __restrict__ values, std::size_t count, int* range) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  float value = i < count ? values[i] : __int_as_float(0x7fc00000);
  const bool known = isfinite(value);
  // A block's extremes first, then one atomic per block.
  __shared__ int lo;
  __shared__ int hi;
  if (threadIdx.x == 0) {
    lo = 0x7fffffff;
    hi = static_cast<int>(0x80000000);
  }
  __syncthreads();
  if (known) {
    atomicMin(&lo, ordered(value));
    atomicMax(&hi, ordered(value));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicMin(&range[0], lo);
    atomicMax(&range[1], hi);
  }
}

// A rainbow from low to high: dark blue, cyan, green, yellow, red, linear
// between them.
__device__ inline void rainbow(float t, std::uint8_t* rgb) {
  constexpr float kStops[5][3] = {
      {0.10F, 0.15F, 0.60F}, {0.00F, 0.75F, 0.95F}, {0.20F, 0.80F, 0.20F},
      {0.95F, 0.90F, 0.10F}, {0.85F, 0.10F, 0.10F},
  };
  const float x = fminf(fmaxf(t, 0.0F), 1.0F) * 4.0F;
  const int k = min(static_cast<int>(x), 3);
  const float f = x - static_cast<float>(k);
  for (int channel = 0; channel < 3; ++channel) {
    const float value = kStops[k][channel] + (kStops[k + 1][channel] - kStops[k][channel]) * f;
    rgb[channel] = static_cast<std::uint8_t>(value * 255.0F + 0.5F);
  }
}

__global__ void elevation_kernel(const float* __restrict__ elevation, std::size_t count, float low,
                                 float high, const int* __restrict__ range,
                                 std::uint8_t* __restrict__ rgb) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  std::uint8_t* out = rgb + 3U * i;
  const float h = elevation[i];
  if (!isfinite(h)) {
    out[0] = out[1] = out[2] = kUnknownGray;
    return;
  }
  if (range != nullptr) {
    low = unordered(range[0]);
    high = unordered(range[1]);
  }
  const float span = high - low;
  rainbow(span > 0.0F ? (h - low) / span : 0.5F, out);
}

__global__ void cost_kernel(const float* __restrict__ cost, std::size_t count,
                            std::uint8_t* __restrict__ rgb) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  std::uint8_t* out = rgb + 3U * i;
  const float c = cost[i];
  if (!isfinite(c)) {
    out[0] = out[1] = out[2] = kUnknownGray;
    return;
  }
  const float t = fminf(fmaxf(c, 0.0F), 1.0F);
  // Green to yellow over the first half, yellow to red over the second.
  out[0] = static_cast<std::uint8_t>(fminf(2.0F * t, 1.0F) * 255.0F + 0.5F);
  out[1] = static_cast<std::uint8_t>(fminf(2.0F * (1.0F - t), 1.0F) * 255.0F + 0.5F);
  out[2] = 0U;
}

bool usable(const void* in, int rows, int cols, const std::uint8_t* rgb) {
  return rows >= 0 && cols >= 0 && rows <= kMaxGridSide && cols <= kMaxGridSide &&
         (rows == 0 || cols == 0 || (in != nullptr && rgb != nullptr));
}

}  // namespace

cudaError_t colorize_elevation(const float* elevation, int rows, int cols, float low, float high,
                               float* range_scratch, std::uint8_t* rgb, cudaStream_t stream) {
  if (!usable(elevation, rows, cols, rgb)) {
    return cudaErrorInvalidValue;
  }
  const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  if (count == 0U) {
    return cudaSuccess;
  }
  const auto blocks = static_cast<unsigned int>((count + 255U) / 256U);
  int* range = nullptr;
  if (!(low < high)) {
    if (range_scratch == nullptr) {
      return cudaErrorInvalidValue;
    }
    range = reinterpret_cast<int*>(range_scratch);
    reset_range_kernel<<<1, 1, 0, stream>>>(range);
    range_kernel<<<blocks, 256, 0, stream>>>(elevation, count, range);
  }
  elevation_kernel<<<blocks, 256, 0, stream>>>(elevation, count, low, high, range, rgb);
  return cudaGetLastError();
}

cudaError_t colorize_cost(const float* cost, int rows, int cols, std::uint8_t* rgb,
                          cudaStream_t stream) {
  if (!usable(cost, rows, cols, rgb)) {
    return cudaErrorInvalidValue;
  }
  const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  if (count == 0U) {
    return cudaSuccess;
  }
  cost_kernel<<<static_cast<unsigned int>((count + 255U) / 256U), 256, 0, stream>>>(cost, count,
                                                                                    rgb);
  return cudaGetLastError();
}

}  // namespace holoscan::height_map
