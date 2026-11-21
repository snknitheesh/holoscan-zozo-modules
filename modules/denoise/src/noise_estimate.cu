// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "noise_estimate.hpp"

namespace holoscan::denoise::detail {
namespace {

constexpr int kThreads = 256;

__device__ int load(const SamplePlane& p, int x, int y) {
  x = x < 0 ? 0 : x >= p.width ? p.width - 1 : x;
  y = y < 0 ? 0 : y >= p.height ? p.height - 1 : y;
  return p.data[static_cast<std::size_t>(y) * p.pitch + static_cast<std::size_t>(x) * p.stride +
                p.channel];
}

// Sums |I * N| over the interior, N = [1 -2 1; -2 4 -2; 1 -2 1], one partial per block.
__global__ void noise_partials_kernel(SamplePlane p, std::uint64_t* partials) {
  __shared__ std::uint64_t sums[kThreads];
  const std::int64_t inner_w = p.width - 2;
  const std::int64_t count = inner_w * (p.height - 2);
  std::uint64_t sum = 0;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const int x = static_cast<int>(i % inner_w) + 1;
    const int y = static_cast<int>(i / inner_w) + 1;
    const int v = load(p, x - 1, y - 1) - 2 * load(p, x, y - 1) + load(p, x + 1, y - 1) -
                  2 * load(p, x - 1, y) + 4 * load(p, x, y) - 2 * load(p, x + 1, y) +
                  load(p, x - 1, y + 1) - 2 * load(p, x, y + 1) + load(p, x + 1, y + 1);
    sum += static_cast<std::uint64_t>(v < 0 ? -v : v);
  }
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (int half = kThreads / 2; half > 0; half /= 2) {
    if (threadIdx.x < half) sums[threadIdx.x] += sums[threadIdx.x + half];
    __syncthreads();
  }
  if (threadIdx.x == 0) partials[blockIdx.x] = sums[0];
}

__global__ void noise_finalize_kernel(const std::uint64_t* partials, int width, int height,
                                      float min_sigma, float* sigma) {
  __shared__ std::uint64_t sums[kThreads];
  std::uint64_t sum = 0;
  for (int i = threadIdx.x; i < kNoiseBlocks; i += kThreads) sum += partials[i];
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (int half = kThreads / 2; half > 0; half /= 2) {
    if (threadIdx.x < half) sums[threadIdx.x] += sums[threadIdx.x + half];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float estimate = min_sigma;
    if (width > 2 && height > 2) {
      // sqrt(pi / 2) / 6 per pixel of the interior (Immerkaer 1996).
      const double pixels = static_cast<double>(width - 2) * static_cast<double>(height - 2);
      estimate =
          static_cast<float>(1.2533141373155003 * static_cast<double>(sums[0]) / (6.0 * pixels));
    }
    *sigma = estimate > min_sigma ? estimate : min_sigma;
  }
}

__device__ float box_mean(const SamplePlane& p, int x, int y) {
  int sum = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) sum += load(p, x + dx, y + dy);
  }
  return static_cast<float>(sum) * (1.0F / 9.0F);
}

__global__ void count_changed_kernel(SamplePlane current, SamplePlane previous, const float* sigma,
                                     float previous_variance, float threshold,
                                     unsigned int* count) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  bool changed = false;
  if (x < current.width && y < current.height) {
    const float limit = threshold * *sigma * sqrtf((1.0F + previous_variance) / 9.0F);
    changed = fabsf(box_mean(current, x, y) - box_mean(previous, x, y)) > limit;
  }
  const int changed_in_block = __syncthreads_count(changed ? 1 : 0);
  if (threadIdx.x == 0 && threadIdx.y == 0 && changed_in_block != 0) {
    atomicAdd(count, static_cast<unsigned int>(changed_in_block));
  }
}

__global__ void set_kernel(float* value, float to) { *value = to; }

}  // namespace

cudaError_t launch_noise_estimate(const SamplePlane& plane, std::uint64_t* partials,
                                  float min_sigma, float* sigma, cudaStream_t stream) {
  noise_partials_kernel<<<kNoiseBlocks, kThreads, 0, stream>>>(plane, partials);
  noise_finalize_kernel<<<1, kThreads, 0, stream>>>(partials, plane.width, plane.height, min_sigma,
                                                    sigma);
  return cudaGetLastError();
}

cudaError_t launch_count_changed(const SamplePlane& current, const SamplePlane& previous,
                                 const float* sigma, float previous_variance, float threshold,
                                 unsigned int* count, cudaStream_t stream) {
  const dim3 block(32, 8);
  const dim3 grid((current.width + 31) / 32, (current.height + 7) / 8);
  count_changed_kernel<<<grid, block, 0, stream>>>(current, previous, sigma, previous_variance,
                                                   threshold, count);
  return cudaGetLastError();
}

cudaError_t launch_set(float* value, float to, cudaStream_t stream) {
  set_kernel<<<1, 1, 0, stream>>>(value, to);
  return cudaGetLastError();
}

}  // namespace holoscan::denoise::detail
