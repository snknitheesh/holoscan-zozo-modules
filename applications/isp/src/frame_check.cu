// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "frame_check.hpp"

namespace isp_demo {
namespace {

constexpr int kThreads = 256;

// Each block: squared error of its pixels, and their histograms in shared memory.
__global__ void error_and_histogram_kernel(holoscan::isp::SyntheticRawScene scene, const float* lut,
                                           int lut_size, const std::uint8_t* rgb,
                                           std::uint32_t* histogram, FrameCheckResult* result) {
  __shared__ std::uint32_t bins[kHistogramEntries];
  __shared__ ErrorSum warp_sums[kThreads / 32];
  for (int i = static_cast<int>(threadIdx.x); i < kHistogramEntries; i += kThreads) bins[i] = 0U;
  __syncthreads();

  const std::int64_t pixels = static_cast<std::int64_t>(scene.width) * scene.height;
  ErrorSum squared = ErrorSum{0};
  for (std::int64_t p = static_cast<std::int64_t>(blockIdx.x) * kThreads + threadIdx.x; p < pixels;
       p += static_cast<std::int64_t>(gridDim.x) * kThreads) {
    const int x = static_cast<int>(p % scene.width);
    const int y = static_cast<int>(p / scene.width);
    int truth[3];
    truth_codes(scene, lut, lut_size, x, y, truth);
    const std::uint8_t* pixel = rgb + 3 * p;
    for (int k = 0; k < 3; ++k) {
      const int d = static_cast<int>(pixel[k]) - truth[k];
      squared += static_cast<ErrorSum>(d * d);
    }
    const int r = pixel[0];
    const int g = pixel[1];
    const int b = pixel[2];
    atomicAdd(
        &bins[holoscan::isp::kHistogramLuma * holoscan::isp::kHistogramBins + luma_bin(r, g, b)],
        1U);
    atomicAdd(&bins[holoscan::isp::kHistogramRed * holoscan::isp::kHistogramBins + r], 1U);
    atomicAdd(&bins[holoscan::isp::kHistogramGreen * holoscan::isp::kHistogramBins + g], 1U);
    atomicAdd(&bins[holoscan::isp::kHistogramBlue * holoscan::isp::kHistogramBins + b], 1U);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    squared += __shfl_down_sync(0xFFFFFFFFU, squared, offset);
  }
  if (threadIdx.x % 32 == 0) warp_sums[threadIdx.x / 32] = squared;
  __syncthreads();
  if (threadIdx.x == 0) {
    ErrorSum block = ErrorSum{0};
    for (const ErrorSum s : warp_sums) block += s;
    atomicAdd(&result->squared_error, block);
  }
  for (int i = static_cast<int>(threadIdx.x); i < kHistogramEntries; i += kThreads) {
    if (bins[i] != 0U) atomicAdd(&histogram[i], bins[i]);
  }
}

// One block: the rebuilt histograms against IspOp's, and IspOp's statistics.
__global__ void compare_kernel(const std::uint32_t* rebuilt, const std::uint32_t* sent,
                               const float* stats, FrameCheckResult* result) {
  unsigned int mismatches = 0U;
  for (int i = static_cast<int>(threadIdx.x); i < kHistogramEntries; i += kThreads) {
    mismatches += rebuilt[i] != sent[i] ? 1U : 0U;
  }
  if (mismatches != 0U) atomicAdd(&result->histogram_mismatches, mismatches);
  if (threadIdx.x < holoscan::isp::stats::kStatsSize) {
    result->stats[threadIdx.x] = stats[threadIdx.x];
  }
}

}  // namespace

cudaError_t check_frame_cuda(const holoscan::isp::SyntheticRawScene& scene, const float* lut,
                             int lut_size, const std::uint8_t* rgb, const std::uint32_t* histogram,
                             const float* stats, std::uint32_t* scratch, FrameCheckResult* result,
                             cudaStream_t stream) {
  if (const cudaError_t status = cudaMemsetAsync(result, 0, sizeof(FrameCheckResult), stream);
      status != cudaSuccess) {
    return status;
  }
  if (const cudaError_t status =
          cudaMemsetAsync(scratch, 0, sizeof(std::uint32_t) * kHistogramEntries, stream);
      status != cudaSuccess) {
    return status;
  }
  int device = 0;
  int sms = 1;
  if (const cudaError_t status = cudaGetDevice(&device); status != cudaSuccess) {
    return status;
  }
  if (const cudaError_t status =
          cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
      status != cudaSuccess) {
    return status;
  }
  const std::int64_t pixels = static_cast<std::int64_t>(scene.width) * scene.height;
  const int blocks = static_cast<int>(std::min<std::int64_t>((pixels + kThreads - 1) / kThreads,
                                                             static_cast<std::int64_t>(sms) * 8));
  error_and_histogram_kernel<<<blocks, kThreads, 0, stream>>>(scene, lut, lut_size, rgb, scratch,
                                                              result);
  compare_kernel<<<1, kThreads, 0, stream>>>(scratch, histogram, stats, result);
  return cudaGetLastError();
}

}  // namespace isp_demo
