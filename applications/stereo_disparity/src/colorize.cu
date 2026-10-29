// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "disparity_color_op.hpp"

namespace stereo_disparity_app {
namespace {

// The jet color map: dark blue (0), blue, cyan, green, yellow, red, dark red (1).
__device__ inline uchar4 color_map(float t) {
  const float r = fminf(fmaxf(1.5F - fabsf(4.0F * t - 3.0F), 0.0F), 1.0F);
  const float g = fminf(fmaxf(1.5F - fabsf(4.0F * t - 2.0F), 0.0F), 1.0F);
  const float b = fminf(fmaxf(1.5F - fabsf(4.0F * t - 1.0F), 0.0F), 1.0F);
  return make_uchar4(static_cast<unsigned char>(255.0F * r + 0.5F),
                     static_cast<unsigned char>(255.0F * g + 0.5F),
                     static_cast<unsigned char>(255.0F * b + 0.5F), 255);
}

__global__ void colorize_kernel(const float* disparity, std::size_t pitch, int width, int height,
                                float max_disparity, uchar4* rgba) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height) {
    return;
  }
  const float d =
      reinterpret_cast<const float*>(reinterpret_cast<const char*>(disparity) + y * pitch)[x];
  rgba[static_cast<std::size_t>(y) * width + x] =
      d > 0.0F ? color_map(fminf(d / max_disparity, 1.0F)) : make_uchar4(0, 0, 0, 255);
}

}  // namespace

cudaError_t colorize_disparity(const float* disparity, std::size_t pitch, int width, int height,
                               float max_disparity, std::uint8_t* rgba, cudaStream_t stream) {
  const dim3 block(32, 8);
  const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  colorize_kernel<<<grid, block, 0, stream>>>(disparity, pitch, width, height, max_disparity,
                                              reinterpret_cast<uchar4*>(rgba));
  return cudaGetLastError();
}

}  // namespace stereo_disparity_app
