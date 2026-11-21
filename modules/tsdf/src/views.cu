// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include <holoscan/tsdf/views.hpp>

namespace holoscan::tsdf {
namespace {

__device__ std::uint8_t to_byte(float value) {
  return static_cast<std::uint8_t>(fminf(fmaxf(value, 0.0F), 1.0F) * 255.0F + 0.5F);
}

__global__ void colorize_kernel(const void* depth, DepthType type, float scale, int width,
                                int height, float depth_min, float depth_max, std::uint8_t* rgb,
                                std::size_t row_bytes) {
  const int u = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int v = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (u >= width || v >= height) {
    return;
  }
  const std::size_t pixel = static_cast<std::size_t>(v) * width + u;
  const float raw = type == DepthType::kUint16
                        ? static_cast<float>(static_cast<const std::uint16_t*>(depth)[pixel])
                        : static_cast<const float*>(depth)[pixel];
  const float d = raw * scale;
  std::uint8_t* out =
      rgb + static_cast<std::size_t>(v) * row_bytes + 3U * static_cast<std::size_t>(u);
  if (!(d > 0.0F && d >= depth_min && d <= depth_max)) {
    out[0] = out[1] = out[2] = 0U;
    return;
  }
  // A blue-cyan-yellow-red ramp, red near.
  const float t = 1.0F - (d - depth_min) / (depth_max - depth_min);
  out[0] = to_byte(1.5F - fabsf(4.0F * t - 3.0F));
  out[1] = to_byte(1.5F - fabsf(4.0F * t - 2.0F));
  out[2] = to_byte(1.5F - fabsf(4.0F * t - 1.0F));
}

__global__ void shade_kernel(const float* normals, const std::uint8_t* colors, int width,
                             int height, std::uint8_t* rgb, std::size_t row_bytes) {
  const int u = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int v = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (u >= width || v >= height) {
    return;
  }
  const std::size_t pixel = static_cast<std::size_t>(v) * width + u;
  std::uint8_t* out =
      rgb + static_cast<std::size_t>(v) * row_bytes + 3U * static_cast<std::size_t>(u);
  const float nz = normals[3 * pixel + 2];
  if (isnan(nz)) {
    out[0] = out[1] = out[2] = 0U;
    return;
  }
  // Facing the camera is -z in the camera frame.
  const float light = 0.25F + 0.75F * fmaxf(-nz, 0.0F);
  for (int c = 0; c < 3; ++c) {
    const float base =
        colors != nullptr ? static_cast<float>(colors[3 * pixel + c]) / 255.0F : 0.85F;
    out[c] = to_byte(base * light);
  }
}

}  // namespace

cudaError_t colorize_depth(const void* depth, DepthType type, float scale, int width, int height,
                           float depth_min, float depth_max, std::uint8_t* rgb,
                           std::size_t row_bytes, cudaStream_t stream) {
  if (width < 1 || height < 1) {
    return cudaSuccess;
  }
  if (depth == nullptr || rgb == nullptr || row_bytes < 3U * static_cast<std::size_t>(width) ||
      !(depth_max > depth_min)) {
    return cudaErrorInvalidValue;
  }
  const dim3 block(16, 16);
  const dim3 grid((width + 15) / 16, (height + 15) / 16);
  colorize_kernel<<<grid, block, 0, stream>>>(depth, type, scale, width, height, depth_min,
                                              depth_max, rgb, row_bytes);
  return cudaGetLastError();
}

cudaError_t shade_raycast(const float* normals, const std::uint8_t* colors, int width, int height,
                          std::uint8_t* rgb, std::size_t row_bytes, cudaStream_t stream) {
  if (width < 1 || height < 1) {
    return cudaSuccess;
  }
  if (normals == nullptr || rgb == nullptr || row_bytes < 3U * static_cast<std::size_t>(width)) {
    return cudaErrorInvalidValue;
  }
  const dim3 block(16, 16);
  const dim3 grid((width + 15) / 16, (height + 15) / 16);
  shade_kernel<<<grid, block, 0, stream>>>(normals, colors, width, height, rgb, row_bytes);
  return cudaGetLastError();
}

}  // namespace holoscan::tsdf
