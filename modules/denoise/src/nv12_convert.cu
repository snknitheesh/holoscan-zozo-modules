// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "nv12_convert.hpp"

namespace holoscan::denoise::detail {
namespace {

__device__ __forceinline__ std::uint8_t to_byte(float v) {
  v = v < 0.0F ? 0.0F : v > 255.0F ? 255.0F : v;
  return static_cast<std::uint8_t>(__float2int_rn(v));
}

__device__ __forceinline__ const std::uint8_t* at(const ImageView& v, int x, int y) {
  x = x < v.width ? x : v.width - 1;
  y = y < v.height ? y : v.height - 1;
  return v.planes[0].data + static_cast<std::size_t>(y) * v.planes[0].pitch +
         static_cast<std::size_t>(x) * pixel_bytes(v.format);
}

__device__ __forceinline__ float luma_of(const ImageView& v, const std::uint8_t* p) {
  if (v.format == PixelFormat::kMono8) return p[0];
  return 0.299F * p[0] + 0.587F * p[1] + 0.114F * p[2];
}

// One thread per 2x2 block: four Y and one UV pair.
__global__ void to_nv12_kernel(ImageView in, Nv12 out) {
  const int bx = blockIdx.x * blockDim.x + threadIdx.x;
  const int by = blockIdx.y * blockDim.y + threadIdx.y;
  if (2 * bx >= out.width || 2 * by >= out.height) return;
  float r = 0.0F;
  float g = 0.0F;
  float b = 0.0F;
  for (int dy = 0; dy < 2; ++dy) {
    for (int dx = 0; dx < 2; ++dx) {
      const int x = 2 * bx + dx;
      const int y = 2 * by + dy;
      const std::uint8_t* p = at(in, x, y);
      out.y.data[static_cast<std::size_t>(y) * out.y.pitch + x] = to_byte(luma_of(in, p));
      if (in.format != PixelFormat::kMono8) {
        r += p[0];
        g += p[1];
        b += p[2];
      }
    }
  }
  std::uint8_t* uv = out.uv.data + static_cast<std::size_t>(by) * out.uv.pitch + 2 * bx;
  if (in.format == PixelFormat::kMono8) {
    uv[0] = 128;
    uv[1] = 128;
    return;
  }
  r *= 0.25F;
  g *= 0.25F;
  b *= 0.25F;
  uv[0] = to_byte(128.0F - 0.168736F * r - 0.331264F * g + 0.5F * b);
  uv[1] = to_byte(128.0F + 0.5F * r - 0.418688F * g - 0.081312F * b);
}

// Chroma at a pixel, from the half-resolution plane, sited at the 2x2 centres.
__device__ __forceinline__ void chroma(const Nv12& in, int x, int y, float& u, float& v) {
  const int cw = in.width / 2;
  const int ch = in.height / 2;
  const float fx = (static_cast<float>(x) - 0.5F) * 0.5F;
  const float fy = (static_cast<float>(y) - 0.5F) * 0.5F;
  int x0 = static_cast<int>(floorf(fx));
  int y0 = static_cast<int>(floorf(fy));
  const float ax = fx - static_cast<float>(x0);
  const float ay = fy - static_cast<float>(y0);
  const int x1 = min(max(x0 + 1, 0), cw - 1);
  const int y1 = min(max(y0 + 1, 0), ch - 1);
  x0 = min(max(x0, 0), cw - 1);
  y0 = min(max(y0, 0), ch - 1);
  const auto sample = [&](int sx, int sy, int c) {
    return static_cast<float>(in.uv.data[static_cast<std::size_t>(sy) * in.uv.pitch + 2 * sx + c]);
  };
  for (int c = 0; c < 2; ++c) {
    const float top = sample(x0, y0, c) + ax * (sample(x1, y0, c) - sample(x0, y0, c));
    const float bottom = sample(x0, y1, c) + ax * (sample(x1, y1, c) - sample(x0, y1, c));
    (c == 0 ? u : v) = top + ay * (bottom - top);
  }
}

__global__ void from_nv12_kernel(Nv12 in, ImageView out, ImageView alpha) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= out.width || y >= out.height) return;
  const float luma = in.y.data[static_cast<std::size_t>(y) * in.y.pitch + x];
  std::uint8_t* o = out.planes[0].data + static_cast<std::size_t>(y) * out.planes[0].pitch +
                    static_cast<std::size_t>(x) * pixel_bytes(out.format);
  if (out.format == PixelFormat::kMono8) {
    o[0] = static_cast<std::uint8_t>(luma);
    return;
  }
  float u = 0.0F;
  float v = 0.0F;
  chroma(in, x, y, u, v);
  u -= 128.0F;
  v -= 128.0F;
  o[0] = to_byte(luma + 1.402F * v);
  o[1] = to_byte(luma - 0.344136F * u - 0.714136F * v);
  o[2] = to_byte(luma + 1.772F * u);
  if (out.format == PixelFormat::kRgba8) o[3] = at(alpha, x, y)[3];
}

}  // namespace

cudaError_t launch_to_nv12(const ImageView& input, const Nv12& output, cudaStream_t stream) {
  if (input.format == PixelFormat::kNv12) {
    cudaError_t status = cudaMemcpy2DAsync(output.y.data, output.y.pitch, input.planes[0].data,
                                           input.planes[0].pitch, input.width, input.height,
                                           cudaMemcpyDeviceToDevice, stream);
    if (status != cudaSuccess) return status;
    return cudaMemcpy2DAsync(output.uv.data, output.uv.pitch, input.planes[1].data,
                             input.planes[1].pitch, input.width, input.height / 2,
                             cudaMemcpyDeviceToDevice, stream);
  }
  const dim3 block(16, 8);
  const dim3 grid((output.width / 2 + 15) / 16, (output.height / 2 + 7) / 8);
  to_nv12_kernel<<<grid, block, 0, stream>>>(input, output);
  return cudaGetLastError();
}

cudaError_t launch_from_nv12(const Nv12& input, const ImageView& output,
                             const ImageView& alpha_source, cudaStream_t stream) {
  if (output.format == PixelFormat::kNv12) {
    cudaError_t status = cudaMemcpy2DAsync(output.planes[0].data, output.planes[0].pitch,
                                           input.y.data, input.y.pitch, output.width, output.height,
                                           cudaMemcpyDeviceToDevice, stream);
    if (status != cudaSuccess) return status;
    return cudaMemcpy2DAsync(output.planes[1].data, output.planes[1].pitch, input.uv.data,
                             input.uv.pitch, output.width, output.height / 2,
                             cudaMemcpyDeviceToDevice, stream);
  }
  const dim3 block(32, 8);
  const dim3 grid((output.width + 31) / 32, (output.height + 7) / 8);
  from_nv12_kernel<<<grid, block, 0, stream>>>(input, output, alpha_source);
  return cudaGetLastError();
}

}  // namespace holoscan::denoise::detail
