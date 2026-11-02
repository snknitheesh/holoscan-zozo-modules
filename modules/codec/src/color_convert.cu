// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// RGB and NV12 conversions on the GPU, BT.709 in limited range: one thread per 2x2 pixel block,
// which shares one chroma sample.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "backend.hpp"

namespace holoscan::codec::detail {
namespace {

// BT.709 luma weights.
constexpr float kKr = 0.2126F;
constexpr float kKb = 0.0722F;
constexpr float kKg = 1.0F - kKr - kKb;

struct Rgb {
  float r, g, b;
};

__device__ __forceinline__ std::uint8_t to_byte(float value) {
  return static_cast<std::uint8_t>(fminf(fmaxf(value + 0.5F, 0.0F), 255.0F));
}

__device__ __forceinline__ Rgb load_rgb(const std::uint8_t* row, int x, int channels, bool bgr) {
  const std::uint8_t* p = row + static_cast<std::size_t>(x) * channels;
  const float r = p[bgr ? 2 : 0];
  const float g = p[1];
  const float b = p[bgr ? 0 : 2];
  return Rgb{r, g, b};
}

__device__ __forceinline__ float luma_of(const Rgb& c) {
  return 16.0F + (219.0F / 255.0F) * (kKr * c.r + kKg * c.g + kKb * c.b);
}

// Packed RGB to NV12; pixels beyond the image repeat its last column and row.
__global__ void rgb_to_nv12(const std::uint8_t* src, std::size_t src_pitch, int src_width,
                            int src_height, int channels, bool bgr, std::uint8_t* luma,
                            std::size_t luma_pitch, std::uint8_t* chroma, std::size_t chroma_pitch,
                            int width, int height) {
  const int bx = blockIdx.x * blockDim.x + threadIdx.x;
  const int by = blockIdx.y * blockDim.y + threadIdx.y;
  if (bx >= width / 2 || by >= height / 2) {
    return;
  }
  float sum_r = 0.0F, sum_g = 0.0F, sum_b = 0.0F;
  for (int dy = 0; dy < 2; ++dy) {
    const int y = 2 * by + dy;
    const int sy = min(y, src_height - 1);
    const std::uint8_t* row = src + static_cast<std::size_t>(sy) * src_pitch;
    for (int dx = 0; dx < 2; ++dx) {
      const int x = 2 * bx + dx;
      const Rgb c = load_rgb(row, min(x, src_width - 1), channels, bgr);
      luma[static_cast<std::size_t>(y) * luma_pitch + x] = to_byte(luma_of(c));
      sum_r += c.r;
      sum_g += c.g;
      sum_b += c.b;
    }
  }
  const Rgb mean{sum_r * 0.25F, sum_g * 0.25F, sum_b * 0.25F};
  const float y = kKr * mean.r + kKg * mean.g + kKb * mean.b;
  const float cb = (mean.b - y) / (2.0F * (1.0F - kKb));
  const float cr = (mean.r - y) / (2.0F * (1.0F - kKr));
  std::uint8_t* uv = chroma + static_cast<std::size_t>(by) * chroma_pitch + 2 * bx;
  uv[0] = to_byte(128.0F + (224.0F / 255.0F) * cb);
  uv[1] = to_byte(128.0F + (224.0F / 255.0F) * cr);
}

// NV12 to a larger (or equal) NV12, repeating the last column and row.
__global__ void pad_nv12(const std::uint8_t* src_luma, std::size_t src_luma_pitch,
                         const std::uint8_t* src_chroma, std::size_t src_chroma_pitch,
                         int src_width, int src_height, std::uint8_t* luma, std::size_t luma_pitch,
                         std::uint8_t* chroma, std::size_t chroma_pitch, int width, int height) {
  const int bx = blockIdx.x * blockDim.x + threadIdx.x;
  const int by = blockIdx.y * blockDim.y + threadIdx.y;
  if (bx >= width / 2 || by >= height / 2) {
    return;
  }
  for (int dy = 0; dy < 2; ++dy) {
    const int y = 2 * by + dy;
    const int sy = min(y, src_height - 1);
    for (int dx = 0; dx < 2; ++dx) {
      const int x = 2 * bx + dx;
      luma[static_cast<std::size_t>(y) * luma_pitch + x] =
          src_luma[static_cast<std::size_t>(sy) * src_luma_pitch + min(x, src_width - 1)];
    }
  }
  const int cx = min(bx, (src_width + 1) / 2 - 1);
  const int cy = min(by, (src_height + 1) / 2 - 1);
  const std::uint8_t* in = src_chroma + static_cast<std::size_t>(cy) * src_chroma_pitch + 2 * cx;
  std::uint8_t* out = chroma + static_cast<std::size_t>(by) * chroma_pitch + 2 * bx;
  out[0] = in[0];
  out[1] = in[1];
}

__device__ __forceinline__ void store_rgb(std::uint8_t* row, int x, int channels, bool bgr,
                                          float yl, float cb, float cr) {
  const float r = yl + 2.0F * (1.0F - kKr) * cr;
  const float b = yl + 2.0F * (1.0F - kKb) * cb;
  const float g = (yl - kKr * r - kKb * b) / kKg;
  std::uint8_t* p = row + static_cast<std::size_t>(x) * channels;
  p[bgr ? 2 : 0] = to_byte(r);
  p[1] = to_byte(g);
  p[bgr ? 0 : 2] = to_byte(b);
  if (channels == 4) {
    p[3] = 255;
  }
}

// NV12 to packed RGB, the top-left width x height pixels.
__global__ void nv12_to_rgb(const std::uint8_t* luma, std::size_t luma_pitch,
                            const std::uint8_t* chroma, std::size_t chroma_pitch, std::uint8_t* out,
                            std::size_t out_pitch, int width, int height, int channels, bool bgr) {
  const int bx = blockIdx.x * blockDim.x + threadIdx.x;
  const int by = blockIdx.y * blockDim.y + threadIdx.y;
  if (2 * bx >= width || 2 * by >= height) {
    return;
  }
  const std::uint8_t* uv = chroma + static_cast<std::size_t>(by) * chroma_pitch + 2 * bx;
  const float cb = (static_cast<float>(uv[0]) - 128.0F) * (255.0F / 224.0F);
  const float cr = (static_cast<float>(uv[1]) - 128.0F) * (255.0F / 224.0F);
  for (int dy = 0; dy < 2; ++dy) {
    const int y = 2 * by + dy;
    if (y >= height) {
      break;
    }
    std::uint8_t* row = out + static_cast<std::size_t>(y) * out_pitch;
    for (int dx = 0; dx < 2; ++dx) {
      const int x = 2 * bx + dx;
      if (x >= width) {
        break;
      }
      const float yl =
          (static_cast<float>(luma[static_cast<std::size_t>(y) * luma_pitch + x]) - 16.0F) *
          (255.0F / 219.0F);
      store_rgb(row, x, channels, bgr, yl, cb, cr);
    }
  }
}

dim3 blocks_for(int width, int height, dim3 threads) {
  return dim3((static_cast<unsigned>(width / 2) + threads.x - 1) / threads.x,
              (static_cast<unsigned>(height / 2) + threads.y - 1) / threads.y);
}

}  // namespace

cudaError_t launch_to_nv12(const DeviceImage& image, std::uint8_t* luma, std::size_t luma_pitch,
                           std::uint8_t* chroma, std::size_t chroma_pitch, int width, int height,
                           cudaStream_t stream) {
  const dim3 threads(32, 8);
  const dim3 blocks = blocks_for(width, height, threads);
  if (image.format == PixelFormat::kNv12) {
    pad_nv12<<<blocks, threads, 0, stream>>>(image.data, image.pitch, image.chroma,
                                             image.chroma_pitch, image.width, image.height, luma,
                                             luma_pitch, chroma, chroma_pitch, width, height);
  } else {
    const bool bgr = image.format == PixelFormat::kBgr8 || image.format == PixelFormat::kBgra8;
    rgb_to_nv12<<<blocks, threads, 0, stream>>>(image.data, image.pitch, image.width, image.height,
                                                bytes_per_pixel(image.format), bgr, luma,
                                                luma_pitch, chroma, chroma_pitch, width, height);
  }
  return cudaGetLastError();
}

cudaError_t launch_from_nv12(const Nv12Frame& frame, PixelFormat format, std::uint8_t* out,
                             std::size_t out_pitch, int width, int height, cudaStream_t stream) {
  if (format == PixelFormat::kNv12) {
    cudaError_t status = cudaMemcpy2DAsync(out, out_pitch, frame.luma, frame.luma_pitch,
                                           static_cast<std::size_t>(width), height,
                                           cudaMemcpyDeviceToDevice, stream);
    if (status != cudaSuccess) {
      return status;
    }
    return cudaMemcpy2DAsync(out + out_pitch * static_cast<std::size_t>(height), out_pitch,
                             frame.chroma, frame.chroma_pitch,
                             2 * static_cast<std::size_t>((width + 1) / 2), (height + 1) / 2,
                             cudaMemcpyDeviceToDevice, stream);
  }
  const dim3 threads(32, 8);
  // Odd sizes need the block that holds the last row or column.
  const dim3 blocks = blocks_for(width + 1, height + 1, threads);
  const bool bgr = format == PixelFormat::kBgr8 || format == PixelFormat::kBgra8;
  nv12_to_rgb<<<blocks, threads, 0, stream>>>(frame.luma, frame.luma_pitch, frame.chroma,
                                              frame.chroma_pitch, out, out_pitch, width, height,
                                              bytes_per_pixel(format), bgr);
  return cudaGetLastError();
}

}  // namespace holoscan::codec::detail
