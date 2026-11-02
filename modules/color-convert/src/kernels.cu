// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The conversion kernel: one thread per destination block (a pixel, or the
// pixels sharing a chroma sample), computing kernel_math.hpp's arithmetic.
// Compiled with --fmad=false, as the reference is without contraction, so the
// two agree to the byte.

#include <algorithm>
#include <cstdint>

#include <holoscan/color_convert/converter.hpp>
#include <holoscan/color_convert/reference.hpp>

#include "kernels.hpp"

namespace holoscan::color_convert {
namespace detail {
namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;

// Whether every pixel of block (bx, by) lies inside the content rectangle.
__device__ bool inside_content(const Plan& p, int bx, int by) {
  const int x0 = bx << p.block_shift_x;
  const int y0 = by << p.block_shift_y;
  const int x1 = min(x0 + (1 << p.block_shift_x), p.dst.width);
  const int y1 = min(y0 + (1 << p.block_shift_y), p.dst.height);
  return x0 >= p.content_x0 && x1 <= p.content_x1 && y0 >= p.content_y0 && y1 <= p.content_y1;
}

__global__ void convert_kernel(const Plan plan, int across, int down, bool border_only) {
  const int bx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int by = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (bx >= across || by >= down) {
    return;
  }
  if (border_only && inside_content(plan, bx, by)) {
    return;
  }
  convert_block(plan, bx, by);
}

__global__ void pattern_kernel(std::uint8_t* data, std::size_t pitch, int width, int height,
                               std::uint32_t frame) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height) {
    return;
  }
  const auto rgb = test_pattern_rgb(x, y, width, height, frame);
  std::uint8_t* p = data + static_cast<std::size_t>(y) * pitch + 3 * static_cast<std::size_t>(x);
  p[0] = rgb[0];
  p[1] = rgb[1];
  p[2] = rgb[2];
}

dim3 grid_for(int across, int down) {
  return dim3(static_cast<unsigned>((across + kBlockX - 1) / kBlockX),
              static_cast<unsigned>((down + kBlockY - 1) / kBlockY));
}

}  // namespace

cudaError_t launch_convert(const Plan& plan, bool border_only, cudaStream_t stream) {
  const int across = blocks_x(plan);
  const int down = blocks_y(plan);
  convert_kernel<<<grid_for(across, down), dim3(kBlockX, kBlockY), 0, stream>>>(plan, across, down,
                                                                                border_only);
  return cudaGetLastError();
}

}  // namespace detail

cudaError_t launch_test_pattern(const ImageView& rgb8, std::uint32_t frame, cudaStream_t stream) {
  if (rgb8.format != PixelFormat::kRgb8 || rgb8.location != MemoryLocation::kDevice ||
      rgb8.width <= 0 || rgb8.height <= 0 || rgb8.planes[0].data == nullptr ||
      rgb8.planes[0].pitch < static_cast<std::size_t>(rgb8.width) * 3U) {
    return cudaErrorInvalidValue;
  }
  detail::pattern_kernel<<<detail::grid_for(rgb8.width, rgb8.height),
                           dim3(detail::kBlockX, detail::kBlockY), 0, stream>>>(
      rgb8.planes[0].data, rgb8.planes[0].pitch, rgb8.width, rgb8.height, frame);
  return cudaGetLastError();
}

}  // namespace holoscan::color_convert
