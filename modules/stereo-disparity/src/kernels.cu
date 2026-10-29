// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/stereo_disparity/stereo_geometry.hpp"
#include "kernels.hpp"

namespace holoscan::stereo_disparity::detail {
namespace {

struct ChannelOrder {
  int stride;
  int red;
  int blue;
};

__host__ __device__ inline ChannelOrder order_of(PixelFormat format) {
  switch (format) {
    case PixelFormat::kRgb8:
      return {3, 0, 2};
    case PixelFormat::kBgr8:
      return {3, 2, 0};
    case PixelFormat::kRgba8:
      return {4, 0, 2};
    case PixelFormat::kBgra8:
      return {4, 2, 0};
    default:
      return {1, 0, 0};
  }
}

dim3 grid_for(int width, int height, dim3 block) {
  return dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
}

__global__ void gray_kernel(const std::uint8_t* input, std::size_t pitch, ChannelOrder order,
                            int width, int height, std::uint8_t* out, std::size_t out_pitch) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height) {
    return;
  }
  const std::uint8_t* pixel = input + y * pitch + static_cast<std::size_t>(x) * order.stride;
  out[y * out_pitch + x] =
      order.stride == 1
          ? pixel[0]
          : static_cast<std::uint8_t>(
                (77U * pixel[order.red] + 150U * pixel[1] + 29U * pixel[order.blue] + 128U) >> 8U);
}

// What VPI's CUDA backend writes where it found no disparity; the OFA and PVA
// write 0.
constexpr std::int16_t kVpiInvalid = 0x7fff;

__global__ void finish_kernel(const std::int16_t* raw, std::size_t raw_pitch,
                              const std::uint16_t* raw_confidence, std::size_t confidence_pitch,
                              int width, int height, float max_disparity,
                              DisparityOutputs outputs) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height) {
    return;
  }
  const std::int16_t value =
      reinterpret_cast<const std::int16_t*>(reinterpret_cast<const char*>(raw) + y * raw_pitch)[x];
  float disparity = static_cast<float>(value) / 32.0F;
  if (value == kVpiInvalid || !(disparity > 0.0F && disparity <= max_disparity)) {
    disparity = 0.0F;
  }
  const std::size_t i = static_cast<std::size_t>(y) * width + x;
  outputs.disparity[i] = disparity;
  if (outputs.depth != nullptr) {
    outputs.depth[i] = depth_from_disparity(disparity, outputs.fx_baseline, outputs.doffs);
  }
  if (outputs.confidence != nullptr) {
    outputs.confidence[i] = reinterpret_cast<const std::uint16_t*>(
        reinterpret_cast<const char*>(raw_confidence) + y * confidence_pitch)[x];
  }
}

}  // namespace

int channels(PixelFormat format) noexcept { return order_of(format).stride; }

cudaError_t to_gray(const std::uint8_t* input, std::size_t pitch, PixelFormat format, int width,
                    int height, std::uint8_t* out, std::size_t out_pitch, cudaStream_t stream) {
  const dim3 block(32, 8);
  gray_kernel<<<grid_for(width, height, block), block, 0, stream>>>(input, pitch, order_of(format),
                                                                    width, height, out, out_pitch);
  return cudaGetLastError();
}

cudaError_t finish_disparity(const std::int16_t* raw, std::size_t raw_pitch,
                             const std::uint16_t* raw_confidence, std::size_t confidence_pitch,
                             int width, int height, float max_disparity,
                             const DisparityOutputs& outputs, cudaStream_t stream) {
  const dim3 block(32, 8);
  finish_kernel<<<grid_for(width, height, block), block, 0, stream>>>(
      raw, raw_pitch, raw_confidence, confidence_pitch, width, height, max_disparity, outputs);
  return cudaGetLastError();
}

}  // namespace holoscan::stereo_disparity::detail
