// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "cuda_buffers.hpp"  // cuda_check
#include "gray_converter.hpp"

namespace holoscan::fiducials {
namespace {

// BT.601 luma in 8-bit fixed point.
__host__ __device__ inline std::uint8_t luma(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
  return static_cast<std::uint8_t>((77U * r + 150U * g + 29U * b + 128U) >> 8U);
}

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

__global__ void to_gray_kernel(const std::uint8_t* input, std::size_t in_pitch, PixelFormat format,
                               int width, int height, std::uint8_t* output) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const ChannelOrder order = order_of(format);
  const std::uint8_t* pixel = input + y * in_pitch + x * order.stride;
  output[y * width + x] = luma(pixel[order.red], pixel[1], pixel[order.blue]);
}

void to_gray_host(const ImageView& input, std::uint8_t* output) {
  const ChannelOrder order = order_of(input.format);
  for (int y = 0; y < input.height; ++y) {
    const std::uint8_t* row = input.data + static_cast<std::size_t>(y) * input.pitch;
    std::uint8_t* out = output + static_cast<std::size_t>(y) * input.width;
    for (int x = 0; x < input.width; ++x) {
      const std::uint8_t* pixel = row + x * order.stride;
      out[x] = luma(pixel[order.red], pixel[1], pixel[order.blue]);
    }
  }
}

}  // namespace

int channels(PixelFormat format) noexcept { return order_of(format).stride; }

void GrayConverter::reserve(const ImageView& input, MemoryLocation target) {
  if (input.width != width_ || input.height != height_) {
    plain_gray_ = {};
    pinned_gray_.reset();
    device_gray_.reset();
    width_ = input.width;
    height_ = input.height;
  }
  const std::size_t bytes = static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);
  const bool on_host = input.location == MemoryLocation::kHost;
  const bool convert = input.format != PixelFormat::kMono8;
  if (on_host && target == MemoryLocation::kHost) {
    plain_gray_.resize(bytes);
    return;
  }
  // Pinned: a host image converted before it goes to the device, or a gray image back from it.
  const bool pinned = (on_host && convert) || target == MemoryLocation::kHost;
  // Device: the backend's input, or a device image converted before it comes back.
  const bool device = target == MemoryLocation::kDevice || convert;
  if (pinned && !pinned_gray_) {
    void* host = nullptr;
    cuda_check(cudaMallocHost(&host, bytes), "cudaMallocHost(gray)");
    pinned_gray_.reset(static_cast<std::uint8_t*>(host));
  }
  if (device && !device_gray_) {
    void* memory = nullptr;
    cuda_check(cudaMalloc(&memory, bytes), "cudaMalloc(gray)");
    device_gray_.reset(static_cast<std::uint8_t*>(memory));
  }
}

GrayImageView GrayConverter::convert(const ImageView& input, MemoryLocation target,
                                     cudaStream_t stream) {
  if (input.format == PixelFormat::kMono8 && input.location == target) {
    return {input.data, input.width, input.height, input.pitch, target};
  }
  reserve(input, target);
  const auto width = static_cast<std::size_t>(input.width);
  const auto height = static_cast<std::size_t>(input.height);
  const bool on_host = input.location == MemoryLocation::kHost;
  if (on_host && target == MemoryLocation::kHost) {
    to_gray_host(input, plain_gray_.data());
    return {plain_gray_.data(), input.width, input.height, width, target};
  }
  // Convert where the image is: a third (or a quarter) of the bytes then moves.
  const std::uint8_t* gray = input.data;
  std::size_t gray_pitch = input.pitch;
  if (input.format != PixelFormat::kMono8) {
    std::uint8_t* out = on_host ? pinned_gray_.get() : device_gray_.get();
    if (on_host) {
      to_gray_host(input, out);
    } else {
      const dim3 block(32, 8);
      const dim3 grid((input.width + block.x - 1) / block.x,
                      (input.height + block.y - 1) / block.y);
      to_gray_kernel<<<grid, block, 0, stream>>>(input.data, input.pitch, input.format, input.width,
                                                 input.height, out);
      cuda_check(cudaGetLastError(), "to_gray_kernel");
    }
    gray = out;
    gray_pitch = width;
  }
  if (input.location == target) {  // On the device, converted there.
    return {gray, input.width, input.height, gray_pitch, target};
  }
  std::uint8_t* result = target == MemoryLocation::kHost ? pinned_gray_.get() : device_gray_.get();
  cuda_check(cudaMemcpy2DAsync(result, width, gray, gray_pitch, width, height,
                               on_host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, stream),
             "cudaMemcpy2DAsync(gray)");
  transfers_.copied_bytes += width * height;
  if (target == MemoryLocation::kHost) {
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    ++transfers_.host_waits;
  }
  return {result, input.width, input.height, width, target};
}

}  // namespace holoscan::fiducials
