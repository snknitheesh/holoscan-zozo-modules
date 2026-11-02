// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include <holoscan/normals/planes.hpp>
#include <holoscan/normals/views.hpp>

namespace holoscan::normals {
namespace {

// Twelve distinct colors, for plane labels 1, 2, ...
__host__ __device__ inline void palette(std::uint8_t label, std::uint8_t rgb[3]) {
  constexpr std::uint8_t kColors[12][3] = {
      {230U, 25U, 75U},  {60U, 180U, 75U},   {255U, 225U, 25U}, {0U, 130U, 200U},
      {245U, 130U, 48U}, {145U, 30U, 180U},  {70U, 240U, 240U}, {240U, 50U, 230U},
      {210U, 245U, 60U}, {250U, 190U, 212U}, {0U, 128U, 128U},  {170U, 110U, 40U},
  };
  if (label == kLabelInvalid) {
    rgb[0] = rgb[1] = rgb[2] = 0U;
  } else if (label == kLabelNoPlane) {
    rgb[0] = rgb[1] = rgb[2] = 60U;
  } else {
    const int k = (label - 1) % 12;
    rgb[0] = kColors[k][0];
    rgb[1] = kColors[k][1];
    rgb[2] = kColors[k][2];
  }
}

__device__ inline std::uint8_t channel(float value) {
  const float scaled = (value + 1.0F) * 127.5F;
  return static_cast<std::uint8_t>(fminf(fmaxf(scaled + 0.5F, 0.0F), 255.0F));
}

__global__ void normals_kernel(const float* __restrict__ normals, int width, int height,
                               std::uint8_t* __restrict__ rgb, std::size_t row_stride) {
  const int u = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int v = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (u >= width || v >= height) {
    return;
  }
  const std::size_t i = static_cast<std::size_t>(v) * width + u;
  std::uint8_t* out = rgb + static_cast<std::size_t>(v) * row_stride + 3U * u;
  const float x = normals[3 * i];
  const float y = normals[3 * i + 1];
  const float z = normals[3 * i + 2];
  const bool valid = isfinite(x) && isfinite(y) && isfinite(z);
  out[0] = valid ? channel(x) : 0U;
  out[1] = valid ? channel(y) : 0U;
  out[2] = valid ? channel(z) : 0U;
}

__global__ void labels_kernel(const std::uint8_t* __restrict__ labels, int width, int height,
                              std::uint8_t* __restrict__ rgb, std::size_t row_stride) {
  const int u = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int v = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (u >= width || v >= height) {
    return;
  }
  palette(labels[static_cast<std::size_t>(v) * width + u],
          rgb + static_cast<std::size_t>(v) * row_stride + 3U * u);
}

bool usable(const void* in, int width, int height, const std::uint8_t* rgb,
            std::size_t row_stride) {
  return width >= 0 && height >= 0 && row_stride >= 3U * static_cast<std::size_t>(width) &&
         (width == 0 || height == 0 || (in != nullptr && rgb != nullptr));
}

}  // namespace

cudaError_t colorize_normals(const float* normals, int width, int height, std::uint8_t* rgb,
                             std::size_t row_stride, cudaStream_t stream) {
  if (!usable(normals, width, height, rgb, row_stride)) {
    return cudaErrorInvalidValue;
  }
  if (width == 0 || height == 0) {
    return cudaSuccess;
  }
  const dim3 block(32, 8);
  const dim3 grid((static_cast<unsigned int>(width) + 31U) / 32U,
                  (static_cast<unsigned int>(height) + 7U) / 8U);
  normals_kernel<<<grid, block, 0, stream>>>(normals, width, height, rgb, row_stride);
  return cudaGetLastError();
}

cudaError_t colorize_labels(const std::uint8_t* labels, int width, int height, std::uint8_t* rgb,
                            std::size_t row_stride, cudaStream_t stream) {
  if (!usable(labels, width, height, rgb, row_stride)) {
    return cudaErrorInvalidValue;
  }
  if (width == 0 || height == 0) {
    return cudaSuccess;
  }
  const dim3 block(32, 8);
  const dim3 grid((static_cast<unsigned int>(width) + 31U) / 32U,
                  (static_cast<unsigned int>(height) + 7U) / 8U);
  labels_kernel<<<grid, block, 0, stream>>>(labels, width, height, rgb, row_stride);
  return cudaGetLastError();
}

void label_color(std::uint8_t label, std::uint8_t rgb[3]) { palette(label, rgb); }

}  // namespace holoscan::normals
