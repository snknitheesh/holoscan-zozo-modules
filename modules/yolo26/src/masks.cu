// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <holoscan/yolo26/kernels.hpp>

namespace holoscan::yolo26 {
namespace {

constexpr int kLogitThreads = 256;
constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
// Boxes staged in shared memory at a time.
constexpr int kBoxChunk = 256;
constexpr int kMaxGridY = 65535;

// Where an image pixel's center lands in the prototypes, per axis:
// p = (pad + (i + 0.5) * resized_per_source) * proto_per_input - 0.5.
struct Geometry {
  float resized_per_source_x;
  float resized_per_source_y;
  float pad_left;
  float pad_top;
  float proto_per_input_x;
  float proto_per_input_y;
};

// The detections' mask logits at the prototypes' resolution: for each slot
// below the count, the coefficients times the prototypes, summed in channel
// order. Slots are spread over the grid's y dimension.
__global__ void logits_kernel(const float* protos, const float* coefficients,
                              const std::int32_t* count, int capacity, int channels, int plane,
                              float* logits) {
  __shared__ float c[kMaxMaskChannels];
  const int n = min(max(*count, 0), capacity);
  for (int k = static_cast<int>(blockIdx.y); k < n; k += static_cast<int>(gridDim.y)) {
    __syncthreads();  // The previous slot's coefficients are no longer read.
    if (static_cast<int>(threadIdx.x) < channels) {
      c[threadIdx.x] = coefficients[static_cast<std::int64_t>(k) * channels + threadIdx.x];
    }
    __syncthreads();
    const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (p < plane) {
      float sum = 0.0f;
      for (int m = 0; m < channels; ++m) {
        sum += c[m] * protos[static_cast<std::int64_t>(m) * plane + p];
      }
      logits[static_cast<std::int64_t>(k) * plane + p] = sum;
    }
  }
}

// The bilinear sample of one logit map at prototype coordinates (sx, sy),
// edges clamped.
__device__ float sample(const float* map, int width, int x0, int x1, int y0, int y1, float wx,
                        float wy) {
  const float a = map[y0 * width + x0];
  const float b = map[y0 * width + x1];
  const float c = map[y1 * width + x0];
  const float d = map[y1 * width + x1];
  const float top = a * (1.0f - wx) + b * wx;
  const float bottom = c * (1.0f - wx) + d * wx;
  return top * (1.0f - wy) + bottom * wy;
}

// One image pixel per thread: its mask value for every slot, and its label.
__global__ void mask_kernel(MaskInputs in, MaskParams p, MaskOutputs out, Geometry g) {
  __shared__ float boxes[kBoxChunk * 4];
  const int width = p.letterbox.source_width;
  const int height = p.letterbox.source_height;
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const bool in_image = x < width && y < height;
  const int n = min(max(*in.count, 0), p.capacity);

  // This pixel's place in the prototypes, as PyTorch's bilinear interpolate
  // without align_corners finds it.
  const float sx = fmaxf(
      (g.pad_left + (static_cast<float>(x) + 0.5f) * g.resized_per_source_x) * g.proto_per_input_x -
          0.5f,
      0.0f);
  const float sy = fmaxf(
      (g.pad_top + (static_cast<float>(y) + 0.5f) * g.resized_per_source_y) * g.proto_per_input_y -
          0.5f,
      0.0f);
  const int x0 = min(static_cast<int>(sx), p.proto_width - 1);
  const int y0 = min(static_cast<int>(sy), p.proto_height - 1);
  const int x1 = min(x0 + 1, p.proto_width - 1);
  const int y1 = min(y0 + 1, p.proto_height - 1);
  const float wx = sx - static_cast<float>(x0);
  const float wy = sy - static_cast<float>(y0);
  const float fx = static_cast<float>(x);
  const float fy = static_cast<float>(y);
  const std::int64_t plane = static_cast<std::int64_t>(p.proto_width) * p.proto_height;
  const std::int64_t pixels = static_cast<std::int64_t>(width) * height;
  const std::int64_t pixel = static_cast<std::int64_t>(y) * width + x;
  const int threads = static_cast<int>(blockDim.x * blockDim.y);
  const int thread = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);

  std::int32_t label = 0;
  for (int base = 0; base < n; base += kBoxChunk) {
    const int chunk = min(kBoxChunk, n - base);
    __syncthreads();  // The previous chunk is no longer read.
    for (int i = thread; i < chunk * 4; i += threads) {
      boxes[i] = in.boxes[static_cast<std::int64_t>(base) * 4 + i];
    }
    __syncthreads();
    if (!in_image) {
      continue;
    }
    for (int j = 0; j < chunk; ++j) {
      const int k = base + j;
      const float* box = boxes + 4 * j;
      bool on = false;
      if (fx >= box[0] && fx < box[2] && fy >= box[1] && fy < box[3]) {
        on = sample(out.logits + k * plane, p.proto_width, x0, x1, y0, y1, wx, wy) >
             p.logit_threshold;
      }
      if (out.masks != nullptr) {
        out.masks[k * pixels + pixel] = on ? 1U : 0U;
      }
      if (on && label == 0) {
        label = k + 1;
      }
    }
  }
  if (!in_image) {
    return;
  }
  if (out.masks != nullptr) {
    for (int k = n; k < p.capacity; ++k) {
      out.masks[k * pixels + pixel] = 0U;
    }
  }
  if (out.labels != nullptr) {
    out.labels[pixel] = label;
  }
}

}  // namespace

cudaError_t launch_masks(const MaskInputs& inputs, const MaskParams& params,
                         const MaskOutputs& outputs, cudaStream_t stream) {
  const Letterbox& lb = params.letterbox;
  if (inputs.protos == nullptr || inputs.coefficients == nullptr || inputs.boxes == nullptr ||
      inputs.count == nullptr || outputs.logits == nullptr || params.capacity < 1 ||
      params.capacity > kMaxRows || params.mask_channels < 1 ||
      params.mask_channels > kMaxMaskChannels || params.proto_width < 1 ||
      params.proto_height < 1 || params.proto_width > kMaxProtoSide ||
      params.proto_height > kMaxProtoSide || std::isnan(params.logit_threshold) ||
      lb.source_width < 1 || lb.source_height < 1 || lb.source_width > kMaxImageSide ||
      lb.source_height > kMaxImageSide || lb.size < 1 || lb.resized_width < 1 ||
      lb.resized_height < 1) {
    return cudaErrorInvalidValue;
  }
  const int plane = params.proto_width * params.proto_height;
  const dim3 logit_grid(static_cast<unsigned>((plane + kLogitThreads - 1) / kLogitThreads),
                        static_cast<unsigned>(std::min(params.capacity, kMaxGridY)));
  logits_kernel<<<logit_grid, kLogitThreads, 0, stream>>>(
      inputs.protos, inputs.coefficients, inputs.count, params.capacity, params.mask_channels,
      plane, outputs.logits);
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return status;
  }
  if (outputs.masks == nullptr && outputs.labels == nullptr) {
    return cudaSuccess;
  }
  const Geometry geometry{
      .resized_per_source_x =
          static_cast<float>(lb.resized_width) / static_cast<float>(lb.source_width),
      .resized_per_source_y =
          static_cast<float>(lb.resized_height) / static_cast<float>(lb.source_height),
      .pad_left = static_cast<float>(lb.pad_left),
      .pad_top = static_cast<float>(lb.pad_top),
      .proto_per_input_x = static_cast<float>(params.proto_width) / static_cast<float>(lb.size),
      .proto_per_input_y = static_cast<float>(params.proto_height) / static_cast<float>(lb.size)};
  const dim3 block(kBlockX, kBlockY);
  const dim3 grid(static_cast<unsigned>((lb.source_width + kBlockX - 1) / kBlockX),
                  static_cast<unsigned>((lb.source_height + kBlockY - 1) / kBlockY));
  mask_kernel<<<grid, block, 0, stream>>>(inputs, params, outputs, geometry);
  return cudaGetLastError();
}

}  // namespace holoscan::yolo26
