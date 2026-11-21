// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <cub/block/block_reduce.cuh>  // NOLINT(build/include_order)
#include <holoscan/mono_depth/colormap.hpp>
#include <holoscan/mono_depth/kernels.hpp>
#include <holoscan/mono_depth/pixel_math.hpp>

namespace holoscan::mono_depth {
namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr int kRangeThreads = 256;
constexpr int kRangeItemsPerThread = 16;

struct Min {
  __device__ std::uint32_t operator()(std::uint32_t a, std::uint32_t b) const { return min(a, b); }
};
struct Max {
  __device__ std::uint32_t operator()(std::uint32_t a, std::uint32_t b) const { return max(a, b); }
};

__constant__ std::uint8_t kTurboDevice[3 * kColormapSize] = {HOLOSCAN_MONO_DEPTH_TURBO_TABLE};

dim3 grid_for(int width, int height) {
  return dim3((static_cast<unsigned>(width) + kBlockX - 1) / kBlockX,
              (static_cast<unsigned>(height) + kBlockY - 1) / kBlockY);
}

// One model input pixel per thread, written to all three planes.
__global__ void preprocess_kernel(PreprocessParams p, const std::uint8_t* image, float* input) {
  const int u = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int v = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (u >= p.input_width || v >= p.input_height) {
    return;
  }
  float rgb[3];
  preprocess_pixel(p, image, u, v, rgb);
  const std::size_t plane =
      static_cast<std::size_t>(p.input_width) * static_cast<std::size_t>(p.input_height);
  const std::size_t index =
      static_cast<std::size_t>(v) * static_cast<std::size_t>(p.input_width) + u;
  input[index] = rgb[0];
  input[plane + index] = rgb[1];
  input[2 * plane + index] = rgb[2];
}

// One image pixel per thread.
__global__ void postprocess_kernel(PostprocessParams p, const float* output, float* inverse,
                                   float* metric) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) {
    return;
  }
  const DepthPair depth = depth_pair(p, sample_output(p, output, x, y));
  const std::size_t index = static_cast<std::size_t>(y) * static_cast<std::size_t>(p.width) + x;
  inverse[index] = depth.inverse;
  if (metric != nullptr) {
    metric[index] = depth.metric;
  }
}

__global__ void range_init_kernel(std::uint32_t* range) {
  range[0] = 0xFFFFFFFFU;
  range[1] = 0U;
}

// Each block reduces its slice, then merges it into `range` atomically.
__global__ void range_kernel(const float* values, std::size_t count, std::uint32_t* range) {
  using Reduce = cub::BlockReduce<std::uint32_t, kRangeThreads>;
  __shared__ typename Reduce::TempStorage low_storage;
  __shared__ typename Reduce::TempStorage high_storage;
  std::uint32_t low = 0xFFFFFFFFU;
  std::uint32_t high = 0U;
  const std::size_t begin =
      static_cast<std::size_t>(blockIdx.x) * kRangeThreads * kRangeItemsPerThread;
  for (int i = 0; i < kRangeItemsPerThread; ++i) {
    const std::size_t index = begin + static_cast<std::size_t>(i) * kRangeThreads + threadIdx.x;
    if (index < count) {
      const float value = values[index];
      if (isfinite(value)) {
        const std::uint32_t bits = ordered_bits(value);
        low = min(low, bits);
        high = max(high, bits);
      }
    }
  }
  const std::uint32_t block_low = Reduce(low_storage).Reduce(low, Min{});
  const std::uint32_t block_high = Reduce(high_storage).Reduce(high, Max{});
  if (threadIdx.x == 0) {
    atomicMin(&range[0], block_low);
    atomicMax(&range[1], block_high);
  }
}

__global__ void colorize_kernel(const float* inverse, int width, int height, ColorRange range,
                                std::uint8_t* rgba) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height) {
    return;
  }
  float low = range.low;
  float high = range.high;
  if (range.range != nullptr) {
    low = from_ordered_bits(range.range[0]);
    high = from_ordered_bits(range.range[1]);
  }
  const std::size_t index = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + x;
  const int entry = colormap_index(inverse[index], low, high);
  std::uint8_t* out = rgba + 4 * index;
  out[0] = kTurboDevice[3 * entry];
  out[1] = kTurboDevice[3 * entry + 1];
  out[2] = kTurboDevice[3 * entry + 2];
  out[3] = 255;
}

}  // namespace

cudaError_t launch_preprocess(const ImageView& image, const ImageFit& fit,
                              const Normalization& normalization, float* input,
                              cudaStream_t stream) {
  if (image.data == nullptr || input == nullptr || image.width != fit.source_width ||
      image.height != fit.source_height ||
      image.pitch < static_cast<std::size_t>(image.width) * channels(image.format)) {
    return cudaErrorInvalidValue;
  }
  const PreprocessParams params =
      make_preprocess_params(fit, image.format, image.pitch, normalization);
  preprocess_kernel<<<grid_for(fit.input_width, fit.input_height), dim3(kBlockX, kBlockY), 0,
                      stream>>>(params, image.data, input);
  return cudaGetLastError();
}

cudaError_t launch_postprocess(const float* output, const ImageFit& fit, ModelOutput model_output,
                               const InverseDepthAffine* affine, const DepthOutputs& outputs,
                               cudaStream_t stream) {
  if (output == nullptr || outputs.inverse == nullptr) {
    return cudaErrorInvalidValue;
  }
  const PostprocessParams params = make_postprocess_params(fit, model_output, affine);
  const bool metric = model_output == ModelOutput::kMetricDepth || affine != nullptr;
  postprocess_kernel<<<grid_for(fit.source_width, fit.source_height), dim3(kBlockX, kBlockY), 0,
                       stream>>>(params, output, outputs.inverse,
                                 metric ? outputs.metric : nullptr);
  return cudaGetLastError();
}

cudaError_t launch_depth_range(const float* depth, std::size_t count, std::uint32_t* range,
                               cudaStream_t stream) {
  if (depth == nullptr || range == nullptr) {
    return cudaErrorInvalidValue;
  }
  range_init_kernel<<<1, 1, 0, stream>>>(range);
  if (count == 0) {
    return cudaGetLastError();
  }
  constexpr std::size_t kPerBlock = static_cast<std::size_t>(kRangeThreads) * kRangeItemsPerThread;
  const auto blocks = static_cast<unsigned>((count + kPerBlock - 1) / kPerBlock);
  range_kernel<<<blocks, kRangeThreads, 0, stream>>>(depth, count, range);
  return cudaGetLastError();
}

cudaError_t launch_colorize(const float* inverse_depth, int width, int height,
                            const ColorRange& range, std::uint8_t* rgba, cudaStream_t stream) {
  if (inverse_depth == nullptr || rgba == nullptr || width < 1 || height < 1) {
    return cudaErrorInvalidValue;
  }
  colorize_kernel<<<grid_for(width, height), dim3(kBlockX, kBlockY), 0, stream>>>(
      inverse_depth, width, height, range, rgba);
  return cudaGetLastError();
}

}  // namespace holoscan::mono_depth
