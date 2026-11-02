// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <holoscan/foundation_stereo/kernels.hpp>
#include <holoscan/foundation_stereo/stereo_geometry.hpp>

namespace holoscan::foundation_stereo {
namespace {

// Where output index `index` of `out` samples an axis of `in` pixels: the two
// neighbors and the weight of the second.
struct Tap {
  int first;
  int second;
  float weight;
};

__device__ Tap tap(int index, int in, int out) {
  const float scale = static_cast<float>(in) / static_cast<float>(out);
  float x = (static_cast<float>(index) + 0.5F) * scale - 0.5F;
  x = fminf(fmaxf(x, 0.0F), static_cast<float>(in - 1));
  const int first = static_cast<int>(x);
  const int second = first + 1 < in ? first + 1 : first;
  return Tap{first, second, x - static_cast<float>(first)};
}

__device__ float lerp(float a, float b, float t) { return a + (b - a) * t; }

__global__ void preprocess_kernel(ImageView image, int model_width, int model_height, float* out) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= model_width || y >= model_height) {
    return;
  }
  const Tap tx = tap(x, image.width, model_width);
  const Tap ty = tap(y, image.height, model_height);
  const int n = channels(image.format);
  const std::uint8_t* row0 = image.data + static_cast<std::size_t>(ty.first) * image.pitch;
  const std::uint8_t* row1 = image.data + static_cast<std::size_t>(ty.second) * image.pitch;
  const bool bgr = image.format == PixelFormat::kBgr8 || image.format == PixelFormat::kBgra8;
  const std::size_t plane = static_cast<std::size_t>(model_width) * model_height;
  const std::size_t at = static_cast<std::size_t>(y) * model_width + x;
  for (int c = 0; c < 3; ++c) {
    // The input byte of RGB channel c.
    const int source = n == 1 ? 0 : bgr ? 2 - c : c;
    const auto value = [&](const std::uint8_t* row, int column) {
      return static_cast<float>(row[column * n + source]);
    };
    const float top = lerp(value(row0, tx.first), value(row0, tx.second), tx.weight);
    const float bottom = lerp(value(row1, tx.first), value(row1, tx.second), tx.weight);
    out[c * plane + at] = (lerp(top, bottom, ty.weight) - image_mean(c)) / image_std(c);
  }
}

__global__ void postprocess_kernel(const float* model, PostprocessParams params, float* disparity,
                                   float* depth) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= params.width || y >= params.height) {
    return;
  }
  const Tap tx = tap(x, params.model_width, params.width);
  const Tap ty = tap(y, params.model_height, params.height);
  const float* row0 = model + static_cast<std::size_t>(ty.first) * params.model_width;
  const float* row1 = model + static_cast<std::size_t>(ty.second) * params.model_width;
  const float top = lerp(row0[tx.first], row0[tx.second], tx.weight);
  const float bottom = lerp(row1[tx.first], row1[tx.second], tx.weight);
  const float scale = static_cast<float>(params.width) / static_cast<float>(params.model_width);
  float d = lerp(top, bottom, ty.weight) * scale;
  // Not positive, not a number, or matched outside the right image: invalid.
  if (!(d > 0.0F) || (params.invalidate_out_of_view && static_cast<float>(x) - d < 0.0F)) {
    d = 0.0F;
  }
  const std::size_t at = static_cast<std::size_t>(y) * params.width + x;
  disparity[at] = d;
  if (depth != nullptr) {
    depth[at] = depth_from_disparity(d, params.fx_baseline, params.doffs);
  }
}

dim3 grid_for(int width, int height, dim3 block) {
  return dim3((static_cast<unsigned>(width) + block.x - 1) / block.x,
              (static_cast<unsigned>(height) + block.y - 1) / block.y);
}

}  // namespace

cudaError_t launch_preprocess(const ImageView& image, int model_width, int model_height, float* out,
                              cudaStream_t stream) {
  if (image.data == nullptr || out == nullptr || image.width < 1 || image.height < 1 ||
      model_width < 1 || model_height < 1 ||
      image.pitch < static_cast<std::size_t>(image.width) * channels(image.format)) {
    return cudaErrorInvalidValue;
  }
  const dim3 block(32, 8);
  preprocess_kernel<<<grid_for(model_width, model_height, block), block, 0, stream>>>(
      image, model_width, model_height, out);
  return cudaGetLastError();
}

cudaError_t launch_postprocess(const float* model_disparity, const PostprocessParams& params,
                               float* disparity, float* depth, cudaStream_t stream) {
  if (model_disparity == nullptr || disparity == nullptr || params.width < 1 || params.height < 1 ||
      params.model_width < 1 || params.model_height < 1) {
    return cudaErrorInvalidValue;
  }
  const dim3 block(32, 8);
  postprocess_kernel<<<grid_for(params.width, params.height, block), block, 0, stream>>>(
      model_disparity, params, disparity, depth);
  return cudaGetLastError();
}

}  // namespace holoscan::foundation_stereo
