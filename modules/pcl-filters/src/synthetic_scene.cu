// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The synthetic scene on the GPU, one thread per pixel, and on the host.
// Compiled without fused multiply-adds, like the host code that checks it.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <holoscan/pcl_filters/synthetic_scene.hpp>

namespace holoscan::pcl_filters {
namespace {

__global__ void render_kernel(SyntheticScene scene, float* __restrict__ depth,
                              std::uint8_t* __restrict__ rgb, float* __restrict__ xyz) {
  const int u = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int v = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (u >= scene.width || v >= scene.height) {
    return;
  }
  const ScenePixel pixel = synthetic_scene_pixel(scene, u, v);
  const std::size_t i = static_cast<std::size_t>(v) * scene.width + u;
  if (depth != nullptr) {
    depth[i] = pixel.depth;
  }
  if (rgb != nullptr) {
    for (int channel = 0; channel < 3; ++channel) {
      rgb[3 * i + channel] = pixel.rgb[channel];
    }
  }
  if (xyz != nullptr) {
    if (pixel.depth > 0.0F) {
      synthetic_scene_point(scene, u, v, pixel.depth, xyz + 3 * i);
    } else {
      const float nan = __int_as_float(0x7FC00000);
      xyz[3 * i] = nan;
      xyz[3 * i + 1] = nan;
      xyz[3 * i + 2] = nan;
    }
  }
}

}  // namespace

void synthetic_scene_cloud(const SyntheticScene& scene, std::vector<float>& xyz,
                           std::vector<std::uint8_t>& rgb, std::vector<SceneLabel>& labels) {
  const std::size_t pixels = static_cast<std::size_t>(scene.width) * scene.height;
  xyz.assign(3 * pixels, std::numeric_limits<float>::quiet_NaN());
  rgb.assign(3 * pixels, 0U);
  labels.assign(pixels, SceneLabel::kNone);
  for (int v = 0; v < scene.height; ++v) {
    for (int u = 0; u < scene.width; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * scene.width + u;
      const ScenePixel pixel = synthetic_scene_pixel(scene, u, v);
      labels[i] = pixel.label;
      for (int channel = 0; channel < 3; ++channel) {
        rgb[3 * i + channel] = pixel.rgb[channel];
      }
      if (pixel.depth > 0.0F) {
        synthetic_scene_point(scene, u, v, pixel.depth, xyz.data() + 3 * i);
      }
    }
  }
}

cudaError_t render_synthetic_scene(const SyntheticScene& scene, float* depth, std::uint8_t* rgb,
                                   float* xyz, cudaStream_t stream) {
  if (scene.width < 1 || scene.height < 1) {
    return cudaErrorInvalidValue;
  }
  const dim3 block(32, 8);
  const dim3 grid((scene.width + 31) / 32, (scene.height + 7) / 8);
  render_kernel<<<grid, block, 0, stream>>>(scene, depth, rgb, xyz);
  return cudaGetLastError();
}

}  // namespace holoscan::pcl_filters
