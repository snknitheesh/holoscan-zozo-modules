// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <stdexcept>
#include <string>

#include "holoscan/stereo_disparity/synthetic_scene.hpp"

namespace holoscan::stereo_disparity {
namespace {

// The camera: a focal length of 0.8 image widths, and a 10 cm baseline.
constexpr float kFocalPerWidth = 0.8F;
constexpr float kBaseline = 0.10F;
// The nearest plane's depth, which sets the largest disparity.
constexpr float kNearest = 1.5F;

__global__ void render_kernel(SyntheticScene scene, int channels, std::uint8_t* left,
                              std::uint8_t* right) {
  const int u = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int v = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (u >= scene.width || v >= scene.height) {
    return;
  }
  const std::size_t offset = (static_cast<std::size_t>(v) * scene.width + u) * channels;
  std::uint8_t rgb[3];
  scene_color(scene, 0.0F, static_cast<float>(u), static_cast<float>(v), rgb);
  for (int c = 0; c < channels; ++c) left[offset + c] = rgb[channels == 1 ? 1 : c];
  scene_color(scene, scene.baseline, static_cast<float>(u), static_cast<float>(v), rgb);
  for (int c = 0; c < channels; ++c) right[offset + c] = rgb[channels == 1 ? 1 : c];
}

}  // namespace

SyntheticScene make_scene(std::uint32_t frame, int width, int height) {
  if (width < kMinSceneWidth || height < kMinSceneHeight || width > kMaxSceneSide ||
      height > kMaxSceneSide) {
    throw std::invalid_argument("synthetic images must be " + std::to_string(kMinSceneWidth) + "x" +
                                std::to_string(kMinSceneHeight) + " to " +
                                std::to_string(kMaxSceneSide) + "x" +
                                std::to_string(kMaxSceneSide));
  }
  SyntheticScene scene;
  scene.width = width;
  scene.height = height;
  scene.focal = kFocalPerWidth * static_cast<float>(width);
  scene.cx = 0.5F * static_cast<float>(width - 1);
  scene.cy = 0.5F * static_cast<float>(height - 1);
  scene.baseline = kBaseline;
  // The rectangles drift sideways, in opposite directions.
  const float shift = 0.15F * std::sin(0.07F * static_cast<float>(frame));
  scene.planes[0] = {.z = kNearest,
                     .x0 = -0.55F + shift,
                     .x1 = -0.05F + shift,
                     .y0 = -0.35F,
                     .y1 = 0.15F,
                     .seed = 11U,
                     .tint = {1.0F, 0.85F, 0.7F}};
  scene.planes[1] = {.z = 2.5F,
                     .x0 = 0.1F - shift,
                     .x1 = 0.9F - shift,
                     .y0 = -0.2F,
                     .y1 = 0.6F,
                     .seed = 23U,
                     .tint = {0.7F, 1.0F, 0.8F}};
  scene.planes[2] = {.z = 4.0F,
                     .x0 = -1.6F,
                     .x1 = 0.4F,
                     .y0 = -1.1F,
                     .y1 = -0.3F,
                     .seed = 37U,
                     .tint = {0.75F, 0.8F, 1.0F}};
  scene.planes[3] = {.z = 8.0F, .background = true, .seed = 51U, .tint = {0.9F, 0.9F, 0.9F}};
  return scene;
}

float max_scene_disparity(int width) {
  return kFocalPerWidth * static_cast<float>(width) * kBaseline / kNearest;
}

StereoGeometry scene_geometry(const SyntheticScene& scene) {
  return StereoGeometry{.fx = scene.focal,
                        .baseline = scene.baseline,
                        .width = scene.width,
                        .height = scene.height,
                        .cx = scene.cx,
                        .cy = scene.cy};
}

cudaError_t render_scene(const SyntheticScene& scene, int channels, std::uint8_t* left,
                         std::uint8_t* right, cudaStream_t stream) {
  if (channels != 1 && channels != 3) {
    return cudaErrorInvalidValue;
  }
  const dim3 block(32, 8);
  const dim3 grid((scene.width + block.x - 1) / block.x, (scene.height + block.y - 1) / block.y);
  render_kernel<<<grid, block, 0, stream>>>(scene, channels, left, right);
  return cudaGetLastError();
}

}  // namespace holoscan::stereo_disparity
