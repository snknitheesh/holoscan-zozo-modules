// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "holoscan/denoise/synthetic_scene.hpp"

namespace holoscan::denoise {
namespace {

// ---- The low-light scene: integer arithmetic, the same on host and device ----

// The colours of the patches, R, G, B.
__host__ __device__ inline void patch_color(int index, int (&rgb)[3]) {
  constexpr int kColors[12][3] = {{90, 40, 30},  {40, 80, 35}, {30, 45, 95}, {85, 80, 30},
                                  {80, 35, 80},  {30, 80, 85}, {60, 60, 60}, {105, 70, 45},
                                  {45, 60, 105}, {70, 95, 60}, {25, 25, 25}, {110, 105, 100}};
  for (int c = 0; c < 3; ++c) rgb[c] = kColors[index % 12][c];
}

// Channel c (R, G, B) of the clean scene at (x, y), 0..255.
__host__ __device__ inline int clean_rgb(int width, int height, bool second, Point object,
                                         int radius, int x, int y, int c) {
  constexpr int kTint[3] = {0, 3, 8};
  const int ramp = (24 * x) / width + (12 * y) / height;
  int value = second ? 58 - ramp + kTint[2 - c] : 18 + ramp + kTint[c];
  // Patches: a 4 x 3 grid in the upper left.
  const int pw = width / 10;
  const int ph = height / 8;
  const int px = x - width / 20;
  const int py = y - height / 16;
  if (pw > 0 && ph > 0 && px >= 0 && py >= 0 && px < 4 * pw && py < 3 * ph &&
      px % pw < pw - pw / 6 && py % ph < ph - ph / 6) {
    int rgb[3];
    patch_color((px / pw) + 4 * (py / ph) + (second ? 5 : 0), rgb);
    value = rgb[c];
  }
  // Stripes in the lower left (vertical, 3 pixels wide; horizontal after a cut),
  // a checkerboard of 2-pixel squares in the lower middle.
  if (y > (5 * height) / 8 && y < (7 * height) / 8) {
    if (x > width / 20 && x < (3 * width) / 10) {
      value += (((second ? y : x) / 3) & 1) != 0 ? 22 : 0;
    } else if (x > (7 * width) / 20 && x < (6 * width) / 10) {
      value += ((x / 2 + y / 2) & 1) != 0 ? 16 : 0;
    }
  }
  if (radius > 0 && object.x >= 0) {
    const int dx = x - object.x;
    const int dy = y - object.y;
    if (dx * dx + dy * dy <= radius * radius) {
      constexpr int kObject[3] = {130, 100, 70};
      constexpr int kObjectAfterCut[3] = {60, 120, 140};
      value = second ? kObjectAfterCut[c] : kObject[c];
    }
  }
  return value < 0 ? 0 : value > 255 ? 255 : value;
}

struct SceneArgs {
  int width;
  int height;
  bool second;
  Point object;
  int radius;
  PixelFormat format;
};

__host__ __device__ inline int clean_sample(const SceneArgs& s, int x, int y, int c) {
  if (s.format == PixelFormat::kMono8) {
    const int r = clean_rgb(s.width, s.height, s.second, s.object, s.radius, x, y, 0);
    const int g = clean_rgb(s.width, s.height, s.second, s.object, s.radius, x, y, 1);
    const int b = clean_rgb(s.width, s.height, s.second, s.object, s.radius, x, y, 2);
    return (r + 2 * g + b) / 4;
  }
  if (c == 3) return 255;
  return clean_rgb(s.width, s.height, s.second, s.object, s.radius, x, y, c);
}

SceneArgs scene_args(const LowLightScene& scene, int frame) {
  return SceneArgs{.width = scene.width,
                   .height = scene.height,
                   .second = scene.cut_frame >= 0 && frame >= scene.cut_frame,
                   .object = object_center(scene, frame),
                   .radius = scene.object_radius,
                   .format = scene.format};
}

// ---- Noise: a counter-based generator, so every sample is reproducible ----

__device__ inline std::uint32_t mix(std::uint32_t h) {
  h ^= h >> 16;
  h *= 0x7feb352dU;
  h ^= h >> 15;
  h *= 0x846ca68bU;
  h ^= h >> 16;
  return h;
}

struct Rng {
  std::uint32_t state;
  __device__ float uniform() {  // (0, 1]
    state = mix(state + 0x9e3779b9U);
    return (static_cast<float>(state >> 8) + 1.0F) * (1.0F / 16777216.0F);
  }
  __device__ float normal() {
    const float u1 = uniform();
    const float u2 = uniform();
    return sqrtf(-2.0F * logf(u1)) * cospif(2.0F * u2);
  }
  __device__ float poisson(float lambda) {
    if (lambda <= 0.0F) return 0.0F;
    if (lambda >= 30.0F) {
      return fmaxf(0.0F, rintf(lambda + sqrtf(lambda) * normal()));
    }
    const float limit = expf(-lambda);
    float product = uniform();
    int k = 0;
    while (product > limit) {
      ++k;
      product *= uniform();
    }
    return static_cast<float>(k);
  }
};

__global__ void noisy_kernel(SceneArgs s, NoiseModel noise, int frame, std::uint8_t* out,
                             std::size_t pitch) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= s.width || y >= s.height) return;
  const int bytes = pixel_bytes(s.format);
  std::uint8_t* o = out + static_cast<std::size_t>(y) * pitch + static_cast<std::size_t>(x) * bytes;
  const int colors = s.format == PixelFormat::kMono8 ? 1 : 3;
  for (int c = 0; c < colors; ++c) {
    Rng rng{mix(noise.seed ^ mix(static_cast<std::uint32_t>(frame) * 0x632be5abU ^
                                 mix(static_cast<std::uint32_t>(y) * 0x85ebca6bU ^
                                     mix(static_cast<std::uint32_t>(x) * 4U + c))))};
    const float clean = static_cast<float>(clean_sample(s, x, y, c));
    float v =
        noise.shot_gain > 0.0F ? rng.poisson(clean / noise.shot_gain) * noise.shot_gain : clean;
    if (noise.read_sigma > 0.0F) v += noise.read_sigma * rng.normal();
    o[c] = static_cast<std::uint8_t>(fminf(fmaxf(rintf(v), 0.0F), 255.0F));
  }
  if (bytes == 4) o[3] = 255;
}

// ---- The HDR scene ----

__host__ __device__ inline float hdr_value(int width, int height, int drift, int frame, int x,
                                           int y, int c) {
  constexpr float kShadowTint[3] = {1.0F, 0.85F, 0.7F};
  constexpr float kWindowTint[3] = {0.8F, 0.9F, 1.0F};
  const int shifted = x + drift * frame;
  if (3 * x < width) {
    // Shadow: stripes 8 pixels wide at 2x contrast, over a vertical ramp.
    const float base = 0.002F + 0.002F * static_cast<float>(y) / static_cast<float>(height);
    const float stripe = ((shifted / 8) & 1) != 0 ? 2.5F : 1.0F;
    return base * stripe * kShadowTint[c];
  }
  if (3 * x < 2 * width) {
    // Mid-tones: a horizontal ramp, and squares every 32 pixels.
    const float t = static_cast<float>(3 * x - width) / static_cast<float>(width);
    float v = 0.02F + 0.48F * t;
    if (((shifted / 32 + y / 32) & 1) != 0) v *= 0.6F;
    return v;
  }
  // Window: horizontal bars 8 pixels high, 1.5 and 6, tinted blue.
  const float bar = (((y + drift * frame) / 8) & 1) != 0 ? 6.0F : 1.5F;
  return bar * kWindowTint[c];
}

__host__ __device__ inline float srgb_encode(float v) {
  v = v < 0.0F ? 0.0F : v > 1.0F ? 1.0F : v;
  return v <= 0.0031308F ? 12.92F * v : 1.055F * powf(v, 1.0F / 2.4F) - 0.055F;
}

__host__ __device__ inline float hdr_gray_or(int channels, int width, int height, int drift,
                                             int frame, int x, int y, int c) {
  if (channels == 1) {
    return 0.2126F * hdr_value(width, height, drift, frame, x, y, 0) +
           0.7152F * hdr_value(width, height, drift, frame, x, y, 1) +
           0.0722F * hdr_value(width, height, drift, frame, x, y, 2);
  }
  return hdr_value(width, height, drift, frame, x, y, c);
}

__global__ void bracket_kernel(HdrScene s, int frame, float exposure, std::uint8_t* out,
                               std::size_t pitch, int bytes) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= s.width || y >= s.height) return;
  std::uint8_t* o = out + static_cast<std::size_t>(y) * pitch + static_cast<std::size_t>(x) * bytes;
  const int colors = bytes == 1 ? 1 : 3;
  for (int c = 0; c < colors; ++c) {
    const float r = hdr_gray_or(bytes == 1 ? 1 : 3, s.width, s.height, s.drift, frame, x, y, c);
    o[c] = static_cast<std::uint8_t>(srgb_encode(r * exposure) * 255.0F + 0.5F);
  }
  if (bytes == 4) o[3] = 255;
}

}  // namespace

double noise_variance(const NoiseModel& noise, double value) noexcept {
  return noise.shot_gain * value + static_cast<double>(noise.read_sigma) * noise.read_sigma;
}

Point object_center(const LowLightScene& scene, int frame) noexcept {
  if (scene.object_radius <= 0) return {};
  // Back and forth across the frame, at mid height.
  const int travel = std::max(1, scene.width - 2 * scene.object_radius);
  const auto distance =
      static_cast<std::int64_t>(std::floor(scene.speed * static_cast<float>(frame)));
  const std::int64_t phase = distance % (2LL * travel);
  const std::int64_t along = phase < travel ? phase : 2LL * travel - phase;
  return Point{.x = scene.object_radius + static_cast<int>(along), .y = (11 * scene.height) / 20};
}

bool object_covers(const LowLightScene& scene, int frame, int x, int y) noexcept {
  const Point center = object_center(scene, frame);
  if (center.x < 0) return false;
  const int dx = x - center.x;
  const int dy = y - center.y;
  return dx * dx + dy * dy <= scene.object_radius * scene.object_radius;
}

std::vector<std::uint8_t> render_clean(const LowLightScene& scene, int frame) {
  const SceneArgs s = scene_args(scene, frame);
  const int bytes = pixel_bytes(scene.format);
  std::vector<std::uint8_t> out(static_cast<std::size_t>(scene.width) * scene.height * bytes);
  for (int y = 0; y < scene.height; ++y) {
    for (int x = 0; x < scene.width; ++x) {
      for (int c = 0; c < bytes; ++c) {
        out[(static_cast<std::size_t>(y) * scene.width + x) * bytes + c] =
            static_cast<std::uint8_t>(clean_sample(s, x, y, c));
      }
    }
  }
  return out;
}

cudaError_t render_noisy(const LowLightScene& scene, const NoiseModel& noise, int frame,
                         const ImageView& output, cudaStream_t stream) {
  if (output.format != scene.format || output.width != scene.width ||
      output.height != scene.height || scene.format == PixelFormat::kNv12) {
    return cudaErrorInvalidValue;
  }
  const dim3 block(32, 8);
  const dim3 grid((scene.width + 31) / 32, (scene.height + 7) / 8);
  noisy_kernel<<<grid, block, 0, stream>>>(scene_args(scene, frame), noise, frame,
                                           output.planes[0].data, output.planes[0].pitch);
  return cudaGetLastError();
}

std::vector<std::uint8_t> render_noisy(const LowLightScene& scene, const NoiseModel& noise,
                                       int frame) {
  const std::size_t row = static_cast<std::size_t>(scene.width) * pixel_bytes(scene.format);
  const std::size_t bytes = row * scene.height;
  void* device = nullptr;
  const auto check = [&](cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
      cudaFree(device);
      throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
  };
  check(cudaMalloc(&device, bytes), "cudaMalloc");
  ImageView view{.format = scene.format, .width = scene.width, .height = scene.height};
  view.planes[0] = Plane{.data = static_cast<std::uint8_t*>(device), .pitch = row};
  check(render_noisy(scene, noise, frame, view, nullptr), "render_noisy");
  std::vector<std::uint8_t> out(bytes);
  check(cudaMemcpy(out.data(), device, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy");
  cudaFree(device);
  return out;
}

double psnr(const std::uint8_t* test, const std::uint8_t* truth, std::size_t count) noexcept {
  double sum = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double d = static_cast<double>(test[i]) - static_cast<double>(truth[i]);
    sum += d * d;
  }
  if (sum == 0.0 || count == 0) return 99.0;
  return 10.0 * std::log10(255.0 * 255.0 / (sum / static_cast<double>(count)));
}

HdrRegion hdr_region(const HdrScene& scene, int x) noexcept {
  return 3 * x < scene.width       ? HdrRegion::kShadow
         : 3 * x < 2 * scene.width ? HdrRegion::kMidtones
                                   : HdrRegion::kWindow;
}

std::vector<float> render_radiance(const HdrScene& scene, int frame) {
  std::vector<float> out(static_cast<std::size_t>(scene.width) * scene.height * scene.channels);
  for (int y = 0; y < scene.height; ++y) {
    for (int x = 0; x < scene.width; ++x) {
      for (int c = 0; c < scene.channels; ++c) {
        out[(static_cast<std::size_t>(y) * scene.width + x) * scene.channels + c] =
            hdr_gray_or(scene.channels, scene.width, scene.height, scene.drift, frame, x, y, c);
      }
    }
  }
  return out;
}

std::vector<std::uint8_t> render_bracket(const HdrScene& scene, int frame, float exposure) {
  const std::vector<float> radiance = render_radiance(scene, frame);
  std::vector<std::uint8_t> out(radiance.size());
  for (std::size_t i = 0; i < radiance.size(); ++i) {
    out[i] = static_cast<std::uint8_t>(srgb_encode(radiance[i] * exposure) * 255.0F + 0.5F);
  }
  return out;
}

cudaError_t render_bracket(const HdrScene& scene, int frame, float exposure,
                           const ImageView& output, cudaStream_t stream) {
  const int bytes = pixel_bytes(output.format);
  if (output.width != scene.width || output.height != scene.height ||
      output.format == PixelFormat::kNv12 || (scene.channels == 1) != (bytes == 1)) {
    return cudaErrorInvalidValue;
  }
  const dim3 block(32, 8);
  const dim3 grid((scene.width + 31) / 32, (scene.height + 7) / 8);
  bracket_kernel<<<grid, block, 0, stream>>>(scene, frame, exposure, output.planes[0].data,
                                             output.planes[0].pitch, bytes);
  return cudaGetLastError();
}

}  // namespace holoscan::denoise
