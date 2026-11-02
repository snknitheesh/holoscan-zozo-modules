// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The cuda backend. Per frame, all on one stream, nothing waiting on the host:
//
// 1. The noise level: configured, or estimated from the frame (noise_estimate.cu).
// 2. Temporal (when enabled): a motion pass compares each pixel's 3x3 luma mean,
//    and the pixel itself, with the history (the previous temporal output, in
//    float) against the noise expected in that difference, and counts the
//    pixels that moved; a blend pass then averages static pixels with the
//    history (a running mean up to history_frames(), then a recursive filter at
//    that weight) and replaces moving ones by the frame, in proportion to how
//    surely they moved. When more than scene_cut_fraction of the pixels moved,
//    the whole history is replaced: a scene cut.
// 3. Spatial: bilateral or non-local means of the temporal output (or of the
//    frame), whose noise level is the frame's times what is left after the
//    pixel's temporal averaging, so that static, well-averaged areas stay sharp
//    and moving ones are smoothed. The history is kept before this step, so
//    spatial smoothing never accumulates.
//
// NV12's UV plane follows its Y plane: the motion of the 2x2 luma pixels it
// covers, and the smaller of their histories.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "denoiser_impl.hpp"
#include "noise_estimate.hpp"

namespace holoscan::denoise::detail {
namespace {

namespace t = tuning;

constexpr int kBilateralTaps = 2 * t::kBilateralRadius + 1;
// exp(-(dx^2 + dy^2) / (2 s^2)) for the bilateral window, set once.
__constant__ float c_spatial_weights[kBilateralTaps * kBilateralTaps];

struct FilterArgs {
  int width{};  // Of the plane being filtered.
  int height{};
  const float* sigma{};  // Device: the frame's noise, code values.
  float strength{};
  int history_limit{};
  // With temporal filtering: the luma-resolution ages after this frame, and
  // whether this plane is at half of that resolution (NV12 UV). Null without.
  const std::uint8_t* age{};
  int age_pitch{};
  bool half_resolution_age{};
};

template <typename T>
__device__ __forceinline__ float value_of(T v) {
  return static_cast<float>(v);
}

__device__ __forceinline__ int clampi(int v, int low, int high) {
  return v < low ? low : v > high ? high : v;
}

__device__ __forceinline__ std::uint8_t to_byte(float v) {
  v = v < 0.0F ? 0.0F : v > 255.0F ? 255.0F : v;
  return static_cast<std::uint8_t>(__float2int_rn(v));
}

// Luma of a pixel for motion: (R + 2G + B) / 4 of a colour pixel.
template <int C, typename T>
__device__ __forceinline__ float luma(const T* p) {
  if constexpr (C == 3) {
    return (value_of(p[0]) + 2.0F * value_of(p[1]) + value_of(p[2])) * 0.25F;
  } else if constexpr (C == 2) {
    return (value_of(p[0]) + value_of(p[1])) * 0.5F;
  } else {
    return value_of(p[0]);
  }
}

// The noise of luma() relative to that of one channel.
template <int C>
constexpr float luma_noise() {
  return C == 3 ? 0.61237244F : C == 2 ? 0.70710678F : 1.0F;
}

// A pixel of a plane of C channels in pixels of S elements, clamped to the plane.
template <int C, int S, typename T>
__device__ __forceinline__ const T* pixel(const T* base, std::size_t pitch_elements, int x, int y,
                                          int width, int height) {
  x = clampi(x, 0, width - 1);
  y = clampi(y, 0, height - 1);
  return base + static_cast<std::size_t>(y) * pitch_elements + static_cast<std::size_t>(x) * S;
}

// The age after this frame of a pixel of the filtered plane.
__device__ __forceinline__ int age_at(const FilterArgs& a, int x, int y) {
  if (!a.half_resolution_age) return a.age[static_cast<std::size_t>(y) * a.age_pitch + x];
  const std::uint8_t* row = a.age + static_cast<std::size_t>(2 * y) * a.age_pitch + 2 * x;
  const int top = min(row[0], row[1]);
  const int bottom = min(row[a.age_pitch], row[a.age_pitch + 1]);
  return min(top, bottom);
}

// The noise sigma left in a pixel of the filtered plane.
__device__ __forceinline__ float residual_sigma(const FilterArgs& a, int x, int y) {
  const float sigma = *a.sigma;
  if (a.age == nullptr) return sigma;
  return sigma * sqrtf(t::residual_variance(age_at(a, x, y) + 1, a.history_limit));
}

// ---- Temporal ----

struct MotionArgs {
  int width{};
  int height{};
  const float* sigma{};
  float threshold{};
  int history_limit{};
};

// Motion confidence of each luma-resolution pixel (255: static, 0: moved), and
// the count of moving pixels.
template <int C, int S>
__global__ void motion_kernel(MotionArgs a, const std::uint8_t* frame, std::size_t frame_pitch,
                              const float* history, std::size_t history_pitch,
                              const std::uint8_t* age, std::uint8_t* still, int map_pitch,
                              unsigned int* moving) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  bool moved = false;
  if (x < a.width && y < a.height) {
    float frame_sum = 0.0F;
    float history_sum = 0.0F;
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        frame_sum += luma<C>(pixel<C, S>(frame, frame_pitch, x + dx, y + dy, a.width, a.height));
        history_sum +=
            luma<C>(pixel<C, C>(history, history_pitch, x + dx, y + dy, a.width, a.height));
      }
    }
    const float center_frame = luma<C>(pixel<C, S>(frame, frame_pitch, x, y, a.width, a.height));
    const float center_history =
        luma<C>(pixel<C, C>(history, history_pitch, x, y, a.width, a.height));
    const std::size_t at = static_cast<std::size_t>(y) * map_pitch + x;
    const float left = t::residual_variance(age[at] + 1, a.history_limit);
    const float noise = *a.sigma * luma_noise<C>() * sqrtf(1.0F + left);
    // The 3x3 mean's difference has a third of the pixel difference's noise.
    const float box = fabsf(frame_sum - history_sum) * (1.0F / 9.0F) / (noise * (1.0F / 3.0F));
    const float single = fabsf(center_frame - center_history) / noise;
    const float m = fmaxf(box, 0.5F * single);
    const float t0 = a.threshold;
    const float stillness = m <= t0 ? 1.0F : m >= 2.0F * t0 ? 0.0F : (2.0F * t0 - m) / t0;
    still[at] = static_cast<std::uint8_t>(__float2int_rn(stillness * 255.0F));
    moved = m > t0;
  }
  const int moved_in_block = __syncthreads_count(moved ? 1 : 0);
  if (threadIdx.x == 0 && threadIdx.y == 0 && moved_in_block != 0) {
    atomicAdd(moving, static_cast<unsigned int>(moved_in_block));
  }
}

struct BlendArgs {
  int width{};  // Of the blended plane.
  int height{};
  int history_limit{};
  bool first{};
  const unsigned int* moving{};
  unsigned int cut_pixels{};
};

// Whether this frame drops the whole history.
__device__ __forceinline__ bool restart(const BlendArgs& a) {
  return a.first || *a.moving > a.cut_pixels;
}

// Blends a luma-resolution plane into its history and ages it.
template <int C, int S>
__global__ void blend_kernel(BlendArgs a, const std::uint8_t* frame, std::size_t frame_pitch,
                             float* history, std::size_t history_pitch, const std::uint8_t* age_in,
                             std::uint8_t* age_out, const std::uint8_t* still, int map_pitch) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= a.width || y >= a.height) return;
  const std::uint8_t* f =
      frame + static_cast<std::size_t>(y) * frame_pitch + static_cast<std::size_t>(x) * S;
  float* h =
      history + static_cast<std::size_t>(y) * history_pitch + static_cast<std::size_t>(x) * C;
  const std::size_t at = static_cast<std::size_t>(y) * map_pitch + x;
  if (restart(a)) {
    for (int c = 0; c < C; ++c) h[c] = f[c];
    age_out[at] = 0;
    return;
  }
  // Frames in the history before this one: age + 1, capped below the limit.
  const int n = min(static_cast<int>(age_in[at]) + 1, a.history_limit - 1);
  const float weight = 1.0F / static_cast<float>(n + 1);
  const int s = still[at];
  const float stillness = static_cast<float>(s) * (1.0F / 255.0F);
  const float alpha = weight + (1.0F - weight) * (1.0F - stillness);
  for (int c = 0; c < C; ++c) h[c] += alpha * (static_cast<float>(f[c]) - h[c]);
  age_out[at] = static_cast<std::uint8_t>(s == 255 ? n : static_cast<int>(n * stillness));
}

// Blends NV12's UV plane by the motion and ages of the luma pixels it covers.
__global__ void blend_uv_kernel(BlendArgs a, const std::uint8_t* frame, std::size_t frame_pitch,
                                float* history, std::size_t history_pitch,
                                const std::uint8_t* age_in, const std::uint8_t* still,
                                int map_pitch) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= a.width || y >= a.height) return;
  const std::uint8_t* f = frame + static_cast<std::size_t>(y) * frame_pitch + 2 * x;
  float* h = history + static_cast<std::size_t>(y) * history_pitch + 2 * x;
  if (restart(a)) {
    h[0] = f[0];
    h[1] = f[1];
    return;
  }
  const std::size_t at = static_cast<std::size_t>(2 * y) * map_pitch + 2 * x;
  const int age =
      min(min(age_in[at], age_in[at + 1]), min(age_in[at + map_pitch], age_in[at + map_pitch + 1]));
  const int s =
      min(min(still[at], still[at + 1]), min(still[at + map_pitch], still[at + map_pitch + 1]));
  const int n = min(age + 1, a.history_limit - 1);
  const float weight = 1.0F / static_cast<float>(n + 1);
  const float alpha = weight + (1.0F - weight) * (1.0F - static_cast<float>(s) * (1.0F / 255.0F));
  h[0] += alpha * (static_cast<float>(f[0]) - h[0]);
  h[1] += alpha * (static_cast<float>(f[1]) - h[1]);
}

// ---- Spatial ----

// Copies (and rounds) a pixel's colour channels to the output.
template <int C, int SIn, typename T>
__device__ __forceinline__ void copy_pixel(const T* in, std::uint8_t* out) {
  for (int c = 0; c < C; ++c) out[c] = to_byte(value_of(in[c]));
}

// 7x7 bilateral filter; with `copy_only`, just rounds and copies.
template <int C, int SIn, int SOut, typename T>
__global__ void bilateral_kernel(FilterArgs a, const T* in, std::size_t in_pitch, std::uint8_t* out,
                                 std::size_t out_pitch, const std::uint8_t* alpha,
                                 std::size_t alpha_pitch, bool copy_only) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= a.width || y >= a.height) return;
  std::uint8_t* o =
      out + static_cast<std::size_t>(y) * out_pitch + static_cast<std::size_t>(x) * SOut;
  if constexpr (SOut == 4) {
    o[3] = alpha[static_cast<std::size_t>(y) * alpha_pitch + static_cast<std::size_t>(x) * 4 + 3];
  }
  const T* center = pixel<C, SIn>(in, in_pitch, x, y, a.width, a.height);
  const float h = t::kBilateralRange * a.strength * residual_sigma(a, x, y);
  if (copy_only || h < 1e-3F) {
    copy_pixel<C, SIn>(center, o);
    return;
  }
  float c0[C];
  for (int c = 0; c < C; ++c) c0[c] = value_of(center[c]);
  const float inv = -1.0F / (2.0F * h * h * static_cast<float>(C));
  float sum[C] = {};
  float total = 0.0F;
  for (int dy = -t::kBilateralRadius; dy <= t::kBilateralRadius; ++dy) {
    for (int dx = -t::kBilateralRadius; dx <= t::kBilateralRadius; ++dx) {
      const T* q = pixel<C, SIn>(in, in_pitch, x + dx, y + dy, a.width, a.height);
      float v[C];
      float distance = 0.0F;
      for (int c = 0; c < C; ++c) {
        v[c] = value_of(q[c]);
        distance += (v[c] - c0[c]) * (v[c] - c0[c]);
      }
      const float w = c_spatial_weights[(dy + t::kBilateralRadius) * kBilateralTaps + dx +
                                        t::kBilateralRadius] *
                      __expf(distance * inv);
      total += w;
      for (int c = 0; c < C; ++c) sum[c] += w * v[c];
    }
  }
  for (int c = 0; c < C; ++c) o[c] = to_byte(sum[c] / total);
}

constexpr int kNlmBlock = 16;
constexpr int kNlmHalo = t::kNlmSearchRadius + t::kNlmPatchRadius;
constexpr int kNlmTile = kNlmBlock + 2 * kNlmHalo;

// Non-local means: patches compared on the guide (luma), colours averaged.
template <int C, int SIn, int SOut, typename T>
__global__ void nlm_kernel(FilterArgs a, const T* in, std::size_t in_pitch, std::uint8_t* out,
                           std::size_t out_pitch, const std::uint8_t* alpha,
                           std::size_t alpha_pitch) {
  __shared__ float guide[kNlmTile][kNlmTile];
  const int x0 = blockIdx.x * kNlmBlock - kNlmHalo;
  const int y0 = blockIdx.y * kNlmBlock - kNlmHalo;
  for (int i = threadIdx.y * kNlmBlock + threadIdx.x; i < kNlmTile * kNlmTile;
       i += kNlmBlock * kNlmBlock) {
    const int tx = i % kNlmTile;
    const int ty = i / kNlmTile;
    guide[ty][tx] = luma<C>(pixel<C, SIn>(in, in_pitch, x0 + tx, y0 + ty, a.width, a.height));
  }
  __syncthreads();
  const int x = blockIdx.x * kNlmBlock + threadIdx.x;
  const int y = blockIdx.y * kNlmBlock + threadIdx.y;
  if (x >= a.width || y >= a.height) return;
  std::uint8_t* o =
      out + static_cast<std::size_t>(y) * out_pitch + static_cast<std::size_t>(x) * SOut;
  if constexpr (SOut == 4) {
    o[3] = alpha[static_cast<std::size_t>(y) * alpha_pitch + static_cast<std::size_t>(x) * 4 + 3];
  }
  const T* center = pixel<C, SIn>(in, in_pitch, x, y, a.width, a.height);
  const float sigma = residual_sigma(a, x, y) * luma_noise<C>();
  const float h = t::kNlmH * a.strength * sigma;
  if (h < 1e-3F) {
    copy_pixel<C, SIn>(center, o);
    return;
  }
  const int cx = threadIdx.x + kNlmHalo;
  const int cy = threadIdx.y + kNlmHalo;
  constexpr float kPatch = 1.0F / ((2 * t::kNlmPatchRadius + 1) * (2 * t::kNlmPatchRadius + 1));
  const float bias = 2.0F * sigma * sigma;
  const float inv = -1.0F / (h * h);
  float sum[C] = {};
  float total = 0.0F;
  for (int dy = -t::kNlmSearchRadius; dy <= t::kNlmSearchRadius; ++dy) {
    for (int dx = -t::kNlmSearchRadius; dx <= t::kNlmSearchRadius; ++dx) {
      float distance = 0.0F;
      for (int py = -t::kNlmPatchRadius; py <= t::kNlmPatchRadius; ++py) {
        for (int px = -t::kNlmPatchRadius; px <= t::kNlmPatchRadius; ++px) {
          const float d = guide[cy + py][cx + px] - guide[cy + dy + py][cx + dx + px];
          distance += d * d;
        }
      }
      const float w = __expf(fmaxf(distance * kPatch - bias, 0.0F) * inv);
      const T* q = pixel<C, SIn>(in, in_pitch, x + dx, y + dy, a.width, a.height);
      total += w;
      for (int c = 0; c < C; ++c) sum[c] += w * value_of(q[c]);
    }
  }
  for (int c = 0; c < C; ++c) o[c] = to_byte(sum[c] / total);
}

// Launches the spatial stage over one plane.
template <int C, int SIn, int SOut, typename T>
cudaError_t launch_spatial(SpatialFilter filter, const FilterArgs& a, const T* in,
                           std::size_t in_pitch, std::uint8_t* out, std::size_t out_pitch,
                           const std::uint8_t* alpha, std::size_t alpha_pitch,
                           cudaStream_t stream) {
  if (filter == SpatialFilter::kNlm) {
    const dim3 block(kNlmBlock, kNlmBlock);
    const dim3 grid((a.width + kNlmBlock - 1) / kNlmBlock, (a.height + kNlmBlock - 1) / kNlmBlock);
    nlm_kernel<C, SIn, SOut, T>
        <<<grid, block, 0, stream>>>(a, in, in_pitch, out, out_pitch, alpha, alpha_pitch);
  } else {
    const dim3 block(32, 8);
    const dim3 grid((a.width + 31) / 32, (a.height + 7) / 8);
    bilateral_kernel<C, SIn, SOut, T><<<grid, block, 0, stream>>>(
        a, in, in_pitch, out, out_pitch, alpha, alpha_pitch, filter == SpatialFilter::kNone);
  }
  return cudaGetLastError();
}

struct DeviceFree {
  void operator()(void* p) const noexcept { cudaFree(p); }
};
template <typename T>
using DeviceBuffer = std::unique_ptr<T, DeviceFree>;

template <typename T>
DeviceBuffer<T> allocate(std::size_t count) {
  void* p = nullptr;
  check_cuda(cudaMalloc(&p, std::max<std::size_t>(count, 1) * sizeof(T)), "cudaMalloc");
  return DeviceBuffer<T>(static_cast<T*>(p));
}

void upload_spatial_weights() {
  static const cudaError_t status = [] {
    float weights[kBilateralTaps * kBilateralTaps];
    for (int dy = -t::kBilateralRadius; dy <= t::kBilateralRadius; ++dy) {
      for (int dx = -t::kBilateralRadius; dx <= t::kBilateralRadius; ++dx) {
        weights[(dy + t::kBilateralRadius) * kBilateralTaps + dx + t::kBilateralRadius] =
            std::exp(-static_cast<float>(dx * dx + dy * dy) /
                     (2.0F * t::kBilateralSpatialSigma * t::kBilateralSpatialSigma));
      }
    }
    return cudaMemcpyToSymbol(c_spatial_weights, weights, sizeof(weights));
  }();
  check_cuda(status, "cudaMemcpyToSymbol");
}

class CudaDenoiser final : public Denoiser::Impl {
 public:
  explicit CudaDenoiser(const DenoiseParams& params)
      : params_(params), limit_(t::history_frames(params)) {
    upload_spatial_weights();
  }

  ~CudaDenoiser() override {
    // cudaFree waits for the work that may still use the buffers.
  }

  Backend backend() const noexcept override { return Backend::kCuda; }
  std::string_view engine() const noexcept override { return "cuda"; }
  void reset() noexcept override { first_ = true; }

  float noise_sigma(cudaStream_t stream) override {
    if (!sigma_) return 0.0F;
    float value = 0.0F;
    check_cuda(cudaMemcpyAsync(&value, sigma_.get(), sizeof(value), cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    return value;
  }

  void process(const ImageView& input, const ImageView& output, cudaStream_t stream) override {
    if (input.width != width_ || input.height != height_ || input.format != format_ || !sigma_) {
      allocate_for(input, stream);
    }
    estimate_noise(input, stream);
    switch (input.format) {
      case PixelFormat::kMono8:
        run<1, 1>(input, output, stream);
        break;
      case PixelFormat::kRgb8:
        run<3, 3>(input, output, stream);
        break;
      case PixelFormat::kRgba8:
        run<3, 4>(input, output, stream);
        break;
      case PixelFormat::kNv12:
        run<1, 1>(input, output, stream);
        run_uv(input, output, stream);
        break;
    }
    if (params_.temporal) {
      std::swap(age_in_, age_out_);
      first_ = false;
    }
    check_cuda(cudaGetLastError(), "denoise kernels");
  }

 private:
  static int color_channels(PixelFormat format) {
    return format == PixelFormat::kRgb8 || format == PixelFormat::kRgba8 ? 3 : 1;
  }

  void allocate_for(const ImageView& input, cudaStream_t stream) {
    // The old buffers may still be read by queued work.
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    width_ = input.width;
    height_ = input.height;
    format_ = input.format;
    const auto pixels = static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);
    sigma_ = allocate<float>(1);
    partials_ = allocate<std::uint64_t>(kNoiseBlocks);
    if (params_.temporal) {
      history_ = allocate<float>(pixels * color_channels(format_));
      history_uv_ = format_ == PixelFormat::kNv12 ? allocate<float>(pixels / 2) : nullptr;
      age_in_ = allocate<std::uint8_t>(pixels);
      age_out_ = allocate<std::uint8_t>(pixels);
      still_ = allocate<std::uint8_t>(pixels);
      moving_ = allocate<unsigned int>(1);
      check_cuda(cudaMemsetAsync(age_in_.get(), 0, pixels, stream), "cudaMemsetAsync");
    }
    if (params_.noise_sigma > 0.0F) {
      check_cuda(launch_set(sigma_.get(), std::max(params_.noise_sigma, t::kMinSigma), stream),
                 "noise sigma");
    }
    first_ = true;
  }

  void estimate_noise(const ImageView& input, cudaStream_t stream) {
    if (params_.noise_sigma > 0.0F) return;
    // Green of a colour image, whose noise is a channel's; luma otherwise.
    const SamplePlane plane{.data = input.planes[0].data,
                            .pitch = input.planes[0].pitch,
                            .width = input.width,
                            .height = input.height,
                            .stride = pixel_bytes(input.format),
                            .channel = color_channels(input.format) == 3 ? 1 : 0};
    check_cuda(launch_noise_estimate(plane, partials_.get(), t::kMinSigma, sigma_.get(), stream),
               "noise estimate");
  }

  FilterArgs filter_args(int width, int height, bool half_resolution_age) const {
    FilterArgs a{.width = width,
                 .height = height,
                 .sigma = sigma_.get(),
                 .strength = params_.strength,
                 .history_limit = limit_};
    if (params_.temporal) {
      a.age = age_out_.get();
      a.age_pitch = width_;
      a.half_resolution_age = half_resolution_age;
    }
    return a;
  }

  BlendArgs blend_args(int width, int height) const {
    const auto pixels = static_cast<double>(width_) * static_cast<double>(height_);
    return BlendArgs{.width = width,
                     .height = height,
                     .history_limit = limit_,
                     .first = first_,
                     .moving = moving_.get(),
                     .cut_pixels = static_cast<unsigned int>(std::min(
                         pixels * static_cast<double>(params_.scene_cut_fraction), 4294967295.0))};
  }

  // The luma-resolution plane: Y of NV12, or the whole image.
  template <int C, int S>
  void run(const ImageView& input, const ImageView& output, cudaStream_t stream) {
    const Plane in = input.planes[0];
    const Plane out = output.planes[0];
    const dim3 block(32, 8);
    const dim3 grid((width_ + 31) / 32, (height_ + 7) / 8);
    if (!params_.temporal) {
      check_cuda(launch_spatial<C, S, S, std::uint8_t>(
                     params_.spatial, filter_args(width_, height_, false), in.data, in.pitch,
                     out.data, out.pitch, in.data, in.pitch, stream),
                 "spatial filter");
      return;
    }
    const std::size_t history_pitch = static_cast<std::size_t>(width_) * C;
    if (!first_) {
      check_cuda(cudaMemsetAsync(moving_.get(), 0, sizeof(unsigned int), stream),
                 "cudaMemsetAsync");
      motion_kernel<C, S>
          <<<grid, block, 0, stream>>>(MotionArgs{.width = width_,
                                                  .height = height_,
                                                  .sigma = sigma_.get(),
                                                  .threshold = params_.motion_threshold,
                                                  .history_limit = limit_},
                                       in.data, in.pitch, history_.get(), history_pitch,
                                       age_in_.get(), still_.get(), width_, moving_.get());
    }
    blend_kernel<C, S><<<grid, block, 0, stream>>>(blend_args(width_, height_), in.data, in.pitch,
                                                   history_.get(), history_pitch, age_in_.get(),
                                                   age_out_.get(), still_.get(), width_);
    check_cuda(launch_spatial<C, C, S, float>(params_.spatial, filter_args(width_, height_, false),
                                              history_.get(), history_pitch, out.data, out.pitch,
                                              in.data, in.pitch, stream),
               "spatial filter");
  }

  // NV12's UV plane, after its Y plane's motion pass.
  void run_uv(const ImageView& input, const ImageView& output, cudaStream_t stream) {
    const Plane in = input.planes[1];
    const Plane out = output.planes[1];
    const int w = width_ / 2;
    const int h = height_ / 2;
    if (!params_.temporal) {
      check_cuda(
          launch_spatial<2, 2, 2, std::uint8_t>(params_.spatial, filter_args(w, h, false), in.data,
                                                in.pitch, out.data, out.pitch, nullptr, 0, stream),
          "spatial filter");
      return;
    }
    const dim3 block(32, 8);
    const dim3 grid((w + 31) / 32, (h + 7) / 8);
    const auto history_pitch = static_cast<std::size_t>(w) * 2;
    blend_uv_kernel<<<grid, block, 0, stream>>>(blend_args(w, h), in.data, in.pitch,
                                                history_uv_.get(), history_pitch, age_in_.get(),
                                                still_.get(), width_);
    check_cuda(
        launch_spatial<2, 2, 2, float>(params_.spatial, filter_args(w, h, true), history_uv_.get(),
                                       history_pitch, out.data, out.pitch, nullptr, 0, stream),
        "spatial filter");
  }

  DenoiseParams params_;
  int limit_;
  int width_{};
  int height_{};
  PixelFormat format_{PixelFormat::kRgb8};
  bool first_{true};
  DeviceBuffer<float> sigma_;
  DeviceBuffer<std::uint64_t> partials_;
  DeviceBuffer<float> history_;
  DeviceBuffer<float> history_uv_;
  DeviceBuffer<std::uint8_t> age_in_;
  DeviceBuffer<std::uint8_t> age_out_;
  DeviceBuffer<std::uint8_t> still_;
  DeviceBuffer<unsigned int> moving_;
};

}  // namespace

std::unique_ptr<Denoiser::Impl> make_cuda_denoiser(const DenoiseParams& params) {
  return std::make_unique<CudaDenoiser>(params);
}

}  // namespace holoscan::denoise::detail
