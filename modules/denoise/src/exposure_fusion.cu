// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Exposure fusion (Mertens, Kautz and Van Reeth, 2007) and Reinhard's global
// tone map, on the GPU.
//
// Fusion: each bracket gets a weight per pixel, the product of its contrast
// (|Laplacian| of its gray), saturation (standard deviation of R, G, B) and
// well-exposedness (a Gaussian around mid-gray, per channel), each raised to
// its exponent; the weights are normalized across the brackets. The brackets'
// Laplacian pyramids are then summed level by level, each weighted by the
// Gaussian pyramid of its weight, and the sum collapsed into the output. The
// pyramids use the 5-tap binomial kernel with mirrored edges, in float.
//
// Tone map: the frame's log-average and largest luminance are reduced on the
// GPU in double, in a fixed order, then every pixel is mapped.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "denoiser_impl.hpp"
#include "holoscan/denoise/exposure_fusion.hpp"

namespace holoscan::denoise {
namespace {

using detail::check_cuda;

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

__device__ __forceinline__ std::uint8_t to_byte(float v) {
  v = v < 0.0F ? 0.0F : v > 1.0F ? 1.0F : v;
  return static_cast<std::uint8_t>(__float2int_rn(v * 255.0F));
}

// A mirrored index into 0..n-1 (-1 is 1, n is n-2).
__device__ __forceinline__ int mirror(int i, int n) {
  if (n == 1) return 0;
  i = i < 0 ? -i : i;
  i = i >= n ? 2 * n - 2 - i : i;
  return i < 0 ? 0 : i >= n ? n - 1 : i;
}

__constant__ float c_binomial[5] = {1.0F / 16, 4.0F / 16, 6.0F / 16, 4.0F / 16, 1.0F / 16};

// ---- Fusion ----

struct WeightArgs {
  int width{};
  int height{};
  float contrast{};
  float saturation{};
  float exposure{};
  float inv_two_sigma2{};  // 1 / (2 sigma^2).
};

template <int C, int S>
__device__ __forceinline__ float gray_at(const std::uint8_t* image, std::size_t pitch, int x, int y,
                                         int width, int height) {
  x = min(max(x, 0), width - 1);
  y = min(max(y, 0), height - 1);
  const std::uint8_t* p =
      image + static_cast<std::size_t>(y) * pitch + static_cast<std::size_t>(x) * S;
  if constexpr (C == 1) {
    return p[0] * (1.0F / 255.0F);
  } else {
    return (0.299F * p[0] + 0.587F * p[1] + 0.114F * p[2]) * (1.0F / 255.0F);
  }
}

// Added to contrast and saturation, so that in flat or gray areas, where every
// bracket's contrast or saturation is 0, well-exposedness still decides.
constexpr float kMeasureFloor = 1.0F / 255.0F;

template <int C, int S>
__global__ void weight_kernel(WeightArgs a, const std::uint8_t* image, std::size_t pitch,
                              float* weight) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= a.width || y >= a.height) return;
  const float laplacian = gray_at<C, S>(image, pitch, x - 1, y, a.width, a.height) +
                          gray_at<C, S>(image, pitch, x + 1, y, a.width, a.height) +
                          gray_at<C, S>(image, pitch, x, y - 1, a.width, a.height) +
                          gray_at<C, S>(image, pitch, x, y + 1, a.width, a.height) -
                          4.0F * gray_at<C, S>(image, pitch, x, y, a.width, a.height);
  const std::uint8_t* p =
      image + static_cast<std::size_t>(y) * pitch + static_cast<std::size_t>(x) * S;
  float exposedness = 1.0F;
  float mean = 0.0F;
  float v[C];
  for (int c = 0; c < C; ++c) {
    v[c] = p[c] * (1.0F / 255.0F);
    exposedness *= expf(-(v[c] - 0.5F) * (v[c] - 0.5F) * a.inv_two_sigma2);
    mean += v[c];
  }
  float w = powf(fabsf(laplacian) + kMeasureFloor, a.contrast) * powf(exposedness, a.exposure);
  if constexpr (C == 3) {
    mean *= 1.0F / 3.0F;
    float variance = 0.0F;
    for (int c = 0; c < 3; ++c) variance += (v[c] - mean) * (v[c] - mean);
    w *= powf(sqrtf(variance * (1.0F / 3.0F)) + kMeasureFloor, a.saturation);
  }
  weight[static_cast<std::size_t>(y) * a.width + x] = w + 1e-12F;
}

__global__ void normalize_kernel(float* w0, float* w1, float* w2, std::size_t count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float sum = w0[i] + w1[i] + (w2 != nullptr ? w2[i] : 0.0F);
  w0[i] /= sum;
  w1[i] /= sum;
  if (w2 != nullptr) w2[i] /= sum;
}

template <int C, int S>
__global__ void to_float_kernel(const std::uint8_t* image, std::size_t pitch, int width, int height,
                                float* out) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const std::uint8_t* p =
      image + static_cast<std::size_t>(y) * pitch + static_cast<std::size_t>(x) * S;
  float* o = out + (static_cast<std::size_t>(y) * width + x) * C;
  for (int c = 0; c < C; ++c) o[c] = p[c] * (1.0F / 255.0F);
}

// Blurs and halves a level of C interleaved channels.
template <int C>
__global__ void downsample_kernel(const float* in, int in_w, int in_h, float* out, int out_w,
                                  int out_h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= out_w || y >= out_h) return;
  float sum[C] = {};
  for (int j = -2; j <= 2; ++j) {
    const int sy = mirror(2 * y + j, in_h);
    for (int i = -2; i <= 2; ++i) {
      const int sx = mirror(2 * x + i, in_w);
      const float k = c_binomial[i + 2] * c_binomial[j + 2];
      const float* p = in + (static_cast<std::size_t>(sy) * in_w + sx) * C;
      for (int c = 0; c < C; ++c) sum[c] += k * p[c];
    }
  }
  float* o = out + (static_cast<std::size_t>(y) * out_w + x) * C;
  for (int c = 0; c < C; ++c) o[c] = sum[c];
}

// The coarser level `in` expanded to (x, y) of the finer one.
template <int C>
__device__ __forceinline__ void expand_at(const float* in, int in_w, int in_h, int x, int y,
                                          float (&value)[C]) {
  for (int c = 0; c < C; ++c) value[c] = 0.0F;
  for (int j = -2; j <= 2; ++j) {
    if (((y - j) & 1) != 0) continue;
    const int sy = mirror((y - j) / 2, in_h);
    for (int i = -2; i <= 2; ++i) {
      if (((x - i) & 1) != 0) continue;
      const int sx = mirror((x - i) / 2, in_w);
      const float k = 4.0F * c_binomial[i + 2] * c_binomial[j + 2];
      const float* p = in + (static_cast<std::size_t>(sy) * in_w + sx) * C;
      for (int c = 0; c < C; ++c) value[c] += k * p[c];
    }
  }
}

// acc (+)= weight * (gaussian - expand(coarser)), or weight * gaussian at the top.
template <int C>
__global__ void accumulate_kernel(const float* gaussian, const float* coarser, int coarse_w,
                                  int coarse_h, const float* weight, float* acc, int width,
                                  int height, bool first) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const std::size_t i = static_cast<std::size_t>(y) * width + x;
  float detail[C];
  if (coarser != nullptr) {
    expand_at<C>(coarser, coarse_w, coarse_h, x, y, detail);
    for (int c = 0; c < C; ++c) detail[c] = gaussian[i * C + c] - detail[c];
  } else {
    for (int c = 0; c < C; ++c) detail[c] = gaussian[i * C + c];
  }
  const float w = weight[i];
  for (int c = 0; c < C; ++c) {
    acc[i * C + c] = first ? w * detail[c] : acc[i * C + c] + w * detail[c];
  }
}

// acc += expand(coarser): collapses one level of the blended pyramid.
template <int C>
__global__ void collapse_kernel(const float* coarser, int coarse_w, int coarse_h, float* acc,
                                int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  float up[C];
  expand_at<C>(coarser, coarse_w, coarse_h, x, y, up);
  float* a = acc + (static_cast<std::size_t>(y) * width + x) * C;
  for (int c = 0; c < C; ++c) a[c] += up[c];
}

template <int C, int S>
__global__ void fusion_output_kernel(const float* acc, int width, int height, std::uint8_t* out,
                                     std::size_t pitch) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const float* a = acc + (static_cast<std::size_t>(y) * width + x) * C;
  std::uint8_t* o = out + static_cast<std::size_t>(y) * pitch + static_cast<std::size_t>(x) * S;
  for (int c = 0; c < C; ++c) o[c] = to_byte(a[c]);
  if constexpr (S == 4) o[3] = 255;
}

dim3 grid_for(int width, int height) { return dim3((width + 31) / 32, (height + 7) / 8); }
const dim3 kBlock(32, 8);

// ---- Tone map ----

constexpr int kStatsBlocks = 256;
constexpr int kStatsThreads = 256;

struct HdrArgs {
  int width{};
  int height{};
  const std::uint8_t* data{};
  std::size_t pitch{};
  int channels{};
  bool is_float{};
  float white_code{};  // 2^bits - 1 for code values.
};

__device__ __forceinline__ float hdr_sample(const HdrArgs& a, int x, int y, int c) {
  const std::uint8_t* row = a.data + static_cast<std::size_t>(y) * a.pitch;
  const std::size_t i = static_cast<std::size_t>(x) * a.channels + c;
  if (a.is_float) {
    const float v = reinterpret_cast<const float*>(row)[i];
    return isfinite(v) && v > 0.0F ? v : 0.0F;
  }
  return static_cast<float>(reinterpret_cast<const std::uint16_t*>(row)[i]) / a.white_code;
}

__device__ __forceinline__ float hdr_luminance(const HdrArgs& a, int x, int y) {
  if (a.channels == 1) return hdr_sample(a, x, y, 0);
  return 0.2126F * hdr_sample(a, x, y, 0) + 0.7152F * hdr_sample(a, x, y, 1) +
         0.0722F * hdr_sample(a, x, y, 2);
}

__global__ void stats_partials_kernel(HdrArgs a, double* log_sums, float* maxima) {
  __shared__ double sums[kStatsThreads];
  __shared__ float peaks[kStatsThreads];
  const std::int64_t count = static_cast<std::int64_t>(a.width) * a.height;
  double sum = 0.0;
  float peak = 0.0F;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const float l = hdr_luminance(a, static_cast<int>(i % a.width), static_cast<int>(i / a.width));
    sum += log(1e-6 + static_cast<double>(l));
    peak = fmaxf(peak, l);
  }
  sums[threadIdx.x] = sum;
  peaks[threadIdx.x] = peak;
  __syncthreads();
  for (int half = kStatsThreads / 2; half > 0; half /= 2) {
    if (threadIdx.x < half) {
      sums[threadIdx.x] += sums[threadIdx.x + half];
      peaks[threadIdx.x] = fmaxf(peaks[threadIdx.x], peaks[threadIdx.x + half]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    log_sums[blockIdx.x] = sums[0];
    maxima[blockIdx.x] = peaks[0];
  }
}

__global__ void stats_finalize_kernel(const double* log_sums, const float* maxima,
                                      std::int64_t count, double* stats) {
  __shared__ double sums[kStatsThreads];
  __shared__ float peaks[kStatsThreads];
  sums[threadIdx.x] = threadIdx.x < kStatsBlocks ? log_sums[threadIdx.x] : 0.0;
  peaks[threadIdx.x] = threadIdx.x < kStatsBlocks ? maxima[threadIdx.x] : 0.0F;
  __syncthreads();
  for (int half = kStatsThreads / 2; half > 0; half /= 2) {
    if (threadIdx.x < half) {
      sums[threadIdx.x] += sums[threadIdx.x + half];
      peaks[threadIdx.x] = fmaxf(peaks[threadIdx.x], peaks[threadIdx.x + half]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    stats[0] = exp(sums[0] / static_cast<double>(count));
    stats[1] = peaks[0];
  }
}

struct MapArgs {
  float key{};
  float white{};  // 0: from the frame's peak.
  ToneCurve curve{};
  float inv_gamma{};
  int out_stride{};
};

__device__ __forceinline__ float encode(float v, const MapArgs& m) {
  v = v < 0.0F ? 0.0F : v > 1.0F ? 1.0F : v;
  switch (m.curve) {
    case ToneCurve::kSrgb:
      return v <= 0.0031308F ? 12.92F * v : 1.055F * powf(v, 1.0F / 2.4F) - 0.055F;
    case ToneCurve::kGamma:
      return powf(v, m.inv_gamma);
    case ToneCurve::kLinear:
      break;
  }
  return v;
}

__global__ void tone_map_kernel(HdrArgs a, MapArgs m, const double* stats, std::uint8_t* out,
                                std::size_t pitch) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= a.width || y >= a.height) return;
  const float scale = static_cast<float>(static_cast<double>(m.key) / stats[0]);
  float white = m.white > 0.0F ? m.white : static_cast<float>(stats[1]) * scale;
  if (!(white > 0.0F)) white = 1.0F;
  const float l = hdr_luminance(a, x, y);
  const float lm = scale * l;
  const float ld = lm * (1.0F + lm / (white * white)) / (1.0F + lm);
  std::uint8_t* o =
      out + static_cast<std::size_t>(y) * pitch + static_cast<std::size_t>(x) * m.out_stride;
  if (a.channels == 1) {
    o[0] = to_byte(encode(ld, m));
    return;
  }
  for (int c = 0; c < 3; ++c) {
    o[c] = to_byte(encode(l > 0.0F ? hdr_sample(a, x, y, c) * ld / l : 0.0F, m));
  }
  if (a.channels == 4) o[3] = to_byte(hdr_sample(a, x, y, 3));
}

void check_bracket(const ImageView& view, const char* what) {
  if (view.format == PixelFormat::kNv12) {
    throw std::invalid_argument(std::string(what) + ": fusion takes MONO8, RGB8 or RGBA8");
  }
  if (view.width < 1 || view.height < 1 || view.planes[0].data == nullptr ||
      view.planes[0].pitch < static_cast<std::size_t>(view.width) * pixel_bytes(view.format)) {
    throw std::invalid_argument(std::string(what) + " has no pixels, or rows too short");
  }
}

}  // namespace

class ExposureFusion::Impl {
 public:
  explicit Impl(const FusionParams& params) : params_(params) {}

  void fuse(std::span<const ImageView> brackets, const ImageView& output, cudaStream_t stream) {
    if (brackets.size() != 2 && brackets.size() != 3) {
      throw std::invalid_argument("fusion takes 2 or 3 brackets");
    }
    check_bracket(output, "the output");
    for (const ImageView& bracket : brackets) {
      check_bracket(bracket, "a bracket");
      if (bracket.format != output.format || bracket.width != output.width ||
          bracket.height != output.height) {
        throw std::invalid_argument("the brackets and the output must share format and size");
      }
    }
    switch (output.format) {
      case PixelFormat::kMono8:
        run<1, 1>(brackets, output, stream);
        break;
      case PixelFormat::kRgb8:
        run<3, 3>(brackets, output, stream);
        break;
      default:
        run<3, 4>(brackets, output, stream);
        break;
    }
    check_cuda(cudaGetLastError(), "fusion kernels");
  }

 private:
  struct Level {
    int width{};
    int height{};
    DeviceBuffer<float> gaussian;  // The current bracket's image.
    DeviceBuffer<float> weight;    // Its weight (level 0: the bracket's own map).
    DeviceBuffer<float> blended;
  };

  void ensure(int width, int height, int channels, std::size_t count, cudaStream_t stream) {
    const int levels = fusion_levels(params_, width, height);
    if (width == width_ && height == height_ && channels == channels_ &&
        levels_.size() == static_cast<std::size_t>(levels) && weights_.size() >= count) {
      return;
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    width_ = width;
    height_ = height;
    channels_ = channels;
    const auto pixels = static_cast<std::size_t>(width) * height;
    weights_.clear();
    for (std::size_t k = 0; k < 3; ++k) weights_.push_back(allocate<float>(pixels));
    levels_.clear();
    levels_.resize(static_cast<std::size_t>(levels));
    int w = width;
    int h = height;
    for (Level& level : levels_) {
      level.width = w;
      level.height = h;
      const auto n = static_cast<std::size_t>(w) * h;
      level.gaussian = allocate<float>(n * channels);
      level.blended = allocate<float>(n * channels);
      if (&level != &levels_.front()) level.weight = allocate<float>(n);
      w = (w + 1) / 2;
      h = (h + 1) / 2;
    }
  }

  template <int C, int S>
  void run(std::span<const ImageView> brackets, const ImageView& output, cudaStream_t stream) {
    const int width = output.width;
    const int height = output.height;
    ensure(width, height, C, brackets.size(), stream);
    const WeightArgs args{
        .width = width,
        .height = height,
        .contrast = params_.contrast_weight,
        .saturation = params_.saturation_weight,
        .exposure = params_.exposure_weight,
        .inv_two_sigma2 = 1.0F / (2.0F * params_.exposure_sigma * params_.exposure_sigma)};
    for (std::size_t k = 0; k < brackets.size(); ++k) {
      weight_kernel<C, S><<<grid_for(width, height), kBlock, 0, stream>>>(
          args, brackets[k].planes[0].data, brackets[k].planes[0].pitch, weights_[k].get());
    }
    const auto pixels = static_cast<std::size_t>(width) * height;
    normalize_kernel<<<static_cast<unsigned>((pixels + 255) / 256), 256, 0, stream>>>(
        weights_[0].get(), weights_[1].get(), brackets.size() == 3 ? weights_[2].get() : nullptr,
        pixels);

    const std::size_t top = levels_.size() - 1;
    for (std::size_t k = 0; k < brackets.size(); ++k) {
      to_float_kernel<C, S><<<grid_for(width, height), kBlock, 0, stream>>>(
          brackets[k].planes[0].data, brackets[k].planes[0].pitch, width, height,
          levels_[0].gaussian.get());
      for (std::size_t l = 0; l + 1 < levels_.size(); ++l) {
        const Level& fine = levels_[l];
        const Level& coarse = levels_[l + 1];
        const dim3 grid = grid_for(coarse.width, coarse.height);
        downsample_kernel<C><<<grid, kBlock, 0, stream>>>(fine.gaussian.get(), fine.width,
                                                          fine.height, coarse.gaussian.get(),
                                                          coarse.width, coarse.height);
        downsample_kernel<1><<<grid, kBlock, 0, stream>>>(
            l == 0 ? weights_[k].get() : fine.weight.get(), fine.width, fine.height,
            coarse.weight.get(), coarse.width, coarse.height);
      }
      for (std::size_t l = 0; l < levels_.size(); ++l) {
        const Level& level = levels_[l];
        const Level* coarse = l < top ? &levels_[l + 1] : nullptr;
        accumulate_kernel<C><<<grid_for(level.width, level.height), kBlock, 0, stream>>>(
            level.gaussian.get(), coarse != nullptr ? coarse->gaussian.get() : nullptr,
            coarse != nullptr ? coarse->width : 0, coarse != nullptr ? coarse->height : 0,
            l == 0 ? weights_[k].get() : level.weight.get(), level.blended.get(), level.width,
            level.height, k == 0);
      }
    }
    for (std::size_t l = top; l-- > 0;) {
      const Level& level = levels_[l];
      const Level& coarse = levels_[l + 1];
      collapse_kernel<C><<<grid_for(level.width, level.height), kBlock, 0, stream>>>(
          coarse.blended.get(), coarse.width, coarse.height, level.blended.get(), level.width,
          level.height);
    }
    fusion_output_kernel<C, S><<<grid_for(width, height), kBlock, 0, stream>>>(
        levels_[0].blended.get(), width, height, output.planes[0].data, output.planes[0].pitch);
  }

  FusionParams params_;
  int width_{};
  int height_{};
  int channels_{};
  std::vector<DeviceBuffer<float>> weights_;
  std::vector<Level> levels_;
};

ExposureFusion::ExposureFusion(const FusionParams& params) {
  validate_fusion_params(params);
  impl_ = std::make_unique<Impl>(params);
}
ExposureFusion::~ExposureFusion() = default;

void ExposureFusion::fuse(std::span<const ImageView> brackets, const ImageView& output,
                          cudaStream_t stream) {
  impl_->fuse(brackets, output, stream);
}

class ToneMapper::Impl {
 public:
  explicit Impl(const ToneMapParams& params)
      : params_(params),
        log_sums_(allocate<double>(kStatsBlocks)),
        maxima_(allocate<float>(kStatsBlocks)),
        stats_(allocate<double>(2)) {}

  void map(const HdrImageView& input, const ImageView& output, cudaStream_t stream) {
    const int channels = hdr_channels(input.format);
    const PixelFormat expected = channels == 1   ? PixelFormat::kMono8
                                 : channels == 3 ? PixelFormat::kRgb8
                                                 : PixelFormat::kRgba8;
    if (input.width < 1 || input.height < 1 || input.data == nullptr) {
      throw std::invalid_argument("the input has no pixels");
    }
    const std::size_t sample = hdr_is_float(input.format) ? sizeof(float) : sizeof(std::uint16_t);
    if (input.pitch < static_cast<std::size_t>(input.width) * channels * sample ||
        input.pitch % sample != 0 || reinterpret_cast<std::uintptr_t>(input.data) % sample != 0) {
      throw std::invalid_argument("the input's rows are too short or its samples misaligned");
    }
    if (!hdr_is_float(input.format) &&
        (input.significant_bits < 1 || input.significant_bits > 16)) {
      throw std::invalid_argument("significant_bits must be in 1..16");
    }
    check_bracket(output, "the output");
    if (output.format != expected || output.width != input.width || output.height != input.height) {
      throw std::invalid_argument(std::string("the output must be ") +
                                  std::string(format_name(expected)) + " of the input's size");
    }
    const HdrArgs a{.width = input.width,
                    .height = input.height,
                    .data = static_cast<const std::uint8_t*>(input.data),
                    .pitch = input.pitch,
                    .channels = channels,
                    .is_float = hdr_is_float(input.format),
                    .white_code = static_cast<float>((1U << input.significant_bits) - 1U)};
    stats_partials_kernel<<<kStatsBlocks, kStatsThreads, 0, stream>>>(a, log_sums_.get(),
                                                                      maxima_.get());
    stats_finalize_kernel<<<1, kStatsThreads, 0, stream>>>(
        log_sums_.get(), maxima_.get(), static_cast<std::int64_t>(input.width) * input.height,
        stats_.get());
    const MapArgs m{.key = params_.key,
                    .white = params_.white,
                    .curve = params_.curve,
                    .inv_gamma = 1.0F / params_.gamma,
                    .out_stride = pixel_bytes(output.format)};
    tone_map_kernel<<<grid_for(input.width, input.height), kBlock, 0, stream>>>(
        a, m, stats_.get(), output.planes[0].data, output.planes[0].pitch);
    check_cuda(cudaGetLastError(), "tone map kernels");
    mapped_ = true;
  }

  ToneMapStats last_stats(cudaStream_t stream) {
    if (!mapped_) return {};
    double stats[2] = {};
    check_cuda(cudaMemcpyAsync(stats, stats_.get(), sizeof(stats), cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    return ToneMapStats{.log_average = stats[0], .max_luminance = stats[1]};
  }

 private:
  ToneMapParams params_;
  DeviceBuffer<double> log_sums_;
  DeviceBuffer<float> maxima_;
  DeviceBuffer<double> stats_;
  bool mapped_{false};
};

ToneMapper::ToneMapper(const ToneMapParams& params) {
  validate_tone_map_params(params);
  impl_ = std::make_unique<Impl>(params);
}
ToneMapper::~ToneMapper() = default;

void ToneMapper::map(const HdrImageView& input, const ImageView& output, cudaStream_t stream) {
  impl_->map(input, output, stream);
}

ToneMapStats ToneMapper::last_stats(cudaStream_t stream) { return impl_->last_stats(stream); }

}  // namespace holoscan::denoise
