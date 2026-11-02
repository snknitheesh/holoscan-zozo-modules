// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpu_pipeline.hpp"
#include "cuda_pipeline.hpp"

namespace holoscan::isp::detail {
namespace {

// The main kernel's block: one tile of 32 x 16 output pixels.
constexpr int kBlockX = 32;
constexpr int kBlockY = 16;
constexpr int kThreads = kBlockX * kBlockY;
// The grid-stride passes.
constexpr int kPassThreads = 256;

template <bool kMalvar, bool kSharpen, bool kStats>
__global__ void __launch_bounds__(kThreads) isp_kernel(FrameParams p, StatsAccum* accum) {
  constexpr int kRadius = kMalvar ? 2 : 1;
  constexpr int kHalo = kRadius + (kSharpen ? 1 : 0);
  constexpr int kTileX = kBlockX + 2 * kHalo;
  constexpr int kTileY = kBlockY + 2 * kHalo;
  __shared__ float tile[kTileY][kTileX];

  const int tid = threadIdx.y * kBlockX + threadIdx.x;
  const int x0 = blockIdx.x * kBlockX;
  const int y0 = blockIdx.y * kBlockY;
  float gains[3];
  for (int k = 0; k < 3; ++k) gains[k] = p.awb != nullptr ? p.awb[k] : p.wb[k];

  // Every sample of the tile and its halo, reflected at the image's edges,
  // preprocessed once.
  for (int i = tid; i < kTileX * kTileY; i += kThreads) {
    const int tx = i % kTileX;
    const int ty = i / kTileX;
    tile[ty][tx] =
        preprocess(p, gains, mirror(x0 + tx - kHalo, p.width), mirror(y0 + ty - kHalo, p.height));
  }
  __syncthreads();

  // RGB at virtual pixel (vx, vy), whose sample is tile[ty][tx].
  const auto demosaic_at = [&](int tx, int ty, int vx, int vy, float rgb[3]) {
    float n[5][5] = {};
#pragma unroll
    for (int dy = -kRadius; dy <= kRadius; ++dy) {
#pragma unroll
      for (int dx = -kRadius; dx <= kRadius; ++dx) {
        n[dy + 2][dx + 2] = tile[ty + dy][tx + dx];
      }
    }
    demosaic<kMalvar>(n, channel_at(p, vx, vy), rgb);
  };

  const int x = x0 + static_cast<int>(threadIdx.x);
  const int y = y0 + static_cast<int>(threadIdx.y);
  float rgb[3];
  if constexpr (kSharpen) {
    // The tile's RGB with a one-pixel border, for the sharpening's neighbours.
    __shared__ float colors[kBlockY + 2][kBlockX + 2][3];
    for (int i = tid; i < (kBlockX + 2) * (kBlockY + 2); i += kThreads) {
      const int sx = i % (kBlockX + 2);
      const int sy = i / (kBlockX + 2);
      demosaic_at(sx + kHalo - 1, sy + kHalo - 1, x0 + sx - 1, y0 + sy - 1, colors[sy][sx]);
    }
    __syncthreads();
    float m[3][3][3];
#pragma unroll
    for (int dy = 0; dy < 3; ++dy) {
#pragma unroll
      for (int dx = 0; dx < 3; ++dx) {
#pragma unroll
        for (int k = 0; k < 3; ++k) m[dy][dx][k] = colors[threadIdx.y + dy][threadIdx.x + dx][k];
      }
    }
    sharpen(m, p.sharpen, rgb);
  } else {
    demosaic_at(threadIdx.x + kHalo, threadIdx.y + kHalo, x, y, rgb);
  }

  AtomicU64 sums[4] = {0, 0, 0, 0};
  if (x < p.width && y < p.height) {
    float linear[3];
    correct_color(p, rgb, linear);
    if constexpr (kStats) {
      for (int k = 0; k < 3; ++k) sums[k] = to_fixed(linear[k]);
      sums[3] = load_raw(p, x, y) >= p.white ? 1U : 0U;
    }
    float toned[3];
    for (int k = 0; k < 3; ++k) toned[k] = tone(p, linear[k]);
    store(p, x, y, toned);
  }

  if constexpr (kStats) {
    // Integer sums, so their order does not matter: the same on every run.
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (int k = 0; k < 4; ++k) sums[k] += __shfl_down_sync(0xFFFFFFFFU, sums[k], offset);
    }
    __shared__ AtomicU64 partial[kThreads / 32][4];
    if ((tid & 31) == 0) {
      for (int k = 0; k < 4; ++k) partial[tid >> 5][k] = sums[k];
    }
    __syncthreads();
    if (tid < 4) {
      AtomicU64 total = 0;
      for (int w = 0; w < kThreads / 32; ++w) total += partial[w][tid];
      atomicAdd(tid < 3 ? &accum->sum[tid] : &accum->saturated, total);
    }
  }
}

template <bool kHistogram>
__global__ void __launch_bounds__(kPassThreads) awb_kernel(FrameParams p, AwbAccum* accum) {
  __shared__ AtomicU64 sums[4];
  __shared__ unsigned int hist[kHistogram ? 3 * kAwbBins : 1];
  if (threadIdx.x < 4) sums[threadIdx.x] = 0;
  if constexpr (kHistogram) {
    for (int i = threadIdx.x; i < 3 * kAwbBins; i += kPassThreads) hist[i] = 0;
  }
  __syncthreads();
  const int quads_x = p.width / 2;
  const std::int64_t quads = static_cast<std::int64_t>(quads_x) * (p.height / 2);
  AtomicU64 local[4] = {0, 0, 0, 0};
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(kPassThreads) + threadIdx.x;
       i < quads; i += static_cast<std::int64_t>(gridDim.x) * kPassThreads) {
    int color[4];
    unsigned int fixed[4];
    int bin[4];
    if (!white_balance_quad(p, static_cast<int>(i % quads_x), static_cast<int>(i / quads_x), color,
                            fixed, bin)) {
      continue;
    }
    for (int k = 0; k < 4; ++k) {
      local[color[k]] += fixed[k];
      if constexpr (kHistogram) atomicAdd(&hist[color[k] * kAwbBins + bin[k]], 1U);
    }
    ++local[3];
  }
  for (int k = 0; k < 4; ++k) {
    if (local[k] != 0) atomicAdd(&sums[k], local[k]);
  }
  __syncthreads();
  if (threadIdx.x < 4 && sums[threadIdx.x] != 0) {
    atomicAdd(threadIdx.x < 3 ? &accum->sum[threadIdx.x] : &accum->quads, sums[threadIdx.x]);
  }
  if constexpr (kHistogram) {
    for (int i = threadIdx.x; i < 3 * kAwbBins; i += kPassThreads) {
      if (hist[i] != 0U) atomicAdd(&accum->hist[i / kAwbBins][i % kAwbBins], hist[i]);
    }
  }
}

__global__ void finish_awb_kernel(const AwbAccum* accum, int mode, float percentile,
                                  float smoothing, AwbState* state) {
  finish_white_balance(*accum, mode, percentile, smoothing, *state);
}

__global__ void __launch_bounds__(kPassThreads) histogram_kernel(FrameParams p, unsigned int* out) {
  __shared__ unsigned int hist[kHistogramChannels * kHistogramBins];
  for (int i = threadIdx.x; i < kHistogramChannels * kHistogramBins; i += kPassThreads) hist[i] = 0;
  __syncthreads();
  const std::int64_t pixels = static_cast<std::int64_t>(p.width) * p.height;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(kPassThreads) + threadIdx.x;
       i < pixels; i += static_cast<std::int64_t>(gridDim.x) * kPassThreads) {
    int q[3];
    output_codes(p, static_cast<int>(i % p.width), static_cast<int>(i / p.width), q);
    atomicAdd(&hist[kHistogramLuma * kHistogramBins + luma_code(q)], 1U);
    atomicAdd(&hist[kHistogramRed * kHistogramBins + q[0]], 1U);
    atomicAdd(&hist[kHistogramGreen * kHistogramBins + q[1]], 1U);
    atomicAdd(&hist[kHistogramBlue * kHistogramBins + q[2]], 1U);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kHistogramChannels * kHistogramBins; i += kPassThreads) {
    if (hist[i] != 0U) atomicAdd(&out[i], hist[i]);
  }
}

__global__ void finish_stats_kernel(FrameParams p, const StatsAccum* accum, AutoExposureParams ae,
                                    FrameExposure exposure, float* out) {
  const float gains[3] = {p.awb != nullptr ? p.awb[0] : p.wb[0],
                          p.awb != nullptr ? p.awb[1] : p.wb[1],
                          p.awb != nullptr ? p.awb[2] : p.wb[2]};
  finish_stats(*accum, static_cast<std::int64_t>(p.width) * p.height, gains, ae, exposure, out);
}

template <bool kMalvar, bool kSharpen, bool kStats>
void launch_isp(const FrameParams& p, StatsAccum* accum, cudaStream_t stream) {
  const dim3 block(kBlockX, kBlockY);
  const dim3 grid((p.width + kBlockX - 1) / kBlockX, (p.height + kBlockY - 1) / kBlockY);
  isp_kernel<kMalvar, kSharpen, kStats><<<grid, block, 0, stream>>>(p, accum);
}

template <bool kMalvar>
void launch_isp(const FrameParams& p, bool sharpen, bool stats, StatsAccum* accum,
                cudaStream_t stream) {
  if (sharpen) {
    stats ? launch_isp<kMalvar, true, true>(p, accum, stream)
          : launch_isp<kMalvar, true, false>(p, accum, stream);
  } else {
    stats ? launch_isp<kMalvar, false, true>(p, accum, stream)
          : launch_isp<kMalvar, false, false>(p, accum, stream);
  }
}

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("IspPipeline (cuda): ") + what + ": " +
                             cudaGetErrorString(status));
  }
}

int pass_blocks(std::int64_t items, int sm_count) {
  const std::int64_t wanted = (items + kPassThreads - 1) / kPassThreads;
  return static_cast<int>(std::clamp<std::int64_t>(wanted, 1, 4LL * sm_count));
}

}  // namespace

CudaPipeline::CudaPipeline(const IspParams& params, const std::vector<float>& lut)
    : params_(params), lut_size_(static_cast<int>(lut.size())) {
  try {
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    check(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device),
          "cudaDeviceGetAttribute");
    check(cudaMalloc(&lut_, lut.size() * sizeof(float)), "cudaMalloc");
    if (params_.lens_shading) {
      check(cudaMalloc(&lsc_, params_.lens_shading->gains.size() * sizeof(float)), "cudaMalloc");
    }
    check(cudaMalloc(&awb_accum_, sizeof(AwbAccum)), "cudaMalloc");
    check(cudaMalloc(&awb_state_, sizeof(AwbState)), "cudaMalloc");
    check(cudaMalloc(&stats_accum_, sizeof(StatsAccum)), "cudaMalloc");
    upload(lut.data(), lut.size() * sizeof(float), lut_);
    if (params_.lens_shading) {
      const std::vector<float>& gains = params_.lens_shading->gains;
      upload(gains.data(), gains.size() * sizeof(float), lsc_);
    }
    reset();
  } catch (...) {
    release();
    throw;
  }
}

CudaPipeline::~CudaPipeline() { release(); }

void CudaPipeline::release() noexcept {
  // cudaFree waits for the work that may still use the buffers.
  cudaFree(lut_);
  cudaFree(lsc_);
  cudaFree(awb_accum_);
  cudaFree(awb_state_);
  cudaFree(stats_accum_);
  lut_ = lsc_ = nullptr;
  awb_accum_ = nullptr;
  awb_state_ = nullptr;
  stats_accum_ = nullptr;
}

void CudaPipeline::reset() {
  const AwbState state = initial_white_balance(params_);
  upload(&state, sizeof(state), awb_state_);
}

void CudaPipeline::upload(const void* source, std::size_t bytes, void* destination) {
  // On a stream of its own, waited for: a plain cudaMemcpy from pageable memory
  // may return before the data has landed, and the frames' non-blocking
  // streams would not wait for it.
  cudaStream_t stream = nullptr;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
  const cudaError_t copied =
      cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream);
  const cudaError_t done = copied == cudaSuccess ? cudaStreamSynchronize(stream) : copied;
  cudaStreamDestroy(stream);
  check(done, "upload");
}

cudaError_t CudaPipeline::process(FrameParams p, const StatsOutput& stats,
                                  const FrameExposure& exposure, cudaStream_t stream) {
  p.lut = lut_;
  p.lut_size = lut_size_;
  if (params_.lens_shading) {
    p.lsc = lsc_;
  }
  // Clear an error an earlier, unrelated call left behind, so the checks below
  // report this frame's launches only.
  static_cast<void>(cudaGetLastError());
  if (params_.white_balance != WhiteBalanceMode::kManual) {
    if (const cudaError_t status = cudaMemsetAsync(awb_accum_, 0, sizeof(AwbAccum), stream);
        status != cudaSuccess) {
      return status;
    }
    const std::int64_t quads = static_cast<std::int64_t>(p.width / 2) * (p.height / 2);
    const int blocks = pass_blocks(quads, sm_count_);
    if (params_.white_balance == WhiteBalanceMode::kWhitePatch) {
      awb_kernel<true><<<blocks, kPassThreads, 0, stream>>>(p, awb_accum_);
    } else {
      awb_kernel<false><<<blocks, kPassThreads, 0, stream>>>(p, awb_accum_);
    }
    finish_awb_kernel<<<1, 1, 0, stream>>>(awb_accum_, static_cast<int>(params_.white_balance),
                                           params_.white_patch_percentile, params_.awb_smoothing,
                                           awb_state_);
    // The gains the estimate leaves in device memory, read by the kernels below.
    p.awb = reinterpret_cast<const float*>(reinterpret_cast<const char*>(awb_state_) +
                                           offsetof(AwbState, gains));
  }
  const bool sums = stats.stats != nullptr;
  if (sums) {
    if (const cudaError_t status = cudaMemsetAsync(stats_accum_, 0, sizeof(StatsAccum), stream);
        status != cudaSuccess) {
      return status;
    }
  }
  const bool sharpen = params_.sharpen > 0.0F;
  if (params_.demosaic == DemosaicMethod::kMalvar) {
    launch_isp<true>(p, sharpen, sums, stats_accum_, stream);
  } else {
    launch_isp<false>(p, sharpen, sums, stats_accum_, stream);
  }
  if (stats.histogram != nullptr) {
    const std::size_t bytes = sizeof(std::uint32_t) * kHistogramChannels * kHistogramBins;
    if (const cudaError_t status = cudaMemsetAsync(stats.histogram, 0, bytes, stream);
        status != cudaSuccess) {
      return status;
    }
    histogram_kernel<<<pass_blocks(static_cast<std::int64_t>(p.width) * p.height, sm_count_),
                       kPassThreads, 0, stream>>>(p, stats.histogram);
  }
  if (sums) {
    finish_stats_kernel<<<1, 1, 0, stream>>>(p, stats_accum_, params_.auto_exposure, exposure,
                                             stats.stats);
  }
  return cudaGetLastError();
}

}  // namespace holoscan::isp::detail
