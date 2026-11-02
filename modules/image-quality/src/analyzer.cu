// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The kernels gather what the CPU reference gathers (reference.cpp), with the
// same integer arithmetic, so the statistics match it exactly:
//   1. luma: per tile part, luma (kept for the next frame's difference), the
//      color sums, the frame hash, the difference to the previous frame, and
//      the luma histogram;
//   2. stencils: per tile part, on interior pixels, the Laplacian and Sobel sums
//      and the histogram of |gx| + |gy|;
//   3. threshold: one block finds the flat pixels' gradient threshold;
//   4. flat: the histogram of the high-pass over the flat pixels.
// Each part's sums are reduced in its block with CUB and written to its own
// slot; histograms are counted in shared memory and added with integer atomics.
// Integer sums do not depend on their order, so neither does the result.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <cub/block/block_reduce.cuh>  // NOLINT(build/include_order)
#include <cub/block/block_scan.cuh>    // NOLINT(build/include_order)
#include <holoscan/image_quality/reference.hpp>

#include "holoscan/image_quality/analyzer.hpp"

namespace holoscan::image_quality {
namespace {

constexpr int kThreads = 256;
// The results buffer: the three histograms, the flat threshold and a pad, in
// 32-bit words, then each part's TileStats.
constexpr std::size_t kLumaOffset = 0;
constexpr std::size_t kGradientOffset = kLumaOffset + kLumaBins;
constexpr std::size_t kHighPassOffset = kGradientOffset + kGradientBins;
constexpr std::size_t kThresholdOffset = kHighPassOffset + kHighPassBins;
constexpr std::size_t kHeaderWords = kThresholdOffset + 2;
constexpr std::size_t kHeaderBytes = kHeaderWords * sizeof(std::uint32_t);
static_assert(kHeaderBytes % alignof(TileStats) == 0);
static_assert(std::is_trivially_copyable_v<TileStats>);
// Most blocks a tile is split into.
constexpr int kMaxParts = 64;

struct Geometry {
  int width;
  int height;
  int tiles_x;
  int tiles_y;
  int parts;
};

// The pixels of part `part` of tile `tile`: [x0, x1) x [y0, y1), a band of the
// tile's rows.
struct Band {
  int x0;
  int x1;
  int y0;
  int y1;
};

__device__ Band band_of(const Geometry& g, int tile, int part) {
  const int tx = tile % g.tiles_x;
  const int ty = tile / g.tiles_x;
  const int ty0 = tile_begin(ty, g.tiles_y, g.height);
  const int ty1 = tile_begin(ty + 1, g.tiles_y, g.height);
  const auto rows = static_cast<std::int64_t>(ty1 - ty0);
  return Band{tile_begin(tx, g.tiles_x, g.width), tile_begin(tx + 1, g.tiles_x, g.width),
              ty0 + static_cast<int>(rows * part / g.parts),
              ty0 + static_cast<int>(rows * (part + 1) / g.parts)};
}

template <typename T>
using BlockSum = cub::BlockReduce<T, kThreads>;

__global__ void luma_kernel(Geometry g, const std::uint8_t* __restrict__ image, std::size_t pitch,
                            int ch, bool bgr, std::uint8_t* __restrict__ luma_out,
                            const std::uint8_t* __restrict__ previous,
                            TileStats* __restrict__ parts,
                            std::uint32_t* __restrict__ luma_histogram) {
  __shared__ union {
    typename BlockSum<std::int64_t>::TempStorage sum;
    typename BlockSum<std::uint64_t>::TempStorage hash;
  } temp;
  __shared__ std::uint32_t histogram[kLumaBins];
  for (int i = threadIdx.x; i < kLumaBins; i += kThreads) histogram[i] = 0U;
  __syncthreads();

  const Band b = band_of(g, blockIdx.x, blockIdx.y);
  const int w = b.x1 - b.x0;
  const auto n = static_cast<std::int64_t>(w) * (b.y1 - b.y0);
  std::int64_t sum_luma = 0;
  std::int64_t sum_red = 0;
  std::int64_t sum_green = 0;
  std::int64_t sum_blue = 0;
  std::int64_t sum_diff = 0;
  std::uint64_t hash = 0;
  for (std::int64_t i = threadIdx.x; i < n; i += kThreads) {
    const int x = b.x0 + static_cast<int>(i % w);
    const int y = b.y0 + static_cast<int>(i / w);
    const std::uint8_t* p =
        image + static_cast<std::size_t>(y) * pitch + static_cast<std::size_t>(x) * ch;
    int r = p[0];
    int gr = p[0];
    int bl = p[0];
    std::uint32_t bytes = p[0];
    int value = p[0];
    if (ch != 1) {
      const int c0 = p[0];
      const int c1 = p[1];
      const int c2 = p[2];
      bytes = static_cast<std::uint32_t>(c0) | (static_cast<std::uint32_t>(c1) << 8U) |
              (static_cast<std::uint32_t>(c2) << 16U);
      if (ch == 4) bytes |= static_cast<std::uint32_t>(p[3]) << 24U;
      r = bgr ? c2 : c0;
      gr = c1;
      bl = bgr ? c0 : c2;
      value = luma(r, gr, bl);
    }
    const std::size_t index = static_cast<std::size_t>(y) * g.width + x;
    luma_out[index] = static_cast<std::uint8_t>(value);
    if (previous != nullptr) {
      sum_diff += abs(value - static_cast<int>(previous[index]));
    }
    sum_luma += value;
    sum_red += r;
    sum_green += gr;
    sum_blue += bl;
    hash += pixel_hash(index, bytes);
    atomicAdd(&histogram[value], 1U);
  }

  const auto reduce = [&](std::int64_t value) {
    const std::int64_t total = BlockSum<std::int64_t>(temp.sum).Sum(value);
    __syncthreads();
    return total;
  };
  sum_luma = reduce(sum_luma);
  sum_red = reduce(sum_red);
  sum_green = reduce(sum_green);
  sum_blue = reduce(sum_blue);
  sum_diff = reduce(sum_diff);
  hash = BlockSum<std::uint64_t>(temp.hash).Sum(hash);
  if (threadIdx.x == 0) {
    TileStats& out = parts[static_cast<std::size_t>(blockIdx.x) * g.parts + blockIdx.y];
    out.hash = hash;
    out.pixels = n;
    out.sum_luma = sum_luma;
    out.sum_red = sum_red;
    out.sum_green = sum_green;
    out.sum_blue = sum_blue;
    out.sum_abs_diff = sum_diff;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kLumaBins; i += kThreads) {
    if (histogram[i] != 0U) atomicAdd(&luma_histogram[i], histogram[i]);
  }
}

// The interior pixels of a band: those whose 3x3 neighborhood is in the image.
__device__ Band interior_of(const Geometry& g, Band b) {
  return Band{max(b.x0, 1), min(b.x1, g.width - 1), max(b.y0, 1), min(b.y1, g.height - 1)};
}

struct Neighborhood {
  int p[3][3];  // [row][column], center at [1][1].
};

__device__ Neighborhood load(const std::uint8_t* __restrict__ luma, int width, int x, int y) {
  Neighborhood n;
  for (int dy = 0; dy < 3; ++dy) {
    const std::uint8_t* row = luma + static_cast<std::size_t>(y + dy - 1) * width + (x - 1);
    for (int dx = 0; dx < 3; ++dx) n.p[dy][dx] = row[dx];
  }
  return n;
}

__device__ int sobel_x(const Neighborhood& n) {
  return (n.p[0][2] + 2 * n.p[1][2] + n.p[2][2]) - (n.p[0][0] + 2 * n.p[1][0] + n.p[2][0]);
}

__device__ int sobel_y(const Neighborhood& n) {
  return (n.p[2][0] + 2 * n.p[2][1] + n.p[2][2]) - (n.p[0][0] + 2 * n.p[0][1] + n.p[0][2]);
}

__global__ void stencil_kernel(Geometry g, const std::uint8_t* __restrict__ luma,
                               TileStats* __restrict__ parts,
                               std::uint32_t* __restrict__ gradient_histogram) {
  __shared__ typename BlockSum<std::int64_t>::TempStorage temp;
  __shared__ std::uint32_t histogram[kGradientBins];
  for (int i = threadIdx.x; i < kGradientBins; i += kThreads) histogram[i] = 0U;
  __syncthreads();

  const Band b = interior_of(g, band_of(g, blockIdx.x, blockIdx.y));
  const int w = max(b.x1 - b.x0, 0);
  const auto n = static_cast<std::int64_t>(w) * max(b.y1 - b.y0, 0);
  std::int64_t sum_lap = 0;
  std::int64_t sum_lap_sq = 0;
  std::int64_t sum_grad_sq = 0;
  for (std::int64_t i = threadIdx.x; i < n; i += kThreads) {
    const int x = b.x0 + static_cast<int>(i % w);
    const int y = b.y0 + static_cast<int>(i / w);
    const Neighborhood nb = load(luma, g.width, x, y);
    const int laplacian = 4 * nb.p[1][1] - nb.p[1][0] - nb.p[1][2] - nb.p[0][1] - nb.p[2][1];
    const int gx = sobel_x(nb);
    const int gy = sobel_y(nb);
    sum_lap += laplacian;
    sum_lap_sq += static_cast<std::int64_t>(laplacian) * laplacian;
    sum_grad_sq += static_cast<std::int64_t>(gx) * gx + static_cast<std::int64_t>(gy) * gy;
    atomicAdd(&histogram[abs(gx) + abs(gy)], 1U);
  }
  const auto reduce = [&](std::int64_t value) {
    const std::int64_t total = BlockSum<std::int64_t>(temp).Sum(value);
    __syncthreads();
    return total;
  };
  sum_lap = reduce(sum_lap);
  sum_lap_sq = reduce(sum_lap_sq);
  sum_grad_sq = reduce(sum_grad_sq);
  if (threadIdx.x == 0) {
    TileStats& out = parts[static_cast<std::size_t>(blockIdx.x) * g.parts + blockIdx.y];
    out.interior = n;
    out.sum_laplacian = sum_lap;
    out.sum_laplacian_sq = sum_lap_sq;
    out.sum_gradient_sq = sum_grad_sq;
  }
  for (int i = threadIdx.x; i < kGradientBins; i += kThreads) {
    if (histogram[i] != 0U) atomicAdd(&gradient_histogram[i], histogram[i]);
  }
}

// One block: the smallest gradient whose cumulative count reaches `fraction` of
// all, as flat_threshold() computes it on the host.
__global__ void threshold_kernel(const std::uint32_t* __restrict__ gradient_histogram,
                                 double fraction, std::uint32_t* __restrict__ threshold) {
  constexpr int kItems = kGradientBins / kThreads;
  using Scan = cub::BlockScan<std::uint64_t, kThreads>;
  __shared__ typename Scan::TempStorage temp;
  __shared__ std::uint64_t cumulative[kGradientBins];
  std::uint64_t items[kItems];
  for (int k = 0; k < kItems; ++k) items[k] = gradient_histogram[threadIdx.x * kItems + k];
  std::uint64_t total = 0;
  Scan(temp).InclusiveSum(items, items, total);
  for (int k = 0; k < kItems; ++k) cumulative[threadIdx.x * kItems + k] = items[k];
  __syncthreads();
  const auto target = static_cast<std::uint64_t>(ceil(fraction * static_cast<double>(total)));
  // Exactly one bin reaches the target first.
  for (int k = 0; k < kItems; ++k) {
    const int bin = threadIdx.x * kItems + k;
    if (cumulative[bin] >= target && (bin == 0 || cumulative[bin - 1] < target)) {
      *threshold = static_cast<std::uint32_t>(bin);
    }
  }
}

__global__ void flat_kernel(Geometry g, const std::uint8_t* __restrict__ luma,
                            const std::uint32_t* __restrict__ threshold,
                            std::uint32_t* __restrict__ high_pass_histogram) {
  __shared__ std::uint32_t histogram[kHighPassBins];
  for (int i = threadIdx.x; i < kHighPassBins; i += kThreads) histogram[i] = 0U;
  __syncthreads();
  const int limit = static_cast<int>(*threshold);
  const Band b = interior_of(g, band_of(g, blockIdx.x, blockIdx.y));
  const int w = max(b.x1 - b.x0, 0);
  const auto n = static_cast<std::int64_t>(w) * max(b.y1 - b.y0, 0);
  for (std::int64_t i = threadIdx.x; i < n; i += kThreads) {
    const int x = b.x0 + static_cast<int>(i % w);
    const int y = b.y0 + static_cast<int>(i / w);
    const Neighborhood nb = load(luma, g.width, x, y);
    if (abs(sobel_x(nb)) + abs(sobel_y(nb)) > limit) continue;
    const int high_pass = (nb.p[0][0] + nb.p[0][2] + nb.p[2][0] + nb.p[2][2]) -
                          2 * (nb.p[0][1] + nb.p[1][0] + nb.p[1][2] + nb.p[2][1]) + 4 * nb.p[1][1];
    atomicAdd(&histogram[abs(high_pass)], 1U);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kHighPassBins; i += kThreads) {
    if (histogram[i] != 0U) atomicAdd(&high_pass_histogram[i], histogram[i]);
  }
}

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("image quality: ") + what + ": " +
                             cudaGetErrorString(status));
  }
}

// A device or pinned host allocation, grown as needed.
template <bool kPinned>
struct Allocation {
  void* data{};
  std::size_t bytes{};

  Allocation() = default;
  Allocation(const Allocation&) = delete;
  Allocation& operator=(const Allocation&) = delete;
  ~Allocation() { release(); }

  void release() noexcept {
    if (data != nullptr) {
      if constexpr (kPinned) {
        cudaFreeHost(data);
      } else {
        cudaFree(data);
      }
    }
    data = nullptr;
    bytes = 0;
  }
  // Grows to at least `size` bytes; the old contents are lost.
  void reserve(std::size_t size) {
    if (bytes >= size) return;
    release();
    if constexpr (kPinned) {
      check(cudaMallocHost(&data, size), "cudaMallocHost");
    } else {
      check(cudaMalloc(&data, size), "cudaMalloc");
    }
    bytes = size;
  }
  template <typename T>
  [[nodiscard]] T* as() const noexcept {
    return static_cast<T*>(data);
  }
};

}  // namespace

struct GpuStats::Buffers {
  Allocation<false> staging;  // A host image's copy.
  Allocation<false> luma[2];  // This frame's and the previous frame's luma.
  Allocation<false> results;  // Histograms, threshold and parts.
  Allocation<true> host;      // The results, copied back.
  int current{};              // Which luma buffer this frame writes.
  int previous_width{};
  int previous_height{};
};

GpuStats::GpuStats(AnalyzerOptions options)
    : options_(std::move(options)), buffers_(std::make_unique<Buffers>()) {
  validate_stats_options(options_.stats);
  if (options_.parts_per_tile < 0 || options_.parts_per_tile > kMaxParts) {
    throw std::invalid_argument("parts_per_tile must be 0..64");
  }
  int device = 0;
  check(cudaGetDevice(&device), "cudaGetDevice");
  check(cudaDeviceGetAttribute(&multiprocessors_, cudaDevAttrMultiProcessorCount, device),
        "cudaDeviceGetAttribute");
}

GpuStats::~GpuStats() = default;

void GpuStats::reset() noexcept { buffers_->previous_width = buffers_->previous_height = 0; }

int GpuStats::parts_per_tile(int width, int height) const noexcept {
  const TileGrid grid = tile_grid(width, height, options_.stats.tiles_x, options_.stats.tiles_y);
  const int rows = std::max(1, height / grid.tiles_y);
  if (options_.parts_per_tile > 0) {
    return std::min(options_.parts_per_tile, rows);
  }
  // Four blocks per multiprocessor keep the GPU busy at any tile count.
  const int tiles = grid.tiles_x * grid.tiles_y;
  const int wanted = (4 * std::max(multiprocessors_, 1) + tiles - 1) / tiles;
  return std::clamp(wanted, 1, std::min(rows, kMaxParts));
}

FrameStats GpuStats::gather(const ImageView& image, bool on_device, cudaStream_t stream) {
  check_image(image);
  Buffers& buf = *buffers_;
  const int width = image.width;
  const int height = image.height;
  const int ch = channels(image.format);
  const TileGrid grid = tile_grid(width, height, options_.stats.tiles_x, options_.stats.tiles_y);
  const int parts = parts_per_tile(width, height);
  const int tiles = grid.tiles_x * grid.tiles_y;
  const auto pixels = static_cast<std::size_t>(width) * height;
  const std::size_t part_count = static_cast<std::size_t>(tiles) * parts;
  const std::size_t results_bytes = kHeaderBytes + part_count * sizeof(TileStats);

  const bool has_previous = buf.previous_width == width && buf.previous_height == height;
  buf.luma[0].reserve(pixels);
  buf.luma[1].reserve(pixels);
  buf.results.reserve(results_bytes);
  buf.host.reserve(results_bytes);

  const std::uint8_t* source = image.data;
  std::size_t pitch = image.pitch;
  if (!on_device) {
    const std::size_t row = static_cast<std::size_t>(width) * ch;
    buf.staging.reserve(row * height);
    check(cudaMemcpy2DAsync(buf.staging.data, row, image.data, image.pitch, row, height,
                            cudaMemcpyHostToDevice, stream),
          "cudaMemcpy2DAsync");
    source = buf.staging.as<std::uint8_t>();
    pitch = row;
  }

  auto* header = buf.results.as<std::uint32_t>();
  auto* part_stats = reinterpret_cast<TileStats*>(buf.results.as<std::uint8_t>() + kHeaderBytes);
  std::uint8_t* luma_plane = buf.luma[buf.current].as<std::uint8_t>();
  const std::uint8_t* previous =
      has_previous ? buf.luma[1 - buf.current].as<std::uint8_t>() : nullptr;
  const Geometry geometry{width, height, grid.tiles_x, grid.tiles_y, parts};
  const dim3 blocks(static_cast<unsigned>(tiles), static_cast<unsigned>(parts));
  const bool bgr = image.format == PixelFormat::kBgr8 || image.format == PixelFormat::kBgra8;

  check(cudaMemsetAsync(header, 0, kHeaderBytes, stream), "cudaMemsetAsync");
  luma_kernel<<<blocks, kThreads, 0, stream>>>(geometry, source, pitch, ch, bgr, luma_plane,
                                               previous, part_stats, header + kLumaOffset);
  check(cudaGetLastError(), "luma kernel");
  stencil_kernel<<<blocks, kThreads, 0, stream>>>(geometry, luma_plane, part_stats,
                                                  header + kGradientOffset);
  check(cudaGetLastError(), "stencil kernel");
  threshold_kernel<<<1, kThreads, 0, stream>>>(
      header + kGradientOffset, options_.stats.flat_fraction, header + kThresholdOffset);
  check(cudaGetLastError(), "threshold kernel");
  flat_kernel<<<blocks, kThreads, 0, stream>>>(geometry, luma_plane, header + kThresholdOffset,
                                               header + kHighPassOffset);
  check(cudaGetLastError(), "flat kernel");
  check(cudaMemcpyAsync(buf.host.data, buf.results.data, results_bytes, cudaMemcpyDeviceToHost,
                        stream),
        "cudaMemcpyAsync");
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

  FrameStats stats;
  stats.width = width;
  stats.height = height;
  stats.tiles_x = grid.tiles_x;
  stats.tiles_y = grid.tiles_y;
  stats.color = ch != 1;
  stats.has_previous = has_previous;
  const auto* words = buf.host.as<std::uint32_t>();
  std::copy_n(words + kLumaOffset, kLumaBins, stats.luma_histogram.begin());
  stats.gradient_histogram.assign(words + kGradientOffset, words + kGradientOffset + kGradientBins);
  stats.high_pass_histogram.assign(words + kHighPassOffset,
                                   words + kHighPassOffset + kHighPassBins);
  stats.flat_threshold = words[kThresholdOffset];
  const std::uint8_t* host_parts = buf.host.as<std::uint8_t>() + kHeaderBytes;
  stats.tiles.assign(static_cast<std::size_t>(tiles), TileStats{});
  for (std::size_t tile = 0; tile < static_cast<std::size_t>(tiles); ++tile) {
    for (int part = 0; part < parts; ++part) {
      TileStats sums;
      std::memcpy(&sums, host_parts + (tile * parts + part) * sizeof(TileStats), sizeof(sums));
      stats.tiles[tile] += sums;
    }
  }

  buf.current = 1 - buf.current;
  buf.previous_width = width;
  buf.previous_height = height;
  return stats;
}

ImageQualityAnalyzer::ImageQualityAnalyzer(AnalyzerOptions options, QualityThresholds thresholds)
    : stats_(std::move(options)), evaluator_(std::move(thresholds)) {}

QualityReport ImageQualityAnalyzer::analyze(const ImageView& image, bool on_device,
                                            cudaStream_t stream) {
  return evaluator_.evaluate(stats_.gather(image, on_device, stream));
}

void ImageQualityAnalyzer::reset() noexcept {
  stats_.reset();
  evaluator_.reset();
}

}  // namespace holoscan::image_quality
