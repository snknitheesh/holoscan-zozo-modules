// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The cuda backend: every step of detection_core.hpp in a kernel, a thread per pixel or tile,
// a warp per candidate or chessboard corner. Components are joined by a lock-free union-find whose
// roots are each component's lowest pixel index, as on the CPU; candidates are numbered in whatever
// order threads reach them, and collect_markers() sorts the markers.

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/fiducials/charuco.hpp>
#include <holoscan/fiducials/detector_backend.hpp>
#include <holoscan/fiducials/dictionary.hpp>

#include "backends.hpp"
#include "cpu_detector.hpp"  // make_params, rotated_codes, collect_markers
#include "cuda_buffers.hpp"
#include "detection_core.hpp"

namespace holoscan::fiducials {
namespace {

using core::Candidate;
using core::CandidateResult;
using core::Image;
using core::Params;

constexpr int kBlock = 256;

unsigned int blocks_for(std::size_t items) {
  return static_cast<unsigned int>((items + kBlock - 1) / kBlock);
}

// A warp's 32 lanes, sharing one candidate's or chessboard corner's work (core::SerialLanes
// on the CPU, in the same order).
struct WarpLanes {
  int lane;

  __device__ static double exchange(double value, int offset) {
    return __shfl_xor_sync(0xFFFFFFFFU, value, offset);
  }
  __device__ static int exchange(int value, int offset) {
    return __shfl_xor_sync(0xFFFFFFFFU, value, offset);
  }
  __device__ static core::Moments exchange(const core::Moments& m, int offset) {
    return {exchange(m.w, offset),     exchange(m.x, offset),  exchange(m.y, offset),
            exchange(m.xx, offset),    exchange(m.xy, offset), exchange(m.yy, offset),
            exchange(m.points, offset)};
  }
  __device__ static core::SaddleSums exchange(const core::SaddleSums& s, int offset) {
    return {exchange(s.a, offset), exchange(s.b, offset), exchange(s.c, offset),
            exchange(s.bx, offset), exchange(s.by, offset)};
  }

  template <typename T, typename Part>
  __device__ T sum(const Part& part) const {
    T value = part(lane);
    for (int offset = core::kLanes / 2; offset > 0; offset /= 2) {
      value = value + exchange(value, offset);
    }
    return value;
  }
  template <typename Part>
  __device__ core::Key min(const Part& part) const {
    core::Key value = part(lane);
    for (int offset = core::kLanes / 2; offset > 0; offset /= 2) {
      const core::Key other = __shfl_xor_sync(0xFFFFFFFFU, value, offset);
      value = other < value ? other : value;
    }
    return value;
  }
  template <typename Value>
  __device__ void fill(double* out, int count, const Value& value) const {
    __syncwarp();  // Every lane is done reading what is there.
    for (int k = lane; k < count; k += core::kLanes) out[k] = value(k);
    __syncwarp();
  }
};

// Warps per block for the per-candidate and per-corner kernels.
constexpr int kWarps = 4;

// Frame counters on the device.
struct Counts {
  std::uint32_t candidates;  // Candidates found; those past the capacity were dropped.
};

__global__ void tile_range_kernel(Image image, Params p, std::uint8_t* lo, std::uint8_t* hi) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= p.tiles_x * p.tiles_y) return;
  core::tile_range(image, p.tile_size, t % p.tiles_x, t / p.tiles_x, lo[t], hi[t]);
}

__global__ void neighborhood_kernel(Params p, const std::uint8_t* lo, const std::uint8_t* hi,
                                    std::uint8_t* near_lo, std::uint8_t* near_hi) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= p.tiles_x * p.tiles_y) return;
  core::neighborhood_range(lo, hi, p.tiles_x, p.tiles_y, t % p.tiles_x, t / p.tiles_x, near_lo[t],
                           near_hi[t]);
}

// Each dark pixel becomes its own component.
__global__ void threshold_kernel(Image image, Params p, const std::uint8_t* near_lo,
                                 const std::uint8_t* near_hi, std::int32_t* labels) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.width * p.height) return;
  const int x = i % p.width;
  const int y = i / p.width;
  const int t = (y / p.tile_size) * p.tiles_x + x / p.tile_size;
  const int value = image.data[static_cast<std::size_t>(y) * image.pitch + x];
  labels[i] = core::is_dark(value, near_lo[t], near_hi[t], p.min_contrast) ? i : -1;
}

__device__ int find_root(const std::int32_t* labels, int node) {
  const volatile std::int32_t* parents = labels;
  int parent = parents[node];
  while (parent != node) {
    node = parent;
    parent = parents[node];
  }
  return node;
}

// Points the larger root at the smaller with atomicMin, retrying from the new root if another
// thread moved it first.
__device__ void unite(std::int32_t* labels, int a, int b) {
  while (true) {
    a = find_root(labels, a);
    b = find_root(labels, b);
    if (a == b) return;
    if (a > b) {
      const int swap = a;
      a = b;
      b = swap;
    }
    const int previous = atomicMin(&labels[b], a);
    if (previous == b) return;
    b = previous;
  }
}

// 8-connectivity: each dark pixel joins its dark left, upper-left, upper and upper-right
// neighbors.
__global__ void union_kernel(Params p, std::int32_t* labels) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.width * p.height || labels[i] < 0) return;
  const int x = i % p.width;
  const int y = i / p.width;
  if (x > 0 && labels[i - 1] >= 0) unite(labels, i, i - 1);
  if (y == 0) return;
  const int up = i - p.width;
  if (x > 0 && labels[up - 1] >= 0) unite(labels, i, up - 1);
  if (labels[up] >= 0) unite(labels, i, up);
  if (x + 1 < p.width && labels[up + 1] >= 0) unite(labels, i, up + 1);
}

__global__ void flatten_kernel(Params p, std::int32_t* labels, std::int32_t* sizes) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.width * p.height || labels[i] < 0) return;
  const int root = find_root(labels, i);
  labels[i] = root;
  atomicAdd(&sizes[root], 1);
}

// Gives each large enough component a candidate slot, or -1, in place of its size.
__global__ void slot_kernel(Params p, const std::int32_t* labels, std::int32_t* slots,
                            Candidate* candidates, Counts* counts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.width * p.height || labels[i] != i) return;
  const auto size = static_cast<std::uint32_t>(slots[i]);
  slots[i] = -1;
  if (size < p.min_component_pixels) return;
  const std::uint32_t slot = atomicAdd(&counts->candidates, 1U);
  if (slot >= core::kMaxCandidates) return;
  core::init_candidate(candidates[slot], i);
  candidates[slot].pixels = size;
  slots[i] = static_cast<std::int32_t>(slot);
}

// The pixel's candidate, or null.
__device__ Candidate* candidate_of(int i, const std::int32_t* labels, const std::int32_t* slots,
                                   Candidate* candidates) {
  const int label = labels[i];
  if (label < 0) return nullptr;
  const int slot = slots[label];
  return slot < 0 ? nullptr : candidates + slot;
}

__global__ void stats_kernel(Params p, const std::int32_t* labels, const std::int32_t* slots,
                             Candidate* candidates) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.width * p.height) return;
  Candidate* c = candidate_of(i, labels, slots, candidates);
  if (c == nullptr) return;
  const int x = i % p.width;
  const int y = i / p.width;
  atomicAdd(&c->sum_x, static_cast<core::Key>(x));
  atomicAdd(&c->sum_y, static_cast<core::Key>(y));
  atomicMin(&c->min_x, x);
  atomicMin(&c->min_y, y);
  atomicMax(&c->max_x, x);
  atomicMax(&c->max_y, y);
}

__global__ void far0_kernel(Params p, const std::int32_t* labels, const std::int32_t* slots,
                            Candidate* candidates) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.width * p.height) return;
  Candidate* c = candidate_of(i, labels, slots, candidates);
  if (c == nullptr) return;
  atomicMax(&c->far0, core::far0_key(*c, i % p.width, i / p.width, i));
}

__global__ void far2_kernel(Params p, const std::int32_t* labels, const std::int32_t* slots,
                            Candidate* candidates) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.width * p.height) return;
  Candidate* c = candidate_of(i, labels, slots, candidates);
  if (c == nullptr) return;
  atomicMax(&c->far2, core::far2_key(*c, p.width, i % p.width, i / p.width, i));
}

__global__ void side_kernel(Params p, const std::int32_t* labels, const std::int32_t* slots,
                            Candidate* candidates) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.width * p.height) return;
  Candidate* c = candidate_of(i, labels, slots, candidates);
  if (c == nullptr) return;
  const std::int64_t side = core::side_of(*c, p.width, i % p.width, i / p.width);
  if (side > 0) {
    atomicMax(&c->side_pos, core::make_key(static_cast<std::uint32_t>(side), i));
  } else if (side < 0) {
    atomicMax(&c->side_neg, core::make_key(static_cast<std::uint32_t>(-side), i));
  }
}

// A warp per candidate.
__global__ void examine_kernel(Image image, Params p, const Candidate* candidates,
                               const Counts* counts, const std::uint64_t* codes,
                               CandidateResult* results) {
  __shared__ double scratch[kWarps][core::kScratch];
  const int warp = static_cast<int>(threadIdx.x) / core::kLanes;
  const std::uint32_t s = blockIdx.x * kWarps + warp;
  const std::uint32_t found =
      counts->candidates < core::kMaxCandidates ? counts->candidates : core::kMaxCandidates;
  if (s >= found) return;  // The whole warp.
  const WarpLanes lanes{static_cast<int>(threadIdx.x) % core::kLanes};
  const CandidateResult result =
      core::examine(lanes, candidates[s], image, p, codes, scratch[warp]);
  if (lanes.lane == 0) results[s] = result;
}

// A chessboard corner to refine, as plain data for the device.
struct Seed {
  double x;
  double y;
  std::int32_t half_window;
};

// A chessboard corner refined, with whether it was found.
struct Saddle {
  double x;
  double y;
  std::int32_t found;
};

// A warp per chessboard corner.
__global__ void saddle_kernel(Image image, int min_contrast, const Seed* seeds, int count,
                              Saddle* out) {
  __shared__ double scratch[kWarps][core::kScratch];
  const int warp = static_cast<int>(threadIdx.x) / core::kLanes;
  const int s = static_cast<int>(blockIdx.x) * kWarps + warp;
  if (s >= count) return;  // The whole warp.
  const WarpLanes lanes{static_cast<int>(threadIdx.x) % core::kLanes};
  double x = seeds[s].x;
  double y = seeds[s].y;
  const bool found =
      core::refine_saddle(lanes, image, seeds[s].half_window, min_contrast, scratch[warp], x, y);
  if (lanes.lane == 0) out[s] = Saddle{x, y, found ? 1 : 0};
}

class CudaBackend final : public DetectorBackend {
 public:
  CudaBackend(const DetectorOptions& options, int width, int height)
      : params_(make_params(options, width, height)) {
    const std::vector<std::uint64_t> codes = rotated_codes(Dictionary(options.dictionary));
    const auto pixels = static_cast<std::size_t>(width) * height;
    const auto tiles = static_cast<std::size_t>(params_.tiles_x) * params_.tiles_y;
    tile_lo_ = DeviceBuffer<std::uint8_t>(tiles);
    tile_hi_ = DeviceBuffer<std::uint8_t>(tiles);
    near_lo_ = DeviceBuffer<std::uint8_t>(tiles);
    near_hi_ = DeviceBuffer<std::uint8_t>(tiles);
    labels_ = DeviceBuffer<std::int32_t>(pixels);
    slots_ = DeviceBuffer<std::int32_t>(pixels);
    candidates_ = DeviceBuffer<Candidate>(core::kMaxCandidates);
    results_ = DeviceBuffer<CandidateResult>(core::kMaxCandidates);
    counts_ = DeviceBuffer<Counts>(1);
    codes_ = DeviceBuffer<std::uint64_t>(codes.size());
    host_counts_ = PinnedBuffer<Counts>(1);
    host_results_ = PinnedBuffer<CandidateResult>(core::kMaxCandidates);
    cuda_check(cudaMemcpy(codes_.get(), codes.data(), codes.size() * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(codes)");
  }

  [[nodiscard]] std::string_view name() const noexcept override { return "cuda"; }
  [[nodiscard]] MemoryLocation input_location() const noexcept override {
    return MemoryLocation::kDevice;
  }

  FrameMarkers detect(const GrayImageView& view, cudaStream_t stream) override {
    check(view);
    const Params& p = params_;
    const Image image{view.data, view.width, view.height, view.pitch};
    const auto pixels = static_cast<std::size_t>(p.width) * p.height;
    const auto tiles = static_cast<std::size_t>(p.tiles_x) * p.tiles_y;
    tile_range_kernel<<<blocks_for(tiles), kBlock, 0, stream>>>(image, p, tile_lo_.get(),
                                                                tile_hi_.get());
    neighborhood_kernel<<<blocks_for(tiles), kBlock, 0, stream>>>(p, tile_lo_.get(), tile_hi_.get(),
                                                                  near_lo_.get(), near_hi_.get());
    threshold_kernel<<<blocks_for(pixels), kBlock, 0, stream>>>(image, p, near_lo_.get(),
                                                                near_hi_.get(), labels_.get());
    union_kernel<<<blocks_for(pixels), kBlock, 0, stream>>>(p, labels_.get());
    cuda_check(cudaMemsetAsync(slots_.get(), 0, pixels * sizeof(std::int32_t), stream),
               "cudaMemsetAsync(slots)");
    cuda_check(cudaMemsetAsync(counts_.get(), 0, sizeof(Counts), stream),
               "cudaMemsetAsync(counts)");
    flatten_kernel<<<blocks_for(pixels), kBlock, 0, stream>>>(p, labels_.get(), slots_.get());
    slot_kernel<<<blocks_for(pixels), kBlock, 0, stream>>>(p, labels_.get(), slots_.get(),
                                                           candidates_.get(), counts_.get());
    stats_kernel<<<blocks_for(pixels), kBlock, 0, stream>>>(p, labels_.get(), slots_.get(),
                                                            candidates_.get());
    far0_kernel<<<blocks_for(pixels), kBlock, 0, stream>>>(p, labels_.get(), slots_.get(),
                                                           candidates_.get());
    far2_kernel<<<blocks_for(pixels), kBlock, 0, stream>>>(p, labels_.get(), slots_.get(),
                                                           candidates_.get());
    side_kernel<<<blocks_for(pixels), kBlock, 0, stream>>>(p, labels_.get(), slots_.get(),
                                                           candidates_.get());
    examine_kernel<<<core::kMaxCandidates / kWarps, kWarps * core::kLanes, 0, stream>>>(
        image, p, candidates_.get(), counts_.get(), codes_.get(), results_.get());
    cuda_check(cudaGetLastError(), "cuda backend kernels");
    cuda_check(cudaMemcpyAsync(host_counts_.get(), counts_.get(), sizeof(Counts),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync(counts)");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    const std::uint32_t total = host_counts_.get()->candidates;
    const std::uint32_t examined = total < core::kMaxCandidates ? total : core::kMaxCandidates;
    if (examined > 0) {
      cuda_check(
          cudaMemcpyAsync(host_results_.get(), results_.get(), examined * sizeof(CandidateResult),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpyAsync(results)");
      cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
    return collect_markers({host_results_.get(), examined}, total - examined);
  }

  std::vector<std::optional<Point2>> refine_saddles(const GrayImageView& view,
                                                    std::span<const SaddleSeed> seeds,
                                                    cudaStream_t stream) override {
    check(view);
    std::vector<std::optional<Point2>> refined;
    if (seeds.empty()) return refined;
    if (seeds.size() > kMaxCharucoCorners) {
      throw std::invalid_argument("cuda backend: more than " + std::to_string(kMaxCharucoCorners) +
                                  " chessboard corners to refine");
    }
    if (!saddle_seeds_.get()) {
      saddle_seeds_ = DeviceBuffer<Seed>(kMaxCharucoCorners);
      saddles_ = DeviceBuffer<Saddle>(kMaxCharucoCorners);
      host_seeds_ = PinnedBuffer<Seed>(kMaxCharucoCorners);
      host_saddles_ = PinnedBuffer<Saddle>(kMaxCharucoCorners);
    }
    const auto count = static_cast<int>(seeds.size());
    for (int s = 0; s < count; ++s) {
      host_seeds_.get()[s] = Seed{seeds[s].position[0], seeds[s].position[1], seeds[s].half_window};
    }
    cuda_check(cudaMemcpyAsync(saddle_seeds_.get(), host_seeds_.get(), count * sizeof(Seed),
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync(seeds)");
    const Image image{view.data, view.width, view.height, view.pitch};
    saddle_kernel<<<(count + kWarps - 1) / kWarps, kWarps * core::kLanes, 0, stream>>>(
        image, params_.min_contrast, saddle_seeds_.get(), count, saddles_.get());
    cuda_check(cudaGetLastError(), "saddle_kernel");
    cuda_check(cudaMemcpyAsync(host_saddles_.get(), saddles_.get(), count * sizeof(Saddle),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync(saddles)");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    refined.reserve(seeds.size());
    for (int s = 0; s < count; ++s) {
      const Saddle& saddle = host_saddles_.get()[s];
      if (saddle.found != 0) {
        refined.emplace_back(Point2{saddle.x, saddle.y});
      } else {
        refined.emplace_back(std::nullopt);
      }
    }
    return refined;
  }

 private:
  void check(const GrayImageView& view) const {
    if (view.location != MemoryLocation::kDevice || view.width != params_.width ||
        view.height != params_.height || view.data == nullptr) {
      throw std::invalid_argument(
          "cuda backend: the image is not of the configured size and location");
    }
  }

  Params params_;
  DeviceBuffer<std::uint8_t> tile_lo_, tile_hi_, near_lo_, near_hi_;
  DeviceBuffer<std::int32_t> labels_;
  DeviceBuffer<std::int32_t> slots_;  // Per root: its size, then its candidate slot or -1.
  DeviceBuffer<Candidate> candidates_;
  DeviceBuffer<CandidateResult> results_;
  DeviceBuffer<Counts> counts_;
  DeviceBuffer<std::uint64_t> codes_;
  PinnedBuffer<Counts> host_counts_;
  PinnedBuffer<CandidateResult> host_results_;
  // For refine_saddles, made on first use.
  DeviceBuffer<Seed> saddle_seeds_;
  DeviceBuffer<Saddle> saddles_;
  PinnedBuffer<Seed> host_seeds_;
  PinnedBuffer<Saddle> host_saddles_;
};

}  // namespace

std::unique_ptr<DetectorBackend> create_cuda_backend(const DetectorOptions& options, int width,
                                                     int height) {
  return std::make_unique<CudaBackend>(options, width, height);
}

}  // namespace holoscan::fiducials
