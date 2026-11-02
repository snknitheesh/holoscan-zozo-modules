// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Euclidean clustering: union-find on the GPU over a graph whose nodes are
// grid cells (kEuclidean) or pixels (kImage).
//
// kEuclidean puts the points in a uniform grid of cells tolerance / sqrt(3) a
// side, slightly less, so that any two points of a cell are within the
// tolerance: a cell is one node. Two cells are linked when some pair of their
// points is within the tolerance; only cells at most two apart on each axis
// can be, and each such pair of cells is tested by one thread, which skips
// cells already joined or whose bounding boxes are too far apart and stops at
// the first close pair. kImage links each pixel to the valid pixels of its
// window that are within the tolerance.
//
// The union-find hooks the larger root under the smaller with atomicCAS and
// halves paths as it goes (as in ECL-CC), so a component's root is its lowest
// node whatever the order of the unions. The components are then counted,
// filtered by size and sorted by (size descending, lowest point index), and
// the points labeled and grouped by label with a stable radix sort. The
// summaries are sums over fixed chunks of each cluster's grouped points,
// reduced in a fixed order, so every result is the same on every run.
//
// Compiled without fused multiply-adds: distances are computed exactly as the
// CPU reference computes them, and a pair of points is linked here exactly
// when it is linked there.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>

#include <cuda/atomic>  // NOLINT(build/include_order): libcu++
#include <holoscan/clustering/clustering.hpp>

#include "device_buffer.hpp"
#include "kernel_common.cuh"

namespace holoscan::clustering {
namespace {

using detail::blocks_for;
using detail::kBlock;
using detail::squared_distance;
using detail::valid_point;

// A cell key: the cell's (x, y, z) index, each biased by kCellRange into 21
// bits, x in the highest. Keys sort like the indices.
constexpr int kCellBits = 21;
constexpr int kKeyBits = 3 * kCellBits;
constexpr std::int32_t kCellRange = std::int32_t{1} << 20;
constexpr std::uint64_t kNoCell = ~std::uint64_t{0};
// The cells at most two apart on each axis that follow a cell in key order:
// half of the 5 x 5 x 5 block around it, without itself.
constexpr int kNeighborOffsets = 62;
// Points per block of the summary sums.
constexpr std::uint32_t kChunk = 2048;
// Shrinks the cell below tolerance / sqrt(3) by far more than rounding can
// add, so that two points of one cell are always within the tolerance.
constexpr double kCellMargin = 1.0 - 1.0e-5;

__device__ inline float nan_value() { return __int_as_float(0x7FC00000); }

// ---- Union-find ---------------------------------------------------------------------------------

__device__ inline int load_relaxed(int* address) {
  return cuda::atomic_ref<int, cuda::thread_scope_device>(*address).load(
      cuda::std::memory_order_relaxed);
}

__device__ inline void store_relaxed(int* address, int value) {
  cuda::atomic_ref<int, cuda::thread_scope_device>(*address).store(value,
                                                                   cuda::std::memory_order_relaxed);
}

// The root of `node`, halving the path on the way. A node's parent is never
// above it, and a root is its own parent.
__device__ inline int representative(int* parent, int node) {
  int current = load_relaxed(parent + node);
  if (current != node) {
    int previous = node;
    int next = 0;
    while (current > (next = load_relaxed(parent + current))) {
      store_relaxed(parent + previous, next);
      previous = current;
      current = next;
    }
  }
  return current;
}

// Joins the components of `a` and `b`: the larger root goes under the smaller.
__device__ inline void unite(int* parent, int a, int b) {
  int ra = representative(parent, a);
  int rb = representative(parent, b);
  while (ra != rb) {
    if (ra < rb) {
      const int old = atomicCAS(parent + rb, rb, ra);
      if (old == rb) {
        return;
      }
      rb = old;
    } else {
      const int old = atomicCAS(parent + ra, ra, rb);
      if (old == ra) {
        return;
      }
      ra = old;
    }
  }
}

__global__ void roots_kernel(const int* __restrict__ parent, std::uint32_t nodes,
                             int* __restrict__ root) {
  const std::uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < nodes) {
    int x = static_cast<int>(n);
    int p = 0;
    while ((p = parent[x]) != x) {
      x = p;
    }
    root[n] = x;
  }
}

// ---- kEuclidean: the grid -----------------------------------------------------------------------

__device__ inline std::uint64_t cell_key(int ix, int iy, int iz) {
  return (static_cast<std::uint64_t>(ix + kCellRange) << (2 * kCellBits)) |
         (static_cast<std::uint64_t>(iy + kCellRange) << kCellBits) |
         static_cast<std::uint64_t>(iz + kCellRange);
}

__device__ inline void decode(std::uint64_t key, int& ix, int& iy, int& iz) {
  constexpr std::uint64_t mask = (std::uint64_t{1} << kCellBits) - 1U;
  ix = static_cast<int>((key >> (2 * kCellBits)) & mask) - kCellRange;
  iy = static_cast<int>((key >> kCellBits) & mask) - kCellRange;
  iz = static_cast<int>(key & mask) - kCellRange;
}

__device__ inline bool in_range(int index) { return index >= -kCellRange && index < kCellRange; }

// In double, so that a point's cell is exact for any float coordinate.
__device__ inline bool cell_index(float value, double cell, int& index) {
  const double c = floor(static_cast<double>(value) / cell);
  if (!(c >= -static_cast<double>(kCellRange) && c < static_cast<double>(kCellRange))) {
    return false;
  }
  index = static_cast<int>(c);
  return true;
}

__global__ void cell_keys_kernel(const float* __restrict__ xyz, std::uint32_t count, double cell,
                                 std::uint64_t* __restrict__ keys,
                                 std::uint32_t* __restrict__ indices) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  indices[i] = i;
  const float x = xyz[3 * i];
  const float y = xyz[3 * i + 1];
  const float z = xyz[3 * i + 2];
  int ix = 0;
  int iy = 0;
  int iz = 0;
  keys[i] = valid_point(x, y, z) && cell_index(x, cell, ix) && cell_index(y, cell, iy) &&
                    cell_index(z, cell, iz)
                ? cell_key(ix, iy, iz)
                : kNoCell;
}

__global__ void heads_kernel(const std::uint64_t* __restrict__ keys, std::uint32_t count,
                             std::uint32_t* __restrict__ heads) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    heads[i] = keys[i] != kNoCell && (i == 0U || keys[i] != keys[i - 1U]) ? 1U : 0U;
  }
}

// Cell c's key and the start of its points; starts[cells] is the end of the
// last cell, and `cells` the number of cells.
__global__ void cells_kernel(const std::uint64_t* __restrict__ keys,
                             const std::uint32_t* __restrict__ heads,
                             const std::uint32_t* __restrict__ offsets, std::uint32_t count,
                             std::uint64_t* __restrict__ cell_keys,
                             std::uint32_t* __restrict__ starts,
                             std::uint32_t* __restrict__ cells) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  if (heads[i] != 0U) {
    cell_keys[offsets[i]] = keys[i];
    starts[offsets[i]] = i;
  }
  const bool last_valid = keys[i] != kNoCell && (i + 1U == count || keys[i + 1U] == kNoCell);
  if (last_valid) {
    starts[offsets[i] + heads[i]] = i + 1U;
  }
  if (i == count - 1U) {
    *cells = offsets[i] + heads[i];
  }
}

// The points in cell order, for locality, and each point's cell (its node),
// -1 for a point in none.
__global__ void gather_kernel(const float* __restrict__ xyz, std::uint32_t count,
                              const std::uint64_t* __restrict__ sorted_keys,
                              const std::uint32_t* __restrict__ sorted_indices,
                              const std::uint32_t* __restrict__ heads,
                              const std::uint32_t* __restrict__ offsets,
                              float* __restrict__ sorted_xyz, int* __restrict__ node_of_point) {
  const std::uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= count) {
    return;
  }
  const std::uint32_t i = sorted_indices[t];
  if (sorted_keys[t] == kNoCell) {
    node_of_point[i] = -1;
    return;
  }
  node_of_point[i] = static_cast<int>(offsets[t] + heads[t] - 1U);
  for (int axis = 0; axis < 3; ++axis) {
    sorted_xyz[3 * t + axis] = xyz[3 * i + axis];
  }
}

// Each cell's bounding box, and its own parent.
__global__ void cell_bounds_kernel(const float* __restrict__ sorted_xyz,
                                   const std::uint32_t* __restrict__ starts, std::uint32_t cells,
                                   float* __restrict__ bounds, int* __restrict__ parent) {
  const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cells) {
    return;
  }
  float lo[3] = {INFINITY, INFINITY, INFINITY};
  float hi[3] = {-INFINITY, -INFINITY, -INFINITY};
  for (std::uint32_t t = starts[c]; t < starts[c + 1U]; ++t) {
    for (int axis = 0; axis < 3; ++axis) {
      lo[axis] = fminf(lo[axis], sorted_xyz[3 * t + axis]);
      hi[axis] = fmaxf(hi[axis], sorted_xyz[3 * t + axis]);
    }
  }
  for (int axis = 0; axis < 3; ++axis) {
    bounds[6 * c + axis] = lo[axis];
    bounds[6 * c + 3 + axis] = hi[axis];
  }
  parent[c] = static_cast<int>(c);
}

// The cell with `key`, or -1.
__device__ inline int find_cell(const std::uint64_t* __restrict__ cell_keys, std::uint32_t cells,
                                std::uint64_t key) {
  std::uint32_t lo = 0;
  std::uint32_t hi = cells;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2U;
    if (cell_keys[mid] < key) {
      lo = mid + 1U;
    } else {
      hi = mid;
    }
  }
  return lo < cells && cell_keys[lo] == key ? static_cast<int>(lo) : -1;
}

// The squared distance from point p to box [lo, hi], never more than from p to
// any point of the box as squared_distance() computes it: rounding is monotonic.
__device__ inline float squared_gap(const float* p, const float* lo, const float* hi) {
  float gap[3];
  for (int axis = 0; axis < 3; ++axis) {
    gap[axis] = fmaxf(fmaxf(lo[axis] - p[axis], p[axis] - hi[axis]), 0.0F);
  }
  return gap[0] * gap[0] + gap[1] * gap[1] + gap[2] * gap[2];
}

// Thread (cell a, offset o) tests whether cell a and the cell at offset o hold
// a pair of points within the tolerance, and joins them if so.
__global__ void cell_pairs_kernel(const float* __restrict__ sorted_xyz,
                                  const std::uint64_t* __restrict__ cell_keys,
                                  const std::uint32_t* __restrict__ starts,
                                  const float* __restrict__ bounds, std::uint32_t cells, float tol2,
                                  int* __restrict__ parent) {
  const std::uint64_t tid = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= static_cast<std::uint64_t>(cells) * kNeighborOffsets) {
    return;
  }
  const auto a = static_cast<std::uint32_t>(tid / kNeighborOffsets);
  // Offsets 63..124 of the 5 x 5 x 5 block in key order follow its center, 62.
  const int index = 63 + static_cast<int>(tid % kNeighborOffsets);
  int ix = 0;
  int iy = 0;
  int iz = 0;
  decode(cell_keys[a], ix, iy, iz);
  const int nx = ix + index / 25 - 2;
  const int ny = iy + (index / 5) % 5 - 2;
  const int nz = iz + index % 5 - 2;
  if (!in_range(nx) || !in_range(ny) || !in_range(nz)) {
    return;
  }
  const int b = find_cell(cell_keys, cells, cell_key(nx, ny, nz));
  if (b < 0 || representative(parent, static_cast<int>(a)) == representative(parent, b)) {
    return;
  }
  const float* a_lo = bounds + 6 * a;
  const float* a_hi = a_lo + 3;
  const float* b_lo = bounds + 6 * b;
  const float* b_hi = b_lo + 3;
  float gap[3];
  for (int axis = 0; axis < 3; ++axis) {
    gap[axis] = fmaxf(fmaxf(a_lo[axis] - b_hi[axis], b_lo[axis] - a_hi[axis]), 0.0F);
  }
  if (gap[0] * gap[0] + gap[1] * gap[1] + gap[2] * gap[2] > tol2) {
    return;
  }
  for (std::uint32_t i = starts[a]; i < starts[a + 1U]; ++i) {
    const float* p = sorted_xyz + 3 * i;
    if (squared_gap(p, b_lo, b_hi) > tol2) {
      continue;
    }
    for (std::uint32_t j = starts[b]; j < starts[b + 1U]; ++j) {
      if (squared_distance(p, sorted_xyz + 3 * j) <= tol2) {
        unite(parent, static_cast<int>(a), b);
        return;
      }
    }
  }
}

// A component's size and its lowest point index, from its cells. Within a
// cell the stable sort keeps the points in input order.
__global__ void cell_sizes_kernel(const std::uint32_t* __restrict__ starts,
                                  const std::uint32_t* __restrict__ sorted_indices,
                                  const int* __restrict__ root, std::uint32_t cells,
                                  std::uint32_t* __restrict__ size,
                                  std::uint32_t* __restrict__ min_index) {
  const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < cells) {
    const int r = root[c];
    atomicAdd(size + r, starts[c + 1U] - starts[c]);
    atomicMin(min_index + r, sorted_indices[starts[c]]);
  }
}

// ---- kImage -------------------------------------------------------------------------------------

__global__ void image_init_kernel(const float* __restrict__ xyz, std::uint32_t count,
                                  int* __restrict__ parent, int* __restrict__ node_of_point) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    parent[i] = static_cast<int>(i);
    node_of_point[i] =
        valid_point(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]) ? static_cast<int>(i) : -1;
  }
}

// Links pixel i to the pixels of its window that follow it, within the tolerance.
__global__ void image_pairs_kernel(const float* __restrict__ xyz, std::uint32_t count,
                                   std::uint32_t width, int window, float tol2,
                                   int* __restrict__ parent) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const float* p = xyz + 3 * i;
  if (!valid_point(p[0], p[1], p[2])) {
    return;
  }
  const int u = static_cast<int>(i % width);
  const int v = static_cast<int>(i / width);
  const int height = static_cast<int>(count / width);
  const int w = static_cast<int>(width);
  for (int dv = 0; dv <= window && v + dv < height; ++dv) {
    for (int du = -window; du <= window; ++du) {
      if ((dv == 0 && du <= 0) || u + du < 0 || u + du >= w) {
        continue;
      }
      const std::uint32_t j =
          static_cast<std::uint32_t>(v + dv) * width + static_cast<std::uint32_t>(u + du);
      const float* q = xyz + 3 * j;
      if (valid_point(q[0], q[1], q[2]) && squared_distance(p, q) <= tol2) {
        unite(parent, static_cast<int>(i), static_cast<int>(j));
      }
    }
  }
}

__global__ void pixel_sizes_kernel(const int* __restrict__ node_of_point,
                                   const int* __restrict__ root, std::uint32_t count,
                                   std::uint32_t* __restrict__ size) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count && node_of_point[i] >= 0) {
    atomicAdd(size + root[i], 1U);
  }
}

// ---- Selecting, sorting and labeling the clusters -----------------------------------------------

__global__ void flags_kernel(const int* __restrict__ root, const std::uint32_t* __restrict__ size,
                             std::uint32_t nodes, std::uint32_t min_size, std::uint32_t max_size,
                             std::uint32_t* __restrict__ flags) {
  const std::uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < nodes) {
    const std::uint32_t s = size[n];
    flags[n] = root[n] == static_cast<int>(n) && s >= min_size && s <= max_size ? 1U : 0U;
  }
}

// The kept components and their sort keys: larger first, then the lower
// lowest point index. Without min_index, a node is its own lowest point.
__global__ void candidates_kernel(const std::uint32_t* __restrict__ flags,
                                  const std::uint32_t* __restrict__ scan,
                                  const std::uint32_t* __restrict__ size,
                                  const std::uint32_t* __restrict__ min_index, std::uint32_t nodes,
                                  std::uint32_t* __restrict__ candidates,
                                  std::uint64_t* __restrict__ keys,
                                  std::uint32_t* __restrict__ total) {
  const std::uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= nodes) {
    return;
  }
  if (flags[n] != 0U) {
    const std::uint32_t lowest = min_index != nullptr ? min_index[n] : n;
    candidates[scan[n]] = n;
    keys[scan[n]] = (static_cast<std::uint64_t>(~size[n]) << 32U) | lowest;
  }
  if (n == nodes - 1U) {
    *total = scan[n] + flags[n];
  }
}

__global__ void rank_kernel(const std::uint32_t* __restrict__ sorted_candidates,
                            const std::uint32_t* __restrict__ size, std::uint32_t clusters,
                            int* __restrict__ rank_of, std::uint32_t* __restrict__ cluster_size) {
  const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < clusters) {
    rank_of[sorted_candidates[r]] = static_cast<int>(r);
    cluster_size[r] = size[sorted_candidates[r]];
  }
}

__global__ void labels_kernel(const int* __restrict__ node_of_point, const int* __restrict__ root,
                              const int* __restrict__ rank_of, std::uint32_t count,
                              std::uint32_t clusters, std::int32_t* __restrict__ labels,
                              std::uint32_t* __restrict__ group_keys,
                              std::uint32_t* __restrict__ group_indices) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const int node = node_of_point[i];
  const int label = node < 0 ? -1 : rank_of[root[node]];
  labels[i] = label;
  group_keys[i] = label < 0 ? clusters : static_cast<std::uint32_t>(label);
  group_indices[i] = i;
}

// ---- Outputs ------------------------------------------------------------------------------------

__global__ void grouped_kernel(const float* __restrict__ xyz, const std::uint8_t* __restrict__ rgb,
                               const std::uint32_t* __restrict__ grouped,
                               const std::int32_t* __restrict__ labels,
                               const std::uint32_t* __restrict__ clustered, std::uint32_t count,
                               float* __restrict__ out_xyz, std::uint8_t* __restrict__ out_rgb,
                               std::int32_t* __restrict__ out_labels) {
  const std::uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= count || t >= *clustered) {
    return;
  }
  const std::uint32_t i = grouped[t];
  if (out_xyz != nullptr) {
    for (int axis = 0; axis < 3; ++axis) {
      out_xyz[3 * t + axis] = xyz[3 * i + axis];
    }
  }
  if (out_rgb != nullptr && rgb != nullptr) {
    for (int channel = 0; channel < 3; ++channel) {
      out_rgb[3 * t + channel] = rgb[3 * i + channel];
    }
  }
  if (out_labels != nullptr) {
    out_labels[t] = labels[i];
  }
}

__global__ void kept_kernel(const float* __restrict__ xyz, const std::uint8_t* __restrict__ rgb,
                            const std::int32_t* __restrict__ labels, std::uint32_t count,
                            float* __restrict__ out_xyz, std::uint8_t* __restrict__ out_rgb) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const bool kept = labels[i] >= 0;
  if (out_xyz != nullptr) {
    for (int axis = 0; axis < 3; ++axis) {
      out_xyz[3 * i + axis] = kept ? xyz[3 * i + axis] : nan_value();
    }
  }
  if (out_rgb != nullptr && rgb != nullptr) {
    for (int channel = 0; channel < 3; ++channel) {
      out_rgb[3 * i + channel] = kept ? rgb[3 * i + channel] : std::uint8_t{0};
    }
  }
}

// ---- Summaries ----------------------------------------------------------------------------------

// The sums of one chunk of a cluster's points, relative to the cluster's
// first point, and their bounds.
struct Partial {
  double sum[3];
  double outer[6];  // xx, xy, xz, yy, yz, zz
  float lo[3];
  float hi[3];
};

// Projections of a chunk's points on the cluster's axes, about its centroid.
struct Extent {
  float lo[3];
  float hi[3];
};

// The cluster of chunk b, and the range of grouped points the chunk covers.
// Chunks of one cluster are consecutive; every cluster has at least one.
__device__ inline std::uint32_t chunk_range(const std::uint32_t* __restrict__ cluster_start,
                                            const std::uint32_t* __restrict__ chunk_start,
                                            std::uint32_t clusters, std::uint32_t b,
                                            std::uint32_t& first, std::uint32_t& end) {
  std::uint32_t lo = 0;
  std::uint32_t hi = clusters;
  while (hi - lo > 1U) {
    const std::uint32_t mid = lo + (hi - lo) / 2U;
    if (chunk_start[mid] <= b) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  first = cluster_start[lo] + (b - chunk_start[lo]) * kChunk;
  end = min(first + kChunk, cluster_start[lo + 1U]);
  return lo;
}

__global__ void chunks_kernel(const std::uint32_t* __restrict__ cluster_size,
                              std::uint32_t clusters, std::uint32_t* __restrict__ chunks) {
  const std::uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k <= clusters) {
    chunks[k] = k < clusters ? (cluster_size[k] + kChunk - 1U) / kChunk : 0U;
  }
}

__global__ void stats_partial_kernel(const float* __restrict__ xyz,
                                     const std::uint32_t* __restrict__ grouped,
                                     const std::uint32_t* __restrict__ cluster_start,
                                     const std::uint32_t* __restrict__ chunk_start,
                                     std::uint32_t clusters, Partial* __restrict__ partials) {
  __shared__ double sums[9][kBlock];
  __shared__ float bounds[6][kBlock];
  const std::uint32_t b = blockIdx.x;
  if (b >= chunk_start[clusters]) {
    return;  // The whole block: the bound is uniform.
  }
  std::uint32_t first = 0;
  std::uint32_t end = 0;
  const std::uint32_t k = chunk_range(cluster_start, chunk_start, clusters, b, first, end);
  const float* ref = xyz + 3 * grouped[cluster_start[k]];
  float s[9] = {};
  float lo[3] = {INFINITY, INFINITY, INFINITY};
  float hi[3] = {-INFINITY, -INFINITY, -INFINITY};
  for (std::uint32_t t = first + threadIdx.x; t < end; t += blockDim.x) {
    const float* p = xyz + 3 * grouped[t];
    const float d[3] = {p[0] - ref[0], p[1] - ref[1], p[2] - ref[2]};
    s[0] += d[0];
    s[1] += d[1];
    s[2] += d[2];
    s[3] += d[0] * d[0];
    s[4] += d[0] * d[1];
    s[5] += d[0] * d[2];
    s[6] += d[1] * d[1];
    s[7] += d[1] * d[2];
    s[8] += d[2] * d[2];
    for (int axis = 0; axis < 3; ++axis) {
      lo[axis] = fminf(lo[axis], p[axis]);
      hi[axis] = fmaxf(hi[axis], p[axis]);
    }
  }
  for (int j = 0; j < 9; ++j) {
    sums[j][threadIdx.x] = s[j];
  }
  for (int axis = 0; axis < 3; ++axis) {
    bounds[axis][threadIdx.x] = lo[axis];
    bounds[3 + axis][threadIdx.x] = hi[axis];
  }
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride /= 2U) {
    if (threadIdx.x < stride) {
      for (int j = 0; j < 9; ++j) {
        sums[j][threadIdx.x] += sums[j][threadIdx.x + stride];
      }
      for (int axis = 0; axis < 3; ++axis) {
        bounds[axis][threadIdx.x] =
            fminf(bounds[axis][threadIdx.x], bounds[axis][threadIdx.x + stride]);
        bounds[3 + axis][threadIdx.x] =
            fmaxf(bounds[3 + axis][threadIdx.x], bounds[3 + axis][threadIdx.x + stride]);
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    Partial& out = partials[b];
    for (int axis = 0; axis < 3; ++axis) {
      out.sum[axis] = sums[axis][0];
      out.lo[axis] = bounds[axis][0];
      out.hi[axis] = bounds[3 + axis][0];
    }
    for (int j = 0; j < 6; ++j) {
      out.outer[j] = sums[3 + j][0];
    }
  }
}

// The eigenvectors (columns of `vectors`) and eigenvalues of symmetric `a`,
// by cyclic Jacobi rotations.
__device__ inline void eigen_symmetric(double a[3][3], double vectors[3][3], double values[3]) {
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      vectors[r][c] = r == c ? 1.0 : 0.0;
    }
  }
  constexpr int kPairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
  for (int sweep = 0; sweep < 32; ++sweep) {
    const double off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    const double diagonal = a[0][0] * a[0][0] + a[1][1] * a[1][1] + a[2][2] * a[2][2];
    if (off <= 1.0e-30 * diagonal || off == 0.0) {
      break;
    }
    for (const auto& pair : kPairs) {
      const int p = pair[0];
      const int q = pair[1];
      if (a[p][q] == 0.0) {
        continue;
      }
      const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
      const double t = (theta >= 0.0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
      const double c = 1.0 / sqrt(t * t + 1.0);
      const double s = t * c;
      for (int k = 0; k < 3; ++k) {
        const double kp = a[k][p];
        const double kq = a[k][q];
        a[k][p] = c * kp - s * kq;
        a[k][q] = s * kp + c * kq;
      }
      for (int k = 0; k < 3; ++k) {
        const double pk = a[p][k];
        const double qk = a[q][k];
        a[p][k] = c * pk - s * qk;
        a[q][k] = s * pk + c * qk;
      }
      for (int k = 0; k < 3; ++k) {
        const double kp = vectors[k][p];
        const double kq = vectors[k][q];
        vectors[k][p] = c * kp - s * kq;
        vectors[k][q] = s * kp + c * kq;
      }
    }
  }
  for (int i = 0; i < 3; ++i) {
    values[i] = a[i][i];
  }
}

// The principal axes of covariance `cov` as unit rows: by decreasing
// variance, each of the first two with its largest component positive, and
// the third their cross product.
__device__ inline void principal_axes(double cov[3][3], double axes[3][3]) {
  double vectors[3][3];
  double values[3];
  eigen_symmetric(cov, vectors, values);
  int order[3] = {0, 1, 2};
  for (int i = 0; i < 2; ++i) {
    for (int j = i + 1; j < 3; ++j) {
      if (values[order[j]] > values[order[i]]) {
        const int swap = order[i];
        order[i] = order[j];
        order[j] = swap;
      }
    }
  }
  for (int i = 0; i < 2; ++i) {
    int largest = 0;
    for (int c = 1; c < 3; ++c) {
      if (fabs(vectors[c][order[i]]) > fabs(vectors[largest][order[i]])) {
        largest = c;
      }
    }
    const double sign = vectors[largest][order[i]] < 0.0 ? -1.0 : 1.0;
    double norm = 0.0;
    for (int c = 0; c < 3; ++c) {
      norm += vectors[c][order[i]] * vectors[c][order[i]];
    }
    norm = sqrt(norm);
    for (int c = 0; c < 3; ++c) {
      axes[i][c] = sign * vectors[c][order[i]] / norm;
    }
  }
  axes[2][0] = axes[0][1] * axes[1][2] - axes[0][2] * axes[1][1];
  axes[2][1] = axes[0][2] * axes[1][0] - axes[0][0] * axes[1][2];
  axes[2][2] = axes[0][0] * axes[1][1] - axes[0][1] * axes[1][0];
  const double norm =
      sqrt(axes[2][0] * axes[2][0] + axes[2][1] * axes[2][1] + axes[2][2] * axes[2][2]);
  for (int c = 0; c < 3; ++c) {
    axes[2][c] /= norm;
  }
}

__global__ void stats_final_kernel(const float* __restrict__ xyz,
                                   const std::uint32_t* __restrict__ grouped,
                                   const std::uint32_t* __restrict__ cluster_start,
                                   const std::uint32_t* __restrict__ chunk_start,
                                   const Partial* __restrict__ partials, std::uint32_t clusters,
                                   bool oriented, float* __restrict__ summaries) {
  const std::uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= clusters) {
    return;
  }
  double sum[3] = {};
  double outer[6] = {};
  float lo[3] = {INFINITY, INFINITY, INFINITY};
  float hi[3] = {-INFINITY, -INFINITY, -INFINITY};
  for (std::uint32_t b = chunk_start[k]; b < chunk_start[k + 1U]; ++b) {
    const Partial& p = partials[b];
    for (int axis = 0; axis < 3; ++axis) {
      sum[axis] += p.sum[axis];
      lo[axis] = fminf(lo[axis], p.lo[axis]);
      hi[axis] = fmaxf(hi[axis], p.hi[axis]);
    }
    for (int j = 0; j < 6; ++j) {
      outer[j] += p.outer[j];
    }
  }
  const std::uint32_t n = cluster_start[k + 1U] - cluster_start[k];
  const float* ref = xyz + 3 * grouped[cluster_start[k]];
  const double inv = 1.0 / static_cast<double>(n);
  const double mean[3] = {sum[0] * inv, sum[1] * inv, sum[2] * inv};
  float* row = summaries + static_cast<std::size_t>(k) * kSummaryFields;
  row[0] = static_cast<float>(k);
  row[1] = static_cast<float>(n);
  for (int axis = 0; axis < 3; ++axis) {
    row[2 + axis] = static_cast<float>(static_cast<double>(ref[axis]) + mean[axis]);
    row[5 + axis] = lo[axis];
    row[8 + axis] = hi[axis];
  }
  if (!oriented) {
    for (int j = 11; j < kSummaryFields; ++j) {
      row[j] = nan_value();
    }
    return;
  }
  double cov[3][3];
  cov[0][0] = outer[0] * inv - mean[0] * mean[0];
  cov[0][1] = outer[1] * inv - mean[0] * mean[1];
  cov[0][2] = outer[2] * inv - mean[0] * mean[2];
  cov[1][1] = outer[3] * inv - mean[1] * mean[1];
  cov[1][2] = outer[4] * inv - mean[1] * mean[2];
  cov[2][2] = outer[5] * inv - mean[2] * mean[2];
  cov[1][0] = cov[0][1];
  cov[2][0] = cov[0][2];
  cov[2][1] = cov[1][2];
  double axes[3][3];
  principal_axes(cov, axes);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      row[14 + 3 * r + c] = static_cast<float>(axes[r][c]);
    }
  }
}

__global__ void extent_partial_kernel(const float* __restrict__ xyz,
                                      const std::uint32_t* __restrict__ grouped,
                                      const std::uint32_t* __restrict__ cluster_start,
                                      const std::uint32_t* __restrict__ chunk_start,
                                      std::uint32_t clusters, const float* __restrict__ summaries,
                                      Extent* __restrict__ extents) {
  __shared__ float bounds[6][kBlock];
  const std::uint32_t b = blockIdx.x;
  if (b >= chunk_start[clusters]) {
    return;
  }
  std::uint32_t first = 0;
  std::uint32_t end = 0;
  const std::uint32_t k = chunk_range(cluster_start, chunk_start, clusters, b, first, end);
  const float* row = summaries + static_cast<std::size_t>(k) * kSummaryFields;
  const float* centroid = row + 2;
  const float* axes = row + 14;
  float lo[3] = {INFINITY, INFINITY, INFINITY};
  float hi[3] = {-INFINITY, -INFINITY, -INFINITY};
  for (std::uint32_t t = first + threadIdx.x; t < end; t += blockDim.x) {
    const float* p = xyz + 3 * grouped[t];
    const float d[3] = {p[0] - centroid[0], p[1] - centroid[1], p[2] - centroid[2]};
    for (int axis = 0; axis < 3; ++axis) {
      const float* a = axes + 3 * axis;
      const float projection = d[0] * a[0] + d[1] * a[1] + d[2] * a[2];
      lo[axis] = fminf(lo[axis], projection);
      hi[axis] = fmaxf(hi[axis], projection);
    }
  }
  for (int axis = 0; axis < 3; ++axis) {
    bounds[axis][threadIdx.x] = lo[axis];
    bounds[3 + axis][threadIdx.x] = hi[axis];
  }
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride /= 2U) {
    if (threadIdx.x < stride) {
      for (int axis = 0; axis < 3; ++axis) {
        bounds[axis][threadIdx.x] =
            fminf(bounds[axis][threadIdx.x], bounds[axis][threadIdx.x + stride]);
        bounds[3 + axis][threadIdx.x] =
            fmaxf(bounds[3 + axis][threadIdx.x], bounds[3 + axis][threadIdx.x + stride]);
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (int axis = 0; axis < 3; ++axis) {
      extents[b].lo[axis] = bounds[axis][0];
      extents[b].hi[axis] = bounds[3 + axis][0];
    }
  }
}

__global__ void extent_final_kernel(const std::uint32_t* __restrict__ chunk_start,
                                    const Extent* __restrict__ extents, std::uint32_t clusters,
                                    float* __restrict__ summaries) {
  const std::uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= clusters) {
    return;
  }
  float lo[3] = {INFINITY, INFINITY, INFINITY};
  float hi[3] = {-INFINITY, -INFINITY, -INFINITY};
  for (std::uint32_t b = chunk_start[k]; b < chunk_start[k + 1U]; ++b) {
    for (int axis = 0; axis < 3; ++axis) {
      lo[axis] = fminf(lo[axis], extents[b].lo[axis]);
      hi[axis] = fmaxf(hi[axis], extents[b].hi[axis]);
    }
  }
  float* row = summaries + static_cast<std::size_t>(k) * kSummaryFields;
  double center[3] = {row[2], row[3], row[4]};
  for (int axis = 0; axis < 3; ++axis) {
    const double middle = 0.5 * (static_cast<double>(lo[axis]) + static_cast<double>(hi[axis]));
    for (int c = 0; c < 3; ++c) {
      center[c] += middle * static_cast<double>(row[14 + 3 * axis + c]);
    }
    row[23 + axis] =
        static_cast<float>(0.5 * (static_cast<double>(hi[axis]) - static_cast<double>(lo[axis])));
  }
  for (int c = 0; c < 3; ++c) {
    row[11 + c] = static_cast<float>(center[c]);
  }
}

}  // namespace

bool valid_params(const ClusterParams& params) {
  return params.tolerance > 0.0F && std::isfinite(params.tolerance) &&
         params.min_cluster_size >= 1U && params.min_cluster_size <= params.max_cluster_size &&
         params.max_clusters >= 1U && params.max_clusters <= kMaxClustersLimit &&
         (params.connectivity == Connectivity::kEuclidean ||
          (params.connectivity == Connectivity::kImage && params.image_window >= 1 &&
           params.image_window <= kMaxImageWindow));
}

struct EuclideanClusterer::Impl {
  // kEuclidean's grid.
  detail::DeviceBuffer keys;
  detail::DeviceBuffer sorted_keys;
  detail::DeviceBuffer indices;
  detail::DeviceBuffer sorted_indices;
  detail::DeviceBuffer heads;
  detail::DeviceBuffer offsets;
  detail::DeviceBuffer cell_keys;
  detail::DeviceBuffer starts;
  detail::DeviceBuffer sorted_xyz;
  detail::DeviceBuffer bounds;
  // The graph's nodes: cells or pixels.
  detail::DeviceBuffer parent;
  detail::DeviceBuffer root;
  detail::DeviceBuffer size;
  detail::DeviceBuffer min_index;
  detail::DeviceBuffer flags;
  detail::DeviceBuffer scan;
  detail::DeviceBuffer candidates;
  detail::DeviceBuffer candidate_keys;
  detail::DeviceBuffer sorted_candidates;
  detail::DeviceBuffer sorted_candidate_keys;
  detail::DeviceBuffer rank_of;
  // Per point.
  detail::DeviceBuffer node_of_point;
  detail::DeviceBuffer labels;
  detail::DeviceBuffer group_keys;
  detail::DeviceBuffer sorted_group_keys;
  detail::DeviceBuffer group_indices;
  detail::DeviceBuffer grouped;
  // Per cluster.
  detail::DeviceBuffer cluster_size;
  detail::DeviceBuffer cluster_start;
  detail::DeviceBuffer chunks;
  detail::DeviceBuffer chunk_start;
  detail::DeviceBuffer partials;
  detail::DeviceBuffer extents;
  detail::DeviceBuffer counters;  // cells, candidates
  detail::DeviceBuffer temp;
  std::uint32_t* host_counters{};  // Pinned.

  // What the last cluster() found.
  std::uint32_t count{};
  std::uint32_t clusters{};

  ~Impl() {
    if (host_counters != nullptr) {
      cudaFreeHost(host_counters);
    }
  }

  cudaError_t read_counter(const std::uint32_t* device, std::uint32_t& value, cudaStream_t stream) {
    if (host_counters == nullptr) {
      if (const cudaError_t status =
              cudaMallocHost(reinterpret_cast<void**>(&host_counters), 2 * sizeof(std::uint32_t));
          status != cudaSuccess) {
        host_counters = nullptr;
        return status;
      }
    }
    if (const cudaError_t status = cudaMemcpyAsync(host_counters, device, sizeof(std::uint32_t),
                                                   cudaMemcpyDeviceToHost, stream);
        status != cudaSuccess) {
      return status;
    }
    if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
      return status;
    }
    value = host_counters[0];
    return cudaSuccess;
  }

  cudaError_t reserve(std::initializer_list<std::pair<detail::DeviceBuffer*, std::size_t>> list,
                      cudaStream_t stream) {
    for (const auto& [buffer, bytes] : list) {
      if (const cudaError_t status = buffer->reserve(std::max<std::size_t>(bytes, 8U), stream);
          status != cudaSuccess) {
        return status;
      }
    }
    return cudaSuccess;
  }

  // The grid's cells as nodes; returns their number.
  cudaError_t euclidean_graph(const float* xyz, const ClusterParams& params, std::uint32_t& nodes,
                              cudaStream_t stream);
  // The pixels as nodes.
  cudaError_t image_graph(const float* xyz, std::uint32_t width, const ClusterParams& params,
                          cudaStream_t stream);
};

cudaError_t EuclideanClusterer::Impl::euclidean_graph(const float* xyz, const ClusterParams& params,
                                                      std::uint32_t& nodes, cudaStream_t stream) {
  const std::size_t n = count;
  if (const cudaError_t status = reserve({{&keys, n * 8U},
                                          {&sorted_keys, n * 8U},
                                          {&indices, n * 4U},
                                          {&sorted_indices, n * 4U},
                                          {&heads, n * 4U},
                                          {&offsets, n * 4U},
                                          {&cell_keys, n * 8U},
                                          {&starts, n * 4U + 4U},
                                          {&sorted_xyz, n * 12U},
                                          {&bounds, n * 24U}},
                                         stream);
      status != cudaSuccess) {
    return status;
  }
  const double cell = static_cast<double>(params.tolerance) / std::sqrt(3.0) * kCellMargin;
  cell_keys_kernel<<<blocks_for(count), kBlock, 0, stream>>>(
      xyz, count, cell, keys.as<std::uint64_t>(), indices.as<std::uint32_t>());
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return status;
  }
  if (const cudaError_t status = detail::sort_pairs(
          keys.as<std::uint64_t>(), sorted_keys.as<std::uint64_t>(), indices.as<std::uint32_t>(),
          sorted_indices.as<std::uint32_t>(), count, kKeyBits + 1, temp, stream);
      status != cudaSuccess) {
    return status;
  }
  heads_kernel<<<blocks_for(count), kBlock, 0, stream>>>(sorted_keys.as<std::uint64_t>(), count,
                                                         heads.as<std::uint32_t>());
  if (const cudaError_t status = detail::exclusive_sum(
          heads.as<std::uint32_t>(), offsets.as<std::uint32_t>(), count, temp, stream);
      status != cudaSuccess) {
    return status;
  }
  cells_kernel<<<blocks_for(count), kBlock, 0, stream>>>(
      sorted_keys.as<std::uint64_t>(), heads.as<std::uint32_t>(), offsets.as<std::uint32_t>(),
      count, cell_keys.as<std::uint64_t>(), starts.as<std::uint32_t>(),
      counters.as<std::uint32_t>());
  gather_kernel<<<blocks_for(count), kBlock, 0, stream>>>(
      xyz, count, sorted_keys.as<std::uint64_t>(), sorted_indices.as<std::uint32_t>(),
      heads.as<std::uint32_t>(), offsets.as<std::uint32_t>(), sorted_xyz.as<float>(),
      node_of_point.as<int>());
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return status;
  }
  // The number of cells sizes the pair search: wait for it.
  if (const cudaError_t status = read_counter(counters.as<std::uint32_t>(), nodes, stream);
      status != cudaSuccess) {
    return status;
  }
  if (nodes == 0U) {
    return cudaSuccess;
  }
  if (const cudaError_t status = reserve({{&parent, std::size_t{nodes} * 4U}}, stream);
      status != cudaSuccess) {
    return status;
  }
  cell_bounds_kernel<<<blocks_for(nodes), kBlock, 0, stream>>>(
      sorted_xyz.as<float>(), starts.as<std::uint32_t>(), nodes, bounds.as<float>(),
      parent.as<int>());
  const std::uint64_t pairs = std::uint64_t{nodes} * kNeighborOffsets;
  cell_pairs_kernel<<<blocks_for(pairs), kBlock, 0, stream>>>(
      sorted_xyz.as<float>(), cell_keys.as<std::uint64_t>(), starts.as<std::uint32_t>(),
      bounds.as<float>(), nodes, params.tolerance * params.tolerance, parent.as<int>());
  return cudaGetLastError();
}

cudaError_t EuclideanClusterer::Impl::image_graph(const float* xyz, std::uint32_t width,
                                                  const ClusterParams& params,
                                                  cudaStream_t stream) {
  image_init_kernel<<<blocks_for(count), kBlock, 0, stream>>>(xyz, count, parent.as<int>(),
                                                              node_of_point.as<int>());
  image_pairs_kernel<<<blocks_for(count), kBlock, 0, stream>>>(
      xyz, count, width, params.image_window, params.tolerance * params.tolerance,
      parent.as<int>());
  return cudaGetLastError();
}

EuclideanClusterer::EuclideanClusterer() : impl_(std::make_unique<Impl>()) {}
EuclideanClusterer::~EuclideanClusterer() = default;

cudaError_t EuclideanClusterer::cluster(const float* xyz, std::uint32_t count, std::uint32_t width,
                                        const ClusterParams& params, std::uint32_t& clusters,
                                        cudaStream_t stream) {
  clusters = 0;
  Impl& s = *impl_;
  s.count = 0;
  s.clusters = 0;
  const bool image = params.connectivity == Connectivity::kImage;
  if (!valid_params(params) || count > kMaxPointsLimit || (width != 0U && count % width != 0U) ||
      (image && width == 0U)) {
    return cudaErrorInvalidValue;
  }
  s.count = count;
  const std::size_t n = count;
  if (const cudaError_t status = s.reserve({{&s.node_of_point, n * 4U},
                                            {&s.labels, n * 4U},
                                            {&s.group_keys, n * 4U},
                                            {&s.sorted_group_keys, n * 4U},
                                            {&s.group_indices, n * 4U},
                                            {&s.grouped, n * 4U},
                                            {&s.counters, 2 * sizeof(std::uint32_t)},
                                            {&s.cluster_size, sizeof(std::uint32_t)},
                                            {&s.cluster_start, sizeof(std::uint32_t)}},
                                           stream);
      status != cudaSuccess) {
    return status;
  }
  // An empty result until the clusters are known: no clustered points.
  if (const cudaError_t status =
          cudaMemsetAsync(s.cluster_start.as<void>(), 0, sizeof(std::uint32_t), stream);
      status != cudaSuccess) {
    return status;
  }
  if (count == 0U) {
    return cudaSuccess;
  }

  std::uint32_t nodes = count;
  if (image) {
    if (const cudaError_t status = s.reserve({{&s.parent, n * 4U}}, stream);
        status != cudaSuccess) {
      return status;
    }
    if (const cudaError_t status = s.image_graph(xyz, width, params, stream);
        status != cudaSuccess) {
      return status;
    }
  } else if (const cudaError_t status = s.euclidean_graph(xyz, params, nodes, stream);
             status != cudaSuccess) {
    return status;
  }

  std::uint32_t total = 0;
  if (nodes != 0U) {
    const std::size_t m = nodes;
    if (const cudaError_t status = s.reserve({{&s.root, m * 4U},
                                              {&s.size, m * 4U},
                                              {&s.min_index, m * 4U},
                                              {&s.flags, m * 4U},
                                              {&s.scan, m * 4U},
                                              {&s.candidates, m * 4U},
                                              {&s.candidate_keys, m * 8U},
                                              {&s.sorted_candidates, m * 4U},
                                              {&s.sorted_candidate_keys, m * 8U},
                                              {&s.rank_of, m * 4U}},
                                             stream);
        status != cudaSuccess) {
      return status;
    }
    roots_kernel<<<blocks_for(nodes), kBlock, 0, stream>>>(s.parent.as<int>(), nodes,
                                                           s.root.as<int>());
    if (const cudaError_t status = cudaMemsetAsync(s.size.as<void>(), 0, m * 4U, stream);
        status != cudaSuccess) {
      return status;
    }
    if (image) {
      pixel_sizes_kernel<<<blocks_for(count), kBlock, 0, stream>>>(
          s.node_of_point.as<int>(), s.root.as<int>(), count, s.size.as<std::uint32_t>());
    } else {
      if (const cudaError_t status = cudaMemsetAsync(s.min_index.as<void>(), 0xFF, m * 4U, stream);
          status != cudaSuccess) {
        return status;
      }
      cell_sizes_kernel<<<blocks_for(nodes), kBlock, 0, stream>>>(
          s.starts.as<std::uint32_t>(), s.sorted_indices.as<std::uint32_t>(), s.root.as<int>(),
          nodes, s.size.as<std::uint32_t>(), s.min_index.as<std::uint32_t>());
    }
    flags_kernel<<<blocks_for(nodes), kBlock, 0, stream>>>(
        s.root.as<int>(), s.size.as<std::uint32_t>(), nodes, params.min_cluster_size,
        params.max_cluster_size, s.flags.as<std::uint32_t>());
    if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
      return status;
    }
    if (const cudaError_t status = detail::exclusive_sum(
            s.flags.as<std::uint32_t>(), s.scan.as<std::uint32_t>(), nodes, s.temp, stream);
        status != cudaSuccess) {
      return status;
    }
    candidates_kernel<<<blocks_for(nodes), kBlock, 0, stream>>>(
        s.flags.as<std::uint32_t>(), s.scan.as<std::uint32_t>(), s.size.as<std::uint32_t>(),
        image ? nullptr : s.min_index.as<std::uint32_t>(), nodes, s.candidates.as<std::uint32_t>(),
        s.candidate_keys.as<std::uint64_t>(), s.counters.as<std::uint32_t>() + 1);
    if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
      return status;
    }
    // The number of components kept sizes their sort: wait for it.
    if (const cudaError_t status =
            s.read_counter(s.counters.as<std::uint32_t>() + 1, total, stream);
        status != cudaSuccess) {
      return status;
    }
  }
  const std::uint32_t kept = std::min(total, params.max_clusters);
  if (const cudaError_t status = s.reserve({{&s.cluster_size, (std::size_t{kept} + 1U) * 4U},
                                            {&s.cluster_start, (std::size_t{kept} + 1U) * 4U}},
                                           stream);
      status != cudaSuccess) {
    return status;
  }
  if (const cudaError_t status =
          cudaMemsetAsync(s.cluster_size.as<void>(), 0, (std::size_t{kept} + 1U) * 4U, stream);
      status != cudaSuccess) {
    return status;
  }
  if (nodes != 0U) {
    if (total != 0U) {
      if (const cudaError_t status = detail::sort_pairs(
              s.candidate_keys.as<std::uint64_t>(), s.sorted_candidate_keys.as<std::uint64_t>(),
              s.candidates.as<std::uint32_t>(), s.sorted_candidates.as<std::uint32_t>(), total, 64,
              s.temp, stream);
          status != cudaSuccess) {
        return status;
      }
    }
    if (const cudaError_t status =
            cudaMemsetAsync(s.rank_of.as<void>(), 0xFF, std::size_t{nodes} * 4U, stream);
        status != cudaSuccess) {
      return status;
    }
    if (kept != 0U) {
      rank_kernel<<<blocks_for(kept), kBlock, 0, stream>>>(
          s.sorted_candidates.as<std::uint32_t>(), s.size.as<std::uint32_t>(), kept,
          s.rank_of.as<int>(), s.cluster_size.as<std::uint32_t>());
    }
    labels_kernel<<<blocks_for(count), kBlock, 0, stream>>>(
        s.node_of_point.as<int>(), s.root.as<int>(), s.rank_of.as<int>(), count, kept,
        s.labels.as<std::int32_t>(), s.group_keys.as<std::uint32_t>(),
        s.group_indices.as<std::uint32_t>());
  } else {
    // No valid point: every label is -1.
    if (const cudaError_t status = cudaMemsetAsync(s.labels.as<void>(), 0xFF, n * 4U, stream);
        status != cudaSuccess) {
      return status;
    }
  }
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return status;
  }
  if (const cudaError_t status =
          detail::exclusive_sum(s.cluster_size.as<std::uint32_t>(),
                                s.cluster_start.as<std::uint32_t>(), kept + 1U, s.temp, stream);
      status != cudaSuccess) {
    return status;
  }
  if (kept != 0U) {
    // Grouped by label, in input order within a cluster; the rest last.
    const int bits = std::bit_width(kept);
    if (const cudaError_t status = detail::sort_pairs(
            s.group_keys.as<std::uint32_t>(), s.sorted_group_keys.as<std::uint32_t>(),
            s.group_indices.as<std::uint32_t>(), s.grouped.as<std::uint32_t>(), count, bits, s.temp,
            stream);
        status != cudaSuccess) {
      return status;
    }
  }
  s.clusters = kept;
  clusters = kept;
  return cudaSuccess;
}

cudaError_t EuclideanClusterer::write_labels(std::int32_t* labels, cudaStream_t stream) const {
  if (impl_->count == 0U) {
    return cudaSuccess;
  }
  return cudaMemcpyAsync(labels, impl_->labels.as<void>(), std::size_t{impl_->count} * 4U,
                         cudaMemcpyDeviceToDevice, stream);
}

cudaError_t EuclideanClusterer::clustered_count(std::uint32_t& points, cudaStream_t stream) {
  points = 0;
  if (impl_->count == 0U) {
    return cudaSuccess;
  }
  return impl_->read_counter(impl_->cluster_start.as<std::uint32_t>() + impl_->clusters, points,
                             stream);
}

cudaError_t EuclideanClusterer::write_grouped(const float* xyz, const std::uint8_t* rgb,
                                              float* out_xyz, std::uint8_t* out_rgb,
                                              std::int32_t* out_labels, cudaStream_t stream) const {
  const Impl& s = *impl_;
  if (s.count == 0U || s.clusters == 0U) {
    return cudaSuccess;
  }
  grouped_kernel<<<blocks_for(s.count), kBlock, 0, stream>>>(
      xyz, rgb, s.grouped.as<std::uint32_t>(), s.labels.as<std::int32_t>(),
      s.cluster_start.as<std::uint32_t>() + s.clusters, s.count, out_xyz, out_rgb, out_labels);
  return cudaGetLastError();
}

cudaError_t EuclideanClusterer::write_kept(const float* xyz, const std::uint8_t* rgb,
                                           float* out_xyz, std::uint8_t* out_rgb,
                                           cudaStream_t stream) const {
  const Impl& s = *impl_;
  if (s.count == 0U) {
    return cudaSuccess;
  }
  kept_kernel<<<blocks_for(s.count), kBlock, 0, stream>>>(xyz, rgb, s.labels.as<std::int32_t>(),
                                                          s.count, out_xyz, out_rgb);
  return cudaGetLastError();
}

cudaError_t EuclideanClusterer::write_summaries(const float* xyz, bool oriented, float* summaries,
                                                cudaStream_t stream) {
  Impl& s = *impl_;
  const std::uint32_t k = s.clusters;
  if (k == 0U) {
    return cudaSuccess;
  }
  // Every cluster's chunks: at most one partial chunk per cluster beyond the
  // whole ones of all clustered points.
  const std::uint64_t max_chunks = (std::uint64_t{s.count} + kChunk - 1U) / kChunk + k;
  if (const cudaError_t status =
          s.reserve({{&s.chunks, (std::size_t{k} + 1U) * 4U},
                     {&s.chunk_start, (std::size_t{k} + 1U) * 4U},
                     {&s.partials, max_chunks * sizeof(Partial)},
                     {&s.extents, oriented ? max_chunks * sizeof(Extent) : 0U}},
                    stream);
      status != cudaSuccess) {
    return status;
  }
  chunks_kernel<<<blocks_for(k + 1U), kBlock, 0, stream>>>(s.cluster_size.as<std::uint32_t>(), k,
                                                           s.chunks.as<std::uint32_t>());
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return status;
  }
  if (const cudaError_t status = detail::exclusive_sum(
          s.chunks.as<std::uint32_t>(), s.chunk_start.as<std::uint32_t>(), k + 1U, s.temp, stream);
      status != cudaSuccess) {
    return status;
  }
  const auto grid = static_cast<unsigned int>(max_chunks);
  stats_partial_kernel<<<grid, kBlock, 0, stream>>>(
      xyz, s.grouped.as<std::uint32_t>(), s.cluster_start.as<std::uint32_t>(),
      s.chunk_start.as<std::uint32_t>(), k, s.partials.as<Partial>());
  stats_final_kernel<<<blocks_for(k), kBlock, 0, stream>>>(
      xyz, s.grouped.as<std::uint32_t>(), s.cluster_start.as<std::uint32_t>(),
      s.chunk_start.as<std::uint32_t>(), s.partials.as<Partial>(), k, oriented, summaries);
  if (oriented) {
    extent_partial_kernel<<<grid, kBlock, 0, stream>>>(
        xyz, s.grouped.as<std::uint32_t>(), s.cluster_start.as<std::uint32_t>(),
        s.chunk_start.as<std::uint32_t>(), k, summaries, s.extents.as<Extent>());
    extent_final_kernel<<<blocks_for(k), kBlock, 0, stream>>>(s.chunk_start.as<std::uint32_t>(),
                                                              s.extents.as<Extent>(), k, summaries);
  }
  return cudaGetLastError();
}

}  // namespace holoscan::clustering
