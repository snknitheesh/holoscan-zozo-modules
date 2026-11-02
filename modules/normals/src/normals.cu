// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Normal estimation. One thread per point gathers its neighborhood, sums the
// neighbors' offsets from the point and their products (in float, in a fixed
// order, so every run gives the same bits), and solves the covariance's
// eigenproblem by Jacobi rotations.
//
// The window search reads the pixels around the point in the image grid. The
// radius and k-nearest searches sort the points into a uniform grid by cell
// key (a stable CUB radix sort), list the occupied cells with where their
// points start, and look each neighbor cell up by binary search over the
// sorted cell keys. The k-nearest search visits rings of cells (a quarter of the
// radius a side) outward and stops once no unvisited point can be nearer; the
// nearest come first, ties to the lower index, so the neighbors are exact and
// do not depend on the visiting order. Compiled without fused multiply-adds,
// so a host reference that computes the same distances finds the same
// neighbors.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <holoscan/normals/normals.hpp>

#include "device_buffer.hpp"
#include "kernel_common.cuh"

namespace holoscan::normals {
namespace {

using detail::blocks_for;
using detail::kBlock;
using detail::kNoCell;
using detail::nan_float;
using detail::valid_point;

// The k-nearest search's cells are a quarter of the radius, so four rings
// of cells cover it; on a dense surface the search mostly stops after one.
constexpr int kKnnRings = 4;

// The sums of a neighborhood: the count, the offsets from the point and their
// products xx, xy, xz, yy, yz, zz.
struct Moments {
  std::uint32_t n{};
  float s[3]{};
  float ss[6]{};

  __device__ void add(float dx, float dy, float dz) {
    ++n;
    s[0] += dx;
    s[1] += dy;
    s[2] += dz;
    ss[0] += dx * dx;
    ss[1] += dx * dy;
    ss[2] += dx * dz;
    ss[3] += dy * dy;
    ss[4] += dy * dz;
    ss[5] += dz * dz;
  }
};

__device__ inline void write_nan(std::uint32_t i, float* __restrict__ normals,
                                 float* __restrict__ curvature) {
  const float nan = nan_float();
  normals[3 * i] = nan;
  normals[3 * i + 1] = nan;
  normals[3 * i + 2] = nan;
  curvature[i] = nan;
}

// The normal of point i at p from its neighborhood's moments, toward `view`.
__device__ void finish(const Moments& m, std::uint32_t min_neighbors, float px, float py, float pz,
                       float3 view, std::uint32_t i, float* __restrict__ normals,
                       float* __restrict__ curvature) {
  if (m.n < min_neighbors) {
    write_nan(i, normals, curvature);
    return;
  }
  const float n = static_cast<float>(m.n);
  const float mean[3] = {m.s[0] / n, m.s[1] / n, m.s[2] / n};
  float cov[3][3];
  cov[0][0] = m.ss[0] / n - mean[0] * mean[0];
  cov[0][1] = m.ss[1] / n - mean[0] * mean[1];
  cov[0][2] = m.ss[2] / n - mean[0] * mean[2];
  cov[1][1] = m.ss[3] / n - mean[1] * mean[1];
  cov[1][2] = m.ss[4] / n - mean[1] * mean[2];
  cov[2][2] = m.ss[5] / n - mean[2] * mean[2];
  cov[1][0] = cov[0][1];
  cov[2][0] = cov[0][2];
  cov[2][1] = cov[1][2];
  float normal[3];
  float variation = 0.0F;
  if (!detail::pca_normal(cov, normal, variation)) {
    write_nan(i, normals, curvature);
    return;
  }
  const float toward =
      normal[0] * (view.x - px) + normal[1] * (view.y - py) + normal[2] * (view.z - pz);
  const float sign = toward < 0.0F ? -1.0F : 1.0F;
  normals[3 * i] = sign * normal[0];
  normals[3 * i + 1] = sign * normal[1];
  normals[3 * i + 2] = sign * normal[2];
  curvature[i] = variation;
}

__device__ inline float range_from(float3 view, float x, float y, float z) {
  const float dx = x - view.x;
  const float dy = y - view.y;
  const float dz = z - view.z;
  return sqrtf(dx * dx + dy * dy + dz * dz);
}

struct WindowArgs {
  int width;
  int height;
  int radius;
  float factor;
  std::uint32_t min_neighbors;
  float3 view;
};

__global__ void window_kernel(const float* __restrict__ xyz, WindowArgs a,
                              float* __restrict__ normals, float* __restrict__ curvature) {
  const int u = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int v = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (u >= a.width || v >= a.height) {
    return;
  }
  const auto i = static_cast<std::uint32_t>(v) * static_cast<std::uint32_t>(a.width) +
                 static_cast<std::uint32_t>(u);
  const float px = xyz[3 * i];
  const float py = xyz[3 * i + 1];
  const float pz = xyz[3 * i + 2];
  if (!valid_point(px, py, pz)) {
    write_nan(i, normals, curvature);
    return;
  }
  const float range = range_from(a.view, px, py, pz);
  const float step_limit = a.factor * range;
  Moments m;
  for (int dv = -a.radius; dv <= a.radius; ++dv) {
    const int row = v + dv;
    if (row < 0 || row >= a.height) {
      continue;
    }
    for (int du = -a.radius; du <= a.radius; ++du) {
      const int column = u + du;
      if (column < 0 || column >= a.width) {
        continue;
      }
      const auto j = static_cast<std::size_t>(row) * static_cast<std::size_t>(a.width) +
                     static_cast<std::size_t>(column);
      const float qx = xyz[3 * j];
      const float qy = xyz[3 * j + 1];
      const float qz = xyz[3 * j + 2];
      if (!valid_point(qx, qy, qz)) {
        continue;
      }
      const int steps = max(abs(du), abs(dv));
      if (fabsf(range_from(a.view, qx, qy, qz) - range) > step_limit * static_cast<float>(steps)) {
        continue;
      }
      m.add(qx - px, qy - py, qz - pz);
    }
  }
  finish(m, a.min_neighbors, px, py, pz, a.view, i, normals, curvature);
}

// ---- The grid ---------------------------------------------------------------------------------

__global__ void cell_keys_kernel(const float* __restrict__ xyz, std::uint32_t count, float size,
                                 std::uint64_t* __restrict__ keys,
                                 std::uint32_t* __restrict__ indices) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    keys[i] = detail::point_cell(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2], size);
    indices[i] = i;
  }
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

struct Grid {
  const std::uint64_t* sorted_keys;
  const std::uint32_t* sorted_indices;
  const std::uint64_t* cell_keys;
  const std::uint32_t* starts;
  const std::uint32_t* cells;
};

// Calls visit(index, squared distance) for every point, the query included,
// in the cells of Chebyshev ring `ring` around cell (ix, iy, iz).
template <typename Visit>
__device__ inline void visit_ring(const Grid& grid, std::uint32_t cells, const float* xyz, float px,
                                  float py, float pz, int ix, int iy, int iz, int ring,
                                  Visit&& visit) {
  for (int dx = -ring; dx <= ring; ++dx) {
    for (int dy = -ring; dy <= ring; ++dy) {
      for (int dz = -ring; dz <= ring; ++dz) {
        if (max(abs(dx), max(abs(dy), abs(dz))) != ring) {
          continue;
        }
        const int nx = ix + dx;
        const int ny = iy + dy;
        const int nz = iz + dz;
        if (!detail::in_range(nx) || !detail::in_range(ny) || !detail::in_range(nz)) {
          continue;
        }
        const int cell = find_cell(grid.cell_keys, cells, detail::cell_key(nx, ny, nz));
        if (cell < 0) {
          continue;
        }
        for (std::uint32_t j = grid.starts[cell]; j < grid.starts[cell + 1]; ++j) {
          const std::uint32_t other = grid.sorted_indices[j];
          const float ex = xyz[3 * other] - px;
          const float ey = xyz[3 * other + 1] - py;
          const float ez = xyz[3 * other + 2] - pz;
          visit(other, ex * ex + ey * ey + ez * ez);
        }
      }
    }
  }
}

__global__ void fill_nan_kernel(std::uint32_t count, float* __restrict__ normals,
                                float* __restrict__ curvature) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    write_nan(i, normals, curvature);
  }
}

struct GridArgs {
  float radius;
  float cell_size;
  std::uint32_t k;
  std::uint32_t min_neighbors;
  float3 view;
};

// Thread t takes the t-th point in cell order, for locality; invalid points,
// sorted last, keep the NaN that fill_nan_kernel wrote.
__global__ void radius_kernel(const float* __restrict__ xyz, std::uint32_t count, Grid grid,
                              GridArgs a, float* __restrict__ normals,
                              float* __restrict__ curvature) {
  const std::uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= count || grid.sorted_keys[t] == kNoCell) {
    return;
  }
  const std::uint32_t cells = *grid.cells;
  const std::uint32_t i = grid.sorted_indices[t];
  const float px = xyz[3 * i];
  const float py = xyz[3 * i + 1];
  const float pz = xyz[3 * i + 2];
  int ix = 0;
  int iy = 0;
  int iz = 0;
  detail::decode_key(grid.sorted_keys[t], ix, iy, iz);
  const float squared_radius = a.radius * a.radius;
  Moments m;
  // Cells of one radius a side: every neighbor is within the first ring.
  const auto add = [&](std::uint32_t other, float squared) {
    if (squared <= squared_radius) {
      m.add(xyz[3 * other] - px, xyz[3 * other + 1] - py, xyz[3 * other + 2] - pz);
    }
  };
  visit_ring(grid, cells, xyz, px, py, pz, ix, iy, iz, 0, add);
  visit_ring(grid, cells, xyz, px, py, pz, ix, iy, iz, 1, add);
  finish(m, a.min_neighbors, px, py, pz, a.view, i, normals, curvature);
}

__global__ void knn_kernel(const float* __restrict__ xyz, std::uint32_t count, Grid grid,
                           GridArgs a, float* __restrict__ normals, float* __restrict__ curvature) {
  const std::uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= count || grid.sorted_keys[t] == kNoCell) {
    return;
  }
  const std::uint32_t cells = *grid.cells;
  const std::uint32_t i = grid.sorted_indices[t];
  const float px = xyz[3 * i];
  const float py = xyz[3 * i + 1];
  const float pz = xyz[3 * i + 2];
  int ix = 0;
  int iy = 0;
  int iz = 0;
  detail::decode_key(grid.sorted_keys[t], ix, iy, iz);
  const float squared_radius = a.radius * a.radius;
  float best[kMaxNeighbors] = {};
  std::uint32_t best_index[kMaxNeighbors] = {};
  std::uint32_t found = 0;
  const std::uint32_t k = a.k;
  // Ascending by (squared distance, index).
  const auto nearer = [](float d1, std::uint32_t i1, float d2, std::uint32_t i2) {
    return d1 < d2 || (d1 == d2 && i1 < i2);
  };
  const auto visit = [&](std::uint32_t other, float squared) {
    if (!(squared <= squared_radius) ||
        (found == k && !nearer(squared, other, best[k - 1U], best_index[k - 1U]))) {
      return;
    }
    std::uint32_t at = found < k ? found++ : k - 1U;
    while (at > 0U && nearer(squared, other, best[at - 1U], best_index[at - 1U])) {
      best[at] = best[at - 1U];
      best_index[at] = best_index[at - 1U];
      --at;
    }
    best[at] = squared;
    best_index[at] = other;
  };
  for (int ring = 0; ring <= kKnnRings; ++ring) {
    visit_ring(grid, cells, xyz, px, py, pz, ix, iy, iz, ring, visit);
    // Every point beyond ring `ring` is farther than ring * cell_size.
    const float reach = static_cast<float>(ring) * a.cell_size;
    if (ring >= 1 && found == k && best[k - 1U] <= reach * reach) {
      break;
    }
  }
  Moments m;
  for (std::uint32_t j = 0; j < found; ++j) {
    const std::uint32_t other = best_index[j];
    m.add(xyz[3 * other] - px, xyz[3 * other + 1] - py, xyz[3 * other + 2] - pz);
  }
  finish(m, a.min_neighbors, px, py, pz, a.view, i, normals, curvature);
}

}  // namespace

bool valid_params(const NormalParams& params, const char** what) {
  const auto fail = [what](const char* field) {
    if (what != nullptr) {
      *what = field;
    }
    return false;
  };
  if (params.search != NeighborSearch::kAuto && params.search != NeighborSearch::kWindow &&
      params.search != NeighborSearch::kRadius && params.search != NeighborSearch::kKnn) {
    return fail("search must be kAuto, kWindow, kRadius or kKnn");
  }
  if (params.window_radius < 1 || params.window_radius > kMaxWindowRadius) {
    return fail("window_radius must be 1..15");
  }
  if (!(params.depth_change_factor > 0.0F) || !std::isfinite(params.depth_change_factor)) {
    return fail("depth_change_factor must be finite and positive");
  }
  if (!(params.radius > 0.0F) || !std::isfinite(params.radius)) {
    return fail("radius must be finite and positive");
  }
  if (params.k < 3U || params.k > kMaxNeighbors) {
    return fail("k must be 3..64");
  }
  constexpr std::uint32_t window = 2U * kMaxWindowRadius + 1U;
  if (params.min_neighbors < 3U || params.min_neighbors > window * window) {
    return fail("min_neighbors must be 3..961");
  }
  if (!std::isfinite(params.viewpoint[0]) || !std::isfinite(params.viewpoint[1]) ||
      !std::isfinite(params.viewpoint[2])) {
    return fail("viewpoint must be finite");
  }
  return true;
}

struct NormalEstimator::Impl {
  detail::DeviceBuffer keys;
  detail::DeviceBuffer sorted_keys;
  detail::DeviceBuffer indices;
  detail::DeviceBuffer sorted_indices;
  detail::DeviceBuffer heads;
  detail::DeviceBuffer offsets;
  detail::DeviceBuffer cell_keys;
  detail::DeviceBuffer starts;
  detail::DeviceBuffer cells;
  detail::DeviceBuffer temp;
};

NormalEstimator::NormalEstimator() : impl_(std::make_unique<Impl>()) {}
NormalEstimator::~NormalEstimator() = default;

cudaError_t NormalEstimator::estimate(const float* xyz, std::uint32_t count, int width, int height,
                                      const NormalParams& params, float* normals, float* curvature,
                                      cudaStream_t stream) {
  if (!valid_params(params) || count > kMaxPointsLimit) {
    return cudaErrorInvalidValue;
  }
  const NeighborSearch search = resolve_search(params.search, height > 1);
  if (search == NeighborSearch::kWindow &&
      (width < 0 || height < 0 ||
       static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) != count)) {
    return cudaErrorInvalidValue;
  }
  if (count == 0U) {
    return cudaSuccess;
  }
  if (xyz == nullptr || normals == nullptr || curvature == nullptr) {
    return cudaErrorInvalidValue;
  }
  const float3 view = make_float3(params.viewpoint[0], params.viewpoint[1], params.viewpoint[2]);

  if (search == NeighborSearch::kWindow) {
    const WindowArgs args{
        width, height, params.window_radius, params.depth_change_factor, params.min_neighbors,
        view};
    const dim3 block(32, 8);
    const dim3 grid((static_cast<unsigned int>(width) + 31U) / 32U,
                    (static_cast<unsigned int>(height) + 7U) / 8U);
    window_kernel<<<grid, block, 0, stream>>>(xyz, args, normals, curvature);
    return cudaGetLastError();
  }

  Impl& s = *impl_;
  const std::size_t n = count;
  for (auto [buffer, bytes] :
       {std::pair{&s.keys, n * 8U}, std::pair{&s.sorted_keys, n * 8U},
        std::pair{&s.indices, n * 4U}, std::pair{&s.sorted_indices, n * 4U},
        std::pair{&s.heads, n * 4U}, std::pair{&s.offsets, n * 4U}, std::pair{&s.cell_keys, n * 8U},
        std::pair{&s.starts, n * 4U + 4U}, std::pair{&s.cells, std::size_t{4}}}) {
    if (const cudaError_t status = buffer->reserve(bytes, stream); status != cudaSuccess) {
      return status;
    }
  }
  const bool knn = search == NeighborSearch::kKnn;
  const float cell_size = knn ? params.radius / static_cast<float>(kKnnRings) : params.radius;
  cell_keys_kernel<<<blocks_for(count), kBlock, 0, stream>>>(
      xyz, count, cell_size, s.keys.as<std::uint64_t>(), s.indices.as<std::uint32_t>());
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return status;
  }
  if (const cudaError_t status =
          detail::sort_pairs(s.keys.as<std::uint64_t>(), s.sorted_keys.as<std::uint64_t>(),
                             s.indices.as<std::uint32_t>(), s.sorted_indices.as<std::uint32_t>(),
                             count, detail::kKeyBits, s.temp, stream);
      status != cudaSuccess) {
    return status;
  }
  heads_kernel<<<blocks_for(count), kBlock, 0, stream>>>(s.sorted_keys.as<std::uint64_t>(), count,
                                                         s.heads.as<std::uint32_t>());
  if (const cudaError_t status = detail::exclusive_sum(
          s.heads.as<std::uint32_t>(), s.offsets.as<std::uint32_t>(), count, s.temp, stream);
      status != cudaSuccess) {
    return status;
  }
  cells_kernel<<<blocks_for(count), kBlock, 0, stream>>>(
      s.sorted_keys.as<std::uint64_t>(), s.heads.as<std::uint32_t>(), s.offsets.as<std::uint32_t>(),
      count, s.cell_keys.as<std::uint64_t>(), s.starts.as<std::uint32_t>(),
      s.cells.as<std::uint32_t>());
  fill_nan_kernel<<<blocks_for(count), kBlock, 0, stream>>>(count, normals, curvature);
  const Grid grid{s.sorted_keys.as<std::uint64_t>(), s.sorted_indices.as<std::uint32_t>(),
                  s.cell_keys.as<std::uint64_t>(), s.starts.as<std::uint32_t>(),
                  s.cells.as<std::uint32_t>()};
  const GridArgs args{params.radius, cell_size, params.k, params.min_neighbors, view};
  if (knn) {
    knn_kernel<<<blocks_for(count), kBlock, 0, stream>>>(xyz, count, grid, args, normals,
                                                         curvature);
  } else {
    radius_kernel<<<blocks_for(count), kBlock, 0, stream>>>(xyz, count, grid, args, normals,
                                                            curvature);
  }
  return cudaGetLastError();
}

}  // namespace holoscan::normals
