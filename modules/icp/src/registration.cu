// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>
#include <vector_types.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <cub/block/block_reduce.cuh>               // NOLINT(build/include_order)
#include <cub/device/device_radix_sort.cuh>         // NOLINT(build/include_order)
#include <cub/device/device_run_length_encode.cuh>  // NOLINT(build/include_order)
#include <cub/device/device_scan.cuh>               // NOLINT(build/include_order)
#include <cub/device/device_select.cuh>             // NOLINT(build/include_order)

#include "holoscan/icp/registration.hpp"
#include "normal_equations.hpp"

namespace holoscan::icp {
namespace {

using detail::kCount;
using detail::kJtJ;
using detail::kJtr;
using detail::kSquaredDistance;
using detail::kSums;

// Grid and voxel keys: 21 bits per axis, x lowest, so the cells x-1..x+1 of
// one row are consecutive keys. Cells -2^20..2^20-1 per axis are keyed.
constexpr int kAxisBits = 21;
constexpr int kAxisOffset = 1 << 20;
constexpr std::uint64_t kInvalidKey = ~std::uint64_t{0};
// The largest cloud: CUB takes int counts, and keys and indices are 32-bit.
constexpr std::uint32_t kMaxCount = std::uint32_t{1} << 30;

// At most this many correspondence distances, halving from the maximum.
constexpr int kMaxLevels = 8;
// Grid cells are this much wider than the search radius, so that rounding in
// a point's cell never puts a neighbor within the radius two cells away.
constexpr float kCellMargin = 1.001F;

constexpr int kBlock = 256;
// The association pass's grid is at most this many blocks, fixed by the
// point count alone, so its sums do not depend on the device.
constexpr int kMaxReduceBlocks = 512;

#define ICP_TRY(expression)                                                 \
  do {                                                                      \
    if (const cudaError_t status_ = (expression); status_ != cudaSuccess) { \
      return status_;                                                       \
    }                                                                       \
  } while (false)

struct DeviceBuffer {
  void* data{};
  std::size_t bytes{};

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data(std::exchange(other.data, nullptr)), bytes(std::exchange(other.bytes, 0U)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    std::swap(data, other.data);
    std::swap(bytes, other.bytes);
    return *this;
  }
  ~DeviceBuffer() {
    if (data != nullptr) {
      cudaFree(data);
    }
  }

  // Grows to at least `wanted` bytes; the old contents are lost. Frees the
  // old buffer only once `stream` is done with it.
  cudaError_t reserve(std::size_t wanted, cudaStream_t stream) {
    if (bytes >= wanted) {
      return cudaSuccess;
    }
    const std::size_t grown = std::max(wanted, bytes + bytes / 2U);
    if (data != nullptr) {
      ICP_TRY(cudaStreamSynchronize(stream));
      cudaFree(data);
      data = nullptr;
      bytes = 0;
    }
    ICP_TRY(cudaMalloc(&data, grown));
    bytes = grown;
    return cudaSuccess;
  }

  template <typename T>
  [[nodiscard]] T* as() const {
    return static_cast<T*>(data);
  }
};

// ---- Device helpers ---------------------------------------------------------

__host__ __device__ inline bool finite3(float3 p) {
  return isfinite(p.x) && isfinite(p.y) && isfinite(p.z);
}

// The cell of `p` in a grid of cells 1 / inv_cell wide, if it is keyed.
__device__ inline bool cell_of(float3 p, float inv_cell, int3& cell) {
  const float x = floorf(p.x * inv_cell);
  const float y = floorf(p.y * inv_cell);
  const float z = floorf(p.z * inv_cell);
  constexpr float kLimit = static_cast<float>(kAxisOffset);
  if (!(x >= -kLimit && x < kLimit && y >= -kLimit && y < kLimit && z >= -kLimit && z < kLimit)) {
    return false;  // Out of range or not finite.
  }
  cell = make_int3(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z));
  return true;
}

__device__ inline std::uint64_t pack_key(int x, int y, int z) {
  return (static_cast<std::uint64_t>(z + kAxisOffset) << (2 * kAxisBits)) |
         (static_cast<std::uint64_t>(y + kAxisOffset) << kAxisBits) |
         static_cast<std::uint64_t>(x + kAxisOffset);
}

__device__ inline std::uint32_t lower_bound(const std::uint64_t* keys, std::uint32_t count,
                                            std::uint64_t key) {
  std::uint32_t low = 0;
  std::uint32_t high = count;
  while (low < high) {
    const std::uint32_t mid = low + (high - low) / 2U;
    if (keys[mid] < key) {
      low = mid + 1U;
    } else {
      high = mid;
    }
  }
  return low;
}

// A grid as the kernels read it: `cells` sorted keys of occupied cells, the
// range starts[c]..starts[c + 1] of `order` holding the indices of the points
// in cell c.
struct GridView {
  const std::uint64_t* keys{};
  const std::uint32_t* starts{};
  const std::uint32_t* order{};
  const std::uint32_t* cells{};  // In device memory.
  float inv_cell{};
};

// Calls visit(index) for every point in the 27 cells around `p`, in a fixed
// order.
template <typename Visit>
__device__ inline void for_each_nearby(const GridView& grid, float3 p, Visit&& visit) {
  int3 c;
  const std::uint32_t cells = *grid.cells;
  if (cells == 0U || !cell_of(p, grid.inv_cell, c)) {
    return;
  }
  const int x_low = max(c.x - 1, -kAxisOffset);
  const int x_high = min(c.x + 1, kAxisOffset - 1);
  for (int z = c.z - 1; z <= c.z + 1; ++z) {
    if (z < -kAxisOffset || z >= kAxisOffset) continue;
    for (int y = c.y - 1; y <= c.y + 1; ++y) {
      if (y < -kAxisOffset || y >= kAxisOffset) continue;
      const std::uint64_t high = pack_key(x_high, y, z);
      for (std::uint32_t cell = lower_bound(grid.keys, cells, pack_key(x_low, y, z));
           cell < cells && grid.keys[cell] <= high; ++cell) {
        for (std::uint32_t j = grid.starts[cell]; j < grid.starts[cell + 1U]; ++j) {
          visit(grid.order[j]);
        }
      }
    }
  }
}

__device__ inline float3 sub(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float3 cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// ---- Kernels ----------------------------------------------------------------

// Each of `capacity` points' voxel or cell key, kInvalidKey if it has none or
// lies beyond `count` (all are points without a count), and its index.
__global__ void key_kernel(const float3* points, const std::uint32_t* count, std::uint32_t capacity,
                           float inv_cell, std::uint64_t* keys, std::uint32_t* indices) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= capacity) return;
  int3 c;
  const bool point = count == nullptr || i < *count;
  keys[i] = point && cell_of(points[i], inv_cell, c) ? pack_key(c.x, c.y, c.z) : kInvalidKey;
  indices[i] = i;
}

// Drops the run of invalid keys, which sorts last, from the runs.
__global__ void valid_runs_kernel(const std::uint64_t* unique_keys, std::uint32_t* run_counts,
                                  std::uint32_t* runs) {
  const std::uint32_t n = *runs;
  if (n > 0U && unique_keys[n - 1U] == kInvalidKey) {
    run_counts[n - 1U] = 0U;
    *runs = n - 1U;
  }
}

__host__ __device__ inline std::uint32_t mix(std::uint32_t x) {
  // A 32-bit integer hash (lowbias32).
  x ^= x >> 16;
  x *= 0x7FEB352DU;
  x ^= x >> 15;
  x *= 0x846CA68BU;
  x ^= x >> 16;
  return x;
}

// One point of each voxel: the one whose index hashes lowest (the lower
// index on a tie). A centroid would be the voxel's center wherever noise fills
// the voxel evenly, and the subsampled cloud a lattice fixed to the camera,
// which ICP would align with the other cloud's lattice instead of the
// surfaces; a pseudo-random member is spread over the voxel like the points.
__global__ void representative_kernel(const float3* points, const std::uint32_t* order,
                                      const std::uint32_t* starts, const std::uint32_t* voxels,
                                      std::uint32_t capacity, float3* out) {
  const std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= capacity || v >= *voxels) return;
  const std::uint32_t begin = starts[v];
  const std::uint32_t end = starts[v + 1U];
  std::uint32_t best = order[begin];
  std::uint32_t best_hash = mix(best);
  for (std::uint32_t j = begin + 1U; j < end; ++j) {
    const std::uint32_t index = order[j];
    const std::uint32_t h = mix(index);
    if (h < best_hash || (h == best_hash && index < best)) {
      best = index;
      best_hash = h;
    }
  }
  out[v] = points[best];
}

struct Finite {
  __host__ __device__ bool operator()(const float3& p) const { return finite3(p); }
};

// The unit eigenvector of the smallest eigenvalue of the symmetric 3x3
// matrix [a00 a01 a02; . a11 a12; . . a22], or false when that eigenvalue is
// not separated from the middle one (no unique normal).
__device__ inline bool smallest_eigenvector(double a00, double a01, double a02, double a11,
                                            double a12, double a22, double3& v) {
  const double p1 = a01 * a01 + a02 * a02 + a12 * a12;
  const double q = (a00 + a11 + a22) / 3.0;
  const double b00 = a00 - q;
  const double b11 = a11 - q;
  const double b22 = a22 - q;
  const double p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * p1;
  if (!(p2 > 0.0)) {
    return false;  // A multiple of the identity.
  }
  const double p = sqrt(p2 / 6.0);
  const double det =
      b00 * (b11 * b22 - a12 * a12) - a01 * (a01 * b22 - a12 * a02) + a02 * (a01 * a12 - b11 * a02);
  const double r = fmin(fmax(det / (2.0 * p * p * p), -1.0), 1.0);
  const double phi = acos(r) / 3.0;
  const double largest = q + 2.0 * p * cos(phi);
  const double smallest = q + 2.0 * p * cos(phi + 2.0943951023931957);  // + 2 pi / 3
  const double middle = 3.0 * q - largest - smallest;
  // Rows of A - smallest I; their cross products span its null space.
  const double3 r0 = make_double3(a00 - smallest, a01, a02);
  const double3 r1 = make_double3(a01, a11 - smallest, a12);
  const double3 r2 = make_double3(a02, a12, a22 - smallest);
  const auto crossd = [](double3 a, double3 b) {
    return make_double3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
  };
  const double3 c[3] = {crossd(r0, r1), crossd(r0, r2), crossd(r1, r2)};
  int best = 0;
  double best_norm = -1.0;
  for (int i = 0; i < 3; ++i) {
    const double n = c[i].x * c[i].x + c[i].y * c[i].y + c[i].z * c[i].z;
    if (n > best_norm) {
      best_norm = n;
      best = i;
    }
  }
  // A line or a point: the two smallest eigenvalues coincide.
  if (!(middle - smallest > 1e-6 * fabs(largest)) || !(best_norm > 0.0)) {
    return false;
  }
  const double inv = 1.0 / sqrt(best_norm);
  v = make_double3(c[best].x * inv, c[best].y * inv, c[best].z * inv);
  return true;
}

// Each point's normal from its neighbors within `radius` (at least 3, itself
// included), facing the origin; NaN where there is none.
__global__ void normal_kernel(const float3* points, const std::uint32_t* count,
                              std::uint32_t capacity, GridView grid, float radius,
                              float3* normals) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= capacity || i >= *count) return;
  const float3 p = points[i];
  const float r2 = radius * radius;
  double n = 0.0;
  double sx = 0.0, sy = 0.0, sz = 0.0;
  double sxx = 0.0, sxy = 0.0, sxz = 0.0, syy = 0.0, syz = 0.0, szz = 0.0;
  for_each_nearby(grid, p, [&](std::uint32_t j) {
    const float3 d = sub(points[j], p);
    if (dot(d, d) > r2) return;
    n += 1.0;
    sx += d.x;
    sy += d.y;
    sz += d.z;
    sxx += static_cast<double>(d.x) * d.x;
    sxy += static_cast<double>(d.x) * d.y;
    sxz += static_cast<double>(d.x) * d.z;
    syy += static_cast<double>(d.y) * d.y;
    syz += static_cast<double>(d.y) * d.z;
    szz += static_cast<double>(d.z) * d.z;
  });
  const float nan = __int_as_float(0x7FC00000);
  double3 v;
  if (n < 3.0) {
    normals[i] = make_float3(nan, nan, nan);
    return;
  }
  const double mx = sx / n, my = sy / n, mz = sz / n;
  if (!smallest_eigenvector(sxx / n - mx * mx, sxy / n - mx * my, sxz / n - mx * mz,
                            syy / n - my * my, syz / n - my * mz, szz / n - mz * mz, v)) {
    normals[i] = make_float3(nan, nan, nan);
    return;
  }
  // Face the sensor at the origin: n . (0 - p) >= 0.
  if (v.x * p.x + v.y * p.y + v.z * p.z > 0.0) {
    v = make_double3(-v.x, -v.y, -v.z);
  }
  normals[i] =
      make_float3(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z));
}

// A rigid transform for the kernels: rows of R, then t.
struct DeviceTransform {
  float r[9];
  float t[3];
};

DeviceTransform to_device(const Matrix4& m) {
  DeviceTransform out{};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      out.r[3 * row + col] = static_cast<float>(m[4 * row + col]);
    }
    out.t[row] = static_cast<float>(m[4 * row + 3]);
  }
  return out;
}

__device__ inline float3 apply(const DeviceTransform& m, float3 p) {
  return make_float3(m.r[0] * p.x + m.r[1] * p.y + m.r[2] * p.z + m.t[0],
                     m.r[3] * p.x + m.r[4] * p.y + m.r[5] * p.z + m.t[1],
                     m.r[6] * p.x + m.r[7] * p.y + m.r[8] * p.z + m.t[2]);
}

// The iterations' state, in device memory: the step kernel updates it, and
// the host reads it once per batch of iterations.
struct LoopState {
  double transform[16];
  // The previous pass's RMSE and correspondences, -1 after a distance change.
  double previous_rmse;
  double previous_count;
  int level;       // The correspondence distance's index, coarse to fine.
  int iterations;  // Updates applied.
  int status;      // An IcpStatus, when finished.
  int finished;
};

struct AssociateParams {
  const float3* source{};
  const std::uint32_t* source_count{};
  const float3* target{};
  const float3* normals{};  // Point-to-plane only.
  GridView grids[kMaxLevels];
  float max_distance2[kMaxLevels]{};
  const LoopState* state{};
  // The distance level to use; -1 for the state's. With one, the pass runs
  // even when the iterations have finished: the final evaluation.
  int level{-1};
  bool point_to_plane{};
  RobustKernel kernel{RobustKernel::kNone};
  float scale{};
};

struct Sums {
  float v[kSums];
};

struct AddSums {
  __device__ Sums operator()(const Sums& a, const Sums& b) const {
    Sums out;
#pragma unroll
    for (int k = 0; k < kSums; ++k) {
      out.v[k] = a.v[k] + b.v[k];
    }
    return out;
  }
};

__device__ inline float robust_weight(RobustKernel kernel, float scale, float residual) {
  const float r = fabsf(residual);
  switch (kernel) {
    case RobustKernel::kHuber:
      return r <= scale ? 1.0F : scale / r;
    case RobustKernel::kTukey: {
      if (r >= scale) return 0.0F;
      const float u = 1.0F - (r / scale) * (r / scale);
      return u * u;
    }
    case RobustKernel::kNone:
    default:
      return 1.0F;
  }
}

// Adds w * J^T J and w * J^T r for one residual r with Jacobian J = [a, b]
// (rotation, then translation).
__device__ inline void add_row(Sums& s, float3 a, float3 b, float r, float w) {
  const float j[6] = {a.x, a.y, a.z, b.x, b.y, b.z};
  int k = kJtJ;
#pragma unroll
  for (int row = 0; row < 6; ++row) {
#pragma unroll
    for (int col = row; col < 6; ++col) {
      s.v[k++] += w * j[row] * j[col];
    }
    s.v[kJtr + row] += w * j[row] * r;
  }
}

// One association pass: each source point, moved by the transform, is paired
// with its nearest target point within the maximum distance (the lower index
// on a tie), and its terms are summed per block, in a fixed order, into
// `partials` (kSums floats per block).
__global__ void associate_kernel(AssociateParams params, float* partials) {
  using Reduce = cub::BlockReduce<Sums, kBlock>;
  __shared__ typename Reduce::TempStorage storage;
  __shared__ DeviceTransform transform;
  const LoopState& state = *params.state;
  if (params.level < 0 && state.finished != 0) {
    return;  // Done: the rest of the batch does nothing.
  }
  const int level = params.level < 0 ? state.level : params.level;
  if (threadIdx.x == 0) {
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        transform.r[3 * row + col] = static_cast<float>(state.transform[4 * row + col]);
      }
      transform.t[row] = static_cast<float>(state.transform[4 * row + 3]);
    }
  }
  __syncthreads();
  const GridView& grid = params.grids[level];
  Sums s;
#pragma unroll
  for (int k = 0; k < kSums; ++k) s.v[k] = 0.0F;

  const std::uint32_t count = *params.source_count;
  const std::uint32_t stride = gridDim.x * blockDim.x;
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
    const float3 p = apply(transform, params.source[i]);
    float best = params.max_distance2[level];
    std::uint32_t match = ~0U;
    for_each_nearby(grid, p, [&](std::uint32_t j) {
      const float3 d = sub(params.target[j], p);
      const float d2 = dot(d, d);
      if (d2 < best || (d2 == best && j < match)) {
        best = d2;
        match = j;
      }
    });
    if (match == ~0U) continue;
    const float3 q = params.target[match];
    const float3 diff = sub(p, q);
    if (params.point_to_plane) {
      const float3 n = params.normals[match];
      if (!finite3(n)) continue;  // No normal: no correspondence.
      const float r = dot(n, diff);
      add_row(s, cross(p, n), n, r, robust_weight(params.kernel, params.scale, r));
    } else {
      const float w = robust_weight(params.kernel, params.scale, sqrtf(best));
      add_row(s, cross(p, make_float3(1.0F, 0.0F, 0.0F)), make_float3(1.0F, 0.0F, 0.0F), diff.x, w);
      add_row(s, cross(p, make_float3(0.0F, 1.0F, 0.0F)), make_float3(0.0F, 1.0F, 0.0F), diff.y, w);
      add_row(s, cross(p, make_float3(0.0F, 0.0F, 1.0F)), make_float3(0.0F, 0.0F, 1.0F), diff.z, w);
    }
    s.v[kSquaredDistance] += best;
    s.v[kCount] += 1.0F;
  }
  const Sums total = Reduce(storage).Reduce(s, AddSums{});
  if (threadIdx.x == 0) {
#pragma unroll
    for (int k = 0; k < kSums; ++k) partials[blockIdx.x * kSums + k] = total.v[k];
  }
}

// Sums the blocks' partials in block order, in double: one thread per sum.
// Does nothing once the iterations have finished, unless `final`.
__global__ void total_kernel(const float* partials, int blocks, const LoopState* state, bool final,
                             double* totals) {
  const int k = threadIdx.x;
  if (k >= kSums || (!final && state->finished != 0)) return;
  double sum = 0.0;
  for (int b = 0; b < blocks; ++b) sum += partials[b * kSums + k];
  totals[k] = sum;
}

struct StepParams {
  double min_correspondences{};
  double relative_rmse_epsilon{};
  double translation_epsilon{};
  double rotation_epsilon{};
  int max_iterations{};
  int finest{};
};

// One iteration's decision, on one thread: give up without enough
// correspondences; move to a finer distance, or finish, once nothing changes
// or an update is small enough; otherwise solve the normal equations and
// update the transform.
__global__ void step_kernel(const double* totals, StepParams p, LoopState* state) {
  LoopState& s = *state;
  if (s.finished != 0) return;
  // Converged at this distance: on to the next, or done at the finest.
  const auto settle = [&]() {
    if (s.level == p.finest) {
      s.status = static_cast<int>(IcpStatus::kConverged);
      s.finished = 1;
      return;
    }
    ++s.level;
    s.previous_rmse = -1.0;
    s.previous_count = -1.0;
  };
  const double matches = totals[kCount];
  if (matches < p.min_correspondences) {
    s.status = static_cast<int>(IcpStatus::kTooFewCorrespondences);
    s.finished = 1;
    return;
  }
  const double rmse = sqrt(totals[kSquaredDistance] / matches);
  // Nothing changes any more: the same correspondences at the same error.
  if (matches == s.previous_count &&
      fabs(rmse - s.previous_rmse) <= p.relative_rmse_epsilon * rmse) {
    settle();
    return;
  }
  s.previous_rmse = rmse;
  s.previous_count = matches;
  double x[6];
  if (!detail::solve_normal_equations(totals, x)) {
    s.status = static_cast<int>(IcpStatus::kDegenerate);
    s.finished = 1;
    return;
  }
  detail::apply_step(x, s.transform);
  ++s.iterations;
  if (sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) <= p.rotation_epsilon &&
      sqrt(x[3] * x[3] + x[4] * x[4] + x[5] * x[5]) <= p.translation_epsilon) {
    settle();
  }
  if (s.finished == 0 && s.iterations >= p.max_iterations) {
    s.status = static_cast<int>(IcpStatus::kMaxIterations);
    s.finished = 1;
  }
}

__global__ void transform_kernel(const float* in, float* out, std::uint64_t count,
                                 DeviceTransform m) {
  const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float3 p = make_float3(in[3 * i], in[3 * i + 1], in[3 * i + 2]);
  if (!finite3(p)) {
    if (out != in) {
      out[3 * i] = p.x;
      out[3 * i + 1] = p.y;
      out[3 * i + 2] = p.z;
    }
    return;
  }
  const float3 q = apply(m, p);
  out[3 * i] = q.x;
  out[3 * i + 1] = q.y;
  out[3 * i + 2] = q.z;
}

unsigned blocks_for(std::uint64_t count) {
  return static_cast<unsigned>((count + kBlock - 1U) / kBlock);
}

// A sorted grid of a cloud: its occupied cells' keys and point ranges.
struct Grid {
  DeviceBuffer keys;    // uint64 per cell.
  DeviceBuffer starts;  // uint32 per cell, plus one.
  DeviceBuffer order;   // uint32 per point: indices by cell.
  DeviceBuffer cells;   // uint32: the occupied cells.
  float cell{};

  [[nodiscard]] GridView view() const {
    return GridView{keys.as<std::uint64_t>(), starts.as<std::uint32_t>(), order.as<std::uint32_t>(),
                    cells.as<std::uint32_t>(), 1.0F / cell};
  }
};

// A prepared cloud: finite points, subsampled, and normals when needed. Its
// point count stays on the GPU, so that preparing it needs no round trip to
// the host; `capacity`, the points it came from, bounds it, and sizes every
// launch over it, so that the sums' order depends only on the input.
struct Cloud {
  DeviceBuffer points;   // float3.
  DeviceBuffer normals;  // float3, point-to-plane targets only.
  DeviceBuffer count;    // uint32.
  std::uint32_t capacity{};
};

}  // namespace

struct IcpRegistration::Impl {
  IcpOptions options;
  Cloud target;
  Cloud source;
  // The correspondence distances, coarse to fine, and the target's grid for
  // each.
  std::array<float, kMaxLevels> distances{};
  int levels{1};
  std::array<Grid, kMaxLevels> grids;
  Grid normal_grid;  // The target's, cells normal_radius wide, when that differs.
  bool has_target{};
  // The passes the last registration took, to size the first batch of the next.
  int last_passes{};

  // Scratch.
  DeviceBuffer keys_in, keys_out, indices_in, run_counts, cub_temp, partials, totals, state;
  DeviceBuffer scratch_order, scratch_unique, scratch_starts, scratch_cells;

  // What the host reads back, in pinned memory.
  struct Mirror {
    LoopState state;
    double totals[kSums];
    std::uint32_t source_count;
    std::uint32_t target_count;
  };
  Mirror* mirror{};

  explicit Impl(IcpOptions o) : options(std::move(o)) {
    const float finest = options.min_correspondence_distance > 0.0F
                             ? options.min_correspondence_distance
                             : options.max_correspondence_distance;
    distances[0] = options.max_correspondence_distance;
    while (levels < kMaxLevels && distances[levels - 1] > finest) {
      distances[levels] = std::max(distances[levels - 1] * 0.5F, finest);
      ++levels;
    }
  }
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  ~Impl() {
    if (mirror != nullptr) cudaFreeHost(mirror);
  }

  [[nodiscard]] float normal_radius() const {
    return options.normal_radius > 0.0F ? options.normal_radius
                                        : options.max_correspondence_distance;
  }

  cudaError_t ensure_host() {
    if (mirror == nullptr) {
      ICP_TRY(cudaMallocHost(reinterpret_cast<void**>(&mirror), sizeof(Mirror)));
    }
    return cudaSuccess;
  }

  // Runs a CUB device algorithm: sizes its temporary storage, then runs it.
  template <typename Algorithm>
  cudaError_t cub_run(Algorithm&& algorithm, cudaStream_t stream) {
    std::size_t bytes = 0;
    ICP_TRY(algorithm(static_cast<void*>(nullptr), bytes));
    ICP_TRY(cub_temp.reserve(std::max<std::size_t>(bytes, 1U), stream));
    bytes = cub_temp.bytes;
    return algorithm(cub_temp.data, bytes);
  }

  // Sorts the first `capacity` points (those before `*count`, or all without
  // a count) by their cell key into `order`, and finds the runs of equal
  // keys: the unique valid keys into `unique_keys`, their starts into
  // `starts`, and their number into `cells`, all on the GPU. Nothing waits for
  // the GPU.
  cudaError_t sort_into_cells(const float3* points, const std::uint32_t* count,
                              std::uint32_t capacity, float cell, DeviceBuffer& order,
                              DeviceBuffer& unique_keys, DeviceBuffer& starts, DeviceBuffer& cells,
                              cudaStream_t stream) {
    const std::size_t n = std::max<std::uint32_t>(capacity, 1U);
    ICP_TRY(keys_in.reserve(n * sizeof(std::uint64_t), stream));
    ICP_TRY(keys_out.reserve(n * sizeof(std::uint64_t), stream));
    ICP_TRY(indices_in.reserve(n * sizeof(std::uint32_t), stream));
    ICP_TRY(order.reserve(n * sizeof(std::uint32_t), stream));
    ICP_TRY(unique_keys.reserve(n * sizeof(std::uint64_t), stream));
    ICP_TRY(run_counts.reserve(n * sizeof(std::uint32_t), stream));
    ICP_TRY(starts.reserve((n + 1U) * sizeof(std::uint32_t), stream));
    ICP_TRY(cells.reserve(sizeof(std::uint32_t), stream));
    ICP_TRY(cudaMemsetAsync(starts.data, 0, sizeof(std::uint32_t), stream));
    if (capacity == 0U) {
      return cudaMemsetAsync(cells.data, 0, sizeof(std::uint32_t), stream);
    }
    key_kernel<<<blocks_for(capacity), kBlock, 0, stream>>>(points, count, capacity, 1.0F / cell,
                                                            keys_in.as<std::uint64_t>(),
                                                            indices_in.as<std::uint32_t>());
    ICP_TRY(cudaGetLastError());
    const int items = static_cast<int>(capacity);
    ICP_TRY(cub_run(
        [&](void* temp, std::size_t& bytes) {
          return cub::DeviceRadixSort::SortPairs(
              temp, bytes, keys_in.as<std::uint64_t>(), keys_out.as<std::uint64_t>(),
              indices_in.as<std::uint32_t>(), order.as<std::uint32_t>(), items, 0, 64, stream);
        },
        stream));
    // Counts past the last run stay 0, so the scan over all of them is exact.
    ICP_TRY(cudaMemsetAsync(run_counts.data, 0, n * sizeof(std::uint32_t), stream));
    ICP_TRY(cub_run(
        [&](void* temp, std::size_t& bytes) {
          return cub::DeviceRunLengthEncode::Encode(
              temp, bytes, keys_out.as<std::uint64_t>(), unique_keys.as<std::uint64_t>(),
              run_counts.as<std::uint32_t>(), cells.as<std::uint32_t>(), items, stream);
        },
        stream));
    valid_runs_kernel<<<1, 1, 0, stream>>>(
        unique_keys.as<std::uint64_t>(), run_counts.as<std::uint32_t>(), cells.as<std::uint32_t>());
    ICP_TRY(cudaGetLastError());
    return cub_run(
        [&](void* temp, std::size_t& bytes) {
          return cub::DeviceScan::InclusiveSum(temp, bytes, run_counts.as<std::uint32_t>(),
                                               starts.as<std::uint32_t>() + 1, items, stream);
        },
        stream);
  }

  // Drops non-finite points and, with a voxel size, keeps one point of each
  // voxel. Nothing waits for the GPU.
  cudaError_t prepare(const float* points, std::uint32_t count, Cloud& cloud, cudaStream_t stream) {
    cloud.capacity = count;
    const std::size_t n = std::max<std::uint32_t>(count, 1U);
    ICP_TRY(cloud.points.reserve(n * sizeof(float3), stream));
    ICP_TRY(cloud.count.reserve(sizeof(std::uint32_t), stream));
    if (count == 0U) {
      return cudaMemsetAsync(cloud.count.data, 0, sizeof(std::uint32_t), stream);
    }
    const auto* in = reinterpret_cast<const float3*>(points);
    if (options.voxel_size > 0.0F) {
      ICP_TRY(sort_into_cells(in, nullptr, count, options.voxel_size, scratch_order, scratch_unique,
                              scratch_starts, scratch_cells, stream));
      representative_kernel<<<blocks_for(count), kBlock, 0, stream>>>(
          in, scratch_order.as<std::uint32_t>(), scratch_starts.as<std::uint32_t>(),
          scratch_cells.as<std::uint32_t>(), count, cloud.points.as<float3>());
      ICP_TRY(cudaGetLastError());
      return cudaMemcpyAsync(cloud.count.data, scratch_cells.data, sizeof(std::uint32_t),
                             cudaMemcpyDeviceToDevice, stream);
    }
    const int items = static_cast<int>(count);
    return cub_run(
        [&](void* temp, std::size_t& bytes) {
          return cub::DeviceSelect::If(temp, bytes, in, cloud.points.as<float3>(),
                                       cloud.count.as<std::uint32_t>(), items, Finite{}, stream);
        },
        stream);
  }

  cudaError_t build_grid(const Cloud& cloud, float cell, Grid& g, cudaStream_t stream) {
    g.cell = cell;
    return sort_into_cells(cloud.points.as<float3>(), cloud.count.as<std::uint32_t>(),
                           cloud.capacity, cell, g.order, g.keys, g.starts, g.cells, stream);
  }

  // Builds the target's grid for every distance and, for point-to-plane, its
  // normals. Nothing waits for the GPU.
  cudaError_t index_target(cudaStream_t stream) {
    for (int level = 0; level < levels; ++level) {
      ICP_TRY(build_grid(target, distances[level] * kCellMargin, grids[level], stream));
    }
    if (options.method != IcpMethod::kPointToPlane) {
      return cudaSuccess;
    }
    ICP_TRY(target.normals.reserve(std::max<std::uint32_t>(target.capacity, 1U) * sizeof(float3),
                                   stream));
    if (target.capacity == 0U) {
      return cudaSuccess;
    }
    const float radius = normal_radius();
    const Grid* neighbors = &grids[0];
    if (radius != distances[0]) {
      ICP_TRY(build_grid(target, radius * kCellMargin, normal_grid, stream));
      neighbors = &normal_grid;
    }
    normal_kernel<<<blocks_for(target.capacity), kBlock, 0, stream>>>(
        target.points.as<float3>(), target.count.as<std::uint32_t>(), target.capacity,
        neighbors->view(), radius, target.normals.as<float3>());
    return cudaGetLastError();
  }

  // Queues one association pass and its sum into `totals`: at the state's
  // distance (`level` -1), skipped once the iterations have finished, or at
  // `level` regardless.
  cudaError_t queue_pass(int level, cudaStream_t stream) {
    const int blocks =
        static_cast<int>(std::min<std::uint32_t>(blocks_for(source.capacity), kMaxReduceBlocks));
    AssociateParams params;
    params.source = source.points.as<float3>();
    params.source_count = source.count.as<std::uint32_t>();
    params.target = target.points.as<float3>();
    params.point_to_plane = options.method == IcpMethod::kPointToPlane;
    params.normals = params.point_to_plane ? target.normals.as<float3>() : nullptr;
    for (int l = 0; l < levels; ++l) {
      params.grids[l] = grids[l].view();
      params.max_distance2[l] = distances[l] * distances[l];
    }
    params.state = state.as<LoopState>();
    params.level = level;
    params.kernel = options.robust_kernel;
    params.scale = options.robust_scale;
    associate_kernel<<<blocks, kBlock, 0, stream>>>(params, partials.as<float>());
    ICP_TRY(cudaGetLastError());
    total_kernel<<<1, kSums, 0, stream>>>(partials.as<float>(), blocks, state.as<LoopState>(),
                                          level >= 0, totals.as<double>());
    return cudaGetLastError();
  }

  // Runs the iterations from `initial` and the final evaluation, leaving the
  // state, the final sums and both clouds' point counts in `mirror`. The host
  // waits for the GPU once per batch of iterations, and once at the end.
  cudaError_t iterate(const Matrix4& initial, cudaStream_t stream) {
    const int blocks =
        static_cast<int>(std::min<std::uint32_t>(blocks_for(source.capacity), kMaxReduceBlocks));
    ICP_TRY(partials.reserve(static_cast<std::size_t>(std::max(blocks, 1)) * kSums * sizeof(float),
                             stream));
    ICP_TRY(totals.reserve(kSums * sizeof(double), stream));
    ICP_TRY(state.reserve(sizeof(LoopState), stream));
    LoopState& start = mirror->state;
    std::copy(initial.begin(), initial.end(), start.transform);
    start.previous_rmse = -1.0;
    start.previous_count = -1.0;
    start.level = 0;
    start.iterations = 0;
    start.status = static_cast<int>(IcpStatus::kMaxIterations);
    start.finished = 0;
    ICP_TRY(cudaMemcpyAsync(state.data, &start, sizeof(LoopState), cudaMemcpyHostToDevice, stream));

    const StepParams step{.min_correspondences = static_cast<double>(options.min_correspondences),
                          .relative_rmse_epsilon = options.relative_rmse_epsilon,
                          .translation_epsilon = options.translation_epsilon,
                          .rotation_epsilon = options.rotation_epsilon,
                          .max_iterations = options.max_iterations,
                          .finest = levels - 1};
    // Each pass either updates the transform or moves to a finer distance,
    // so this many passes always finish. The first batch is as long as the
    // last registration took, the rest are short; passes past the end do
    // nothing.
    const int budget = options.max_iterations + levels;
    int queued = 0;
    int batch = std::clamp(last_passes + 1, 4, budget);
    if (source.capacity != 0U) {
      while (queued < budget) {
        batch = std::min(batch, budget - queued);
        for (int pass = 0; pass < batch; ++pass) {
          ICP_TRY(queue_pass(-1, stream));
          step_kernel<<<1, 1, 0, stream>>>(totals.as<double>(), step, state.as<LoopState>());
          ICP_TRY(cudaGetLastError());
        }
        queued += batch;
        ICP_TRY(cudaMemcpyAsync(&mirror->state, state.data, sizeof(LoopState),
                                cudaMemcpyDeviceToHost, stream));
        ICP_TRY(cudaStreamSynchronize(stream));
        if (mirror->state.finished != 0) break;
        batch = 8;
      }
      last_passes = mirror->state.iterations + mirror->state.level + 1;
      // The fitness and error at the final transform, at the finest distance.
      ICP_TRY(queue_pass(levels - 1, stream));
      ICP_TRY(cudaMemcpyAsync(mirror->totals, totals.data, kSums * sizeof(double),
                              cudaMemcpyDeviceToHost, stream));
    } else {
      std::fill(std::begin(mirror->totals), std::end(mirror->totals), 0.0);
      mirror->state.status = static_cast<int>(IcpStatus::kTooFewCorrespondences);
      mirror->state.finished = 1;
    }
    ICP_TRY(cudaMemcpyAsync(&mirror->source_count, source.count.data, sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost, stream));
    ICP_TRY(cudaMemcpyAsync(&mirror->target_count, target.count.data, sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost, stream));
    return cudaStreamSynchronize(stream);
  }
};

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

}  // namespace

void validate_options(const IcpOptions& options) {
  require(options.method == IcpMethod::kPointToPoint || options.method == IcpMethod::kPointToPlane,
          "method must be point-to-point or point-to-plane");
  require(std::isfinite(options.max_correspondence_distance) &&
              options.max_correspondence_distance > 0.0F,
          "max_correspondence_distance must be finite and positive");
  require(std::isfinite(options.min_correspondence_distance) &&
              options.min_correspondence_distance >= 0.0F &&
              options.min_correspondence_distance <= options.max_correspondence_distance &&
              (options.min_correspondence_distance == 0.0F ||
               options.max_correspondence_distance <= 128.0F * options.min_correspondence_distance),
          "min_correspondence_distance must be 0, or at most the maximum and at least 1/128 of it");
  require(options.max_iterations >= 1 && options.max_iterations <= 1000,
          "max_iterations must be 1..1000");
  require(std::isfinite(options.translation_epsilon) && options.translation_epsilon >= 0.0 &&
              std::isfinite(options.rotation_epsilon) && options.rotation_epsilon >= 0.0 &&
              std::isfinite(options.relative_rmse_epsilon) && options.relative_rmse_epsilon >= 0.0,
          "the convergence epsilons must be finite and not negative");
  require(std::isfinite(options.voxel_size) && options.voxel_size >= 0.0F,
          "voxel_size must be finite and not negative");
  require(std::isfinite(options.normal_radius) && options.normal_radius >= 0.0F,
          "normal_radius must be finite and not negative");
  require(options.robust_kernel == RobustKernel::kNone ||
              options.robust_kernel == RobustKernel::kHuber ||
              options.robust_kernel == RobustKernel::kTukey,
          "robust_kernel must be none, Huber or Tukey");
  require(options.robust_kernel == RobustKernel::kNone ||
              (std::isfinite(options.robust_scale) && options.robust_scale > 0.0F),
          "robust_scale must be finite and positive with a robust kernel");
  require(options.min_correspondences >= 6U,
          "min_correspondences must be at least 6, the unknowns of a rigid transform");
}

const char* to_string(IcpStatus status) {
  switch (status) {
    case IcpStatus::kConverged:
      return "converged";
    case IcpStatus::kMaxIterations:
      return "max_iterations";
    case IcpStatus::kTooFewCorrespondences:
      return "too_few_correspondences";
    case IcpStatus::kDegenerate:
      return "degenerate";
  }
  return "unknown";
}

IcpRegistration::IcpRegistration(IcpOptions options) {
  validate_options(options);
  impl_ = std::make_unique<Impl>(std::move(options));
}

IcpRegistration::~IcpRegistration() = default;
IcpRegistration::IcpRegistration(IcpRegistration&&) noexcept = default;
IcpRegistration& IcpRegistration::operator=(IcpRegistration&&) noexcept = default;

const IcpOptions& IcpRegistration::options() const noexcept { return impl_->options; }

bool IcpRegistration::has_target() const noexcept { return impl_->has_target; }

cudaError_t IcpRegistration::target_points(cudaStream_t stream, std::uint32_t& count) {
  Impl& impl = *impl_;
  count = 0;
  if (!impl.has_target) {
    return cudaSuccess;
  }
  ICP_TRY(impl.ensure_host());
  ICP_TRY(cudaMemcpyAsync(&impl.mirror->target_count, impl.target.count.data, sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost, stream));
  ICP_TRY(cudaStreamSynchronize(stream));
  count = impl.mirror->target_count;
  return cudaSuccess;
}

cudaError_t IcpRegistration::set_target(const float* points, std::uint32_t count,
                                        cudaStream_t stream) {
  Impl& impl = *impl_;
  impl.has_target = false;
  if (count > kMaxCount || (count != 0U && points == nullptr)) {
    return cudaErrorInvalidValue;
  }
  ICP_TRY(impl.prepare(points, count, impl.target, stream));
  ICP_TRY(impl.index_target(stream));
  impl.has_target = true;
  return cudaSuccess;
}

cudaError_t IcpRegistration::promote_source_to_target(cudaStream_t stream) {
  Impl& impl = *impl_;
  impl.has_target = false;
  std::swap(impl.target, impl.source);
  ICP_TRY(impl.index_target(stream));
  impl.has_target = true;
  return cudaSuccess;
}

cudaError_t IcpRegistration::align(const float* points, std::uint32_t count, const Matrix4& initial,
                                   cudaStream_t stream, IcpResult& result) {
  Impl& impl = *impl_;
  result = IcpResult{};
  result.transform = initial;
  if (count > kMaxCount || (count != 0U && points == nullptr) || !is_rigid_shaped(initial)) {
    return cudaErrorInvalidValue;
  }
  ICP_TRY(impl.ensure_host());
  ICP_TRY(impl.prepare(points, count, impl.source, stream));
  if (!impl.has_target) {
    ICP_TRY(cudaMemcpyAsync(&impl.mirror->source_count, impl.source.count.data,
                            sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
    ICP_TRY(cudaStreamSynchronize(stream));
    result.source_points = impl.mirror->source_count;
    return cudaSuccess;
  }
  ICP_TRY(impl.iterate(initial, stream));
  const Impl::Mirror& m = *impl.mirror;
  std::copy(std::begin(m.state.transform), std::end(m.state.transform), result.transform.begin());
  result.iterations = m.state.iterations;
  result.source_points = m.source_count;
  result.target_points = m.target_count;
  const double matches = m.totals[kCount];
  result.correspondences = static_cast<std::uint32_t>(matches);
  result.fitness = m.source_count == 0U ? 0.0 : matches / static_cast<double>(m.source_count);
  result.rmse = matches > 0.0 ? std::sqrt(m.totals[kSquaredDistance] / matches) : 0.0;
  auto status = static_cast<IcpStatus>(m.state.status);
  if (status == IcpStatus::kConverged &&
      matches < static_cast<double>(impl.options.min_correspondences)) {
    status = IcpStatus::kTooFewCorrespondences;
  }
  result.status = status;
  result.converged = status == IcpStatus::kConverged;
  return cudaSuccess;
}

cudaError_t launch_transform_points(const float* in, float* out, std::uint64_t count,
                                    const Matrix4& transform, cudaStream_t stream) {
  if (count == 0U) {
    return cudaSuccess;
  }
  if (in == nullptr || out == nullptr || !is_rigid_shaped(transform)) {
    return cudaErrorInvalidValue;
  }
  transform_kernel<<<blocks_for(count), kBlock, 0, stream>>>(in, out, count, to_device(transform));
  return cudaGetLastError();
}

}  // namespace holoscan::icp
