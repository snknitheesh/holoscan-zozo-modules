// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include <cub/device/device_radix_sort.cuh>         // NOLINT(build/include_order): CUB
#include <cub/device/device_run_length_encode.cuh>  // NOLINT(build/include_order): CUB
#include <cub/device/device_scan.cuh>               // NOLINT(build/include_order): CUB
#include <holoscan/height_map/height_mapper.hpp>

#include "device_buffer.hpp"
#include "kernel_common.cuh"

namespace holoscan::height_map {
namespace {

// A cell never observed, or cleared, has this stamp.
constexpr std::int64_t kNever = std::numeric_limits<std::int64_t>::min();

// The map's per-cell state, structure of arrays, `cells` long each.
struct CellState {
  std::int64_t* stamp;  // Of the latest observation; kNever when unknown.
  float* fused;         // kFused: the estimate.
  float* fused_var;     // kFused: its variance.
  float* lo;            // Latest observation: lowest height,
  float* hi;            // highest,
  float* mean;          // mean,
  float* spread;        // population variance of the heights,
  float* count;         // and its points.
};

constexpr std::size_t kStateFloats = 7;
constexpr std::size_t kStateBytesPerCell = sizeof(std::int64_t) + kStateFloats * sizeof(float);

CellState state_view(void* base, std::size_t cells) {
  auto* stamp = static_cast<std::int64_t*>(base);
  auto* floats = reinterpret_cast<float*>(stamp + cells);
  return CellState{stamp,
                   floats,
                   floats + cells,
                   floats + 2 * cells,
                   floats + 3 * cells,
                   floats + 4 * cells,
                   floats + 5 * cells,
                   floats + 6 * cells};
}

// The transform in float, for the kernels.
struct Pose {
  float m[12];
};

__device__ inline void clear_cell(const CellState& s, std::size_t i) {
  const float nan = __int_as_float(0x7fc00000);
  s.stamp[i] = kNever;
  s.fused[i] = nan;
  s.fused_var[i] = nan;
  s.lo[i] = nan;
  s.hi[i] = nan;
  s.mean[i] = nan;
  s.spread[i] = nan;
  s.count[i] = nan;
}

__global__ void clear_kernel(CellState s, std::size_t cells) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < cells) {
    clear_cell(s, i);
  }
}

// Cell (r, c) of the moved window was cell (r + dy, c + dx) of the old one.
__global__ void shift_kernel(CellState from, CellState to, int rows, int cols, std::int64_t dx,
                             std::int64_t dy) {
  const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int r = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (c >= cols || r >= rows) {
    return;
  }
  const std::size_t i = static_cast<std::size_t>(r) * cols + c;
  const std::int64_t sc = c + dx;
  const std::int64_t sr = r + dy;
  if (sc < 0 || sr < 0 || sc >= cols || sr >= rows) {
    clear_cell(to, i);
    return;
  }
  const std::size_t j = static_cast<std::size_t>(sr) * cols + static_cast<std::size_t>(sc);
  to.stamp[i] = from.stamp[j];
  to.fused[i] = from.fused[j];
  to.fused_var[i] = from.fused_var[j];
  to.lo[i] = from.lo[j];
  to.hi[i] = from.hi[j];
  to.mean[i] = from.mean[j];
  to.spread[i] = from.spread[j];
  to.count[i] = from.count[j];
}

struct AgeParams {
  std::int64_t now;
  float elapsed;            // Seconds since the previous update.
  std::int64_t max_age_ns;  // 0: never.
  float growth;             // m^2/s.
  float max_variance;       // 0: never.
  bool accumulate;
  bool fused;
};

__global__ void age_kernel(CellState s, std::size_t cells, AgeParams p) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= cells || s.stamp[i] == kNever) {
    return;
  }
  if (!p.accumulate || (p.max_age_ns > 0 && p.now - s.stamp[i] > p.max_age_ns)) {
    clear_cell(s, i);
    return;
  }
  if (p.fused) {
    const float var = s.fused_var[i] + p.growth * p.elapsed;
    if (p.max_variance > 0.0F && var > p.max_variance) {
      clear_cell(s, i);
      return;
    }
    s.fused_var[i] = var;
  }
}

struct BinParams {
  Pose pose;
  float min_height;
  float max_height;
  float min_range;
  float max_range;
  float noise_base;
  float noise_quadratic;
  // The pose's translation is relative to the window's corner.
  float resolution;
  int rows;
  int cols;
  std::uint32_t invalid_key;
};

// Each point's cell (invalid_key when it is dropped), map-frame height and
// inverse variance.
__global__ void bin_kernel(const float* __restrict__ xyz, std::uint32_t count, BinParams p,
                           std::uint32_t* __restrict__ keys, float2* __restrict__ values) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const float x = xyz[3 * static_cast<std::size_t>(i)];
  const float y = xyz[3 * static_cast<std::size_t>(i) + 1];
  const float z = xyz[3 * static_cast<std::size_t>(i) + 2];
  std::uint32_t key = p.invalid_key;
  float2 value{0.0F, 0.0F};
  if (isfinite(x) && isfinite(y) && isfinite(z)) {
    const float range = sqrtf(x * x + y * y + z * z);
    const float* m = p.pose.m;
    const float mx = m[0] * x + m[1] * y + m[2] * z + m[3];
    const float my = m[4] * x + m[5] * y + m[6] * z + m[7];
    const float mz = m[8] * x + m[9] * y + m[10] * z + m[11];
    const float u = floorf(mx / p.resolution);
    const float v = floorf(my / p.resolution);
    if (range >= p.min_range && range <= p.max_range && mz >= p.min_height && mz <= p.max_height &&
        u >= 0.0F && v >= 0.0F && u < static_cast<float>(p.cols) &&
        v < static_cast<float>(p.rows)) {
      key = static_cast<std::uint32_t>(v) * static_cast<std::uint32_t>(p.cols) +
            static_cast<std::uint32_t>(u);
      const float sigma = p.noise_base + p.noise_quadratic * (range * range);
      value = float2{mz, 1.0F / (sigma * sigma)};
    }
  }
  keys[i] = key;
  values[i] = value;
}

struct FuseParams {
  std::int64_t now;
  std::uint32_t cells;
  float min_variance;
  float gate;
  float rejection_variance;
  bool fused;
};

// One thread per run of points in one cell, in input order: the observation,
// and its fusion into the cell's estimate.
__global__ void observe_kernel(const std::uint32_t* __restrict__ cells_of_runs,
                               const std::uint32_t* __restrict__ run_counts,
                               const std::uint32_t* __restrict__ run_offsets,
                               const int* __restrict__ runs, const float2* __restrict__ values,
                               CellState s, FuseParams p) {
  const int r = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (r >= *runs) {
    return;
  }
  const std::uint32_t cell = cells_of_runs[r];
  if (cell >= p.cells) {
    return;  // The dropped points.
  }
  const std::uint32_t begin = run_offsets[r];
  const std::uint32_t n = run_counts[r];
  // Sums relative to the first height, which keeps them small.
  const float z0 = values[begin].x;
  float lo = z0;
  float hi = z0;
  float sum = 0.0F;
  for (std::uint32_t k = 0; k < n; ++k) {
    const float z = values[begin + k].x;
    lo = fminf(lo, z);
    hi = fmaxf(hi, z);
    sum += z - z0;
  }
  const float count = static_cast<float>(n);
  const float mean_offset = sum / count;
  float squares = 0.0F;
  float weights = 0.0F;
  float weighted = 0.0F;
  for (std::uint32_t k = 0; k < n; ++k) {
    const float2 value = values[begin + k];
    const float d = (value.x - z0) - mean_offset;
    squares += d * d;
    weights += value.y;
    weighted += value.y * (value.x - z0);
  }
  const float spread = squares / count;
  s.stamp[cell] = p.now;
  s.lo[cell] = lo;
  s.hi[cell] = hi;
  s.mean[cell] = z0 + mean_offset;
  s.spread[cell] = spread;
  s.count[cell] = count;
  if (!p.fused) {
    return;
  }
  const float measured = z0 + weighted / weights;
  const float variance = fmaxf(1.0F / weights + spread, p.min_variance);
  const float estimate = s.fused[cell];
  const float estimate_var = s.fused_var[cell];
  if (!isfinite(estimate) || !isfinite(estimate_var)) {
    s.fused[cell] = measured;
    s.fused_var[cell] = variance;
    return;
  }
  const float innovation = measured - estimate;
  const float total = estimate_var + variance;
  if (p.gate > 0.0F && innovation * innovation > (p.gate * p.gate) * total) {
    if (innovation > 0.0F) {
      s.fused[cell] = measured;
      s.fused_var[cell] = variance;
    } else {
      s.fused_var[cell] = estimate_var + p.rejection_variance;
    }
    return;
  }
  const float gain = estimate_var / total;
  s.fused[cell] = estimate + gain * innovation;
  s.fused_var[cell] = (estimate_var * variance) / total;
}

struct WriteParams {
  std::int64_t now;
  std::size_t cells;
  ElevationMode mode;
};

__global__ void write_kernel(CellState s, WriteParams p, MapLayers out) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= p.cells) {
    return;
  }
  const float nan = __int_as_float(0x7fc00000);
  const bool known = s.stamp[i] != kNever;
  if (out.elevation != nullptr) {
    float value = nan;
    if (known) {
      value = p.mode == ElevationMode::kMax    ? s.hi[i]
              : p.mode == ElevationMode::kMean ? s.mean[i]
                                               : s.fused[i];
    }
    out.elevation[i] = value;
  }
  if (out.variance != nullptr) {
    out.variance[i] = !known ? nan : p.mode == ElevationMode::kFused ? s.fused_var[i] : s.spread[i];
  }
  if (out.count != nullptr) {
    out.count[i] = known ? s.count[i] : nan;
  }
  if (out.min_elevation != nullptr) {
    out.min_elevation[i] = known ? s.lo[i] : nan;
  }
  if (out.max_elevation != nullptr) {
    out.max_elevation[i] = known ? s.hi[i] : nan;
  }
  if (out.time_since_update != nullptr) {
    out.time_since_update[i] = known ? static_cast<float>(p.now - s.stamp[i]) * 1e-9F : nan;
  }
}

// Bits that hold every key up to and including `max_key`.
int key_bits(std::uint32_t max_key) {
  int bits = 1;
  while (bits < 32 && (max_key >> bits) != 0U) {
    ++bits;
  }
  return bits;
}

std::int64_t floor_index(double value) { return static_cast<std::int64_t>(std::floor(value)); }

}  // namespace

bool valid_params(const HeightMapParams& p, const char** what) {
  const auto fail = [what](const char* message) {
    if (what != nullptr) {
      *what = message;
    }
    return false;
  };
  const MapGeometry& g = p.geometry;
  if (!(std::isfinite(g.resolution) && g.resolution > 0.0F)) {
    return fail("resolution must be positive");
  }
  if (g.rows < 1 || g.cols < 1 || g.rows > kMaxGridSide || g.cols > kMaxGridSide) {
    return fail("rows and cols must be 1..8192");
  }
  if (!std::isfinite(g.origin_x) || !std::isfinite(g.origin_y)) {
    return fail("origin must be finite");
  }
  if (p.mode != MapMode::kFixed && p.mode != MapMode::kFollow) {
    return fail("mode must be kFixed or kFollow");
  }
  if (p.elevation != ElevationMode::kMax && p.elevation != ElevationMode::kMean &&
      p.elevation != ElevationMode::kFused) {
    return fail("elevation must be kMax, kMean or kFused");
  }
  if (std::isnan(p.min_height) || std::isnan(p.max_height) || !(p.min_height <= p.max_height)) {
    return fail("min_height must not exceed max_height");
  }
  if (!(p.min_range >= 0.0F) || std::isnan(p.max_range) || !(p.min_range <= p.max_range)) {
    return fail("min_range must be 0..max_range");
  }
  if (!(std::isfinite(p.noise_base) && p.noise_base >= 0.0F) ||
      !(std::isfinite(p.noise_quadratic) && p.noise_quadratic >= 0.0F) ||
      !(p.noise_base > 0.0F || p.noise_quadratic > 0.0F)) {
    return fail("noise_base and noise_quadratic must be finite, not negative, not both 0");
  }
  if (!(std::isfinite(p.min_variance) && p.min_variance > 0.0F)) {
    return fail("min_variance must be positive");
  }
  if (!(std::isfinite(p.mahalanobis_gate) && p.mahalanobis_gate >= 0.0F)) {
    return fail("mahalanobis_gate must not be negative");
  }
  if (!(std::isfinite(p.rejection_variance) && p.rejection_variance >= 0.0F)) {
    return fail("rejection_variance must not be negative");
  }
  if (!(std::isfinite(p.max_age) && p.max_age >= 0.0F)) {
    return fail("max_age must not be negative");
  }
  if (!(std::isfinite(p.variance_growth) && p.variance_growth >= 0.0F)) {
    return fail("variance_growth must not be negative");
  }
  if (!(std::isfinite(p.max_variance) && p.max_variance >= 0.0F)) {
    return fail("max_variance must not be negative");
  }
  return true;
}

bool is_rigid(const Transform& t, double tolerance) {
  for (const double value : t.m) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  const double* m = t.m;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const double dot =
          m[4 * a] * m[4 * b] + m[4 * a + 1] * m[4 * b + 1] + m[4 * a + 2] * m[4 * b + 2];
      if (std::fabs(dot - (a == b ? 1.0 : 0.0)) > tolerance) {
        return false;
      }
    }
  }
  const double det = m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) +
                     m[2] * (m[4] * m[9] - m[5] * m[8]);
  return det > 0.0;
}

struct HeightMapper::Impl {
  HeightMapParams params;
  MapGeometry geometry;     // Where the window is.
  std::int64_t corner_x{};  // The window's corner, in cells from params.geometry.origin.
  std::int64_t corner_y{};
  bool placed{};   // The window has been placed (kFollow).
  bool cleared{};  // The state holds a map (or all unknown).
  std::int64_t last_stamp{};
  bool updated{};
  detail::DeviceBuffer state;
  detail::DeviceBuffer spare;  // The shift's destination.
  detail::DeviceBuffer keys;
  detail::DeviceBuffer sorted_keys;
  detail::DeviceBuffer values;
  detail::DeviceBuffer sorted_values;
  detail::DeviceBuffer run_cells;
  detail::DeviceBuffer run_counts;
  detail::DeviceBuffer run_offsets;
  detail::DeviceBuffer runs;
  detail::DeviceBuffer temp;
};

HeightMapper::HeightMapper(const HeightMapParams& params) : impl_(std::make_unique<Impl>()) {
  const char* what = nullptr;
  if (!valid_params(params, &what)) {
    throw std::invalid_argument(what);
  }
  impl_->params = params;
  impl_->geometry = params.geometry;
}

HeightMapper::~HeightMapper() = default;

const HeightMapParams& HeightMapper::params() const noexcept { return impl_->params; }

MapGeometry HeightMapper::geometry() const noexcept { return impl_->geometry; }

void HeightMapper::reset() noexcept {
  impl_->geometry = impl_->params.geometry;
  impl_->corner_x = 0;
  impl_->corner_y = 0;
  impl_->placed = false;
  impl_->cleared = false;
  impl_->updated = false;
  impl_->last_stamp = 0;
}

cudaError_t HeightMapper::update(const float* xyz, std::uint32_t count,
                                 const Transform& sensor_to_map, std::int64_t stamp_ns,
                                 cudaStream_t stream) {
  Impl& s = *impl_;
  const HeightMapParams& p = s.params;
  const MapGeometry& base = p.geometry;
  const std::size_t cells = base.cells();
  if (count > kMaxPointsLimit || (count != 0U && xyz == nullptr)) {
    return cudaErrorInvalidValue;
  }
  if (const cudaError_t status = s.state.reserve(cells * kStateBytesPerCell, stream);
      status != cudaSuccess) {
    return status;
  }
  const unsigned int cell_blocks = static_cast<unsigned int>((cells + 255U) / 256U);
  if (!s.cleared) {
    clear_kernel<<<cell_blocks, 256, 0, stream>>>(state_view(s.state.as<void>(), cells), cells);
    if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
      return status;
    }
    s.cleared = true;
  }

  // 1. The window.
  if (p.mode == MapMode::kFollow) {
    const double sx = sensor_to_map.m[3];
    const double sy = sensor_to_map.m[7];
    if (!std::isfinite(sx) || !std::isfinite(sy)) {
      return cudaErrorInvalidValue;
    }
    const std::int64_t cx =
        floor_index((sx - base.origin_x) / static_cast<double>(base.resolution)) - base.cols / 2;
    const std::int64_t cy =
        floor_index((sy - base.origin_y) / static_cast<double>(base.resolution)) - base.rows / 2;
    if (s.placed && (cx != s.corner_x || cy != s.corner_y)) {
      const std::int64_t dx = cx - s.corner_x;
      const std::int64_t dy = cy - s.corner_y;
      if (const cudaError_t status = s.spare.reserve(cells * kStateBytesPerCell, stream);
          status != cudaSuccess) {
        return status;
      }
      const dim3 block(32, 8);
      const dim3 grid((static_cast<unsigned int>(base.cols) + 31U) / 32U,
                      (static_cast<unsigned int>(base.rows) + 7U) / 8U);
      shift_kernel<<<grid, block, 0, stream>>>(state_view(s.state.as<void>(), cells),
                                               state_view(s.spare.as<void>(), cells), base.rows,
                                               base.cols, dx, dy);
      if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
        return status;
      }
      s.state.swap(s.spare);
    }
    s.corner_x = cx;
    s.corner_y = cy;
    s.placed = true;
    s.geometry.origin_x =
        base.origin_x + static_cast<double>(cx) * static_cast<double>(base.resolution);
    s.geometry.origin_y =
        base.origin_y + static_cast<double>(cy) * static_cast<double>(base.resolution);
  }
  const CellState state = state_view(s.state.as<void>(), cells);

  // 2. Ageing.
  const std::int64_t now = stamp_ns;
  const float elapsed =
      s.updated && now > s.last_stamp ? static_cast<float>(now - s.last_stamp) * 1e-9F : 0.0F;
  const AgeParams age{
      .now = now,
      .elapsed = elapsed,
      .max_age_ns = static_cast<std::int64_t>(std::llround(static_cast<double>(p.max_age) * 1e9)),
      .growth = p.variance_growth,
      .max_variance = p.max_variance,
      .accumulate = p.accumulate,
      .fused = p.elevation == ElevationMode::kFused};
  age_kernel<<<cell_blocks, 256, 0, stream>>>(state, cells, age);
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return status;
  }
  s.last_stamp = now;
  s.updated = true;
  if (count == 0U) {
    return cudaSuccess;
  }

  // 3. Binning: every point's cell, sorted by cell, stably.
  for (detail::DeviceBuffer* buffer :
       {&s.keys, &s.sorted_keys, &s.run_cells, &s.run_counts, &s.run_offsets}) {
    if (const cudaError_t status =
            buffer->reserve(std::size_t{count} * sizeof(std::uint32_t), stream);
        status != cudaSuccess) {
      return status;
    }
  }
  for (detail::DeviceBuffer* buffer : {&s.values, &s.sorted_values}) {
    if (const cudaError_t status = buffer->reserve(std::size_t{count} * sizeof(float2), stream);
        status != cudaSuccess) {
      return status;
    }
  }
  if (const cudaError_t status = s.runs.reserve(sizeof(int), stream); status != cudaSuccess) {
    return status;
  }
  const auto invalid_key = static_cast<std::uint32_t>(cells);
  const int end_bit = key_bits(invalid_key);
  std::size_t sort_bytes = 0;
  std::size_t encode_bytes = 0;
  std::size_t scan_bytes = 0;
  if (const cudaError_t status = cub::DeviceRadixSort::SortPairs(
          nullptr, sort_bytes, s.keys.as<std::uint32_t>(), s.sorted_keys.as<std::uint32_t>(),
          s.values.as<float2>(), s.sorted_values.as<float2>(), count, 0, end_bit, stream);
      status != cudaSuccess) {
    return status;
  }
  if (const cudaError_t status = cub::DeviceRunLengthEncode::Encode(
          nullptr, encode_bytes, s.sorted_keys.as<std::uint32_t>(), s.run_cells.as<std::uint32_t>(),
          s.run_counts.as<std::uint32_t>(), s.runs.as<int>(), count, stream);
      status != cudaSuccess) {
    return status;
  }
  if (const cudaError_t status =
          cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, s.run_counts.as<std::uint32_t>(),
                                        s.run_offsets.as<std::uint32_t>(), count, stream);
      status != cudaSuccess) {
    return status;
  }
  if (const cudaError_t status =
          s.temp.reserve(std::max({sort_bytes, encode_bytes, scan_bytes, std::size_t{1}}), stream);
      status != cudaSuccess) {
    return status;
  }
  std::size_t temp_bytes = s.temp.bytes();

  const MapGeometry& g = s.geometry;
  BinParams bin{};
  for (int k = 0; k < 12; ++k) {
    bin.pose.m[k] = static_cast<float>(sensor_to_map.m[k]);
  }
  // Cells count from the window's corner; the translation is taken relative
  // to it in double, so that a map far from the origin keeps its precision.
  bin.pose.m[3] = static_cast<float>(sensor_to_map.m[3] - g.origin_x);
  bin.pose.m[7] = static_cast<float>(sensor_to_map.m[7] - g.origin_y);
  bin.min_height = p.min_height;
  bin.max_height = p.max_height;
  bin.min_range = p.min_range;
  bin.max_range = p.max_range;
  bin.noise_base = p.noise_base;
  bin.noise_quadratic = p.noise_quadratic;
  bin.resolution = g.resolution;
  bin.rows = g.rows;
  bin.cols = g.cols;
  bin.invalid_key = invalid_key;
  bin_kernel<<<(count + 255U) / 256U, 256, 0, stream>>>(xyz, count, bin, s.keys.as<std::uint32_t>(),
                                                        s.values.as<float2>());
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return status;
  }
  if (const cudaError_t status =
          cub::DeviceRadixSort::SortPairs(s.temp.as<void>(), temp_bytes, s.keys.as<std::uint32_t>(),
                                          s.sorted_keys.as<std::uint32_t>(), s.values.as<float2>(),
                                          s.sorted_values.as<float2>(), count, 0, end_bit, stream);
      status != cudaSuccess) {
    return status;
  }
  temp_bytes = s.temp.bytes();
  if (const cudaError_t status = cub::DeviceRunLengthEncode::Encode(
          s.temp.as<void>(), temp_bytes, s.sorted_keys.as<std::uint32_t>(),
          s.run_cells.as<std::uint32_t>(), s.run_counts.as<std::uint32_t>(), s.runs.as<int>(),
          count, stream);
      status != cudaSuccess) {
    return status;
  }
  temp_bytes = s.temp.bytes();
  if (const cudaError_t status = cub::DeviceScan::ExclusiveSum(
          s.temp.as<void>(), temp_bytes, s.run_counts.as<std::uint32_t>(),
          s.run_offsets.as<std::uint32_t>(), count, stream);
      status != cudaSuccess) {
    return status;
  }

  // 4. The observations, and their fusion. At most one run per cell, plus
  // the dropped points'.
  const std::size_t most_runs = std::min<std::size_t>(count, cells + 1U);
  const FuseParams fuse{.now = now,
                        .cells = invalid_key,
                        .min_variance = p.min_variance,
                        .gate = p.mahalanobis_gate,
                        .rejection_variance = p.rejection_variance,
                        .fused = p.elevation == ElevationMode::kFused};
  observe_kernel<<<static_cast<unsigned int>((most_runs + 127U) / 128U), 128, 0, stream>>>(
      s.run_cells.as<std::uint32_t>(), s.run_counts.as<std::uint32_t>(),
      s.run_offsets.as<std::uint32_t>(), s.runs.as<int>(), s.sorted_values.as<float2>(), state,
      fuse);
  return cudaGetLastError();
}

cudaError_t HeightMapper::write_layers(const MapLayers& layers, cudaStream_t stream) const {
  const Impl& s = *impl_;
  const std::size_t cells = s.params.geometry.cells();
  if (!s.cleared) {
    // Nothing observed yet: every cell unknown.
    for (float* layer : {layers.elevation, layers.variance, layers.count, layers.min_elevation,
                         layers.max_elevation, layers.time_since_update}) {
      if (layer != nullptr) {
        if (const cudaError_t status = detail::fill_nan(layer, cells, stream);
            status != cudaSuccess) {
          return status;
        }
      }
    }
    return cudaSuccess;
  }
  const WriteParams params{.now = s.last_stamp, .cells = cells, .mode = s.params.elevation};
  write_kernel<<<static_cast<unsigned int>((cells + 255U) / 256U), 256, 0, stream>>>(
      state_view(s.state.as<void>(), cells), params, layers);
  return cudaGetLastError();
}

}  // namespace holoscan::height_map
