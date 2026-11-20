// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The 2.5D elevation map on the GPU, without a Holoscan graph. A cloud is
// `count` packed float32 x, y, z triples in device memory, in the sensor's
// frame, the layout of holoscan-point-cloud's clouds; a point with a
// non-finite coordinate is invalid. A rigid transform takes the points into
// the map frame (see grid.hpp), where each point falls into one cell.
//
// Each update:
//
// 1. With MapMode::kFollow, moves the grid by whole cells so that the sensor
//    stays in its middle: cells still inside keep what they hold, cells that
//    enter are unknown.
// 2. Ages the map: without `accumulate` every cell is cleared; otherwise a
//    cell not observed for longer than `max_age` is cleared, and with
//    kFused the variance of every known cell grows by `variance_growth` per
//    second (the process noise), and a cell whose variance exceeds
//    `max_variance` is cleared.
// 3. Bins the valid points that pass the range and height limits by cell (a
//    stable CUB radix sort, so every cell sees its points in input order),
//    and gives every cell that received points this observation: the count,
//    the lowest, the highest and the mean height, and the spread (the
//    population variance of the heights).
// 4. kFused also fuses the observation into the cell's estimate, a scalar
//    Kalman filter: the measurement is the inverse-variance weighted mean of
//    the points' heights, each point's standard deviation
//    noise_base + noise_quadratic * range^2, and its variance is
//    1 / sum(1 / variance) plus the spread, at least `min_variance`. A
//    measurement further than `mahalanobis_gate` standard deviations from the
//    estimate is not fused: a higher one replaces it (something now stands
//    there); a lower one, which may only have seen past an edge, is dropped
//    and adds `rejection_variance` to the estimate's variance, so that lower
//    terrain that persists is taken after a few observations.
//
// The layers then give, per cell: the elevation (the latest observation's
// highest or mean height, or the fused estimate), its variance (the fused
// variance, or the latest observation's spread), the latest observation's
// count, lowest and highest heights, and the seconds since that observation.
// Every layer is NaN where the cell is unknown.
//
// Everything is deterministic: the same clouds give the same bits. The
// kernels are compiled without fused multiply-adds, so a host evaluation of
// the same expressions (tests/reference.hpp) matches them bit for bit.
// A mapper keeps its device state and scratch between calls and is for one
// stream at a time; the host never waits for the stream: the state and the
// scratch are allocated and grown in stream order (CUDA's stream-ordered
// allocator; on a GPU without it, growing waits for the stream). Destroying
// the mapper waits for the device.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>

#include <holoscan/height_map/grid.hpp>
#include <holoscan/height_map/params.hpp>

namespace holoscan::height_map {

// Where write_layers() puts each layer: float32 [rows, cols] in device
// memory, or null to skip it.
struct MapLayers {
  float* elevation{};
  float* variance{};
  float* count{};
  float* min_elevation{};
  float* max_elevation{};
  float* time_since_update{};
};

class HeightMapper {
 public:
  // @throws std::invalid_argument naming the first unusable field.
  explicit HeightMapper(const HeightMapParams& params);
  ~HeightMapper();
  HeightMapper(const HeightMapper&) = delete;
  HeightMapper& operator=(const HeightMapper&) = delete;

  // Integrates a cloud captured at `stamp_ns` (nanoseconds, any epoch, not
  // decreasing) into the map. `sensor_to_map` takes its points into the map
  // frame; it should be rigid, as the range is measured in the sensor frame.
  cudaError_t update(const float* xyz, std::uint32_t count, const Transform& sensor_to_map,
                     std::int64_t stamp_ns, cudaStream_t stream);

  // Writes the map as of the last update. Before the first, every cell is
  // unknown.
  cudaError_t write_layers(const MapLayers& layers, cudaStream_t stream) const;

  // The grid as of the last update: in kFollow, where the window is now.
  [[nodiscard]] MapGeometry geometry() const noexcept;
  [[nodiscard]] const HeightMapParams& params() const noexcept;

  // Forgets everything: every cell unknown, the window where the next update
  // puts it.
  void reset() noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::height_map
