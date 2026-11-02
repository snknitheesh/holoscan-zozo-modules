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
// stream at a time; nothing waits for the stream, except that growing a
// scratch buffer first waits for the stream to finish with the old one.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <memory>

#include <holoscan/height_map/grid.hpp>

namespace holoscan::height_map {

enum class MapMode : std::uint8_t {
  kFixed,   // The grid stays at geometry.origin.
  kFollow,  // A moving window: the grid follows the sensor, cell by cell.
};

enum class ElevationMode : std::uint8_t {
  kMax,    // The highest point of the latest observation.
  kMean,   // The mean height of the latest observation.
  kFused,  // The variance-weighted estimate, integrated over time.
};

struct HeightMapParams {
  // kFixed: the grid. kFollow: its size and resolution; origin_x and origin_y
  // only set where the cell boundaries lie (the lattice); the window's
  // corner is the cell boundary rows / 2 and cols / 2 cells from the cell the
  // sensor is in.
  MapGeometry geometry;
  MapMode mode{MapMode::kFixed};
  ElevationMode elevation{ElevationMode::kFused};
  // Points outside [min_height, max_height] in the map frame, or whose
  // distance from the sensor is outside [min_range, max_range], are dropped.
  float min_height{-std::numeric_limits<float>::infinity()};
  float max_height{std::numeric_limits<float>::infinity()};
  float min_range{0.0F};
  float max_range{std::numeric_limits<float>::infinity()};
  // A point's height standard deviation in meters:
  // noise_base + noise_quadratic * range^2.
  float noise_base{0.005F};
  float noise_quadratic{0.002F};
  // Floor of a measurement's variance, m^2; positive.
  float min_variance{1e-6F};
  // kFused: measurements further than this many standard deviations from the
  // estimate are not fused; 0 fuses every measurement.
  float mahalanobis_gate{3.0F};
  // kFused: variance added to an estimate for each lower measurement dropped
  // by the gate, m^2; not negative.
  float rejection_variance{1e-4F};
  // Keep cells between observations; without, the map is the latest cloud's.
  bool accumulate{true};
  // Clear cells observed longer ago than this, seconds; 0 never.
  float max_age{0.0F};
  // kFused: variance added per second, m^2/s; 0 none.
  float variance_growth{0.0F};
  // kFused: clear cells whose variance exceeds this, m^2; 0 never.
  float max_variance{0.0F};
};

// Whether `params` are usable; if not, `what` (when given) names the first
// unusable field.
[[nodiscard]] bool valid_params(const HeightMapParams& params, const char** what = nullptr);

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
