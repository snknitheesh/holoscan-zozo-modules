// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The settings of the map (HeightMapper, HeightMapOp) and of its
// traversability (compute_traversability(), TraversabilityOp), and the rules
// they follow. Needs neither the SDK nor CUDA; height_mapper.hpp and
// traversability.hpp include it.

#pragma once

#include <cstdint>
#include <limits>

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
  // noise_base + noise_quadratic * range^2; noise_base positive, so that a
  // point at the sensor has a finite weight.
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

struct TraversabilityParams {
  // The plane fit's window, 1..10 cells around the cell.
  int window_radius{2};
  // The share of the plane fit's window that must be known, 0..1.
  float min_valid_fraction{0.5F};
  // The step height's window, 1..10 cells around the cell.
  int step_radius{2};
  // Where the cost reaches 1; positive.
  float max_slope_deg{30.0F};
  float max_step{0.2F};
  float max_roughness{0.05F};
  // The cost's weights below the maxima; not negative, not all 0.
  float slope_weight{0.5F};
  float step_weight{0.25F};
  float roughness_weight{0.25F};
};

// Whether `params` are usable; if not, `what` (when given) names the first
// unusable field.
[[nodiscard]] bool valid_params(const TraversabilityParams& params, const char** what = nullptr);

// Where compute_traversability() puts each layer, [rows, cols] in device
// memory, or null to skip it. `unknown` is uint8, 1 where unknown; the others
// are float32, NaN where unknown.
}  // namespace holoscan::height_map
