// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Traversability of an elevation map on the GPU, without a Holoscan graph:
// one thread per cell of a float32 [rows, cols] elevation layer (grid.hpp),
// NaN where unknown.
//
// For each known cell:
// - slope: the least-squares plane z = a x + b y + c through the known cells
//   of the (2 window_radius + 1)^2 window around it, at least
//   min_valid_fraction of the window; the slope is atan(|(a, b)|), in
//   degrees.
// - roughness: the root mean square of the cells' distances from that plane,
//   along z, in meters.
// - step height: the highest minus the lowest known cell of the
//   (2 step_radius + 1)^2 window around the cell, in meters. A mean or fused
//   elevation mixes a riser's points into the cells next to it; a window two
//   cells wide either side still reaches clear cells on both sides.
// - cost: 1 when the slope, the step height or the roughness reaches its
//   maximum; otherwise their weighted mean, each divided by its maximum. It
//   grows with each of them, and lies in 0..1.
//
// A cell whose elevation is unknown, or whose window holds too few known
// cells, or only collinear ones, is unknown: its mask is 1 and every other
// layer NaN. The kernel is compiled without fused multiply-adds, like the
// mapper's.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace holoscan::height_map {

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
struct TraversabilityLayers {
  float* cost{};
  std::uint8_t* unknown{};
  float* slope{};
  float* step_height{};
  float* roughness{};
};

// Computes the layers of a `rows` x `cols` elevation layer with cells
// `resolution` meters wide. Returns cudaErrorInvalidValue for unusable
// arguments.
cudaError_t compute_traversability(const float* elevation, int rows, int cols, float resolution,
                                   const TraversabilityParams& params,
                                   const TraversabilityLayers& layers, cudaStream_t stream);

}  // namespace holoscan::height_map
