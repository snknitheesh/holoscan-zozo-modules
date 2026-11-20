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

#include <holoscan/height_map/params.hpp>

namespace holoscan::height_map {

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
