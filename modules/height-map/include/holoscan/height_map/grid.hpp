// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The grid every part of the module shares: a fixed number of square cells in
// the map frame's x-y plane, z up.
//
// Cell (row, col) covers x in [origin_x + col * resolution, ... + resolution)
// and y in [origin_y + row * resolution, ... + resolution): columns run along
// +x, rows along +y, and (origin_x, origin_y) is the corner of cell (0, 0).
// A layer is float32 [rows, cols], row-major, NaN where the cell is unknown.

#pragma once

#include <cstddef>
#include <cstdint>

namespace holoscan::height_map {

// Largest point cloud and largest grid side the module accepts.
inline constexpr std::uint32_t kMaxPointsLimit = std::uint32_t{1} << 30;
inline constexpr int kMaxGridSide = 8192;

struct MapGeometry {
  // Cell side in meters.
  float resolution{0.05F};
  // Cells along y and along x, 1..kMaxGridSide each.
  int rows{200};
  int cols{200};
  // The corner of cell (0, 0) in the map frame, in meters.
  double origin_x{-5.0};
  double origin_y{-5.0};

  [[nodiscard]] std::size_t cells() const noexcept {
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  }
};

// A rigid transform as a 3x4 row-major matrix [R | t]: p_out = R * p_in + t.
struct Transform {
  double m[12]{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
};

// Whether `transform` is rigid: finite, its rotation orthonormal within
// `tolerance` and proper.
[[nodiscard]] bool is_rigid(const Transform& transform, double tolerance = 1e-4);

// The `map_info` the operators emit with every map: float64 [kMapInfoSize],
// in this order. The capture time is the sample's.
inline constexpr std::size_t kInfoOriginX = 0;     // Meters, the corner of cell (0, 0).
inline constexpr std::size_t kInfoOriginY = 1;     // Meters.
inline constexpr std::size_t kInfoResolution = 2;  // Meters per cell.
inline constexpr std::size_t kInfoRows = 3;
inline constexpr std::size_t kInfoCols = 4;
inline constexpr std::size_t kMapInfoSize = 5;

}  // namespace holoscan::height_map
