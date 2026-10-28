// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A plain CPU remap, the reference the backends are compared with: the same
// sampling as the cuda backend (pixel centers at integers, zero for a source
// outside [0, size - 1], rounding half up), in double precision.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <holoscan/rectify/rectification.hpp>

namespace holoscan::rectify::test {

// Remaps a packed image of `T` with `channels` per pixel through `table`.
template <typename T>
std::vector<T> reference_remap(const std::vector<T>& input, int width, int height, int channels,
                               const RemapTable& table, bool linear) {
  std::vector<T> output(static_cast<std::size_t>(table.width) * table.height * channels);
  const double max_value = sizeof(T) == 1 ? 255.0 : 65535.0;
  const auto at = [&](int x, int y, int c) -> double {
    if (x < 0 || y < 0 || x >= width || y >= height) return 0.0;
    return input[(static_cast<std::size_t>(y) * width + x) * channels + c];
  };
  for (int v = 0; v < table.height; ++v) {
    for (int u = 0; u < table.width; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * table.width + u;
      const double sx = table.xy[2 * i];
      const double sy = table.xy[2 * i + 1];
      const bool inside = sx >= 0 && sy >= 0 && sx <= width - 1 && sy <= height - 1;
      for (int c = 0; c < channels; ++c) {
        double value = 0.0;
        if (!inside) {
          // Zero outside [0, size - 1], as every backend.
        } else if (linear) {
          const double fx = std::floor(sx);
          const double fy = std::floor(sy);
          const int x0 = static_cast<int>(fx);
          const int y0 = static_cast<int>(fy);
          const double ax = sx - fx;
          const double ay = sy - fy;
          value = at(x0, y0, c) * (1 - ax) * (1 - ay) + at(x0 + 1, y0, c) * ax * (1 - ay) +
                  at(x0, y0 + 1, c) * (1 - ax) * ay + at(x0 + 1, y0 + 1, c) * ax * ay;
        } else {
          value =
              at(static_cast<int>(std::floor(sx + 0.5)), static_cast<int>(std::floor(sy + 0.5)), c);
        }
        output[i * channels + c] =
            static_cast<T>(std::clamp(std::floor(value + 0.5), 0.0, max_value));
      }
    }
  }
  return output;
}

}  // namespace holoscan::rectify::test
