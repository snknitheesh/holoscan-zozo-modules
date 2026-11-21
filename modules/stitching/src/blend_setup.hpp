// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What the CPU reference and the Stitcher share outside the per-pixel
// arithmetic: a camera's region of interest from its coverage, and the blend
// parameters of a frame.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <holoscan/stitching/remap.hpp>

#include "stitch_math.hpp"

namespace holoscan::stitching::detail {

struct Roi {
  int x0{};
  int y0{};
  int width{};
  int height{};
};

// The smallest region holding every covered pixel, from the columns that hold
// any (`columns`, one flag per output column) and the first and last covered
// rows (y_last < y_first when none). When the output wraps, the columns may
// run across its right edge: the region then starts after the widest gap.
[[nodiscard]] Roi region_of_interest(const std::vector<std::uint8_t>& columns, int y_first,
                                     int y_last, bool wraps);

// The parameters of a blend: settings, size, and no cameras yet.
[[nodiscard]] BlendParams blend_params(const BlendSettings& settings, int output_width,
                                       int output_height, bool wraps);

// Sets camera `index` of `params`: its image and its table's region, in
// whatever memory the blend will read them from.
void set_camera(BlendParams& params, int index, const std::uint8_t* image, std::size_t pitch,
                int width, int height, PixelFormat format, const Roi& roi, const float* uv,
                const float* margin);

// The weights 1 / sigma_N^2 and 1 / sigma_g^2 of the gain solve.
[[nodiscard]] double gain_alpha(const BlendSettings& settings);
[[nodiscard]] double gain_beta(const BlendSettings& settings);

}  // namespace holoscan::stitching::detail
