// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Color views of the map layers on the GPU, for display: RGB8 images, one
// pixel per cell, rows as the layer's (row 0, the lowest y, first), that a
// HoloViz image layer shows as they are. Unknown cells are dark gray.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace holoscan::height_map {

// The color of unknown cells.
inline constexpr std::uint8_t kUnknownGray = 40U;

// Each elevation in a rainbow, from dark blue at `low` through cyan, green
// and yellow to red at `high` meters, clamped. With low >= high, the range is the layer's own: the
// lowest and highest known elevations, found on the GPU; `range_scratch`
// then holds two floats of device memory for them (unused otherwise).
cudaError_t colorize_elevation(const float* elevation, int rows, int cols, float low, float high,
                               float* range_scratch, std::uint8_t* rgb, cudaStream_t stream);

// Each cost from green at 0 through yellow to red at 1.
cudaError_t colorize_cost(const float* cost, int rows, int cols, std::uint8_t* rgb,
                          cudaStream_t stream);

}  // namespace holoscan::height_map
