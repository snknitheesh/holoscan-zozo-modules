// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Color views of normals and plane labels on the GPU, for display: RGB8
// images a HoloViz image layer shows as they are. Each writes `height` rows
// of `width` pixels into `rgb`, whose rows start `row_stride` bytes apart, so
// that views can sit side by side in one image.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace holoscan::normals {

// Each normal (x, y, z) as the color ((x + 1) / 2, (y + 1) / 2, (z + 1) / 2);
// black without a normal.
cudaError_t colorize_normals(const float* normals, int width, int height, std::uint8_t* rgb,
                             std::size_t row_stride, cudaStream_t stream);

// Each plane label in a color of its own (twelve, repeating); dark gray for
// kLabelNoPlane and black for kLabelInvalid.
cudaError_t colorize_labels(const std::uint8_t* labels, int width, int height, std::uint8_t* rgb,
                            std::size_t row_stride, cudaStream_t stream);

// The color colorize_labels() gives a label, on the host.
void label_color(std::uint8_t label, std::uint8_t rgb[3]);

}  // namespace holoscan::normals
