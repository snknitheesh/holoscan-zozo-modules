// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Pictures for display, on the GPU: a depth image color-mapped, and a ray
// cast shaded by its normals. Each writes RGB8 pixels into rows `row_bytes`
// apart, so two pictures can sit side by side in one image.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include <holoscan/tsdf/tsdf_volume.hpp>

namespace holoscan::tsdf {

// Depth in meters (uint16 raw units times `scale`, or float32 times
// `scale`) from near (warm) to far (cool) over [depth_min, depth_max];
// black where there is no depth.
cudaError_t colorize_depth(const void* depth, DepthType type, float scale, int width, int height,
                           float depth_min, float depth_max, std::uint8_t* rgb,
                           std::size_t row_bytes, cudaStream_t stream);

// A ray cast lit from the camera: the normals' (camera frame) facing the
// camera, times the ray-cast colors when given (else light gray); black
// where a ray hit nothing (NaN normal).
cudaError_t shade_raycast(const float* normals, const std::uint8_t* colors, int width, int height,
                          std::uint8_t* rgb, std::size_t row_bytes, cudaStream_t stream);

}  // namespace holoscan::tsdf
