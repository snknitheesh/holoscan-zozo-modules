// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The color map DisparityColorOp shows: the jet map from blue (far) to dark red
// (near), black where the disparity is invalid.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace stereo_disparity_app {

// Float disparity, rows `pitch` bytes apart, as packed RGBA8 rows: `max_disparity`
// and nearer dark red, invalid (0) black. Queued on `stream`.
[[nodiscard]] cudaError_t colorize_disparity(const float* disparity, std::size_t pitch, int width,
                                             int height, float max_disparity, std::uint8_t* rgba,
                                             cudaStream_t stream);

}  // namespace stereo_disparity_app
