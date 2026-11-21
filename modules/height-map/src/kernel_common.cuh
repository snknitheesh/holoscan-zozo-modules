// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Small kernels the module's other kernels share.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace holoscan::height_map::detail {

// Writes NaN into `count` floats.
cudaError_t fill_nan(float* data, std::size_t count, cudaStream_t stream);

}  // namespace holoscan::height_map::detail
