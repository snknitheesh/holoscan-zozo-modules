// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The CUDA kernels' launchers.

#pragma once

#include <cuda_runtime_api.h>

#include "kernel_math.hpp"

namespace holoscan::color_convert::detail {

// Converts the whole destination of `plan` on `stream`; with `border_only`,
// only the blocks outside the content rectangle (the letterbox fill).
[[nodiscard]] cudaError_t launch_convert(const Plan& plan, bool border_only, cudaStream_t stream);

}  // namespace holoscan::color_convert::detail
