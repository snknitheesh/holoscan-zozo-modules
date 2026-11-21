// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The synthetic raw camera of synthetic_raw.hpp, rendered on the GPU.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include <holoscan/isp/synthetic_raw.hpp>

namespace holoscan::isp {

// Renders the raw frame into device memory on `stream`, bit for bit as
// render_synthetic_raw() does on the host.
[[nodiscard]] cudaError_t render_synthetic_raw_cuda(const SyntheticRawScene& scene, void* data,
                                                    std::size_t pitch, cudaStream_t stream);

}  // namespace holoscan::isp
