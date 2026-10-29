// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Keeping host input alive while the GPU still copies it: shared by the
// module's operators, which copy host input to the GPU asynchronously.

#pragma once

#include <cuda_runtime_api.h>

#include <string_view>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>

namespace holoscan::pcl_filters::detail {

// Keeps `tensor`'s memory from reuse until the work queued on `stream` so far
// is done: its allocator waits for the stream if it can, otherwise this
// waits. For host memory that the operator copies to the GPU asynchronously.
// An error names `op`, the operator that asked.
[[nodiscard]] holoscan::expected<void, holoscan::Error> hold_until_done(
    const holoscan::Tensor& tensor, cudaStream_t stream, std::string_view op);

}  // namespace holoscan::pcl_filters::detail
