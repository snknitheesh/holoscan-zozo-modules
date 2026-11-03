// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Keeping host input alive while the GPU still copies it: shared by the
// module's operators, which copy host input to the GPU asynchronously.

#pragma once

#include <cuda_runtime_api.h>

#include <span>
#include <string_view>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>

namespace holoscan::pcl_filters::detail {

// Keeps the memory of `tensors` from reuse until the work queued on `stream`
// so far is done: their allocators wait for the stream if they all can,
// otherwise this waits once (the runtime's own pools cannot). For host memory
// that the operator copies to the GPU asynchronously. An error names `op`,
// the operator that asked.
[[nodiscard]] holoscan::expected<void, holoscan::Error> hold_until_done(
    std::span<const holoscan::Tensor* const> tensors, cudaStream_t stream, std::string_view op);

}  // namespace holoscan::pcl_filters::detail
