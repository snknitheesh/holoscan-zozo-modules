// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "host_input.hpp"

#include <string>

namespace holoscan::pcl_filters::detail {

holoscan::expected<void, holoscan::Error> hold_until_done(
    std::span<const holoscan::Tensor* const> tensors, cudaStream_t stream, std::string_view op) {
  bool held = true;
  for (const holoscan::Tensor* tensor : tensors) {
    holoscan::Tensor handle = *tensor;
    held = handle.set_deallocation_stream(stream) && held;
  }
  if (held) {
    return {};
  }
  // Some memory cannot wait for the stream, so wait here.
  if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure,
        std::string(op) + ": cudaStreamSynchronize: " + cudaGetErrorString(status)});
  }
  return {};
}

}  // namespace holoscan::pcl_filters::detail
