// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "host_input.hpp"

#include <string>

namespace holoscan::point_cloud::detail {

holoscan::expected<void, holoscan::Error> hold_until_done(const holoscan::Tensor& tensor,
                                                          cudaStream_t stream,
                                                          std::string_view op) {
  holoscan::Tensor handle = tensor;
  if (handle.set_deallocation_stream(stream)) {
    return {};
  }
  // The memory cannot wait for the stream, so wait here.
  if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure,
        std::string(op) + ": cudaStreamSynchronize: " + cudaGetErrorString(status)});
  }
  return {};
}

}  // namespace holoscan::point_cloud::detail
