// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "host_input.hpp"

#include <mutex>
#include <string>

#include <holoscan/logger/logger.hpp>

namespace holoscan::height_map::detail {

holoscan::expected<void, holoscan::Error> hold_until_done(const holoscan::Tensor& tensor,
                                                          cudaStream_t stream,
                                                          std::string_view op) {
  holoscan::Tensor handle = tensor;
  if (handle.set_deallocation_stream(stream)) {
    return {};
  }
  // The memory cannot wait for the stream, so wait here; said once per
  // process, as it happens every frame for such a producer.
  static std::once_flag logged;
  std::call_once(logged, [op] {
    HOLOSCAN_LOG_INFO(
        "{}: the pinned host input's memory cannot wait for the copy's stream; the host waits for "
        "the copy every frame (send device memory to avoid it)",
        std::string(op));
  });
  if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure,
        std::string(op) + ": cudaStreamSynchronize: " + cudaGetErrorString(status)});
  }
  return {};
}

}  // namespace holoscan::height_map::detail
