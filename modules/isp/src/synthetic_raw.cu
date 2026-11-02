// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/isp/synthetic_raw_cuda.hpp"

namespace holoscan::isp {
namespace {

constexpr int kBlock = 256;

__global__ void render_kernel(SyntheticRawScene scene, unsigned char* data, std::size_t pitch,
                              int units) {
  const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(kBlock) + threadIdx.x;
  if (i >= static_cast<std::int64_t>(units) * scene.height) {
    return;
  }
  const int y = static_cast<int>(i / units);
  synthetic_write_unit(scene, static_cast<int>(i % units), y,
                       data + static_cast<std::size_t>(y) * pitch);
}

}  // namespace

cudaError_t render_synthetic_raw_cuda(const SyntheticRawScene& scene, void* data, std::size_t pitch,
                                      cudaStream_t stream) {
  const int units = synthetic_units(scene);
  const std::int64_t n = static_cast<std::int64_t>(units) * scene.height;
  if (n <= 0) {
    return cudaSuccess;
  }
  // Clear an error an earlier, unrelated call left behind, so the check after
  // the launch reports this launch only.
  static_cast<void>(cudaGetLastError());
  render_kernel<<<static_cast<unsigned int>((n + kBlock - 1) / kBlock), kBlock, 0, stream>>>(
      scene, static_cast<unsigned char*>(data), pitch, units);
  return cudaGetLastError();
}

}  // namespace holoscan::isp
