// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "kernel_common.cuh"

namespace holoscan::height_map::detail {
namespace {

__global__ void fill_nan_kernel(float* data, std::size_t count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    data[i] = __int_as_float(0x7fc00000);
  }
}

}  // namespace

cudaError_t fill_nan(float* data, std::size_t count, cudaStream_t stream) {
  if (count == 0U) {
    return cudaSuccess;
  }
  if (data == nullptr) {
    return cudaErrorInvalidValue;
  }
  fill_nan_kernel<<<static_cast<unsigned int>((count + 255U) / 256U), 256, 0, stream>>>(data,
                                                                                        count);
  return cudaGetLastError();
}

}  // namespace holoscan::height_map::detail
