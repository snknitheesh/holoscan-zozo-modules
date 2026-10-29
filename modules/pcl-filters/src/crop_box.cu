// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The crop box: one thread per point. Compiled without fused multiply-adds,
// so the transform gives what the host computes from the same expression.

#include <cstdint>

#include <holoscan/pcl_filters/filters.hpp>

#include "kernel_common.cuh"

namespace holoscan::pcl_filters {
namespace {

using detail::blocks_for;
using detail::kBlock;
using detail::valid_point;

__global__ void crop_box_kernel(const float* __restrict__ xyz, std::uint32_t count,
                                CropBoxParams params, std::uint8_t* __restrict__ keep) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  float p[3] = {xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]};
  if (!valid_point(p[0], p[1], p[2])) {
    keep[i] = 0U;
    return;
  }
  if (params.use_transform) {
    const float* m = params.transform.matrix;
    const float q[3] = {p[0], p[1], p[2]};
    for (int row = 0; row < 3; ++row) {
      p[row] = m[4 * row] * q[0] + m[4 * row + 1] * q[1] + m[4 * row + 2] * q[2] + m[4 * row + 3];
    }
  }
  bool inside = true;
  for (int axis = 0; axis < 3; ++axis) {
    inside = inside && p[axis] >= params.min[axis] && p[axis] <= params.max[axis];
  }
  const float squared = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
  inside = inside && squared >= params.min_distance * params.min_distance &&
           squared <= params.max_distance * params.max_distance;
  keep[i] = inside != params.keep_outside ? 1U : 0U;
}

}  // namespace

cudaError_t launch_crop_box(const float* xyz, std::uint32_t count, const CropBoxParams& params,
                            std::uint8_t* keep, cudaStream_t stream) {
  if (count == 0U) {
    return cudaSuccess;
  }
  crop_box_kernel<<<blocks_for(count), kBlock, 0, stream>>>(xyz, count, params, keep);
  return cudaGetLastError();
}

}  // namespace holoscan::pcl_filters
