// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What the operators check of the tensors they receive.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace holoscan::height_map::detail {

inline bool is_dtype(const DLDataType& dtype, const DLDataType& expected) {
  return dtype.code == expected.code && dtype.bits == expected.bits &&
         dtype.lanes == expected.lanes;
}

inline bool on_gpu(const holoscan::Tensor& tensor) {
  const DLDevice device = tensor.device();
  return device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
}

// Whether the tensor is packed. Any stride is accepted on an axis of extent 1,
// which no element steps along.
inline bool packed(const holoscan::Tensor& tensor) {
  const std::span<const std::int64_t> strides = tensor.dlpack_strides_span();
  if (strides.empty()) {
    return true;  // DLPack's compact row-major layout.
  }
  const std::span<const std::int64_t> shape = tensor.shape_span();
  std::int64_t expected = 1;  // In elements, as DLPack counts strides.
  for (std::size_t axis = shape.size(); axis-- > 0;) {
    if (shape[axis] > 1 && strides[axis] != expected) {
      return false;
    }
    expected *= shape[axis];
  }
  return true;
}

}  // namespace holoscan::height_map::detail
