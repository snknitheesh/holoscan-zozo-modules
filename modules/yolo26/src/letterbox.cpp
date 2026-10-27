// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <holoscan/yolo26/letterbox.hpp>

namespace holoscan::yolo26 {

Letterbox make_letterbox(int source_width, int source_height, int size) {
  if (source_width < 1 || source_height < 1 || source_width > kMaxImageSide ||
      source_height > kMaxImageSide) {
    throw std::invalid_argument("image sides must be 1.." + std::to_string(kMaxImageSide) +
                                ", got " + std::to_string(source_width) + "x" +
                                std::to_string(source_height));
  }
  if (size < kMinInputSize || size > kMaxInputSize) {
    throw std::invalid_argument("input size must be " + std::to_string(kMinInputSize) + ".." +
                                std::to_string(kMaxInputSize));
  }
  Letterbox result;
  result.source_width = source_width;
  result.source_height = source_height;
  result.size = size;
  const double scale =
      std::min(static_cast<double>(size) / source_width, static_cast<double>(size) / source_height);
  // nearbyint rounds half to even in the default rounding mode, as Python's
  // round() does; the result never exceeds `size`.
  result.resized_width =
      std::clamp(static_cast<int>(std::nearbyint(source_width * scale)), 1, size);
  result.resized_height =
      std::clamp(static_cast<int>(std::nearbyint(source_height * scale)), 1, size);
  // round(d / 2 - 0.1) for the whole padding d: the left or top half, rounded
  // down when d is odd.
  result.pad_left = (size - result.resized_width) / 2;
  result.pad_top = (size - result.resized_height) / 2;
  result.scale = static_cast<float>(scale);
  return result;
}

}  // namespace holoscan::yolo26
