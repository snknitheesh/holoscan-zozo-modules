// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <holoscan/mono_depth/image_fit.hpp>

namespace holoscan::mono_depth {

ImageFit make_fit(int source_width, int source_height, int input_width, int input_height,
                  int output_width, int output_height, ResizeMode mode) {
  if (source_width < 1 || source_height < 1 || source_width > kMaxImageSide ||
      source_height > kMaxImageSide) {
    throw std::invalid_argument("image sides must be 1.." + std::to_string(kMaxImageSide) +
                                ", got " + std::to_string(source_width) + "x" +
                                std::to_string(source_height));
  }
  for (const int side : {input_width, input_height, output_width, output_height}) {
    if (side < 1 || side > kMaxModelSide) {
      throw std::invalid_argument("model input and output sides must be 1.." +
                                  std::to_string(kMaxModelSide));
    }
  }
  ImageFit fit;
  fit.source_width = source_width;
  fit.source_height = source_height;
  fit.input_width = input_width;
  fit.input_height = input_height;
  fit.output_width = output_width;
  fit.output_height = output_height;
  if (mode == ResizeMode::kStretch) {
    fit.resized_width = input_width;
    fit.resized_height = input_height;
  } else {
    const double scale = std::min(static_cast<double>(input_width) / source_width,
                                  static_cast<double>(input_height) / source_height);
    // nearbyint rounds half to even in the default rounding mode.
    fit.resized_width =
        std::clamp(static_cast<int>(std::nearbyint(source_width * scale)), 1, input_width);
    fit.resized_height =
        std::clamp(static_cast<int>(std::nearbyint(source_height * scale)), 1, input_height);
    fit.pad_left = (input_width - fit.resized_width) / 2;
    fit.pad_top = (input_height - fit.resized_height) / 2;
  }
  // The output pixels whose centers lie inside the resized image, or the one
  // nearest to it when none does (a very thin image).
  const auto valid = [](int pad, int resized, int input, int output, int& low, int& high) {
    const double ratio = static_cast<double>(output) / input;
    low = static_cast<int>(std::ceil(pad * ratio - 0.5));
    high = static_cast<int>(std::ceil((pad + resized) * ratio - 0.5)) - 1;
    low = std::clamp(low, 0, output - 1);
    high = std::clamp(high, low, output - 1);
  };
  valid(fit.pad_left, fit.resized_width, input_width, output_width, fit.valid_left,
        fit.valid_right);
  valid(fit.pad_top, fit.resized_height, input_height, output_height, fit.valid_top,
        fit.valid_bottom);
  return fit;
}

}  // namespace holoscan::mono_depth
