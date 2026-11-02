// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Host references of what MonoDepthOp computes, from the same per-pixel
// arithmetic as its kernels (pixel_math.hpp): the preprocessing, the stub
// model, the postprocessing, the range and the colorization. With the stub
// model they predict the operator's every output value from the image. For
// tests and the demo's checks; no SDK, no CUDA.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <holoscan/mono_depth/image_fit.hpp>
#include <holoscan/mono_depth/pixel_math.hpp>

namespace holoscan::mono_depth {

// The model input: float32 [3, input_height, input_width] planes of `image`,
// rows of `pitch` bytes in `format`, as launch_preprocess() writes them.
[[nodiscard]] std::vector<float> reference_preprocess(const std::uint8_t* image, std::size_t pitch,
                                                      PixelFormat format, const ImageFit& fit,
                                                      const Normalization& normalization);

// The stub model's output for that input: float32 [height, width].
[[nodiscard]] std::vector<float> reference_stub(std::span<const float> input, int width,
                                                int height);

// The inverse and, for a metric model or with `affine`, metric depth of the
// model's output, as launch_postprocess() writes them: source_height x
// source_width values each; `metric` is left empty otherwise.
struct ReferenceDepth {
  std::vector<float> inverse;
  std::vector<float> metric;
};

[[nodiscard]] ReferenceDepth reference_postprocess(std::span<const float> output,
                                                   const ImageFit& fit, ModelOutput model_output,
                                                   const InverseDepthAffine* affine);

// The smallest and largest finite value; {0, 0} when there is none.
[[nodiscard]] std::array<float, 2> reference_range(std::span<const float> values);

// RGBA8 [height, width, 4] of `inverse_depth` between `low` and `high`, as
// launch_colorize() writes it.
[[nodiscard]] std::vector<std::uint8_t> reference_colorize(std::span<const float> inverse_depth,
                                                           float low, float high);

}  // namespace holoscan::mono_depth
