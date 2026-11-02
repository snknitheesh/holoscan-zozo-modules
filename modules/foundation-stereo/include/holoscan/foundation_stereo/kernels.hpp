// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// FoundationStereoOp's CUDA kernels, public so that tests and other operators
// can use them: an 8-bit image resized and normalized into the model's input,
// and the model's disparity resized and rescaled back to the image, with its
// depth. Each queues its work on `stream` and returns the launch status.
//
// Both resize bilinearly with pixel centers at half-integer coordinates
// (OpenCV's INTER_LINEAR, PyTorch's align_corners=False): output pixel x
// samples the input at (x + 0.5) * in / out - 0.5, clamped to the image, and
// likewise for y.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include <holoscan/foundation_stereo/model_input.hpp>

namespace holoscan::foundation_stereo {

// Packed 8-bit pixels in device memory, rows `pitch` bytes apart.
struct ImageView {
  const std::uint8_t* data{};
  std::size_t pitch{};
  int width{};
  int height{};
  PixelFormat format{PixelFormat::kRgb8};
};

// `image` resized to model_width x model_height and normalized into `out`,
// float32 planar RGB [3, model_height, model_width]. MONO8 is repeated in all
// three channels; alpha is ignored.
[[nodiscard]] cudaError_t launch_preprocess(const ImageView& image, int model_width,
                                            int model_height, float* out, cudaStream_t stream);

struct PostprocessParams {
  int model_width{};
  int model_height{};
  int width{};  // The image's size, which the outputs have.
  int height{};
  // For `depth`: focal length in pixels times baseline in meters, and doffs
  // (see stereo_geometry.hpp).
  float fx_baseline{};
  float doffs{};
  // Also invalidate pixels whose match would lie left of the right image
  // (x - disparity < 0), which the right camera cannot have seen.
  bool invalidate_out_of_view{true};
};

// The model's disparity, float32 [model_height, model_width] in model pixels,
// resized to width x height and multiplied by width / model_width into
// `disparity`, packed float32 [height, width] in image pixels; 0 where it is
// not positive (and, with invalidate_out_of_view, out of view). With a
// non-null `depth`, also depth_from_disparity() of each pixel, 0 where the
// disparity is 0.
[[nodiscard]] cudaError_t launch_postprocess(const float* model_disparity,
                                             const PostprocessParams& params, float* disparity,
                                             float* depth, cudaStream_t stream);

}  // namespace holoscan::foundation_stereo
