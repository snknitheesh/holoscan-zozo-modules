// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The CUDA kernels around a monocular depth model, usable without a Holoscan
// graph:
//
// - the preprocessing, which resizes an image into the model's input and
//   normalizes it;
// - the postprocessing, which resamples the model's output back onto the
//   image as inverse depth and, optionally, metric depth;
// - the range of a depth image, and its colorization.
//
// MonoDepthOp runs them around TensorRT. The per-pixel arithmetic is in
// pixel_math.hpp, and host references in reference.hpp.

#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include <holoscan/mono_depth/image_fit.hpp>
#include <holoscan/mono_depth/pixel_math.hpp>

namespace holoscan::mono_depth {

// An image in GPU memory: rows of `width` packed pixels, `pitch` bytes apart.
struct ImageView {
  const std::uint8_t* data{};
  std::size_t pitch{};
  int width{};
  int height{};
  PixelFormat format{PixelFormat::kRgb8};
};

// Writes float32 [3, input_height, input_width] RGB planes (preprocess_pixel()
// per pixel). `image` must match the fit's source size. Alpha is ignored.
cudaError_t launch_preprocess(const ImageView& image, const ImageFit& fit,
                              const Normalization& normalization, float* input,
                              cudaStream_t stream);

// Where the postprocessing writes: float32 [height, width] each, packed.
// `metric` may be null, and is ignored unless the model gives metric depth or
// `affine` is given.
struct DepthOutputs {
  float* inverse{};
  float* metric{};
};

// Resamples the model's float32 [output_height, output_width] output onto
// the fit's source image (sample_output() and depth_pair() per pixel).
// `affine`, for a relative model, converts its inverse depth to meters.
cudaError_t launch_postprocess(const float* output, const ImageFit& fit, ModelOutput model_output,
                               const InverseDepthAffine* affine, const DepthOutputs& outputs,
                               cudaStream_t stream);

// The smallest and largest finite value of `count` floats, written to
// `range` (GPU memory) as two ordered_bits() values. With no finite value,
// `range` holds {0xFFFFFFFF, 0}, which decodes to NaNs: an empty range, which
// colorizes everything as the middle entry, as reference_range()'s {0, 0}.
cudaError_t launch_depth_range(const float* depth, std::size_t count, std::uint32_t* range,
                               cudaStream_t stream);

// Where the colorization takes its range: a fixed [low, high], or the
// ordered_bits() pair `range` that launch_depth_range() wrote on the GPU.
struct ColorRange {
  const std::uint32_t* range{};
  float low{};
  float high{};
};

// Writes RGBA8 [height, width, 4], packed and opaque: each inverse depth's
// Turbo entry (colormap_index()).
cudaError_t launch_colorize(const float* inverse_depth, int width, int height,
                            const ColorRange& range, std::uint8_t* rgba, cudaStream_t stream);

}  // namespace holoscan::mono_depth
