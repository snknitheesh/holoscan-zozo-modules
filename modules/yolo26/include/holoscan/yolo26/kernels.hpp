// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The two CUDA kernels around an end-to-end YOLO26 model, usable without a
// Holoscan graph:
//
// - the letterbox, which writes an image into the model's input;
// - the decode, which keeps the model's output rows that pass the score
//   threshold, maps their corners back to the image and compacts them, in
//   model order, into fixed-capacity outputs. The model emits a fixed number
//   of rows, [x1, y1, x2, y2, score, class], with corners in letterboxed input
//   pixels and no suppression left to do.
//
// Yolo26DetectionOp runs both around TensorRT.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include <holoscan/yolo26/letterbox.hpp>

namespace holoscan::yolo26 {

// An image in GPU memory: rows of `width` packed pixels, `pitch` bytes apart.
struct ImageView {
  const std::uint8_t* data{};
  std::size_t pitch{};
  int width{};
  int height{};
  PixelFormat format{PixelFormat::kRgb8};
};

// Writes the letterboxed image as float32 [3, size, size] RGB planes, each
// value in [0, 1]: bilinear samples of the image (pixel centers aligned, edges
// clamped, as OpenCV's INTER_LINEAR) divided by 255, and `pad_value` / 255
// outside it. Alpha is ignored. `image` must match the letterbox's source size.
cudaError_t launch_letterbox(const ImageView& image, const Letterbox& letterbox,
                             std::uint8_t pad_value, float* output, cudaStream_t stream);

// Floats per model output row.
inline constexpr int kRowFields = 6;
// Largest number of model rows and of kept detections.
inline constexpr int kMaxRows = 1 << 20;

struct DecodeParams {
  // Model rows to read.
  int rows{};
  // A row is kept when its score is finite and at least this.
  float score_threshold{};
  // Class values must round to 0..num_classes - 1.
  int num_classes{};
  // Detections the outputs hold; rows kept beyond it are dropped, the
  // lowest-ranked first.
  int capacity{};
  // The input's letterbox, to map corners back to the image.
  Letterbox letterbox;
};

// GPU outputs: `boxes` holds capacity x [x1, y1, x2, y2] in image pixels,
// `scores` and `class_ids` capacity values, and `count` one int32, the
// detections written. Slots from `count` on are zero. `passed`, if given,
// gets the number of rows kept before the capacity cut, so a caller can tell
// that some were dropped; it may be mapped host memory.
struct DecodeOutputs {
  float* boxes{};
  float* scores{};
  std::int32_t* class_ids{};
  std::int32_t* count{};
  std::int32_t* passed{};
};

// Decodes `rows` (params.rows x kRowFields floats in GPU memory) on `stream`.
// A row is kept when its score passes the threshold, its class value is a
// valid class, and its box, mapped back to the image as
// ((x - pad_left) / scale, (y - pad_top) / scale) and clamped to
// [0, width] x [0, height], still has a positive width and height. Kept rows
// keep their order.
cudaError_t launch_decode(const float* rows, const DecodeParams& params,
                          const DecodeOutputs& outputs, cudaStream_t stream);

}  // namespace holoscan::yolo26
