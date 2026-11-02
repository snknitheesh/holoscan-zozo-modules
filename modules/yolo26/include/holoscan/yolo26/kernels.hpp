// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The CUDA kernels around an end-to-end YOLO26 model, usable without a
// Holoscan graph:
//
// - the letterbox, which writes an image into the model's input;
// - the decode, which keeps the model's output rows that pass the score
//   threshold, maps their corners back to the image and compacts them, in
//   model order, into fixed-capacity outputs. The model emits a fixed number
//   of rows, [x1, y1, x2, y2, score, class], with corners in letterboxed input
//   pixels and no suppression left to do;
// - for a segmentation model, the decode that also gathers each kept row's
//   mask coefficients, and the mask decode, which turns them and the model's
//   prototype masks into instance masks and a label image at the image's size.
//
// Yolo26DetectionOp and Yolo26SegmentationOp run them around TensorRT.

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

// Segmentation. An end-to-end YOLO26 segmentation model gives two outputs:
// rows of [x1, y1, x2, y2, score, class, c_0 .. c_M-1], the detection's six
// fields followed by M mask coefficients, and M prototype masks of
// proto_height x proto_width (a quarter of the input side). An instance's
// mask is where the sum over m of c_m * prototype_m, upsampled bilinearly to
// the image, exceeds a threshold, cropped to the instance's box.

// Most mask coefficients per row (YOLO26 models have 32).
inline constexpr int kMaxMaskChannels = 64;
// Largest prototype mask side.
inline constexpr int kMaxProtoSide = kMaxInputSize;

// The decode of segmentation rows: as launch_decode() for the rows' first six
// fields, with rows of 6 + mask_channels floats, and the mask coefficients of
// each detection written to `coefficients`: capacity x mask_channels floats in
// GPU memory, a detection's in its slot, zero in the slots from `count` on.
cudaError_t launch_segmentation_decode(const float* rows, const DecodeParams& params,
                                       int mask_channels, const DecodeOutputs& outputs,
                                       float* coefficients, cudaStream_t stream);

struct MaskParams {
  // Detection slots: the length of `boxes` and `coefficients`, and of `masks`.
  int capacity{};
  int mask_channels{};
  int proto_width{};
  int proto_height{};
  // A pixel belongs to an instance when the mask's value there, a logit,
  // exceeds this: 0 is a probability of one half.
  float logit_threshold{};
  // The input's letterbox; the masks have its source size.
  Letterbox letterbox;
};

// GPU inputs: the model's prototypes (mask_channels x proto_height x
// proto_width floats), and launch_segmentation_decode()'s coefficients, boxes
// (in image pixels) and count, which is read on the GPU.
struct MaskInputs {
  const float* protos{};
  const float* coefficients{};
  const float* boxes{};
  const std::int32_t* count{};
};

// GPU outputs, each optional:
// - `masks`: capacity x height x width uint8, the source image's size: 1 where
//   the detection in that slot covers the pixel, 0 elsewhere and in the slots
//   from `count` on;
// - `labels`: height x width int32: 0 where no instance covers the pixel,
//   otherwise 1 + the slot of the first (best) detection that does.
// `logits` is scratch of capacity x proto_height x proto_width floats.
struct MaskOutputs {
  std::uint8_t* masks{};
  std::int32_t* labels{};
  float* logits{};
};

// Decodes the instance masks on `stream`. Each detection's mask logits are
// its coefficients times the prototypes, at the prototypes' resolution. Each
// image pixel (x, y) inside a detection's box (x1 <= x < x2, y1 <= y < y2)
// samples them at its center as the letterbox placed it in the input:
// input u = pad_left + (x + 0.5) * resized_width / width (v likewise), then
// prototype p = u * proto_width / size - 0.5, bilinearly with the edges
// clamped (as PyTorch's interpolate without align_corners), and belongs to
// the detection when the sample exceeds the threshold.
cudaError_t launch_masks(const MaskInputs& inputs, const MaskParams& params,
                         const MaskOutputs& outputs, cudaStream_t stream);

}  // namespace holoscan::yolo26
