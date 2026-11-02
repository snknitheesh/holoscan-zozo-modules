// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The CUDA kernels around SAM 3's encoder and decoder, usable without a
// Holoscan graph:
//
// - the preprocessing, which stretches an image into the encoder's input and
//   normalizes it;
// - the selection, which keeps one prompt's best instances from the
//   decoder's outputs, with their low-resolution mask logits;
// - the finalization, which ranks every prompt's instances together, best
//   first, and upsamples their masks to the image as binary masks and a label
//   image;
// - the overlay, which colors a label image for HoloViz.
//
// Sam3SegmentationOp and Sam3OverlayOp run them. The per-pixel arithmetic is
// in pixel_math.hpp, and host references in reference.hpp.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include <holoscan/sam3/pixel_math.hpp>

namespace holoscan::sam3 {

// An image in GPU memory: rows of `width` packed pixels, `pitch` bytes apart.
struct ImageView {
  const std::uint8_t* data{};
  std::size_t pitch{};
  int width{};
  int height{};
  PixelFormat format{PixelFormat::kRgb8};
};

// Writes float32 [3, input_height, input_width] RGB planes (preprocess_pixel()
// per pixel). Alpha is ignored.
cudaError_t launch_preprocess(const ImageView& image, int input_width, int input_height,
                              const Normalization& normalization, float* input,
                              cudaStream_t stream);

// Every prompt's candidate instances of one image, in GPU memory: `capacity`
// slots per prompt, prompt p's from slot p * capacity, best first.
struct Candidates {
  // [prompts * capacity]: the score, in [0, 1].
  float* scores{};
  // [prompts * capacity][4]: the box, x1, y1, x2, y2 in fractions of the image.
  float* boxes{};
  // [prompts * capacity]: the decoder query it came from.
  std::int32_t* queries{};
  // [prompts * capacity][mask_height * mask_width]: its mask logits.
  float* logits{};
  // [prompts]: how many slots hold an instance.
  std::int32_t* counts{};
  // [prompts], optional (host-mapped memory, for example): how many passed
  // the score threshold, before the capacity cut.
  std::int32_t* passed{};
};

struct SelectParams {
  // The prompt whose decoder outputs these are.
  int prompt{};
  // The decoder's query count.
  int queries{};
  // Slots per prompt.
  int capacity{};
  int mask_width{};
  int mask_height{};
  // Instances need a score above this.
  float score_threshold{0.5F};
};

// Keeps the instances of one prompt that score above the threshold, at most
// `capacity`, best first (ranks_before(), keyed by query), from the decoder's
// float32 `scores` [queries], `boxes` [queries][4] and `masks`
// [queries][mask_height][mask_width].
cudaError_t launch_select(const float* scores, const float* boxes, const float* masks,
                          const SelectParams& params, const Candidates& candidates,
                          cudaStream_t stream);

// One image's instances, in GPU memory, `capacity` of each, best first; the
// rows beyond `count` hold zeros (prompt_ids: -1).
struct Instances {
  // uint8 [capacity][height][width]: 1 inside the instance, 0 outside;
  // optional.
  std::uint8_t* masks{};
  // uint16 [height][width]: 1 + the index of the best instance covering the
  // pixel, 0 for none.
  std::uint16_t* labels{};
  // [capacity]: the score.
  float* scores{};
  // [capacity][4]: the box, x1, y1, x2, y2 in image pixels.
  float* boxes{};
  // [capacity]: the index of its prompt.
  std::int32_t* prompt_ids{};
  // [1]: how many instances there are.
  std::int32_t* count{};
  // [capacity]: scratch, each instance's candidate slot.
  std::int32_t* order{};
};

struct FinalizeParams {
  int prompts{};
  // Slots per prompt, and instances in the output.
  int capacity{};
  int mask_width{};
  int mask_height{};
  // The image's size.
  int width{};
  int height{};
  // A mask keeps the pixels whose upsampled logit exceeds this.
  float mask_logit_threshold{0.0F};
};

// Ranks every prompt's candidates together (ranks_before(), keyed by slot),
// keeps the best `capacity`, and writes them as instances: scores, boxes in
// pixels (box_to_pixels()), prompt ids, the count, and, upsampled to the image
// (sample_mask()), their masks and the label image.
cudaError_t launch_finalize(const Candidates& candidates, const FinalizeParams& params,
                            const Instances& instances, cudaStream_t stream);

// Writes RGBA8 [height][width][4] (overlay_pixel() per pixel).
cudaError_t launch_overlay(const std::uint16_t* labels, int width, int height,
                           std::uint8_t fill_alpha, std::uint8_t* rgba, cudaStream_t stream);

}  // namespace holoscan::sam3
