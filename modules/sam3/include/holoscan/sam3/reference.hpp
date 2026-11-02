// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Host references of what Sam3SegmentationOp and Sam3OverlayOp compute, from
// the same per-pixel arithmetic as their kernels (pixel_math.hpp and
// kernels.hpp): the preprocessing, the selection of each prompt's instances,
// their ranking, masks and label image, and the overlay. With the stub model
// (stub_model.hpp) they predict the operator's outputs from the image. For
// tests and the demo's checks; no SDK, no CUDA.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <holoscan/sam3/pixel_math.hpp>

namespace holoscan::sam3 {

// The encoder input: float32 [3, input_height, input_width] planes of
// `image`, rows of `pitch` bytes in `format`, as launch_preprocess() writes
// them.
[[nodiscard]] std::vector<float> reference_preprocess(const std::uint8_t* image, std::size_t pitch,
                                                      PixelFormat format, int width, int height,
                                                      int input_width, int input_height,
                                                      const Normalization& normalization);

// One prompt's decoder outputs: `queries` scores, boxes (x1, y1, x2, y2 in
// fractions of the image) and mask_height x mask_width mask logits each.
struct DecoderOutput {
  int queries{};
  int mask_width{};
  int mask_height{};
  std::vector<float> scores;
  std::vector<float> boxes;
  std::vector<float> masks;
};

// One image's instances, as Sam3SegmentationOp gives them: `capacity` rows,
// best first, zero beyond `count` (prompt_ids: -1).
struct ReferenceInstances {
  int count{};
  std::vector<float> scores;
  std::vector<float> boxes;
  std::vector<std::int32_t> prompt_ids;
  // [capacity][height][width], 0 or 1.
  std::vector<std::uint8_t> masks;
  // [height][width].
  std::vector<std::uint16_t> labels;
  // [count][height][width]: each instance's upsampled mask logits, which
  // tell how near a pixel is to its mask's edge.
  std::vector<float> logits;
  // Per prompt, how many instances passed the score threshold.
  std::vector<int> passed;
};

// What launch_select() and launch_finalize() make of every prompt's decoder
// outputs, in prompt order, for a width x height image.
[[nodiscard]] ReferenceInstances reference_instances(std::span<const DecoderOutput> prompts,
                                                     int capacity, float score_threshold,
                                                     float mask_logit_threshold, int width,
                                                     int height);

// RGBA8 [height][width][4] of a label image, as launch_overlay() writes it.
[[nodiscard]] std::vector<std::uint8_t> reference_overlay(std::span<const std::uint16_t> labels,
                                                          int width, int height,
                                                          std::uint8_t fill_alpha);

}  // namespace holoscan::sam3
