// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include <holoscan/sam3/pixel_math.hpp>
#include <holoscan/sam3/reference.hpp>

namespace holoscan::sam3 {
namespace {

// A candidate instance: its score, its key in the ranking, and where it came
// from.
struct Candidate {
  float score{};
  int key{};
  int prompt{};
  int query{};
};

// `items` ordered as the kernels rank them.
void rank(std::vector<Candidate>& items) {
  std::sort(items.begin(), items.end(), [](const Candidate& a, const Candidate& b) {
    return ranks_before(a.score, a.key, b.score, b.key);
  });
}

}  // namespace

std::vector<float> reference_preprocess(const std::uint8_t* image, std::size_t pitch,
                                        PixelFormat format, int width, int height, int input_width,
                                        int input_height, const Normalization& normalization) {
  const PreprocessParams p = make_preprocess_params(width, height, format, pitch, input_width,
                                                    input_height, normalization);
  const std::size_t plane =
      static_cast<std::size_t>(input_width) * static_cast<std::size_t>(input_height);
  std::vector<float> input(3 * plane);
  for (int v = 0; v < input_height; ++v) {
    for (int u = 0; u < input_width; ++u) {
      float rgb[3];
      preprocess_pixel(p, image, u, v, rgb);
      const std::size_t index = static_cast<std::size_t>(v) * input_width + u;
      input[index] = rgb[0];
      input[plane + index] = rgb[1];
      input[2 * plane + index] = rgb[2];
    }
  }
  return input;
}

ReferenceInstances reference_instances(std::span<const DecoderOutput> prompts, int capacity,
                                       float score_threshold, float mask_logit_threshold, int width,
                                       int height) {
  if (capacity < 1 || width < 1 || height < 1) {
    throw std::invalid_argument("reference_instances: capacity and size must be positive");
  }
  ReferenceInstances result;
  // Each prompt's best `capacity`, keyed by query, then all of them keyed by
  // their slot, which orders them by prompt and then by rank.
  std::vector<Candidate> all;
  for (std::size_t p = 0; p < prompts.size(); ++p) {
    const DecoderOutput& out = prompts[p];
    std::vector<Candidate> passing;
    for (int q = 0; q < out.queries; ++q) {
      const float score = out.scores[static_cast<std::size_t>(q)];
      if (score > score_threshold) {
        passing.push_back({.score = score, .key = q, .prompt = static_cast<int>(p), .query = q});
      }
    }
    result.passed.push_back(static_cast<int>(passing.size()));
    rank(passing);
    for (std::size_t r = 0; r < passing.size() && r < static_cast<std::size_t>(capacity); ++r) {
      Candidate candidate = passing[r];
      candidate.key = static_cast<int>(p) * capacity + static_cast<int>(r);
      all.push_back(candidate);
    }
  }
  rank(all);
  result.count = std::min(static_cast<int>(all.size()), capacity);

  const auto rows = static_cast<std::size_t>(capacity);
  const std::size_t plane = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  result.scores.assign(rows, 0.0F);
  result.boxes.assign(4 * rows, 0.0F);
  result.prompt_ids.assign(rows, -1);
  result.masks.assign(rows * plane, 0);
  result.labels.assign(plane, 0);
  result.logits.assign(static_cast<std::size_t>(result.count) * plane, 0.0F);
  for (int k = 0; k < result.count; ++k) {
    const Candidate& c = all[static_cast<std::size_t>(k)];
    const DecoderOutput& out = prompts[static_cast<std::size_t>(c.prompt)];
    const auto row = static_cast<std::size_t>(k);
    result.scores[row] = c.score;
    box_to_pixels(out.boxes.data() + 4 * static_cast<std::size_t>(c.query), width, height,
                  result.boxes.data() + 4 * row);
    result.prompt_ids[row] = c.prompt;
    const MaskSampling s = make_mask_sampling(out.mask_width, out.mask_height, width, height);
    const float* logits = out.masks.data() + static_cast<std::size_t>(c.query) *
                                                 static_cast<std::size_t>(out.mask_width) *
                                                 static_cast<std::size_t>(out.mask_height);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const std::size_t index = static_cast<std::size_t>(y) * width + x;
        const float logit = sample_mask(s, logits, x, y);
        result.logits[row * plane + index] = logit;
        if (logit > mask_logit_threshold) {
          result.masks[row * plane + index] = 1;
          if (result.labels[index] == 0) {
            result.labels[index] = static_cast<std::uint16_t>(k + 1);
          }
        }
      }
    }
  }
  return result;
}

std::vector<std::uint8_t> reference_overlay(std::span<const std::uint16_t> labels, int width,
                                            int height, std::uint8_t fill_alpha) {
  if (labels.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(height)) {
    throw std::invalid_argument("reference_overlay: the labels do not hold the image");
  }
  std::vector<std::uint8_t> rgba(labels.size() * 4U);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      overlay_pixel(labels.data(), width, height, x, y, fill_alpha,
                    rgba.data() + 4 * (static_cast<std::size_t>(y) * width + x));
    }
  }
  return rgba;
}

}  // namespace holoscan::sam3
