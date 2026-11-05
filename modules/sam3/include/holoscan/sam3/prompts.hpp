// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What Sam3SegmentationOp looks for, and the precomputed text embeddings its
// text prompts come from. Host code only: no SDK, no CUDA.
//
// SAM 3 segments every instance of a concept. A prompt names the concept
// with a short noun phrase ("apple", "red circle"), with exemplar boxes
// around instances of it (positive) or around things that are not (negative),
// or with both. A prompt with boxes and no text uses the text "visual", as
// SAM 3's own predictor does.
//
// SAM 3's text encoder is not part of the exported model: it is a large
// CLIP-style transformer with a byte-pair tokenizer, run once per prompt, not
// once per frame. tools/prepare_model.py runs it for a list of prompts and
// writes their embeddings to a prompt table; the operator reads the table
// and can use any prompt in it. To use a new phrase, add it to the table
// with the tool.
//
// A prompt table file (little-endian):
//
//   char[8]  "SAM3PRM1"
//   uint32   count, tokens, dim
//   count times:
//     uint32   text length in bytes, then the UTF-8 text
//     float32  mask[tokens]           1 for a token, 0 for padding
//     float32  features[tokens][dim]  the projected text features

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace holoscan::sam3 {

// The text SAM 3 uses for a prompt of boxes alone. Every table has it.
inline constexpr std::string_view kVisualPrompt = "visual";

// The most prompts one operator runs per image, and the longest prompt text.
inline constexpr int kMaxPrompts = 16;
inline constexpr std::size_t kMaxPromptTextBytes = 256;

// An exemplar box, in fractions of the image's width and height: x1 < x2 and
// y1 < y2, all in [0, 1]. A positive box marks an instance of the concept, a
// negative one something that is not.
struct PromptBox {
  std::array<float, 4> xyxy{};
  bool positive{true};
};

struct Sam3Prompt {
  // A phrase in the prompt table; empty for boxes alone ("visual").
  std::string text;
  // Exemplar boxes; at most as many as the decoder was exported for (four
  // with tools/prepare_model.py).
  std::vector<PromptBox> boxes;
};

// The table's name for a prompt's text: its text, or "visual".
[[nodiscard]] std::string_view table_text(const Sam3Prompt& prompt) noexcept;

// A short name for a prompt, for labels: its text, or "box" for boxes alone.
[[nodiscard]] std::string display_name(const Sam3Prompt& prompt);

// @throws std::invalid_argument naming what makes `prompt` unusable: no text
// and no boxes, a text longer than kMaxPromptTextBytes, or a box that is not
// finite and ordered within [0, 1].
void validate_prompt(const Sam3Prompt& prompt);

// One phrase's precomputed text embedding.
struct TextEmbedding {
  std::string text;
  // `tokens` values: 1 for a token, 0 for padding.
  std::vector<float> mask;
  // tokens x dim values, row by row.
  std::vector<float> features;
};

struct PromptTable {
  int tokens{};
  int dim{};
  std::vector<TextEmbedding> entries;

  // The entry for `text`, if any.
  [[nodiscard]] const TextEmbedding* find(std::string_view text) const noexcept;
};

// @throws std::runtime_error if the file cannot be read or is not a valid
// table: a bad header, an entry of the wrong size, a duplicate or empty text.
[[nodiscard]] PromptTable read_prompt_table(const std::filesystem::path& path);

// Writes `table`, replacing the file atomically.
// @throws std::invalid_argument for an inconsistent table;
// std::runtime_error if the file cannot be written.
void write_prompt_table(const std::filesystem::path& path, const PromptTable& table);

// The decoder's per-prompt inputs, as the exported decoder takes them.
struct DecoderPrompt {
  // [tokens][dim] and [tokens]: the text's embedding.
  std::vector<float> text_features;
  std::vector<float> text_mask;
  // [max_boxes][4] boxes as center x, center y, width, height in fractions of
  // the image, and [max_boxes] labels (1 positive, 0 negative) and validity
  // (1 for a box, 0 for padding).
  std::vector<float> boxes;
  std::vector<float> box_labels;
  std::vector<float> box_valid;
};

// The decoder inputs of `prompt` for a decoder with `max_boxes` box slots.
// @throws std::invalid_argument if the prompt is invalid, has more boxes than
// `max_boxes`, or its text is not in `table`.
[[nodiscard]] DecoderPrompt make_decoder_prompt(const Sam3Prompt& prompt, const PromptTable& table,
                                                int max_boxes);

// Prompts that change while a graph runs. Give the same control to
// Sam3SegmentationConfig::prompt_control and keep it: set() replaces the
// prompts from the operator's next image on. The runtime may construct the
// operator more than once; every copy shares the control. Thread-safe.
class Sam3PromptControl {
 public:
  // Checks prompts against `table`, the operator's prompt table.
  explicit Sam3PromptControl(PromptTable table);

  // @throws std::invalid_argument for an empty or too long list, an invalid
  // prompt or a text that is not in the table; the prompts stay as they were.
  // A prompt with more boxes than the decoder takes is refused by the
  // operator, which then keeps its prompts and logs a warning.
  void set(std::vector<Sam3Prompt> prompts);

  // The prompts set last and how many times set() has succeeded; 0 and no
  // prompts before the first.
  [[nodiscard]] std::pair<std::uint64_t, std::vector<Sam3Prompt>> latest() const;

  [[nodiscard]] const PromptTable& table() const noexcept { return table_; }

 private:
  const PromptTable table_;
  mutable std::mutex mutex_;
  std::uint64_t generation_{};
  std::vector<Sam3Prompt> prompts_;
};

// @throws std::invalid_argument for 0 or more than kMaxPrompts prompts, an
// invalid prompt, or a text that is not in `table`.
void validate_prompts(const std::vector<Sam3Prompt>& prompts, const PromptTable& table);

}  // namespace holoscan::sam3
