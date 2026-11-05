// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A stand-in for SAM 3, for tests and the `ci` mode, which need no downloaded
// weights: a small ONNX encoder and decoder with the tensors of
// tools/prepare_model.py's export, and a prompt table, written by the module
// itself. Their response is known, so through TensorRT a test predicts every
// output of Sam3SegmentationOp from the image and the prompts:
//
// - The encoder averages the normalized image over square cells: fpn0 is
//   [1, 3, M, M] for an S x S input, fpn1 and fpn2 halve it twice.
// - The decoder reads, for each of its Q queries q, the q-th token of the
//   prompt's text features: weights w (3), a bias b, a score, a box (4) and a
//   zero factor z. Query q's mask logits are w . fpn0 + b + z * (mean(fpn1) -
//   mean(fpn2)); its score is the score times the token's mask value; its box
//   is the box plus the sum of the prompt's positive boxes (center x, center
//   y, width, height). The z term only keeps fpn1 and fpn2 bound; it is 0.
//
// No Holoscan, no CUDA.

#pragma once

#include <array>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <holoscan/sam3/prompts.hpp>
#include <holoscan/sam3/reference.hpp>

namespace holoscan::sam3 {

struct StubShape {
  // The encoder's input side S, a multiple of mask_size.
  int input_size{64};
  // The mask side M, a multiple of 4.
  int mask_size{16};
  // Decoder queries Q, text tokens (at least Q) and feature size (at least
  // 10), and box slots.
  int queries{4};
  int tokens{8};
  int dim{16};
  int boxes{2};
};

// @throws std::invalid_argument for a shape the stub cannot have.
void validate_stub_shape(const StubShape& shape);

// What one decoder query of a stub prompt does.
struct StubQuery {
  // Mask logits: weights of fpn0's red, green and blue, and a bias.
  std::array<float, 3> weights{};
  float bias{};
  float score{};
  // x1, y1, x2, y2 in fractions of the image.
  std::array<float, 4> box{};
};

// A prompt table entry for the stub whose first queries do `queries`; the
// others give score 0 and an empty mask.
// @throws std::invalid_argument if there are more queries than the shape has.
[[nodiscard]] TextEmbedding stub_embedding(std::string text, const StubShape& shape,
                                           std::span<const StubQuery> queries);

// The prompts of the stub's own table, for the synthetic scene
// (synthetic_scene.hpp): "red" finds its red circle, "blue" its blue
// rectangle and "visual" (boxes alone) its yellow triangle; each also has a
// second query that never passes a 0.5 threshold.
[[nodiscard]] PromptTable stub_prompt_table(const StubShape& shape);

// The serialized ONNX models (opset 13).
[[nodiscard]] std::string stub_encoder_onnx(const StubShape& shape);
[[nodiscard]] std::string stub_decoder_onnx(const StubShape& shape);

struct StubModel {
  std::filesystem::path encoder;
  std::filesystem::path decoder;
  std::filesystem::path prompt_table;
};

// Writes the stub's encoder, decoder and prompt table (stub_prompt_table())
// to `directory`, each unless the file already holds it, and returns their
// paths. Files are replaced atomically, so concurrent runs can share the
// directory, and TensorRT engines cached beside the models stay valid as long
// as they do not change.
// @throws std::runtime_error if a file cannot be written.
StubModel write_stub_model(const std::filesystem::path& directory, const StubShape& shape = {});

// The stub encoder's fpn0 for an encoder input (reference_preprocess()):
// [3][M][M].
[[nodiscard]] std::vector<float> reference_stub_encoder(std::span<const float> input,
                                                        const StubShape& shape);

// The stub decoder's outputs for fpn0 and one prompt's decoder inputs.
[[nodiscard]] DecoderOutput reference_stub_decoder(std::span<const float> fpn0,
                                                   const DecoderPrompt& prompt,
                                                   const StubShape& shape);

}  // namespace holoscan::sam3
