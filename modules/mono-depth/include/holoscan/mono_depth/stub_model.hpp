// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A stand-in for a depth model, for tests and the `ci` mode, which need no
// downloaded weights. It is a small ONNX network with the tensors of a Depth
// Anything export, `image` float32 [1, 3, H, W] in and `depth` float32
// [1, H, W] out, and a known response: each output pixel is a fixed weighted
// sum of the same input pixel's normalized red, green and blue, plus a bias
// (stub_response()). Its "depth" means nothing, but through TensorRT it lets
// a test predict every output value of MonoDepthOp from the image alone, so
// the pre- and postprocessing are checked exactly; swapped red and blue
// change the output. Its values are positive, so it also serves as a metric
// model. No Holoscan.

#pragma once

#include <filesystem>
#include <string>

namespace holoscan::mono_depth {

// The stub's weights of normalized red, green and blue, and its bias.
inline constexpr float kStubWeights[3] = {0.5F, 0.25F, 0.125F};
inline constexpr float kStubBias = 2.5F;

// The stub's output for one pixel's normalized red, green and blue, in the
// order its graph adds them.
[[nodiscard]] constexpr float stub_response(float red, float green, float blue) noexcept {
  return red * kStubWeights[0] + green * kStubWeights[1] + blue * kStubWeights[2] + kStubBias;
}

// The serialized ONNX model (opset 13) for a width x height input.
// @throws std::invalid_argument for a side outside 1..kMaxModelSide.
[[nodiscard]] std::string stub_model_onnx(int width, int height);

// Writes the model for width x height to
// `directory`/mono_depth_stub_<width>x<height>.onnx unless that file already
// holds it, and returns its path. The file is replaced atomically, so
// concurrent runs can share the directory, and a TensorRT engine cached beside
// it stays valid as long as the file does not change.
// @throws std::runtime_error if the file cannot be written.
std::filesystem::path write_stub_model(const std::filesystem::path& directory, int width,
                                       int height);

}  // namespace holoscan::mono_depth
