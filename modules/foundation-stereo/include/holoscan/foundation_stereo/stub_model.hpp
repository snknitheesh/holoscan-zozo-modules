// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A stand-in for a Fast-FoundationStereo model, for tests and the `ci` mode,
// which need no downloaded weights. It is a small ONNX network with the
// tensors of a single-file export, `left_image` and `right_image` float32
// [1, 3, H, W] in (ImageNet-normalized RGB) and `disparity` float32
// [1, 1, H, W] out, whose "disparity" is an exact function of the pixels it
// is given, stub_disparity(), so a test can recompute it on the CPU:
// FoundationStereoOp's preprocessing (resize, channel order, normalization),
// TensorRT, and its postprocessing (resize, rescale, invalidation, depth) all
// show in the result. It does not match stereo. No Holoscan.

#pragma once

#include <filesystem>
#include <string>

namespace holoscan::foundation_stereo {

// Weights of the left image's R, G and B (0..255) and of the right image's R.
inline constexpr float kStubLeftWeights[3] = {0.125F, 0.0625F, 0.03125F};
inline constexpr float kStubRightWeight = 0.015625F;

// The stub's output for one pixel, from the 0..255 values that its
// normalized inputs stand for.
[[nodiscard]] constexpr float stub_disparity(const float left_rgb[3], float right_red) noexcept {
  return kStubLeftWeights[0] * left_rgb[0] + kStubLeftWeights[1] * left_rgb[1] +
         kStubLeftWeights[2] * left_rgb[2] - kStubRightWeight * right_red;
}

// Smallest and largest stub model sides.
inline constexpr int kMinStubSide = 8;
inline constexpr int kMaxStubSide = 4096;

// The serialized ONNX model (opset 13) for width x height inputs.
// @throws std::invalid_argument for a side outside kMinStubSide..kMaxStubSide.
[[nodiscard]] std::string stub_model_onnx(int width, int height);

// Writes the model to `directory`/foundation_stereo_stub_<width>x<height>.onnx
// unless that file already holds it, and returns its path. The file is
// replaced atomically, so concurrent runs can share the directory, and a
// TensorRT engine cached beside it stays valid as long as the file does not
// change.
// @throws std::runtime_error if the file cannot be written.
std::filesystem::path write_stub_model(const std::filesystem::path& directory, int width,
                                       int height);

}  // namespace holoscan::foundation_stereo
