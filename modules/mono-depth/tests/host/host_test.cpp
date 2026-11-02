// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The module's host-only pieces, without the SDK, CUDA or a GPU: the fit of
// an image into the model input and of the output back onto it; the host
// references, against independent evaluations; the stub model file; the
// synthetic scene's geometry and truth; Spearman's correlation and the
// ordering check; and the colormap.
//
// Usage: mono_depth_host_test DIR (a scratch folder for the files)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/mono_depth/colormap.hpp>
#include <holoscan/mono_depth/image_fit.hpp>
#include <holoscan/mono_depth/pixel_math.hpp>
#include <holoscan/mono_depth/reference.hpp>
#include <holoscan/mono_depth/stub_model.hpp>
#include <holoscan/mono_depth/synthetic_scene.hpp>

#include "test_support.hpp"

namespace holoscan::mono_depth {
namespace {

std::filesystem::path g_directory;

void check_fit(int width, int height, int input_width, int input_height, int output_width,
               int output_height, ResizeMode mode, std::array<int, 4> placement,
               std::array<int, 4> valid) {
  const ImageFit f =
      make_fit(width, height, input_width, input_height, output_width, output_height, mode);
  const std::string what =
      std::to_string(width) + "x" + std::to_string(height) + ": " +
      std::to_string(f.resized_width) + "x" + std::to_string(f.resized_height) + " at (" +
      std::to_string(f.pad_left) + ", " + std::to_string(f.pad_top) + "), valid " +
      std::to_string(f.valid_left) + ".." + std::to_string(f.valid_right) + " x " +
      std::to_string(f.valid_top) + ".." + std::to_string(f.valid_bottom);
  VERIFY_MSG(f.resized_width == placement[0] && f.resized_height == placement[1] &&
                 f.pad_left == placement[2] && f.pad_top == placement[3],
             what);
  VERIFY_MSG(f.valid_left == valid[0] && f.valid_right == valid[1] && f.valid_top == valid[2] &&
                 f.valid_bottom == valid[3],
             what);
}

void check_fits() {
  const auto letterbox = ResizeMode::kLetterbox;
  // Wide: 1280x720 * 518/1280 = 518x291.375, rounded to 291; 227 rows of padding,
  // 113 above and 114 below.
  check_fit(1280, 720, 518, 518, 518, 518, letterbox, {518, 291, 0, 113}, {0, 517, 113, 403});
  // Tall, and an output at half the input's size: input columns 113..403 are
  // output columns 56.5..202, whose pixel centers 56.5..201.5 lie inside.
  check_fit(720, 1280, 518, 518, 259, 259, letterbox, {291, 518, 113, 0}, {56, 201, 0, 258});
  // Half to even: 1036x519 * 0.5 = 518x259.5, rounded to 260.
  check_fit(1036, 519, 518, 518, 518, 518, letterbox, {518, 260, 0, 129}, {0, 517, 129, 388});
  // The same aspect ratio: no padding.
  check_fit(1036, 588, 518, 294, 518, 294, letterbox, {518, 294, 0, 0}, {0, 517, 0, 293});
  // Stretched: the whole input, whatever the image.
  check_fit(640, 480, 392, 224, 784, 448, ResizeMode::kStretch, {392, 224, 0, 0}, {0, 783, 0, 447});
  // A one-pixel-high image still has a valid output row.
  check_fit(1000, 1, 28, 28, 28, 28, letterbox, {28, 1, 0, 13}, {0, 27, 13, 13});
  VERIFY(test::throws<std::invalid_argument>(
      [] { (void)make_fit(0, 10, 14, 14, 14, 14, ResizeMode::kLetterbox); }));
  VERIFY(test::throws<std::invalid_argument>(
      [] { (void)make_fit(10, 10, 14, 14, 14, 5000, ResizeMode::kLetterbox); }));
}

// The preprocessing reference against a direct evaluation: a 2x1 image, a
// red and a blue pixel, stretched to 4x1; its bilinear samples at input pixel
// centers 0.25, 0.75, 1.25, 1.75 source pixels are clamped at the edges.
void check_reference_preprocess() {
  const std::vector<std::uint8_t> image{255, 0, 0, 0, 0, 255};
  const ImageFit fit = make_fit(2, 1, 4, 1, 4, 1, ResizeMode::kStretch);
  const Normalization n{.mean = {0.5F, 0.5F, 0.5F}, .std = {0.5F, 0.5F, 0.5F}};
  const std::vector<float> input =
      reference_preprocess(image.data(), 6, PixelFormat::kRgb8, fit, n);
  // Red: 1, 0.75, 0.25, 0 of the way to the blue pixel; normalized, 2v - 1.
  const std::array<float, 4> red{1.0F, 0.5F, -0.5F, -1.0F};
  for (int u = 0; u < 4; ++u) {
    VERIFY(std::fabs(input[u] - red[u]) < 1e-6F);
    VERIFY(std::fabs(input[4 + u] + 1.0F) < 1e-6F);    // No green anywhere.
    VERIFY(std::fabs(input[8 + u] + red[u]) < 1e-6F);  // Blue the other way.
  }
  // BGR8 bytes of the same image give the same planes.
  const std::vector<std::uint8_t> bgr{0, 0, 255, 255, 0, 0};
  VERIFY(reference_preprocess(bgr.data(), 6, PixelFormat::kBgr8, fit, n) == input);
}

// The postprocessing reference: a linear ramp output resamples to the ramp at
// the image's pixel centers.
void check_reference_postprocess() {
  const int size = 28;
  std::vector<float> output(static_cast<std::size_t>(size) * size);
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      output[static_cast<std::size_t>(y) * size + x] = static_cast<float>(x);
    }
  }
  // Stretched from 56 wide: image pixel x is output coordinate (x + 0.5) / 2 - 0.5.
  const ImageFit fit = make_fit(56, 10, size, size, size, size, ResizeMode::kStretch);
  const ReferenceDepth depth =
      reference_postprocess(output, fit, ModelOutput::kRelativeInverseDepth, nullptr);
  VERIFY(depth.metric.empty() && depth.inverse.size() == 560U);
  for (int x = 0; x < 56; ++x) {
    const float expected = std::clamp((static_cast<float>(x) + 0.5F) / 2.0F - 0.5F, 0.0F, 27.0F);
    VERIFY_MSG(std::fabs(depth.inverse[static_cast<std::size_t>(x)] - expected) < 1e-5F,
               "column " + std::to_string(x));
  }
  const InverseDepthAffine affine{.scale = 2.0F, .shift = 1.0F};
  const ReferenceDepth metric =
      reference_postprocess(output, fit, ModelOutput::kRelativeInverseDepth, &affine);
  VERIFY(metric.metric.size() == 560U && std::fabs(metric.metric[0] - 1.0F) < 1e-6F);
}

void check_depth_pairs() {
  const ImageFit fit = make_fit(14, 14, 14, 14, 14, 14, ResizeMode::kStretch);
  const InverseDepthAffine affine{.scale = 1.0F, .shift = -1.0F};
  const PostprocessParams relative =
      make_postprocess_params(fit, ModelOutput::kRelativeInverseDepth, &affine);
  VERIFY(depth_pair(relative, 3.0F).inverse == 3.0F && depth_pair(relative, 3.0F).metric == 0.5F);
  // 1 / z would be 0 or negative: no metric depth.
  VERIFY(depth_pair(relative, 1.0F).metric == 0.0F && depth_pair(relative, 0.5F).metric == 0.0F);
  const PostprocessParams metric = make_postprocess_params(fit, ModelOutput::kMetricDepth, nullptr);
  VERIFY(depth_pair(metric, 4.0F).inverse == 0.25F && depth_pair(metric, 4.0F).metric == 4.0F);
  VERIFY(depth_pair(metric, -1.0F).inverse == 0.0F && depth_pair(metric, -1.0F).metric == 0.0F);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  VERIFY(depth_pair(metric, nan).inverse == 0.0F && depth_pair(relative, nan).inverse == 0.0F);
}

void check_ordered_bits() {
  const std::vector<float> values{
      -std::numeric_limits<float>::infinity(), -3.5F, -0.0F, 0.0F, 1e-30F, 2.0F,
      std::numeric_limits<float>::infinity()};
  for (std::size_t i = 0; i < values.size(); ++i) {
    VERIFY(from_ordered_bits(ordered_bits(values[i])) == values[i]);
    if (i > 0) {
      VERIFY(ordered_bits(values[i - 1]) < ordered_bits(values[i]));
    }
  }
}

void check_colormap() {
  VERIFY(colormap_index(0.0F, 0.0F, 1.0F) == 0);
  VERIFY(colormap_index(1.0F, 0.0F, 1.0F) == 255);
  VERIFY(colormap_index(-5.0F, 0.0F, 1.0F) == 0 && colormap_index(5.0F, 0.0F, 1.0F) == 255);
  VERIFY(colormap_index(0.5F, 0.0F, 1.0F) == 128);  // 127.5 rounds up.
  VERIFY(colormap_index(1.0F, 1.0F, 1.0F) == 128 && colormap_index(0.0F, 2.0F, 1.0F) == 128);
  // Turbo: dark blue far, red near, bright green in the middle.
  VERIFY(kTurbo[2] > kTurbo[0] && kTurbo[765] > kTurbo[767] && kTurbo[3 * 128 + 1] > 200);
  const std::vector<std::uint8_t> rgba = reference_colorize(std::vector<float>{0.0F, 1.0F}, 0, 1);
  VERIFY(rgba.size() == 8U && rgba[3] == 255 && rgba[7] == 255);
  VERIFY(rgba[0] == kTurbo[0] && rgba[4] == kTurbo[765]);
  VERIFY((reference_range(std::vector<float>{3.0F, std::numeric_limits<float>::quiet_NaN(),
                                             -1.0F}) == std::array<float, 2>{-1.0F, 3.0F}));
}

// The file holds the stub, is rewritten only when it changed, and encodes the
// tensors MonoDepthOp expects.
void check_stub_model() {
  const auto path = write_stub_model(g_directory, 518, 294);
  VERIFY(path.filename() == "mono_depth_stub_518x294.onnx");
  std::ifstream in(path, std::ios::binary);
  const std::string bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  VERIFY(bytes == stub_model_onnx(518, 294));
  VERIFY(bytes.find("image") != std::string::npos && bytes.find("depth") != std::string::npos);
  const auto written = std::filesystem::last_write_time(path);
  VERIFY(write_stub_model(g_directory, 518, 294) == path);
  VERIFY(std::filesystem::last_write_time(path) == written);
  VERIFY(stub_model_onnx(14, 14) != stub_model_onnx(14, 28));
  VERIFY(test::throws<std::invalid_argument>([] { (void)stub_model_onnx(0, 14); }));
  // The response: red counts most, blue least, and it stays positive.
  VERIFY(stub_response(1.0F, 0.0F, 0.0F) > stub_response(0.0F, 1.0F, 0.0F));
  VERIFY(stub_response(0.0F, 1.0F, 0.0F) > stub_response(0.0F, 0.0F, 1.0F));
  VERIFY(stub_response(-2.2F, -2.1F, -1.9F) > 0.0F);
}

void check_scene() {
  const DepthScene scene = make_depth_scene(640, 360, 0);
  std::vector<std::uint8_t> rgb(640U * 360U * 3U);
  std::vector<std::uint8_t> bgra(640U * 360U * 4U);
  std::vector<float> depth(640U * 360U);
  std::vector<SceneRegion> regions(640U * 360U);
  render_depth_scene(scene, PixelFormat::kRgb8, rgb.data(), 640U * 3U, depth.data(),
                     regions.data());
  render_depth_scene(scene, PixelFormat::kBgra8, bgra.data(), 640U * 4U);
  for (std::size_t i = 0; i < depth.size(); ++i) {
    VERIFY(depth[i] > 0.5F && depth[i] <= scene.wall_z);
    VERIFY(rgb[3 * i] == bgra[4 * i + 2] && rgb[3 * i + 2] == bgra[4 * i] &&
           bgra[4 * i + 3] == 255);
  }
  // The top row shows the wall, the left middle the panel, the bottom row the
  // floor, which is nearer at the bottom.
  const auto at = [](int u, int v) { return static_cast<std::size_t>(v) * 640U + u; };
  VERIFY(regions[at(320, 0)] == SceneRegion::kWall && depth[at(320, 0)] == scene.wall_z);
  VERIFY(regions[at(20, 180)] == SceneRegion::kPanel && depth[at(20, 180)] == scene.panel_z);
  VERIFY(regions[at(400, 359)] == SceneRegion::kFloor && depth[at(400, 359)] < depth[at(400, 300)]);
  // The sphere is in view, between the panel and the wall.
  const auto sphere = std::count(regions.begin(), regions.end(), SceneRegion::kSphere);
  VERIFY(sphere > 2000);
  for (std::size_t i = 0; i < regions.size(); ++i) {
    if (regions[i] == SceneRegion::kSphere) {
      VERIFY(depth[i] > scene.panel_z && depth[i] < scene.wall_z);
    }
  }
  // It moves.
  VERIFY(make_depth_scene(640, 360, 5).sphere_center != scene.sphere_center);
  VERIFY(test::throws<std::invalid_argument>([] { (void)make_depth_scene(16, 360, 0); }));

  // The truth itself is perfectly ordered, and its reverse perfectly not.
  std::vector<float> inverse(depth.size());
  std::vector<float> reversed(depth.size());
  for (std::size_t i = 0; i < depth.size(); ++i) {
    inverse[i] = 3.0F / depth[i] + 1.0F;  // Any scale and shift.
    reversed[i] = depth[i];
  }
  const OrderingResult good = check_ordering(scene, inverse);
  VERIFY(good.ordered && good.spearman > 0.999);
  const OrderingResult bad = check_ordering(scene, reversed);
  VERIFY(!bad.ordered && bad.spearman < -0.999);
  VERIFY(test::throws<std::invalid_argument>(
      [&scene] { (void)check_ordering(scene, std::vector<float>(10)); }));
}

void check_spearman() {
  const std::vector<float> a{1, 2, 3, 4, 5};
  VERIFY(std::fabs(spearman(a, std::vector<float>{10, 20, 30, 40, 50}) - 1.0) < 1e-12);
  VERIFY(std::fabs(spearman(a, std::vector<float>{5, 4, 3, 2, 1}) + 1.0) < 1e-12);
  // Monotonic, not linear: still 1.
  VERIFY(std::fabs(spearman(a, std::vector<float>{1, 8, 27, 64, 125}) - 1.0) < 1e-12);
  // Ties get their average rank: ranks 0, 1.5, 1.5, 3 against 0, 1, 2, 3.
  const double tied = spearman(std::vector<float>{1, 2, 2, 3}, std::vector<float>{1, 2, 3, 4});
  VERIFY(std::fabs(tied - 4.5 / std::sqrt(4.5 * 5.0)) < 1e-12);
  VERIFY(spearman(a, std::vector<float>{7, 7, 7, 7, 7}) == 0.0);
  VERIFY(spearman(a, std::vector<float>{1, 2}) == 0.0);
}

}  // namespace
}  // namespace holoscan::mono_depth

int main(int argc, char** argv) {
  namespace md = holoscan::mono_depth;
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " DIR\n";
    return 2;
  }
  md::g_directory = argv[1];
  std::filesystem::create_directories(md::g_directory);
  return md::test::run({
      {"image fits", md::check_fits},
      {"reference preprocessing", md::check_reference_preprocess},
      {"reference postprocessing", md::check_reference_postprocess},
      {"inverse and metric depth", md::check_depth_pairs},
      {"ordered float bits", md::check_ordered_bits},
      {"colormap", md::check_colormap},
      {"stub model", md::check_stub_model},
      {"synthetic scene", md::check_scene},
      {"spearman", md::check_spearman},
  });
}
