// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The kernels against their host references (reference.hpp), bit for bit:
// the preprocessing of random images in every encoding, letterboxed and
// stretched, wide and tall, padded rows, down- and upscaled; the
// postprocessing of model outputs at the input's size and at other sizes,
// into inverse and metric depth; the range of depth with non-finite values;
// and the colorization. Plus what the values mean, independently of the
// references: the normalization, the padding, the channel order, and the
// colormap's ends.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/mono_depth/colormap.hpp>
#include <holoscan/mono_depth/image_fit.hpp>
#include <holoscan/mono_depth/kernels.hpp>
#include <holoscan/mono_depth/reference.hpp>

#include "test_support.hpp"

namespace holoscan::mono_depth {
namespace {

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

// GPU memory for `count` values of T, freed at scope exit.
template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    check_cuda(cudaMalloc(&data_, count * sizeof(T) + 1), "cudaMalloc");
  }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() const { return static_cast<T*>(data_); }
  void upload(const std::vector<T>& values) {
    check_cuda(cudaMemcpy(data_, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice),
               "upload");
  }
  std::vector<T> download() const {
    std::vector<T> values(count_);
    check_cuda(cudaMemcpy(values.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost),
               "download");
    return values;
  }

 private:
  void* data_{};
  std::size_t count_{};
};

std::vector<std::uint8_t> random_bytes(std::size_t count, std::uint32_t seed) {
  std::mt19937 generator(seed);
  std::uniform_int_distribution<int> byte(0, 255);
  std::vector<std::uint8_t> bytes(count);
  for (auto& b : bytes) {
    b = static_cast<std::uint8_t>(byte(generator));
  }
  return bytes;
}

std::vector<float> random_floats(std::size_t count, float low, float high, std::uint32_t seed) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> value(low, high);
  std::vector<float> values(count);
  for (auto& v : values) {
    v = value(generator);
  }
  return values;
}

bool all_near(const std::vector<float>& values, float expected) {
  return std::all_of(values.begin(), values.end(),
                     [expected](float v) { return std::fabs(v - expected) <= 1e-6F; });
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// An image of width x height in `format`, rows `extra` bytes longer than its
// pixels, through the preprocessing at input_width x input_height.
void check_preprocess(PixelFormat format, int width, int height, int extra, int input_width,
                      int input_height, ResizeMode mode) {
  const std::size_t pitch = static_cast<std::size_t>(width) * channels(format) + extra;
  const std::vector<std::uint8_t> image =
      random_bytes(pitch * height, static_cast<std::uint32_t>(width * 7 + height));
  const ImageFit fit =
      make_fit(width, height, input_width, input_height, input_width, input_height, mode);
  const Normalization normalization;
  DeviceBuffer<std::uint8_t> device_image(image.size());
  device_image.upload(image);
  DeviceBuffer<float> input(3U * input_width * input_height);
  check_cuda(launch_preprocess(ImageView{device_image.get(), pitch, width, height, format}, fit,
                               normalization, input.get(), nullptr),
             "launch_preprocess");
  check_cuda(cudaDeviceSynchronize(), "preprocess");
  const std::vector<float> expected =
      reference_preprocess(image.data(), pitch, format, fit, normalization);
  VERIFY_MSG(same_bits(input.download(), expected),
             std::to_string(width) + "x" + std::to_string(height) + " into " +
                 std::to_string(input_width) + "x" + std::to_string(input_height) +
                 ": the kernel and the reference differ");
}

// What the values mean: a uniform image normalizes to its color's
// normalized values everywhere it covers, the padding is 0, and BGR and RGB
// order the channels.
void check_preprocess_meaning() {
  const int width = 200;
  const int height = 100;
  for (const PixelFormat format : {PixelFormat::kRgb8, PixelFormat::kBgra8}) {
    const int n = channels(format);
    std::vector<std::uint8_t> image(static_cast<std::size_t>(width) * height * n);
    const std::uint8_t rgb[3] = {255, 128, 0};
    for (std::size_t p = 0; p < image.size(); p += n) {
      image[p] = blue_first(format) ? rgb[2] : rgb[0];
      image[p + 1] = rgb[1];
      image[p + 2] = blue_first(format) ? rgb[0] : rgb[2];
      if (n == 4) image[p + 3] = 7;
    }
    const ImageFit fit = make_fit(width, height, 70, 70, 70, 70, ResizeMode::kLetterbox);
    VERIFY(fit.resized_width == 70 && fit.resized_height == 35 && fit.pad_top == 17);
    DeviceBuffer<std::uint8_t> device_image(image.size());
    device_image.upload(image);
    DeviceBuffer<float> input(3U * 70 * 70);
    const Normalization normalization;
    check_cuda(launch_preprocess(ImageView{device_image.get(), static_cast<std::size_t>(width) * n,
                                           width, height, format},
                                 fit, normalization, input.get(), nullptr),
               "launch_preprocess");
    const std::vector<float> values = input.download();
    for (int c = 0; c < 3; ++c) {
      const float expected =
          (static_cast<float>(rgb[c]) / 255.0F - normalization.mean[c]) / normalization.std[c];
      const std::size_t plane = static_cast<std::size_t>(c) * 70 * 70;
      VERIFY(std::fabs(values[plane + 35 * 70 + 35] - expected) < 1e-5F);
      VERIFY(values[plane + 0] == 0.0F && values[plane + 69 * 70 + 69] == 0.0F);
    }
  }
}

// A model output of output_width x output_height through the
// postprocessing onto a width x height image.
void check_postprocess(int width, int height, int input_width, int input_height, int output_width,
                       int output_height, ResizeMode mode, ModelOutput model_output,
                       const InverseDepthAffine* affine) {
  const ImageFit fit =
      make_fit(width, height, input_width, input_height, output_width, output_height, mode);
  const std::vector<float> output =
      random_floats(static_cast<std::size_t>(output_width) * output_height, 0.05F, 8.0F,
                    static_cast<std::uint32_t>(output_width + 3 * output_height));
  DeviceBuffer<float> device_output(output.size());
  device_output.upload(output);
  const std::size_t pixels = static_cast<std::size_t>(width) * height;
  DeviceBuffer<float> inverse(pixels);
  DeviceBuffer<float> metric(pixels);
  check_cuda(launch_postprocess(device_output.get(), fit, model_output, affine,
                                DepthOutputs{inverse.get(), metric.get()}, nullptr),
             "launch_postprocess");
  check_cuda(cudaDeviceSynchronize(), "postprocess");
  const ReferenceDepth expected = reference_postprocess(output, fit, model_output, affine);
  const std::string what = std::to_string(output_width) + "x" + std::to_string(output_height) +
                           " onto " + std::to_string(width) + "x" + std::to_string(height);
  VERIFY_MSG(same_bits(inverse.download(), expected.inverse), what + ": inverse depth differs");
  if (!expected.metric.empty()) {
    VERIFY_MSG(same_bits(metric.download(), expected.metric), what + ": metric depth differs");
  }
}

// What the values mean: a constant output gives that constant everywhere; a
// metric output gives its reciprocal as inverse depth; the affine gives
// 1 / (scale * d + shift), and 0 where that is not positive; and the image's
// corners take the valid output's corners, never the padding's.
void check_postprocess_meaning() {
  const int width = 64;
  const int height = 32;
  const ImageFit fit = make_fit(width, height, 28, 28, 28, 28, ResizeMode::kLetterbox);
  VERIFY(fit.valid_top == 7 && fit.valid_bottom == 20 && fit.valid_left == 0 &&
         fit.valid_right == 27);
  std::vector<float> output(28 * 28, -100.0F);  // The padding.
  for (int y = fit.valid_top; y <= fit.valid_bottom; ++y) {
    for (int x = 0; x < 28; ++x) {
      output[static_cast<std::size_t>(y) * 28 + x] = 2.0F;
    }
  }
  DeviceBuffer<float> device_output(output.size());
  device_output.upload(output);
  DeviceBuffer<float> inverse(width * height);
  DeviceBuffer<float> metric(width * height);
  const auto run = [&](ModelOutput kind, const InverseDepthAffine* affine) {
    check_cuda(launch_postprocess(device_output.get(), fit, kind, affine,
                                  DepthOutputs{inverse.get(), metric.get()}, nullptr),
               "launch_postprocess");
    return std::pair{inverse.download(), metric.download()};
  };
  const auto [relative, unused] = run(ModelOutput::kRelativeInverseDepth, nullptr);
  VERIFY(all_near(relative, 2.0F));
  const auto [inverse_of_metric, meters] = run(ModelOutput::kMetricDepth, nullptr);
  VERIFY(all_near(meters, 2.0F));
  VERIFY(all_near(inverse_of_metric, 0.5F));
  const InverseDepthAffine affine{.scale = 0.25F, .shift = 0.5F};
  const auto [unchanged, fitted] = run(ModelOutput::kRelativeInverseDepth, &affine);
  VERIFY(all_near(fitted, 1.0F));
  VERIFY(unchanged == relative);
  const InverseDepthAffine negative{.scale = -1.0F, .shift = 0.5F};
  const auto [ignored, invalid] = run(ModelOutput::kRelativeInverseDepth, &negative);
  VERIFY(std::all_of(invalid.begin(), invalid.end(), [](float v) { return v == 0.0F; }));
}

void check_range_and_colorize() {
  const int width = 300;
  const int height = 211;  // Not a multiple of any block.
  std::vector<float> depth =
      random_floats(static_cast<std::size_t>(width) * height, -3.0F, 40.0F, 17U);
  depth[5] = std::numeric_limits<float>::quiet_NaN();
  depth[77] = std::numeric_limits<float>::infinity();
  depth[1000] = -std::numeric_limits<float>::infinity();
  depth[60000] = -7.5F;  // The smallest finite value.
  depth[60001] = 55.0F;  // The largest.
  DeviceBuffer<float> device_depth(depth.size());
  device_depth.upload(depth);
  DeviceBuffer<std::uint32_t> range(2);
  check_cuda(launch_depth_range(device_depth.get(), depth.size(), range.get(), nullptr),
             "launch_depth_range");
  const std::vector<std::uint32_t> bits = range.download();
  const std::array<float, 2> expected = reference_range(depth);
  VERIFY(expected[0] == -7.5F && expected[1] == 55.0F);
  VERIFY(from_ordered_bits(bits[0]) == expected[0] && from_ordered_bits(bits[1]) == expected[1]);

  DeviceBuffer<std::uint8_t> rgba(depth.size() * 4U);
  check_cuda(launch_colorize(device_depth.get(), width, height, ColorRange{.range = range.get()},
                             rgba.get(), nullptr),
             "launch_colorize");
  VERIFY(rgba.download() == reference_colorize(depth, expected[0], expected[1]));
  check_cuda(launch_colorize(device_depth.get(), width, height,
                             ColorRange{.low = 0.0F, .high = 10.0F}, rgba.get(), nullptr),
             "launch_colorize");
  const std::vector<std::uint8_t> fixed = rgba.download();
  VERIFY(fixed == reference_colorize(depth, 0.0F, 10.0F));
  // The far end is Turbo's first entry, the near end its last.
  const auto entry = [&fixed](std::size_t i) {
    return std::array<std::uint8_t, 4>{fixed[4 * i], fixed[4 * i + 1], fixed[4 * i + 2],
                                       fixed[4 * i + 3]};
  };
  VERIFY((entry(60000) == std::array<std::uint8_t, 4>{kTurbo[0], kTurbo[1], kTurbo[2], 255}));
  VERIFY((entry(60001) == std::array<std::uint8_t, 4>{kTurbo[765], kTurbo[766], kTurbo[767], 255}));

  // Nothing finite: an empty range, the middle entry everywhere.
  std::vector<float> nothing(100, std::numeric_limits<float>::quiet_NaN());
  device_depth.upload(nothing);
  check_cuda(launch_depth_range(device_depth.get(), nothing.size(), range.get(), nullptr),
             "launch_depth_range");
  check_cuda(launch_colorize(device_depth.get(), 10, 10, ColorRange{.range = range.get()},
                             rgba.get(), nullptr),
             "launch_colorize");
  const std::vector<std::uint8_t> middle = rgba.download();
  VERIFY(middle[0] == kTurbo[384] && middle[1] == kTurbo[385] && middle[2] == kTurbo[386]);
}

void check_rejects() {
  const ImageFit fit = make_fit(10, 10, 14, 14, 14, 14, ResizeMode::kStretch);
  float* none = nullptr;
  VERIFY(launch_preprocess(ImageView{nullptr, 30, 10, 10, PixelFormat::kRgb8}, fit, Normalization{},
                           none, nullptr) == cudaErrorInvalidValue);
  VERIFY(launch_postprocess(nullptr, fit, ModelOutput::kRelativeInverseDepth, nullptr,
                            DepthOutputs{}, nullptr) == cudaErrorInvalidValue);
  VERIFY(launch_depth_range(nullptr, 10, nullptr, nullptr) == cudaErrorInvalidValue);
  VERIFY(launch_colorize(nullptr, 10, 10, ColorRange{}, nullptr, nullptr) == cudaErrorInvalidValue);
}

}  // namespace
}  // namespace holoscan::mono_depth

int main() {
  namespace md = holoscan::mono_depth;
  using md::ModelOutput;
  using md::PixelFormat;
  using md::ResizeMode;
  const md::InverseDepthAffine affine{.scale = 0.3F, .shift = 0.05F};
  return md::test::run({
      {"preprocess RGB8, wide, letterboxed",
       [] {
         md::check_preprocess(PixelFormat::kRgb8, 1280, 720, 0, 518, 518, ResizeMode::kLetterbox);
       }},
      {"preprocess BGR8, tall, padded rows",
       [] {
         md::check_preprocess(PixelFormat::kBgr8, 333, 1000, 5, 518, 518, ResizeMode::kLetterbox);
       }},
      {"preprocess RGBA8, stretched to a wide input",
       [] {
         md::check_preprocess(PixelFormat::kRgba8, 640, 480, 0, 518, 294, ResizeMode::kStretch);
       }},
      {"preprocess BGRA8, upscaled",
       [] {
         md::check_preprocess(PixelFormat::kBgra8, 61, 37, 3, 364, 364, ResizeMode::kLetterbox);
       }},
      {"preprocess values", md::check_preprocess_meaning},
      {"postprocess at the input size",
       [] {
         md::check_postprocess(1280, 720, 518, 518, 518, 518, ResizeMode::kLetterbox,
                               ModelOutput::kRelativeInverseDepth, nullptr);
       }},
      {"postprocess a half-size output, tall",
       [] {
         md::check_postprocess(333, 1000, 518, 518, 259, 259, ResizeMode::kLetterbox,
                               ModelOutput::kRelativeInverseDepth, nullptr);
       }},
      {"postprocess stretched, a larger output, metric model",
       [] {
         md::check_postprocess(640, 480, 392, 224, 784, 448, ResizeMode::kStretch,
                               ModelOutput::kMetricDepth, nullptr);
       }},
      {"postprocess with an affine to meters",
       [&affine] {
         md::check_postprocess(1000, 333, 518, 518, 518, 518, ResizeMode::kLetterbox,
                               ModelOutput::kRelativeInverseDepth, &affine);
       }},
      {"postprocess values", md::check_postprocess_meaning},
      {"range and colorize", md::check_range_and_colorize},
      {"rejects null buffers", md::check_rejects},
  });
}
