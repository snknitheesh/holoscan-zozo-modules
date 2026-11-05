// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The kernels against their host references, bit for bit, without TensorRT:
// the preprocessing of every pixel format, downscaled and upscaled; the
// selection and finalization of instances from made-up decoder outputs, with
// tied scores, NaN scores, more instances than fit, prompts with none,
// overlapping masks, and with and without the masks output; and the overlay.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/sam3/kernels.hpp>
#include <holoscan/sam3/reference.hpp>
#include <holoscan/sam3/synthetic_scene.hpp>

#include "test_support.hpp"

namespace holoscan::sam3 {
namespace {

void cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

// GPU memory holding a copy of `values`.
template <typename T>
class Device {
 public:
  explicit Device(std::size_t count) : count_(count) {
    cuda(cudaMalloc(&data_, std::max<std::size_t>(count, 1) * sizeof(T)));
    cuda(cudaMemset(data_, 0xA5, std::max<std::size_t>(count, 1) * sizeof(T)));
  }
  explicit Device(const std::vector<T>& values) : Device(values.size()) {
    cuda(cudaMemcpy(data_, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  ~Device() { cudaFree(data_); }
  T* get() const { return static_cast<T*>(data_); }
  std::vector<T> read() const {
    std::vector<T> out(count_);
    cuda(cudaMemcpy(out.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
    return out;
  }

 private:
  void* data_{};
  std::size_t count_;
};

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

void check_preprocess(PixelFormat format, int width, int height, int input) {
  const std::string what =
      std::to_string(width) + "x" + std::to_string(height) + " to " + std::to_string(input);
  // Rows padded by a few bytes, to exercise the pitch.
  const std::size_t pitch = static_cast<std::size_t>(width) * channels(format) + 5;
  std::vector<std::uint8_t> pixels(pitch * height, 0);
  render_shape_scene(make_shape_scene(width, height, 7), format, pixels.data(), pitch);
  Device<std::uint8_t> image(pixels);
  Device<float> planes(3 * static_cast<std::size_t>(input) * input);
  const Normalization normalization{};
  cuda(launch_preprocess(
      ImageView{
          .data = image.get(), .pitch = pitch, .width = width, .height = height, .format = format},
      input, input, normalization, planes.get(), nullptr));
  cuda(cudaDeviceSynchronize());
  VERIFY_MSG(same_bits(planes.read(), reference_preprocess(pixels.data(), pitch, format, width,
                                                           height, input, input, normalization)),
             what + ": the preprocessing differs from the reference");
}

// Made-up decoder outputs for `prompts` prompts: random scores with ties and
// NaNs, boxes partly outside the image, and smooth mask logits.
std::vector<DecoderOutput> decoder_outputs(int prompts, int queries, int mask_width,
                                           int mask_height, std::uint32_t seed) {
  std::mt19937 random(seed);
  std::uniform_real_distribution<float> unit(0.0F, 1.0F);
  std::vector<DecoderOutput> outputs;
  for (int p = 0; p < prompts; ++p) {
    DecoderOutput out;
    out.queries = queries;
    out.mask_width = mask_width;
    out.mask_height = mask_height;
    for (int q = 0; q < queries; ++q) {
      float score = unit(random);
      if (q % 7 == 3) score = 0.75F;  // Ties, within and across prompts.
      if (q % 11 == 5) score = std::numeric_limits<float>::quiet_NaN();
      out.scores.push_back(p == prompts - 1 && prompts > 1 ? 0.1F * score : score);
      const float x = unit(random) * 1.2F - 0.1F;
      const float y = unit(random) * 1.2F - 0.1F;
      for (const float value : {x, y, x + 0.3F, y + 0.2F}) {
        out.boxes.push_back(value);
      }
      const float cx = unit(random) * mask_width;
      const float cy = unit(random) * mask_height;
      const float r = 2.0F + unit(random) * mask_width * 0.4F;
      for (int y0 = 0; y0 < mask_height; ++y0) {
        for (int x0 = 0; x0 < mask_width; ++x0) {
          const float d = std::hypot(static_cast<float>(x0) - cx, static_cast<float>(y0) - cy);
          out.masks.push_back(r - d);
        }
      }
    }
    outputs.push_back(std::move(out));
  }
  return outputs;
}

void check_instances(int prompts, int queries, int capacity, int mask_size, int width, int height,
                     bool with_masks, float threshold) {
  const std::string what = std::to_string(prompts) + " prompts, " + std::to_string(queries) +
                           " queries, capacity " + std::to_string(capacity) + ", " +
                           std::to_string(width) + "x" + std::to_string(height);
  const std::vector<DecoderOutput> outputs =
      decoder_outputs(prompts, queries, mask_size, mask_size, 17U + prompts + capacity);
  const float mask_threshold = logit_of(0.6F);
  const ReferenceInstances expected =
      reference_instances(outputs, capacity, threshold, mask_threshold, width, height);

  const auto slots = static_cast<std::size_t>(prompts) * capacity;
  const std::size_t cells = static_cast<std::size_t>(mask_size) * mask_size;
  const std::size_t plane = static_cast<std::size_t>(width) * height;
  Device<float> scores(slots);
  Device<float> boxes(4 * slots);
  Device<std::int32_t> queries_of(slots);
  Device<float> logits(slots * cells);
  Device<std::int32_t> counts(static_cast<std::size_t>(prompts));
  Device<std::int32_t> passed(static_cast<std::size_t>(prompts));
  const Candidates candidates{.scores = scores.get(),
                              .boxes = boxes.get(),
                              .queries = queries_of.get(),
                              .logits = logits.get(),
                              .counts = counts.get(),
                              .passed = passed.get()};
  for (int p = 0; p < prompts; ++p) {
    const DecoderOutput& out = outputs[static_cast<std::size_t>(p)];
    Device<float> s(out.scores);
    Device<float> b(out.boxes);
    Device<float> m(out.masks);
    cuda(launch_select(s.get(), b.get(), m.get(),
                       SelectParams{.prompt = p,
                                    .queries = queries,
                                    .capacity = capacity,
                                    .mask_width = mask_size,
                                    .mask_height = mask_size,
                                    .score_threshold = threshold},
                       candidates, nullptr));
    cuda(cudaDeviceSynchronize());
  }
  Device<std::uint8_t> masks(static_cast<std::size_t>(capacity) * plane);
  Device<std::uint16_t> labels(plane);
  Device<float> out_scores(static_cast<std::size_t>(capacity));
  Device<float> out_boxes(4 * static_cast<std::size_t>(capacity));
  Device<std::int32_t> prompt_ids(static_cast<std::size_t>(capacity));
  Device<std::int32_t> count(1);
  Device<std::int32_t> order(static_cast<std::size_t>(capacity));
  cuda(launch_finalize(candidates,
                       FinalizeParams{.prompts = prompts,
                                      .capacity = capacity,
                                      .mask_width = mask_size,
                                      .mask_height = mask_size,
                                      .width = width,
                                      .height = height,
                                      .mask_logit_threshold = mask_threshold},
                       Instances{.masks = with_masks ? masks.get() : nullptr,
                                 .labels = labels.get(),
                                 .scores = out_scores.get(),
                                 .boxes = out_boxes.get(),
                                 .prompt_ids = prompt_ids.get(),
                                 .count = count.get(),
                                 .order = order.get()},
                       nullptr));
  cuda(cudaDeviceSynchronize());
  VERIFY_MSG(count.read()[0] == expected.count, what + ": count " +
                                                    std::to_string(count.read()[0]) +
                                                    ", expected " + std::to_string(expected.count));
  VERIFY_MSG(same_bits(out_scores.read(), expected.scores), what + ": scores differ");
  VERIFY_MSG(same_bits(out_boxes.read(), expected.boxes), what + ": boxes differ");
  VERIFY_MSG(prompt_ids.read() == expected.prompt_ids, what + ": prompt ids differ");
  VERIFY_MSG(labels.read() == expected.labels, what + ": the label image differs");
  if (with_masks) {
    VERIFY_MSG(masks.read() == expected.masks, what + ": the masks differ");
  }
  std::vector<std::int32_t> passed_counts = passed.read();
  VERIFY_MSG(std::vector<int>(passed_counts.begin(), passed_counts.end()) == expected.passed,
             what + ": passed counts differ");
  // The case is meaningful where it should be: some instances pass.
  VERIFY_MSG(threshold > 0.9F || expected.count > 0, what + ": no instances to test");
}

void check_overlay(int width, int height) {
  // A label image from the synthetic scene, three instances and background.
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * 3);
  std::vector<SceneShape> shapes(static_cast<std::size_t>(width) * height);
  render_shape_scene(make_shape_scene(width, height, 3), PixelFormat::kRgb8, pixels.data(),
                     static_cast<std::size_t>(width) * 3, shapes.data());
  std::vector<std::uint16_t> labels(shapes.size());
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    labels[i] = static_cast<std::uint16_t>(shapes[i]) * 5U;  // Labels 0, 5, 10 and 15.
  }
  Device<std::uint16_t> device_labels(labels);
  Device<std::uint8_t> rgba(labels.size() * 4);
  cuda(launch_overlay(device_labels.get(), width, height, 99, rgba.get(), nullptr));
  cuda(cudaDeviceSynchronize());
  const std::vector<std::uint8_t> expected = reference_overlay(labels, width, height, 99);
  VERIFY_MSG(rgba.read() == expected, "the overlay differs");
  // Spot checks: background is transparent, insides translucent, outlines
  // opaque.
  bool transparent = false;
  bool inside = false;
  bool outline = false;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    transparent |= labels[i] == 0 && expected[4 * i + 3] == 0;
    inside |= labels[i] != 0 && expected[4 * i + 3] == 99;
    outline |= labels[i] != 0 && expected[4 * i + 3] == 255;
  }
  VERIFY(transparent && inside && outline);
}

}  // namespace
}  // namespace holoscan::sam3

int main() {
  namespace s3 = holoscan::sam3;
  using s3::PixelFormat;
  return s3::test::run({
      {"preprocess RGB8, downscaled",
       [] { s3::check_preprocess(PixelFormat::kRgb8, 1280, 720, 1008); }},
      {"preprocess BGR8, upscaled",
       [] { s3::check_preprocess(PixelFormat::kBgr8, 333, 251, 1008); }},
      {"preprocess RGBA8, tall", [] { s3::check_preprocess(PixelFormat::kRgba8, 240, 640, 64); }},
      {"preprocess BGRA8, tiny", [] { s3::check_preprocess(PixelFormat::kBgra8, 32, 32, 112); }},
      {"instances, SAM 3 sizes",
       [] { s3::check_instances(3, 200, 16, 288, 640, 480, true, 0.5F); }},
      {"instances, more than fit, without masks",
       [] { s3::check_instances(2, 200, 4, 288, 1280, 720, false, 0.2F); }},
      {"instances, many prompts, few queries",
       [] { s3::check_instances(16, 4, 3, 16, 333, 501, true, 0.3F); }},
      {"instances, none pass", [] { s3::check_instances(2, 50, 8, 32, 64, 48, true, 0.99F); }},
      {"overlay", [] { s3::check_overlay(640, 480); }},
      {"overlay, odd size", [] { s3::check_overlay(37, 91); }},
  });
}
