// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Sam3SegmentationOp with the real SAM 3, exported by tools/prepare_model.py,
// on the synthetic scene, whose true instances are known: each text prompt
// ("circle", "blue rectangle", "triangle") finds its shape, and an exemplar
// box around a shape (no text) finds that shape, each with a mask whose
// intersection over union with the truth is at least kMinIou. The engines are
// FP16, as by default. Skipped (exit status 77) unless the folder holds the
// export; CI runs it only when its nightly job provides the model.
//
// Usage: sam3_real_model_test DIR (with sam3_encoder.onnx, sam3_decoder.onnx
// and sam3_prompts.bin)

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/sam3/sam3_segmentation_op.hpp>
#include <holoscan/sam3/synthetic_scene.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "test_support.hpp"

namespace holoscan::sam3 {
namespace {

using holoscan::testing::TensorSnapshot;

constexpr double kMinIou = 0.9;
constexpr int kWidth = 1280;
constexpr int kHeight = 720;
constexpr std::uint32_t kFrames = 3;

std::filesystem::path g_directory;

holoscan::schema::ImageT scene_image(std::uint32_t frame, std::vector<SceneShape>& shapes) {
  std::vector<std::uint8_t> pixels(std::size_t{kWidth} * kHeight * 3U);
  shapes.resize(std::size_t{kWidth} * kHeight);
  render_shape_scene(make_shape_scene(kWidth, kHeight, frame), PixelFormat::kRgb8, pixels.data(),
                     std::size_t{kWidth} * 3U, shapes.data());
  const std::vector<std::int64_t> shape{kHeight, kWidth, 3};
  auto tensor = holoscan::make_tensor(shape, DLDataType{kDLUInt, 8, 1}, std::nullopt,
                                      holoscan::MemoryStorageType::kDevice,
                                      std::make_shared<holoscan::UnboundedAllocator>());
  if (!tensor || cudaMemcpy(*tensor->mutable_data(), pixels.data(), pixels.size(),
                            cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("cannot stage the image on the GPU");
  }
  holoscan::schema::ImageT image;
  image.width = kWidth;
  image.height = kHeight;
  image.encoding = holoscan::schema::ImageEncoding_RGB8;
  image.data = std::make_shared<holoscan::Tensor>(std::move(*tensor));
  return image;
}

template <typename T>
std::vector<T> values(const TensorSnapshot& snapshot) {
  std::vector<T> out(snapshot.bytes().size() / sizeof(T));
  std::memcpy(out.data(), snapshot.bytes().data(), out.size() * sizeof(T));
  return out;
}

// Runs `prompts` on kFrames frames, `step` frames of the scene apart, and
// checks that each prompt's best instance is `expected[prompt]`, with an IoU
// of at least kMinIou.
void check_prompts(const std::vector<Sam3Prompt>& prompts, const std::vector<SceneShape>& expected,
                   std::uint32_t step) {
  const Sam3SegmentationConfig config{.encoder = g_directory / "sam3_encoder.onnx",
                                      .decoder = g_directory / "sam3_decoder.onnx",
                                      .prompt_table = g_directory / "sam3_prompts.bin",
                                      .prompts = prompts,
                                      .max_width = kWidth,
                                      .max_height = kHeight};
  prepare_engines(config);
  auto h = holoscan::testing::create_operator_test<Sam3SegmentationOp>(config);
  std::vector<std::vector<SceneShape>> truths(kFrames);
  std::vector<holoscan::testing::TestSample<holoscan::schema::ImageT>> samples;
  for (std::uint32_t f = 0; f < kFrames; ++f) {
    samples.push_back({.data = scene_image(f * step, truths[f]),
                       .capture_coordinate = std::chrono::milliseconds(10 * (f + 1))});
  }
  h.template push_samples<holoscan::schema::ImageT>("image", std::move(samples));
  // The images come from GPU memory.
  h.input_source_tensor_spec(
      "image", holoscan::TensorOutputSpec{
                   .representation =
                       holoscan::TensorRepresentation{
                           .memory_kind = holoscan::MemoryKind::kCudaDevice,
                           .dtype = DLDataType{kDLUInt, 8, 1},
                           .rank = 3U,
                           .strides = holoscan::TensorStridePolicy::kContiguousOnly,
                       },
                   .bounds = holoscan::tensor_bounds(std::uint64_t{kWidth} * kHeight * 3U),
                   .storage = holoscan::TensorOutputStorage::kRuntimePool,
               });
  h.input_source_device("image", holoscan::DeviceId{0});
  for (const char* name : {"masks", "labels", "scores", "boxes", "prompt_ids", "count"}) {
    h.output_device(name, holoscan::DeviceId{0});
    h.template capture_snapshot<holoscan::Tensor>(name, kFrames);
  }
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto masks = result.template snapshots<holoscan::Tensor>("masks");
  const auto scores = result.template snapshots<holoscan::Tensor>("scores");
  const auto prompt_ids = result.template snapshots<holoscan::Tensor>("prompt_ids");
  const auto counts = result.template snapshots<holoscan::Tensor>("count");
  VERIFY(masks.size() == kFrames && counts.size() == kFrames);
  const std::size_t plane = std::size_t{kWidth} * kHeight;
  for (std::uint32_t f = 0; f < kFrames && f < masks.size(); ++f) {
    const std::vector<std::uint8_t> all = values<std::uint8_t>(masks[f]);
    const std::vector<float> frame_scores = values<float>(scores[f]);
    const std::vector<std::int32_t> ids = values<std::int32_t>(prompt_ids[f]);
    const int count = values<std::int32_t>(counts[f])[0];
    for (std::size_t p = 0; p < prompts.size(); ++p) {
      const std::string name = display_name(prompts[p]);
      // Instances are best first: the prompt's first is its best.
      const auto first = std::find(ids.begin(), ids.begin() + count, static_cast<std::int32_t>(p));
      if (first == ids.begin() + count) {
        VERIFY_MSG(false, "frame " + std::to_string(f) + ": nothing found for '" + name + "'");
        continue;
      }
      const auto k = static_cast<std::size_t>(first - ids.begin());
      const double iou = mask_iou(std::span(all.data() + k * plane, plane), truths[f], expected[p]);
      std::cout << "  frame " << f << " '" << name << "': score " << std::fixed
                << std::setprecision(3) << frame_scores[k] << ", IoU " << iou << ", " << count
                << " instances in all\n";
      VERIFY_MSG(iou >= kMinIou, "'" + name + "' IoU " + std::to_string(iou));
    }
  }
}

}  // namespace
}  // namespace holoscan::sam3

int main(int argc, char** argv) {
  namespace s3 = holoscan::sam3;
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " DIR\n";
    return 2;
  }
  s3::g_directory = argv[1];
  for (const char* file : {"sam3_encoder.onnx", "sam3_decoder.onnx", "sam3_prompts.bin"}) {
    if (!std::filesystem::is_regular_file(s3::g_directory / file)) {
      std::cout << "SKIPPED: no " << (s3::g_directory / file).string()
                << "; export SAM 3 with tools/prepare_model.py\n";
      return 77;
    }
  }
  using s3::SceneShape;
  const s3::ShapeScene scene = s3::make_shape_scene(s3::kWidth, s3::kHeight, 0);
  return s3::test::run({
      {"text prompts",
       [] {
         s3::check_prompts({{.text = "circle"}, {.text = "blue rectangle"}, {.text = "triangle"}},
                           {SceneShape::kCircle, SceneShape::kRectangle, SceneShape::kTriangle},
                           40U);
       }},
      // The box is the circle's in frame 0, so the scene stands still.
      {"an exemplar box",
       [&scene] {
         s3::check_prompts(
             {{.boxes = {s3::PromptBox{.xyxy = s3::shape_box(scene, SceneShape::kCircle)}}}},
             {SceneShape::kCircle}, 0U);
       }},
  });
}
