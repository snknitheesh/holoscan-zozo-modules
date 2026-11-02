// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Yolo26SegmentationOp through the SDK's operator test harness.
//
// With the stub segmentation model (no download): synthetic frames in every
// encoding, from GPU, host and pinned memory, wide, tall, odd and upscaled,
// so through the letterbox both ways; the detections checked against the
// scene as Yolo26DetectionOp's are, each mask against its rectangle (an IoU
// of at least 0.9, and every pixel right that is more than the prototypes'
// resolution from an edge), the masks cropped to their boxes, the label
// image against the masks, empty slots past the count, and the capture time
// and frame id on all six outputs; the score and mask thresholds and the
// capacity; an image size that changes; and the images, models and settings
// it must reject. And, when it is built, Yolo26SegmentationOverlayOp: masks
// filled in their class's color, boxes and labels over them.
//
// With --real MODEL, the real YOLO26 segmentation model instead, on rendered
// stop signs: the sign must be found, and its mask must overlap the octagon
// by an IoU of at least 0.9. Exits 77, a skip, when MODEL does not exist.
//
// Usage: yolo26_segmentation_op_test DIR [--real MODEL]

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>
#include <holoscan/yolo26/letterbox.hpp>
#include <holoscan/yolo26/stub_model.hpp>
#include <holoscan/yolo26/synthetic_scene.hpp>
#include <holoscan/yolo26/yolo26_detection_op.hpp>
#include <holoscan/yolo26/yolo26_segmentation_op.hpp>

#include "operator_support.hpp"
#include "stop_sign.hpp"
#include "test_support.hpp"
#if HOLOSCAN_YOLO26_WITH_OVERLAY
#include <holoscan/visualization/holoviz_op.hpp>
#include <holoscan/yolo26/yolo26_segmentation_overlay_op.hpp>
#endif

namespace holoscan::yolo26 {
namespace {

using holoscan::testing::TensorSnapshot;
using test::capture_time;
using test::check_truth;
using test::Frame;
using test::frames_of;
using test::image_tensor;
using test::Memory;
using test::scene_image;
using test::stage_on_gpu;
using test::values;

std::filesystem::path g_directory;

constexpr int kCapacity = 6;
const char* const kOutputs[] = {"boxes", "scores", "class_ids", "count", "masks", "label_image"};

Yolo26SegmentationConfig stub_config(int input_size = 640) {
  return Yolo26SegmentationConfig{.model = write_stub_segmentation_model(g_directory, input_size),
                                  .input_size = input_size,
                                  .max_detections = kCapacity,
                                  .num_classes = kStubClasses};
}

// Captures every output of up to `frames` frames. The capture's capacity
// sizes the output pools, so it stays near what a test sends: the masks are
// large.
void declare_outputs(holoscan::testing::OperatorTestHarness<Yolo26SegmentationOp>& h,
                     std::size_t frames) {
  for (const char* name : kOutputs) {
    h.output_device(name, holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::Tensor>(name, frames, std::size_t{256} << 20U);
  }
}

// One frame's masks and label image, after checking their shapes, the
// capture time and frame id they carry, the masks against the boxes and the
// count, and the label image against the masks.
struct FrameMasks {
  int width{};
  int height{};
  std::vector<std::uint8_t> masks;
  std::vector<std::int32_t> labels;

  [[nodiscard]] const std::uint8_t* mask(std::size_t k) const {
    return masks.data() + k * static_cast<std::size_t>(width) * height;
  }
};

std::vector<FrameMasks> masks_of(const holoscan::testing::TestResult& result,
                                 const std::vector<Frame>& frames, int capacity,
                                 const std::vector<std::pair<int, int>>& sizes) {
  const auto masks = result.snapshots<holoscan::Tensor>("masks");
  const auto labels = result.snapshots<holoscan::Tensor>("label_image");
  VERIFY(masks.size() == frames.size() && labels.size() == frames.size() &&
         sizes.size() == frames.size());
  std::vector<FrameMasks> out;
  for (std::size_t f = 0; f < frames.size() && f < masks.size() && f < labels.size(); ++f) {
    const auto [width, height] = sizes[f];
    for (const TensorSnapshot* snapshot : {&masks[f], &labels[f]}) {
      VERIFY(snapshot->sample_metadata().capture_timestamp_ns ==
                 frames[f].metadata.capture_timestamp_ns &&
             snapshot->sample_metadata().frame_id == frames[f].metadata.frame_id);
    }
    FrameMasks m{.width = width,
                 .height = height,
                 .masks = values<std::uint8_t>(masks[f], {capacity, height, width}),
                 .labels = values<std::int32_t>(labels[f], {height, width})};
    VERIFY(masks[f].dtype().code == kDLUInt && labels[f].dtype().code == kDLInt);
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    if (m.masks.size() != pixels * capacity || m.labels.size() != pixels) {
      continue;
    }
    const auto& detections = frames[f].detections;
    std::size_t outside_box = 0;
    std::size_t past_count = 0;
    std::size_t wrong_labels = 0;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const std::size_t p = static_cast<std::size_t>(y) * width + x;
        std::int32_t first = 0;
        for (std::size_t k = 0; k < static_cast<std::size_t>(capacity); ++k) {
          const std::uint8_t v = m.mask(k)[p];
          if (k >= detections.size()) {
            past_count += v != 0;
            continue;
          }
          const BoxDetection& d = detections[k];
          const bool in_box = x >= d.x1 && x < d.x2 && y >= d.y1 && y < d.y2;
          outside_box += v != 0 && !in_box;
          VERIFY(v <= 1);
          if (v != 0 && first == 0) {
            first = static_cast<std::int32_t>(k) + 1;
          }
        }
        wrong_labels += m.labels[p] != first;
      }
    }
    VERIFY_MSG(outside_box == 0, std::to_string(outside_box) + " mask pixels outside their box");
    VERIFY_MSG(past_count == 0, std::to_string(past_count) + " mask pixels past the count");
    VERIFY_MSG(wrong_labels == 0, std::to_string(wrong_labels) + " labels disagree with the masks");
    out.push_back(std::move(m));
  }
  return out;
}

// Intersection over union of a mask and a silhouette of the same size.
float mask_iou(const std::uint8_t* mask, const std::vector<std::uint8_t>& silhouette) {
  std::size_t both = 0;
  std::size_t either = 0;
  for (std::size_t i = 0; i < silhouette.size(); ++i) {
    both += mask[i] != 0 && silhouette[i] != 0;
    either += mask[i] != 0 || silhouette[i] != 0;
  }
  return either == 0 ? 0.0F : static_cast<float>(both) / static_cast<float>(either);
}

// Each detection's mask is its class's rectangle: an IoU of at least 0.9, or
// 0.8 for a rectangle under 24 pixels on a side, where the letterbox's blur
// of its edges weighs more; and right at every pixel farther than the prototypes'
// resolution (4 input pixels) from the rectangle's edges.
void check_scene_masks(const Frame& frame, const FrameMasks& m, const SyntheticScene& scene,
                       int input_size) {
  const float margin = static_cast<float>(kStubProtoStride) /
                           make_letterbox(scene.width, scene.height, input_size).scale +
                       1.0F;
  for (std::size_t k = 0; k < frame.detections.size(); ++k) {
    const auto c = static_cast<std::size_t>(frame.detections[k].class_id);
    if (c >= static_cast<std::size_t>(kStubClasses)) continue;
    const SceneRect& rect = scene.rects[c];
    std::vector<std::uint8_t> silhouette(static_cast<std::size_t>(m.width) * m.height, 0);
    std::size_t wrong = 0;
    for (int y = 0; y < m.height; ++y) {
      for (int x = 0; x < m.width; ++x) {
        const bool inside = x >= rect.x0 && x < rect.x1 && y >= rect.y0 && y < rect.y1;
        silhouette[static_cast<std::size_t>(y) * m.width + x] = inside ? 1 : 0;
        const float cx = x + 0.5F;
        const float cy = y + 0.5F;
        const float depth = std::min({cx - rect.x0, rect.x1 - cx, cy - rect.y0, rect.y1 - cy});
        const std::uint8_t v = m.mask(k)[static_cast<std::size_t>(y) * m.width + x];
        wrong += (depth > margin && v != 1) || (depth < -margin && v != 0);
      }
    }
    const float overlap = mask_iou(m.mask(k), silhouette);
    const bool tiny = std::min(rect.x1 - rect.x0, rect.y1 - rect.y0) < 24;
    VERIFY_MSG(overlap >= (tiny ? 0.8F : 0.9F),
               std::to_string(scene.width) + "x" + std::to_string(scene.height) + " class " +
                   std::to_string(c) + ": mask IoU " + std::to_string(overlap));
    VERIFY_MSG(wrong == 0, std::to_string(scene.width) + "x" + std::to_string(scene.height) +
                               " class " + std::to_string(c) + ": " + std::to_string(wrong) +
                               " pixels wrong away from the edges");
  }
}

// Frames of a width x height scene in `format` from `memory`.
void check_segmentation(PixelFormat format, Memory memory, int width, int height, int input_size) {
  Yolo26SegmentationConfig config = stub_config(input_size);
  config.max_image_width = std::max(width, 64);
  config.max_image_height = std::max(height, 64);
  auto h = holoscan::testing::create_operator_test<Yolo26SegmentationOp>(config);
  constexpr std::uint32_t kFrames = 3;
  for (std::uint32_t f = 0; f < kFrames; ++f) {
    holoscan::testing::TestSample<holoscan::schema::ImageT> sample{
        .data = scene_image(make_scene(width, height, f * 9U), format, memory),
        .capture_coordinate = capture_time(f)};
    sample.metadata.frame_id = 1U + f;
    h.push_samples<holoscan::schema::ImageT>("image", {std::move(sample)});
  }
  if (memory == Memory::kDevice) {
    stage_on_gpu(h, static_cast<std::uint64_t>(width) * height * channels(format));
  }
  declare_outputs(h, kFrames);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto frames = frames_of(result, kCapacity);
  VERIFY(frames.size() == kFrames);
  const auto masks = masks_of(result, frames, kCapacity,
                              std::vector<std::pair<int, int>>(frames.size(), {width, height}));
  for (std::uint32_t f = 0; f < frames.size() && f < masks.size(); ++f) {
    const SyntheticScene scene = make_scene(width, height, f * 9U);
    check_truth(frames[f], scene, input_size);
    check_scene_masks(frames[f], masks[f], scene, input_size);
    VERIFY(frames[f].metadata.capture_timestamp_ns == capture_time(f).count());
    VERIFY(frames[f].metadata.frame_id == 1U + f);
  }
}

// The masks follow each image's size, up to the largest allowed.
void check_size_change() {
  Yolo26SegmentationConfig config = stub_config();
  config.max_image_width = 1280;
  config.max_image_height = 720;
  auto h = holoscan::testing::create_operator_test<Yolo26SegmentationOp>(config);
  const std::vector<SyntheticScene> scenes{make_scene(1280, 720, 1), make_scene(333, 500, 2),
                                           make_scene(64, 48, 4), make_scene(720, 640, 5)};
  std::vector<std::pair<int, int>> sizes;
  for (const SyntheticScene& scene : scenes) {
    h.push("image", scene_image(scene, PixelFormat::kRgba8, Memory::kDevice));
    sizes.emplace_back(scene.width, scene.height);
  }
  stage_on_gpu(h, 1280ULL * 720ULL * 4ULL);
  declare_outputs(h, scenes.size());
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto frames = frames_of(result, kCapacity);
  VERIFY(frames.size() == scenes.size());
  const auto masks = masks_of(result, frames, kCapacity, sizes);
  for (std::size_t i = 0; i < frames.size() && i < masks.size(); ++i) {
    check_truth(frames[i], scenes[i], 640);
    check_scene_masks(frames[i], masks[i], scenes[i], 640);
  }
}

// One frame of `scene` with `config`: its detections and masks.
std::optional<std::pair<Frame, FrameMasks>> run_one(const Yolo26SegmentationConfig& config,
                                                    const SyntheticScene& scene) {
  auto h = holoscan::testing::create_operator_test<Yolo26SegmentationOp>(config);
  h.push("image", scene_image(scene, PixelFormat::kRgb8));
  declare_outputs(h, 1);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return std::nullopt;
  const auto frames = frames_of(result, config.max_detections);
  const auto masks = masks_of(result, frames, config.max_detections, {{scene.width, scene.height}});
  if (frames.size() != 1 || masks.size() != 1) return std::nullopt;
  return std::pair{frames[0], masks[0]};
}

std::size_t pixels_set(const FrameMasks& m, std::size_t k) {
  return static_cast<std::size_t>(
      std::count(m.mask(k), m.mask(k) + static_cast<std::size_t>(m.width) * m.height, 1));
}

// The score threshold and the capacity limit the detections, and so the
// masks; a mask threshold above every mask value empties the masks but keeps
// the detections.
void check_thresholds_and_capacity() {
  const SyntheticScene scene = make_scene(320, 240, 0);
  Yolo26SegmentationConfig config = stub_config();
  config.max_image_width = 320;
  config.max_image_height = 240;

  config.score_threshold = 0.85F;
  if (const auto out = run_one(config, scene)) {
    VERIFY(out->first.detections.size() == 1 && out->first.detections[0].class_id == 0);
    VERIFY(pixels_set(out->second, 0) > 0);
  }
  config.score_threshold = 0.25F;
  config.max_detections = 1;
  if (const auto out = run_one(config, scene)) {
    VERIFY(out->first.detections.size() == 1 && pixels_set(out->second, 0) > 0);
  }
  // Every row, the low-score ones included, whose masks are empty.
  config.score_threshold = 0.0F;
  config.max_detections = kStubRows;
  if (const auto out = run_one(config, scene)) {
    VERIFY(out->first.detections.size() == static_cast<std::size_t>(kStubRows));
    VERIFY(pixels_set(out->second, 0) > 0 && pixels_set(out->second, 1) > 0);
    std::size_t filler = 0;
    for (std::size_t k = 2; k < static_cast<std::size_t>(kStubRows); ++k) {
      filler += pixels_set(out->second, k);
    }
    VERIFY_MSG(filler == 0, std::to_string(filler) + " pixels in the low-score rows' masks");
  }
  // The stub's mask values stay below 0.3, a probability of 0.57.
  config = stub_config();
  config.max_image_width = 320;
  config.max_image_height = 240;
  config.mask_threshold = 0.7F;
  if (const auto out = run_one(config, scene)) {
    VERIFY(out->first.detections.size() == 2);
    VERIFY(pixels_set(out->second, 0) == 0 && pixels_set(out->second, 1) == 0);
    VERIFY(std::all_of(out->second.labels.begin(), out->second.labels.end(),
                       [](std::int32_t v) { return v == 0; }));
  }
}

// A run with `image` fails, saying `message`.
template <typename Op, typename Config>
void check_run_fails(Config config, holoscan::schema::ImageT image, const std::string& message) {
  auto h = holoscan::testing::create_operator_test<Op>(std::move(config));
  h.push("image", std::move(image));
  if constexpr (std::is_same_v<Op, Yolo26SegmentationOp>) {
    declare_outputs(h, 1);
  } else {
    for (const char* name : {"boxes", "scores", "class_ids", "count"}) {
      h.output_device(name, holoscan::DeviceId{0});
      h.template capture_snapshot<holoscan::Tensor>(name, 1);
    }
  }
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), "expected a failure: " + message);
  VERIFY_MSG(result.describe_failure().find(message) != std::string::npos,
             "expected '" + message + "' in: " + result.describe_failure());
}

void check_rejects() {
  const SyntheticScene scene = make_scene(320, 240, 0);
  Yolo26SegmentationConfig small = stub_config();
  small.max_image_width = 320;
  small.max_image_height = 200;
  check_run_fails<Yolo26SegmentationOp>(small, scene_image(scene, PixelFormat::kRgb8),
                                        "larger than max_image_width x max_image_height");
  holoscan::schema::ImageT mono = scene_image(scene, PixelFormat::kRgb8);
  mono.encoding = holoscan::schema::ImageEncoding_MONO8;
  check_run_fails<Yolo26SegmentationOp>(stub_config(), mono, "unsupported image encoding MONO8");
  // A detection model, which has no masks, and a segmentation model where a
  // detection model belongs.
  Yolo26SegmentationConfig detection_model = stub_config();
  detection_model.model = write_stub_model(g_directory, 640);
  check_run_fails<Yolo26SegmentationOp>(detection_model, scene_image(scene, PixelFormat::kRgb8),
                                        "export a YOLO26 segmentation model");
  check_run_fails<Yolo26DetectionOp>(
      Yolo26DetectionConfig{.model = write_stub_segmentation_model(g_directory, 640),
                            .num_classes = kStubClasses},
      scene_image(scene, PixelFormat::kRgb8), "Yolo26DetectionOp: ");
  Yolo26SegmentationConfig other_size = stub_config(640);
  other_size.model = write_stub_segmentation_model(g_directory, 320);
  check_run_fails<Yolo26SegmentationOp>(other_size, scene_image(scene, PixelFormat::kRgb8),
                                        "export it for this input size");
}

void check_rejects_settings() {
  const auto rejects = [](auto edit) {
    Yolo26SegmentationConfig config{.model = "model.onnx"};
    edit(config);
    return test::throws<std::invalid_argument>([&] { Yolo26SegmentationOp op(config); });
  };
  VERIFY(rejects([](auto& c) { c.model.clear(); }));
  VERIFY(rejects([](auto& c) { c.input_size = 100; }));
  VERIFY(rejects([](auto& c) { c.score_threshold = 1.5F; }));
  VERIFY(rejects([](auto& c) { c.score_threshold = std::numeric_limits<float>::quiet_NaN(); }));
  VERIFY(rejects([](auto& c) { c.max_detections = 0; }));
  VERIFY(rejects([](auto& c) { c.max_detections = kMaxSegmentationDetections + 1; }));
  VERIFY(rejects([](auto& c) { c.num_classes = 0; }));
  VERIFY(rejects([](auto& c) { c.mask_threshold = 0.0F; }));
  VERIFY(rejects([](auto& c) { c.mask_threshold = 1.0F; }));
  VERIFY(rejects([](auto& c) { c.mask_threshold = std::numeric_limits<float>::quiet_NaN(); }));
  VERIFY(rejects([](auto& c) { c.max_image_width = 0; }));
  VERIFY(rejects([](auto& c) { c.max_image_height = kMaxImageSide + 1; }));
  VERIFY(rejects([](auto& c) { c.build_workspace_bytes = 1024; }));
  VERIFY(!rejects([](auto&) {}));
}

// The real model on rendered stop signs, in RGB8 from host memory.
int run_real(const std::filesystem::path& model) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(model, error)) {
    std::cout << "skipped: no model at " << model.string()
              << "; prepare it with tools/prepare_model.py --task segment\n";
    return 77;
  }
  const std::vector<test::StopSign> signs{{640, 480, 320.0F, 220.0F, 120.0F},
                                          {1280, 720, 500.0F, 330.0F, 200.0F},
                                          {640, 480, 200.0F, 200.0F, 70.0F},
                                          {480, 640, 240.0F, 300.0F, 150.0F}};
  return test::run(
      {{"real model: rendered stop signs", [&] {
          Yolo26SegmentationConfig config{.model = model,
                                          .max_detections = 20,
                                          .max_image_width = 1280,
                                          .max_image_height = 1280};
          auto h = holoscan::testing::create_operator_test<Yolo26SegmentationOp>(config);
          std::vector<std::pair<int, int>> sizes;
          for (const test::StopSign& sign : signs) {
            holoscan::schema::ImageT image;
            image.width = sign.width;
            image.height = sign.height;
            image.encoding = holoscan::schema::ImageEncoding_RGB8;
            image.data = image_tensor(test::render_stop_sign(sign), {sign.height, sign.width, 3},
                                      Memory::kHost);
            h.push("image", std::move(image));
            sizes.emplace_back(sign.width, sign.height);
          }
          declare_outputs(h, signs.size());
          const auto result = h.run();
          VERIFY_MSG(result.ok(), result.describe_failure());
          if (!result.ok()) return;
          const auto frames = frames_of(result, config.max_detections);
          const auto masks = masks_of(result, frames, config.max_detections, sizes);
          VERIFY(frames.size() == signs.size() && masks.size() == signs.size());
          for (std::size_t i = 0; i < frames.size() && i < masks.size(); ++i) {
            const auto& d = frames[i].detections;
            const auto sign = std::find_if(d.begin(), d.end(), [](const auto& x) {
              return x.class_id == test::kStopSignClass;
            });
            const std::string what = std::to_string(signs[i].width) + "x" +
                                     std::to_string(signs[i].height) + " r" +
                                     std::to_string(signs[i].radius);
            VERIFY_MSG(sign != d.end() && sign->score >= 0.5F, what + ": no stop sign found");
            if (sign == d.end()) continue;
            const auto k = static_cast<std::size_t>(sign - d.begin());
            const float overlap = mask_iou(masks[i].mask(k), test::stop_sign_silhouette(signs[i]));
            std::cout << what << ": stop sign " << sign->score << ", mask IoU " << overlap << '\n';
            VERIFY_MSG(overlap >= 0.9F, what + ": mask IoU " + std::to_string(overlap));
          }
        }}});
}

#if HOLOSCAN_YOLO26_WITH_OVERLAY
// `values` as a host tensor of `shape` and `dtype`.
template <typename T>
holoscan::Tensor host_tensor(const std::vector<T>& values, std::vector<std::int64_t> shape,
                             DLDataType dtype) {
  std::vector<std::uint8_t> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return *image_tensor(std::move(bytes), std::move(shape), Memory::kHost, dtype);
}

constexpr std::chrono::nanoseconds kOverlayCaptured = std::chrono::milliseconds(4321);
namespace viz = holoscan::visualization;
using Rgba = std::array<std::uint8_t, 4>;

// Two detections of three slots in a 200x100 image, and a label image with
// instance 1 on [10, 60) x [10, 50), instance 2 on [100, 150) x [50, 90),
// and a stray 3, past the count, at [0, 5) x [95, 100). Its two layers, or
// nothing if the run failed; with `label_shape`, a label image of that shape.
using ImageSnapshot = holoscan::testing::SchemaSnapshot<holoscan::schema::ImageT>;
using OverlayLayers = std::pair<ImageSnapshot, TensorSnapshot>;

std::optional<OverlayLayers> run_overlay(std::int32_t n,
                                         std::vector<std::int64_t> label_shape = {100, 200},
                                         const std::string& failure = "") {
  std::vector<std::int32_t> label_image(static_cast<std::size_t>(label_shape[0] * label_shape[1]));
  if (label_shape == std::vector<std::int64_t>{100, 200}) {
    for (int y = 0; y < 100; ++y) {
      for (int x = 0; x < 200; ++x) {
        std::int32_t& v = label_image[static_cast<std::size_t>(y) * 200 + x];
        if (x >= 10 && x < 60 && y >= 10 && y < 50) v = 1;
        if (x >= 100 && x < 150 && y >= 50 && y < 90) v = 2;
        if (x < 5 && y >= 95) v = 3;
      }
    }
  }
  auto h = holoscan::testing::create_operator_test<Yolo26SegmentationOverlayOp>(
      Yolo26SegmentationOverlayConfig{
          .width = 200, .height = 100, .max_detections = 3, .labels = {"person"}});
  namespace ht = holoscan::testing;
  h.push_frame(ht::frame_at(
      kOverlayCaptured,
      ht::input("boxes",
                host_tensor(std::vector<float>{10, 10, 60, 50, 100, 50, 150, 90, 0, 0, 200, 100},
                            {3, 4}, {kDLFloat, 32, 1})),
      ht::input("scores",
                host_tensor(std::vector<float>{0.9F, 0.6F, 0.1F}, {3}, {kDLFloat, 32, 1})),
      ht::input("class_ids", host_tensor(std::vector<std::int32_t>{0, 1, 0}, {3}, {kDLInt, 32, 1})),
      ht::input("count", host_tensor(std::vector<std::int32_t>{n}, {1}, {kDLInt, 32, 1})),
      ht::input("label_image", host_tensor(label_image, label_shape, {kDLInt, 32, 1}))));
  h.capture_snapshot<holoscan::schema::ImageT>("overlay");
  h.capture_snapshot<holoscan::Tensor>("labels");
  const auto result = h.run();
  if (!failure.empty()) {
    VERIFY_MSG(!result.ok() && result.describe_failure().find(failure) != std::string::npos,
               "expected '" + failure + "' in: " + result.describe_failure());
    return std::nullopt;
  }
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return std::nullopt;
  return std::pair{result.snapshot<holoscan::schema::ImageT>("overlay"),
                   result.snapshot<holoscan::Tensor>("labels")};
}

Rgba pixel(const TensorSnapshot& image, int x, int y) {
  const auto* p = reinterpret_cast<const std::uint8_t*>(image.bytes().data()) + (y * 200 + x) * 4;
  return {p[0], p[1], p[2], p[3]};
}

// The instances filled in their class's color at the masks' opacity, the
// boxes opaque over them, and a label each, as Yolo26OverlayOp's.
void check_overlay() {
  const auto out = run_overlay(2);
  if (!out) return;
  const auto& [overlay, labels] = *out;
  VERIFY(overlay.sample_metadata().capture_timestamp_ns == kOverlayCaptured.count() &&
         labels.sample_metadata().capture_timestamp_ns == kOverlayCaptured.count());
  VERIFY(overlay.metadata().encoding == holoscan::schema::ImageEncoding_RGBA8);
  const auto& image = overlay.tensor(holoscan::schema::ImageTensorFields::data);
  VERIFY(image.shape() == (std::vector<std::int64_t>{100, 200, 4}));
  constexpr Rgba kClass0Fill{230, 25, 75, 110};
  constexpr Rgba kClass1Fill{60, 180, 75, 110};
  constexpr Rgba kClass0Box{230, 25, 75, 255};
  constexpr Rgba kClass1Box{60, 180, 75, 255};
  constexpr Rgba kClear{0, 0, 0, 0};
  VERIFY(pixel(image, 30, 30) == kClass0Fill && pixel(image, 120, 70) == kClass1Fill);
  VERIFY(pixel(image, 10, 30) == kClass0Box && pixel(image, 149, 70) == kClass1Box);
  VERIFY(pixel(image, 80, 30) == kClear && pixel(image, 2, 97) == kClear);
  VERIFY(labels.shape() ==
         (std::vector<std::int64_t>{
             6, static_cast<std::int64_t>(viz::kTextRunHeaderBytes + kOverlayLabelBytes)}));
  std::string first(reinterpret_cast<const char*>(labels.bytes().data()) +
                        viz::kTextRunHeaderBytes + viz::kTextRunHeaderBytes + kOverlayLabelBytes,
                    kOverlayLabelBytes);
  first.erase(std::min(first.find('\0'), first.size()));
  VERIFY_MSG(first == "person 0.90", "label '" + first + "'");
}

// No detections: fully transparent, even where the label image says
// otherwise. A label image of another size fails.
void check_overlay_empty_and_mismatch() {
  if (const auto out = run_overlay(0)) {
    const auto bytes = out->first.tensor(holoscan::schema::ImageTensorFields::data).bytes();
    VERIFY(std::all_of(bytes.begin(), bytes.end(), [](std::byte b) { return b == std::byte{0}; }));
  }
  (void)run_overlay(2, {50, 200}, "the label image must be int32 [100, 200]");
}
#endif

}  // namespace
}  // namespace holoscan::yolo26

int main(int argc, char** argv) {
  namespace yd = holoscan::yolo26;
  if (argc != 2 && !(argc == 4 && std::string_view(argv[2]) == "--real")) {
    std::cerr << "usage: " << argv[0] << " DIR [--real MODEL]\n";
    return 2;
  }
  yd::g_directory = argv[1];
  std::filesystem::create_directories(yd::g_directory);
  if (argc == 4) {
    return yd::run_real(argv[3]);
  }
  using yd::PixelFormat;
  using yd::test::Memory;
  std::vector<yd::test::TestCase> cases{
      {"RGB8 from the GPU, wide",
       [] { yd::check_segmentation(PixelFormat::kRgb8, Memory::kDevice, 1280, 720, 640); }},
      {"BGR8 from the GPU, tall",
       [] { yd::check_segmentation(PixelFormat::kBgr8, Memory::kDevice, 333, 1000, 640); }},
      {"RGBA8 from host memory, odd padding",
       [] { yd::check_segmentation(PixelFormat::kRgba8, Memory::kHost, 1000, 333, 640); }},
      {"BGRA8 from pinned memory, input 320",
       [] { yd::check_segmentation(PixelFormat::kBgra8, Memory::kPinned, 1920, 1080, 320); }},
      {"small image, upscaled",
       [] { yd::check_segmentation(PixelFormat::kRgb8, Memory::kHost, 100, 64, 640); }},
      {"image size changes", yd::check_size_change},
      {"thresholds and capacity", yd::check_thresholds_and_capacity},
      {"rejects unusable images and models", yd::check_rejects},
      {"rejects unusable settings", yd::check_rejects_settings},
  };
#if HOLOSCAN_YOLO26_WITH_OVERLAY
  cases.push_back({"overlay masks, boxes and labels", yd::check_overlay});
  cases.push_back(
      {"overlay without detections, and of the wrong size", yd::check_overlay_empty_and_mismatch});
#endif
  return yd::test::run(cases);
}
