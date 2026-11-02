// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Sam3SegmentationOp and Sam3OverlayOp on their own, through the SDK's
// operator test harness, with the stub model, so through TensorRT without
// downloaded weights. The stub's known response lets the host references
// predict every output from the image and the prompts: synthetic frames in
// every encoding, from GPU, host and pinned host memory; text and box
// prompts; more instances than fit; without the masks output; an FP16
// engine; an image size that changes; the capture time and frame id on every
// output; and images, settings, prompts and models it must reject.
//
// Usage: sam3_operator_test DIR (a folder for the stub models and engines)

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
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/sam3/reference.hpp>
#include <holoscan/sam3/sam3_segmentation_op.hpp>
#include <holoscan/sam3/stub_model.hpp>
#include <holoscan/sam3/synthetic_scene.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#if HOLOSCAN_SAM3_TEST_OVERLAY
#include <holoscan/sam3/sam3_overlay_op.hpp>
#include <holoscan/visualization/holoviz_op.hpp>
#endif

#include "test_support.hpp"

namespace holoscan::sam3 {
namespace {

using holoscan::schema::ImageEncoding;
using holoscan::testing::TensorSnapshot;

std::filesystem::path g_directory;

constexpr std::size_t kCaptureCapacity = 8;

enum class Memory { kDevice, kHost, kPinned };

// A uint8 tensor holding `bytes`: in plain host memory, or copied into pinned
// host or GPU memory.
std::shared_ptr<holoscan::Tensor> image_tensor(std::vector<std::uint8_t> bytes,
                                               std::vector<std::int64_t> shape, Memory memory) {
  const DLDataType dtype{kDLUInt, 8, 1};
  if (memory == Memory::kDevice) {
    // From the SDK's allocator, whose memory says when it is ready to read.
    auto tensor =
        holoscan::make_tensor(shape, dtype, std::nullopt, holoscan::MemoryStorageType::kDevice,
                              std::make_shared<holoscan::UnboundedAllocator>());
    if (!tensor) {
      throw std::runtime_error(tensor.error().message);
    }
    auto destination = tensor->mutable_data();
    if (!destination || cudaMemcpy(*destination, bytes.data(), bytes.size(),
                                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("cannot copy the image to the GPU");
    }
    return std::make_shared<holoscan::Tensor>(std::move(*tensor));
  }
  struct Storage {
    DLManagedTensor managed{};
    std::vector<std::int64_t> shape;
    std::vector<std::uint8_t> bytes;
    void* pinned{};
    ~Storage() {
      if (pinned != nullptr) cudaFreeHost(pinned);
    }
  };
  auto storage = std::make_unique<Storage>();
  storage->shape = std::move(shape);
  storage->bytes = std::move(bytes);
  void* data = storage->bytes.data();
  DLDeviceType type = kDLCPU;
  if (memory == Memory::kPinned) {
    if (cudaMallocHost(&storage->pinned, storage->bytes.size()) != cudaSuccess) {
      throw std::runtime_error("cudaMallocHost failed");
    }
    std::memcpy(storage->pinned, storage->bytes.data(), storage->bytes.size());
    data = storage->pinned;
    type = kDLCUDAHost;
  }
  storage->managed.dl_tensor = {.data = data,
                                .device = {type, 0},
                                .ndim = static_cast<std::int32_t>(storage->shape.size()),
                                .dtype = dtype,
                                .shape = storage->shape.data(),
                                .strides = nullptr,
                                .byte_offset = 0};
  storage->managed.manager_ctx = storage.get();
  storage->managed.deleter = [](DLManagedTensor* tensor) {
    delete static_cast<Storage*>(tensor->manager_ctx);
  };
  return std::make_shared<holoscan::Tensor>(&storage.release()->managed);
}

ImageEncoding encoding_of(PixelFormat format) {
  switch (format) {
    case PixelFormat::kBgr8:
      return holoscan::schema::ImageEncoding_BGR8;
    case PixelFormat::kRgba8:
      return holoscan::schema::ImageEncoding_RGBA8;
    case PixelFormat::kBgra8:
      return holoscan::schema::ImageEncoding_BGRA8;
    default:
      return holoscan::schema::ImageEncoding_RGB8;
  }
}

// A synthetic frame's pixels in `format`, packed.
std::vector<std::uint8_t> scene_pixels(int width, int height, std::uint32_t frame,
                                       PixelFormat format) {
  const std::size_t pitch = static_cast<std::size_t>(width) * channels(format);
  std::vector<std::uint8_t> pixels(pitch * height);
  render_shape_scene(make_shape_scene(width, height, frame), format, pixels.data(), pitch);
  return pixels;
}

holoscan::schema::ImageT scene_image(int width, int height, std::uint32_t frame, PixelFormat format,
                                     Memory memory = Memory::kHost,
                                     std::optional<ImageEncoding> label = std::nullopt) {
  holoscan::schema::ImageT image;
  image.width = width;
  image.height = height;
  image.encoding = label.value_or(encoding_of(format));
  image.data = image_tensor(scene_pixels(width, height, frame, format),
                            {height, width, channels(format)}, memory);
  return image;
}

StubShape g_shape{};

// The stub with three prompts: "red", "blue", and a box around the
// triangle of frame 0.
Sam3SegmentationConfig stub_config(StubShape shape = g_shape) {
  const StubModel files = write_stub_model(g_directory, shape);
  const ShapeScene scene = make_shape_scene(640, 480, 0);
  return Sam3SegmentationConfig{
      .encoder = files.encoder,
      .decoder = files.decoder,
      .prompt_table = files.prompt_table,
      .prompts = {{.text = "red"},
                  {.text = "blue"},
                  {.boxes = {PromptBox{.xyxy = shape_box(scene, SceneShape::kTriangle)}}}},
      .max_instances = 8,
      .fp16 = false};
}

// The harness stages the images in GPU memory, `max_bytes` for the largest.
template <typename Harness>
void stage_on_gpu(Harness& h, std::uint64_t max_bytes) {
  h.input_source_tensor_spec("image",
                             holoscan::TensorOutputSpec{
                                 .representation =
                                     holoscan::TensorRepresentation{
                                         .memory_kind = holoscan::MemoryKind::kCudaDevice,
                                         .dtype = DLDataType{kDLUInt, 8, 1},
                                         .rank = 3U,
                                         .strides = holoscan::TensorStridePolicy::kContiguousOnly,
                                     },
                                 .bounds = holoscan::tensor_bounds(max_bytes),
                                 .storage = holoscan::TensorOutputStorage::kRuntimePool,
                             });
  h.input_source_device("image", holoscan::DeviceId{0});
}

// The outputs `config` declares.
std::vector<const char*> outputs_of(const Sam3SegmentationConfig& config) {
  std::vector<const char*> names;
  if (config.output_masks) names.push_back("masks");
  for (const char* name : {"labels", "scores", "boxes", "prompt_ids", "count"}) {
    names.push_back(name);
  }
  return names;
}

template <typename Harness>
void declare_outputs(Harness& h, const Sam3SegmentationConfig& config) {
  for (const char* name : outputs_of(config)) {
    h.output_device(name, holoscan::DeviceId{0});
    // Room for every frame of a run, and no more: the capture's capacity sizes
    // the output's GPU pool.
    h.template capture_snapshot<holoscan::Tensor>(name, kCaptureCapacity);
  }
}

template <typename T>
std::vector<T> values(const TensorSnapshot& snapshot) {
  std::vector<T> out(snapshot.bytes().size() / sizeof(T));
  std::memcpy(out.data(), snapshot.bytes().data(), out.size() * sizeof(T));
  return out;
}

// The reference of one frame through the stub: every prompt's decoder
// outputs, then the instances.
ReferenceInstances expected_instances(const Sam3SegmentationConfig& config, const StubShape& shape,
                                      const std::vector<std::uint8_t>& pixels, PixelFormat format,
                                      int width, int height) {
  const std::vector<float> input = reference_preprocess(
      pixels.data(), static_cast<std::size_t>(width) * channels(format), format, width, height,
      shape.input_size, shape.input_size, Normalization{});
  const std::vector<float> fpn0 = reference_stub_encoder(input, shape);
  const PromptTable table = read_prompt_table(config.prompt_table);
  std::vector<DecoderOutput> outputs;
  // A control's prompts, once set, replace the configured ones.
  const std::vector<Sam3Prompt> prompts =
      config.prompt_control && config.prompt_control->latest().first != 0
          ? config.prompt_control->latest().second
          : config.prompts;
  for (const Sam3Prompt& prompt : prompts) {
    outputs.push_back(
        reference_stub_decoder(fpn0, make_decoder_prompt(prompt, table, shape.boxes), shape));
  }
  return reference_instances(outputs, config.max_instances, config.score_threshold,
                             logit_of(config.mask_threshold), width, height);
}

// Pixels whose reference logit is this near the threshold may go either way:
// TensorRT sums the stub's terms in its own order.
constexpr float kLogitSlack = 1e-3F;

// Checks one frame's outputs against the reference.
void check_frame(const Sam3SegmentationConfig& config, const StubShape& shape,
                 const std::vector<std::uint8_t>& pixels, PixelFormat format, int width, int height,
                 const std::vector<const TensorSnapshot*>& frame, float slack) {
  const std::string what = std::to_string(width) + "x" + std::to_string(height);
  const ReferenceInstances expected =
      expected_instances(config, shape, pixels, format, width, height);
  const std::int64_t capacity = config.max_instances;
  const std::size_t plane = static_cast<std::size_t>(width) * height;
  const float threshold = logit_of(config.mask_threshold);
  std::size_t port = 0;
  // Near the threshold, either answer is right.
  const auto ambiguous = [&](std::size_t k, std::size_t i) {
    return k < static_cast<std::size_t>(expected.count) &&
           std::fabs(expected.logits[k * plane + i] - threshold) <= slack;
  };
  if (config.output_masks) {
    const TensorSnapshot& masks = *frame[port++];
    VERIFY(masks.shape() == (std::vector<std::int64_t>{capacity, height, width}));
    const std::vector<std::uint8_t> actual = values<std::uint8_t>(masks);
    std::size_t wrong = 0;
    for (std::size_t k = 0; k < static_cast<std::size_t>(capacity); ++k) {
      for (std::size_t i = 0; i < plane; ++i) {
        if (actual[k * plane + i] != expected.masks[k * plane + i] && !ambiguous(k, i)) {
          ++wrong;
        }
      }
    }
    VERIFY_MSG(wrong == 0, what + ": " + std::to_string(wrong) + " mask pixels differ");
  }
  const TensorSnapshot& labels = *frame[port++];
  VERIFY(labels.shape() == (std::vector<std::int64_t>{height, width}) && labels.dtype().bits == 16);
  const std::vector<std::uint16_t> label_image = values<std::uint16_t>(labels);
  std::size_t wrong_labels = 0;
  for (std::size_t i = 0; i < plane; ++i) {
    if (label_image[i] != expected.labels[i]) {
      // Allowed only where an instance up to either label is ambiguous.
      const std::size_t last = std::max(label_image[i], expected.labels[i]);
      bool excused = false;
      for (std::size_t k = 0; k < last && !excused; ++k) {
        excused = ambiguous(k, i);
      }
      wrong_labels += excused ? 0U : 1U;
    }
  }
  VERIFY_MSG(wrong_labels == 0,
             what + ": " + std::to_string(wrong_labels) + " label pixels differ");
  // FP32 gives the stub's scores and boxes exactly; FP16 rounds them.
  const auto max_error = [](const std::vector<float>& a, const std::vector<float>& b) {
    float error = a.size() == b.size() ? 0.0F : std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
      error = std::max(error, std::fabs(a[i] - b[i]));
    }
    return error;
  };
  const float score_error = max_error(values<float>(*frame[port++]), expected.scores);
  VERIFY_MSG(score_error <= (config.fp16 ? 1e-3F : 0.0F),
             what + ": scores off by " + std::to_string(score_error));
  const float box_error = max_error(values<float>(*frame[port++]), expected.boxes);
  const float box_tolerance = (config.fp16 ? 1e-3F : 1e-5F) * std::max(width, height);
  VERIFY_MSG(box_error <= box_tolerance, what + ": boxes off by " + std::to_string(box_error));
  VERIFY_MSG(values<std::int32_t>(*frame[port++]) == expected.prompt_ids,
             what + ": prompt ids differ");
  VERIFY_MSG(values<std::int32_t>(*frame[port++]) == std::vector<std::int32_t>{expected.count},
             what + ": count differs");
}

std::chrono::nanoseconds capture_time(std::uint32_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

struct Image {
  int width;
  int height;
};

// Frames of the given sizes in `format` from `memory` through `config`, each
// checked against the references, with its capture time and frame id.
void check_frames(Sam3SegmentationConfig config, PixelFormat format, Memory memory,
                  const std::vector<Image>& images, float slack = kLogitSlack,
                  const StubShape& shape = g_shape) {
  // The engines are built first, as an application does, rather than in
  // start() under the harness's watchdog.
  prepare_engines(config);
  auto h = holoscan::testing::create_operator_test<Sam3SegmentationOp>(config);
  std::uint64_t max_bytes = 0;
  for (std::uint32_t f = 0; f < images.size(); ++f) {
    const auto [width, height] = images[f];
    holoscan::testing::TestSample<holoscan::schema::ImageT> sample{
        .data = scene_image(width, height, f * 3U, format, memory),
        .capture_coordinate = capture_time(f)};
    sample.metadata.frame_id = 1U + f;
    h.template push_samples<holoscan::schema::ImageT>("image", {std::move(sample)});
    max_bytes = std::max<std::uint64_t>(
        max_bytes, static_cast<std::uint64_t>(width) * height * channels(format));
  }
  if (memory == Memory::kDevice) {
    stage_on_gpu(h, max_bytes);
  }
  declare_outputs(h, config);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  std::vector<std::vector<TensorSnapshot>> ports;
  for (const char* name : outputs_of(config)) {
    ports.push_back(result.template snapshots<holoscan::Tensor>(name));
    VERIFY_MSG(ports.back().size() == images.size(),
               std::string(name) + ": " + std::to_string(ports.back().size()) + " frames");
  }
  for (std::uint32_t f = 0; f < images.size(); ++f) {
    std::vector<const TensorSnapshot*> frame;
    for (const auto& port : ports) {
      if (f >= port.size()) return;
      frame.push_back(&port[f]);
      VERIFY(port[f].sample_metadata().capture_timestamp_ns == capture_time(f).count());
      VERIFY(port[f].sample_metadata().frame_id == 1U + f);
    }
    const auto [width, height] = images[f];
    check_frame(config, shape, scene_pixels(width, height, f * 3U, format), format, width, height,
                frame, slack);
  }
}

// A run with `image` fails, saying `message`.
void check_run_fails(const Sam3SegmentationConfig& config, holoscan::schema::ImageT image,
                     const std::string& message) {
  auto h = holoscan::testing::create_operator_test<Sam3SegmentationOp>(config);
  h.push("image", std::move(image));
  declare_outputs(h, config);
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), "expected a failure: " + message);
  VERIFY_MSG(result.describe_failure().find(message) != std::string::npos,
             "expected '" + message + "' in: " + result.describe_failure());
}

void check_rejects_images() {
  holoscan::schema::ImageT mono = scene_image(320, 240, 0, PixelFormat::kRgb8);
  mono.encoding = holoscan::schema::ImageEncoding_MONO8;
  check_run_fails(stub_config(), mono, "unsupported image encoding MONO8");
  // RGBA pixels labeled RGB8: four channels where three belong.
  check_run_fails(stub_config(),
                  scene_image(320, 240, 0, PixelFormat::kRgba8, Memory::kHost,
                              holoscan::schema::ImageEncoding_RGB8),
                  "uint8 [height, width, channels]");
  Sam3SegmentationConfig small = stub_config();
  small.max_width = 300;
  check_run_fails(small, scene_image(320, 240, 0, PixelFormat::kRgb8), "max_width x max_height");
}

void check_rejects_models() {
  const holoscan::schema::ImageT image = scene_image(320, 240, 0, PixelFormat::kRgb8);
  // An encoder and a decoder of different exports.
  Sam3SegmentationConfig mixed = stub_config();
  mixed.decoder = write_stub_model(g_directory, StubShape{.mask_size = 8}).decoder;
  check_run_fails(mixed, image, "use an encoder and a decoder of the same export");
  // A prompt table of another export.
  Sam3SegmentationConfig other_table = stub_config();
  other_table.prompt_table = write_stub_model(g_directory, StubShape{.tokens = 6}).prompt_table;
  check_run_fails(other_table, image, "use the prompt table of the same export");
  // More boxes than the decoder takes.
  Sam3SegmentationConfig boxes = stub_config();
  boxes.prompts = {{.boxes = {PromptBox{.xyxy = {0, 0, 1, 1}}, PromptBox{.xyxy = {0, 0, 1, 1}},
                              PromptBox{.xyxy = {0, 0, 1, 1}}}}};
  check_run_fails(boxes, image, "the decoder takes at most 2");
  // The models swapped.
  Sam3SegmentationConfig swapped = stub_config();
  std::swap(swapped.encoder, swapped.decoder);
  check_run_fails(swapped, image, "Sam3SegmentationOp");
  Sam3SegmentationConfig missing = stub_config();
  missing.encoder = g_directory / "no_such_model.onnx";
  check_run_fails(missing, image, "cannot load");
}

void check_rejects_settings() {
  const auto rejects = [](auto edit) {
    Sam3SegmentationConfig config = stub_config();
    edit(config);
    return test::throws<std::invalid_argument>([&] { Sam3SegmentationOp op(config); });
  };
  const float nan = std::numeric_limits<float>::quiet_NaN();
  VERIFY(rejects([](auto& c) { c.encoder.clear(); }));
  VERIFY(rejects([](auto& c) { c.prompts.clear(); }));
  VERIFY(rejects([](auto& c) { c.prompts.resize(kMaxPrompts + 1, Sam3Prompt{.text = "red"}); }));
  VERIFY(rejects([](auto& c) { c.prompts = {Sam3Prompt{.text = "green"}}; }));
  VERIFY(rejects([](auto& c) { c.prompts = {Sam3Prompt{}}; }));
  VERIFY(rejects([](auto& c) { c.score_threshold = 1.0F; }));
  VERIFY(rejects([nan](auto& c) { c.score_threshold = nan; }));
  VERIFY(rejects([](auto& c) { c.mask_threshold = 0.0F; }));
  VERIFY(rejects([](auto& c) { c.max_instances = 0; }));
  VERIFY(rejects([](auto& c) { c.max_instances = kMaxInstances + 1; }));
  VERIFY(rejects([](auto& c) { c.max_width = 0; }));
  VERIFY(rejects([](auto& c) { c.max_height = 20000; }));
  VERIFY(rejects([](auto& c) { c.build_workspace_bytes = 1024; }));
  VERIFY(!rejects([](auto&) {}));
  // A missing prompt table is a runtime error, as an unreadable one.
  Sam3SegmentationConfig no_table = stub_config();
  no_table.prompt_table = g_directory / "no_such_table.bin";
  VERIFY(test::throws<std::runtime_error>([&] { Sam3SegmentationOp op(no_table); }));
  // A prompt control checks as the constructor does, and keeps the prompts on
  // failure.
  Sam3PromptControl control(read_prompt_table(stub_config().prompt_table));
  VERIFY(test::throws<std::invalid_argument>([&] { control.set({}); }));
  VERIFY(test::throws<std::invalid_argument>([&] { control.set({{.text = "green"}}); }));
  VERIFY(control.latest().first == 0 && control.latest().second.empty());
  control.set({{.text = "blue"}});
  VERIFY(control.latest().first == 1 && control.latest().second.size() == 1);
  VERIFY(test::throws<std::invalid_argument>([&] { control.set({{.text = "green"}}); }));
  VERIFY(control.latest().first == 1 && control.latest().second[0].text == "blue");
}

#if HOLOSCAN_SAM3_TEST_OVERLAY
// Sam3OverlayOp on a reference frame's instances: the overlay is the
// reference's, and each instance has its label row.
void check_overlay() {
  namespace viz = holoscan::visualization;
  const int width = 320;
  const int height = 240;
  const Sam3SegmentationConfig config = stub_config();
  const std::vector<std::uint8_t> pixels = scene_pixels(width, height, 0, PixelFormat::kRgb8);
  const ReferenceInstances expected =
      expected_instances(config, g_shape, pixels, PixelFormat::kRgb8, width, height);
  const std::int64_t capacity = config.max_instances;

  auto h = holoscan::testing::create_operator_test<Sam3OverlayOp>(
      Sam3OverlayConfig{.max_width = width,
                        .max_height = height,
                        .max_instances = config.max_instances,
                        .prompt_names = {"red", "blue"},
                        .fill_alpha = 90});
  const auto host_tensor = [](auto values, std::vector<std::int64_t> shape, DLDataType dtype) {
    using T = typename decltype(values)::value_type;
    auto tensor = holoscan::make_tensor(std::span<const std::int64_t>(shape), dtype, std::nullopt,
                                        holoscan::MemoryStorageType::kHost,
                                        std::make_shared<holoscan::UnboundedAllocator>());
    if (!tensor) throw std::runtime_error(tensor.error().message);
    std::memcpy(*tensor->mutable_data(), values.data(), values.size() * sizeof(T));
    return std::move(*tensor);
  };
  const std::vector<std::int64_t> label_shape{height, width};
  auto labels = holoscan::make_tensor(label_shape, DLDataType{kDLUInt, 16, 1}, std::nullopt,
                                      holoscan::MemoryStorageType::kDevice,
                                      std::make_shared<holoscan::UnboundedAllocator>());
  if (!labels || cudaMemcpy(*labels->mutable_data(), expected.labels.data(),
                            expected.labels.size() * 2, cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("cannot stage the label image");
  }
  const auto at = std::chrono::milliseconds(5);
  h.template push_samples<holoscan::Tensor>(
      "labels", {holoscan::testing::TestSample<holoscan::Tensor>{.data = std::move(*labels),
                                                                 .capture_coordinate = at}});
  h.template push_samples<holoscan::Tensor>(
      "scores", {{.data = host_tensor(expected.scores, {capacity}, DLDataType{kDLFloat, 32, 1}),
                  .capture_coordinate = at}});
  h.template push_samples<holoscan::Tensor>(
      "boxes", {{.data = host_tensor(expected.boxes, {capacity, 4}, DLDataType{kDLFloat, 32, 1}),
                 .capture_coordinate = at}});
  h.template push_samples<holoscan::Tensor>(
      "prompt_ids",
      {{.data = host_tensor(expected.prompt_ids, {capacity}, DLDataType{kDLInt, 32, 1}),
        .capture_coordinate = at}});
  h.template push_samples<holoscan::Tensor>(
      "count", {{.data = host_tensor(std::vector<std::int32_t>{expected.count}, {1},
                                     DLDataType{kDLInt, 32, 1}),
                 .capture_coordinate = at}});
  // The label image comes from GPU memory, as the segmenter gives it.
  h.input_source_tensor_spec(
      "labels", holoscan::TensorOutputSpec{
                    .representation =
                        holoscan::TensorRepresentation{
                            .memory_kind = holoscan::MemoryKind::kCudaDevice,
                            .dtype = DLDataType{kDLUInt, 16, 1},
                            .rank = 2U,
                            .strides = holoscan::TensorStridePolicy::kContiguousOnly,
                        },
                    .bounds = holoscan::tensor_bounds(std::uint64_t{2} * width * height),
                    .storage = holoscan::TensorOutputStorage::kRuntimePool,
                });
  h.input_source_device("labels", holoscan::DeviceId{0});
  h.output_device("overlay", holoscan::DeviceId{0});
  h.template capture_snapshot<holoscan::schema::ImageT>("overlay", 2);
  h.template capture_snapshot<holoscan::Tensor>("text", 2);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto overlays = result.template snapshots<holoscan::schema::ImageT>("overlay");
  const auto texts = result.template snapshots<holoscan::Tensor>("text");
  VERIFY(overlays.size() == 1 && texts.size() == 1);
  if (overlays.size() != 1 || texts.size() != 1) return;
  const auto& image = overlays[0].tensor(holoscan::schema::ImageTensorFields::data);
  VERIFY(overlays[0].metadata().encoding == holoscan::schema::ImageEncoding_RGBA8);
  VERIFY(image.shape() == (std::vector<std::int64_t>{height, width, 4}));
  VERIFY_MSG(values<std::uint8_t>(image) == reference_overlay(expected.labels, width, height, 90),
             "the overlay differs from the reference");
  // Both layers carry one capture time, so HoloViz draws them together.
  VERIFY(overlays[0].sample_metadata().capture_timestamp_ns ==
         texts[0].sample_metadata().capture_timestamp_ns);
  // Rows 2k and 2k + 1 are instance k's shadow and label; the third instance's
  // prompt has no name.
  const std::vector<std::uint8_t> text = values<std::uint8_t>(texts[0]);
  const std::size_t row = viz::kTextRunHeaderBytes + kOverlayLabelBytes;
  const auto label = [&](std::size_t index) {
    const char* start =
        reinterpret_cast<const char*>(text.data() + index * row + viz::kTextRunHeaderBytes);
    return std::string(start, strnlen(start, kOverlayLabelBytes));
  };
  VERIFY(expected.count == 3);
  VERIFY_MSG(label(1) == "red 0.90" && label(0) == label(1), label(1));
  VERIFY_MSG(label(3) == "blue 0.80", label(3));
  VERIFY_MSG(label(5) == "prompt 2 0.70", label(5));
  VERIFY(label(7).empty());
}
#endif

}  // namespace
}  // namespace holoscan::sam3

int main(int argc, char** argv) {
  namespace s3 = holoscan::sam3;
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " DIR\n";
    return 2;
  }
  s3::g_directory = argv[1];
  std::filesystem::create_directories(s3::g_directory);
  using s3::Memory;
  using s3::PixelFormat;
  return s3::test::run({
      {"RGB8 from the GPU, text and box prompts",
       [] {
         s3::check_frames(s3::stub_config(), PixelFormat::kRgb8, Memory::kDevice,
                          {{640, 480}, {640, 480}});
       }},
      {"BGR8 from host memory, wide",
       [] {
         s3::check_frames(s3::stub_config(), PixelFormat::kBgr8, Memory::kHost, {{1280, 720}});
       }},
      {"RGBA8 from pinned memory, tall",
       [] {
         s3::check_frames(s3::stub_config(), PixelFormat::kRgba8, Memory::kPinned, {{333, 500}});
       }},
      {"BGRA8 without the masks output",
       [] {
         s3::Sam3SegmentationConfig config = s3::stub_config();
         config.output_masks = false;
         s3::check_frames(config, PixelFormat::kBgra8, Memory::kDevice, {{640, 360}});
       }},
      {"more instances than fit, other thresholds",
       [] {
         s3::Sam3SegmentationConfig config = s3::stub_config();
         config.max_instances = 2;
         config.score_threshold = 0.2F;
         config.mask_threshold = 0.3F;
         s3::check_frames(config, PixelFormat::kRgb8, Memory::kDevice, {{640, 480}});
       }},
      {"image size changes",
       [] {
         s3::check_frames(s3::stub_config(), PixelFormat::kRgb8, Memory::kDevice,
                          {{1280, 720}, {333, 500}, {1920, 1080}, {32, 32}});
       }},
      {"prompts from a control",
       [] {
         s3::Sam3SegmentationConfig config = s3::stub_config();
         config.prompts = {{.text = "red"}};
         config.prompt_control =
             std::make_shared<s3::Sam3PromptControl>(s3::read_prompt_table(config.prompt_table));
         // Boxes alone first, then blue: the prompt ids follow the new list.
         config.prompt_control->set(
             {{.boxes = {s3::PromptBox{.xyxy = {0.3F, 0.55F, 0.7F, 0.95F}}}}, {.text = "blue"}});
         s3::check_frames(config, PixelFormat::kRgb8, Memory::kDevice, {{640, 480}});
       }},
      {"FP16 engines",
       [] {
         s3::Sam3SegmentationConfig config = s3::stub_config();
         config.fp16 = true;
         s3::check_frames(config, PixelFormat::kRgb8, Memory::kDevice, {{640, 480}}, 2e-2F);
       }},
      {"rejects unusable images", s3::check_rejects_images},
      {"rejects unusable models", s3::check_rejects_models},
      {"rejects unusable settings", s3::check_rejects_settings},
#if HOLOSCAN_SAM3_TEST_OVERLAY
      {"overlay", s3::check_overlay},
#endif
  });
}
