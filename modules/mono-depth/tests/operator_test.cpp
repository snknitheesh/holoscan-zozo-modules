// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// MonoDepthOp on its own, through the SDK's operator test harness, with the
// stub model, so through TensorRT without downloaded weights. The stub's
// known response lets the host references predict every output value from
// the image: synthetic frames in every encoding, from GPU, host and pinned
// host memory, wide and tall, letterboxed and stretched, at the image's size;
// the colorized output, with a per-frame and a fixed range; metric depth from
// an affine and from a metric model, with the camera info they need and the
// depth's camera info they send; an FP16 engine; inference without a CUDA
// graph; an image size that changes; the capture time, frame id and header
// on every output; and images, camera infos, settings and models it must
// reject.
//
// Usage: mono_depth_operator_test DIR (a folder for the stub models and engines)

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
#include <holoscan/messages/conversion.hpp>
#include <holoscan/mono_depth/image_fit.hpp>
#include <holoscan/mono_depth/mono_depth_op.hpp>
#include <holoscan/mono_depth/reference.hpp>
#include <holoscan/mono_depth/stub_model.hpp>
#include <holoscan/mono_depth/synthetic_scene.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "test_support.hpp"

namespace holoscan::mono_depth {
namespace {

namespace hm = holoscan::messages;
namespace ht = holoscan::testing;
using holoscan::schema::ImageEncoding;
using ImageSnapshot = holoscan::testing::SchemaSnapshot<holoscan::schema::ImageT>;

std::filesystem::path g_directory;

constexpr int kInputWidth = 518;
constexpr int kInputHeight = 518;
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

// A host tensor of uint16 samples holding `bytes`.
std::shared_ptr<holoscan::Tensor> uint16_tensor(std::vector<std::uint8_t> bytes,
                                                std::vector<std::int64_t> shape) {
  struct Storage {
    DLManagedTensor managed{};
    std::vector<std::int64_t> shape;
    std::vector<std::uint8_t> bytes;
  };
  auto storage = std::make_unique<Storage>();
  storage->shape = std::move(shape);
  storage->bytes = std::move(bytes);
  storage->managed.dl_tensor = {.data = storage->bytes.data(),
                                .device = {kDLCPU, 0},
                                .ndim = static_cast<std::int32_t>(storage->shape.size()),
                                .dtype = DLDataType{kDLUInt, 16, 1},
                                .shape = storage->shape.data(),
                                .strides = nullptr,
                                .byte_offset = 0};
  storage->managed.manager_ctx = storage.get();
  storage->managed.deleter = [](DLManagedTensor* tensor) {
    delete static_cast<Storage*>(tensor->manager_ctx);
  };
  return std::make_shared<holoscan::Tensor>(&storage.release()->managed);
}

// A host RGB8 tensor of width x height pixels that are every other pixel of
// the packed `wide` image, twice as wide: its pixels are not packed in a row.
std::shared_ptr<holoscan::Tensor> strided_tensor(std::vector<std::uint8_t> wide, int width,
                                                 int height) {
  struct Storage {
    DLManagedTensor managed{};
    std::array<std::int64_t, 3> shape{};
    std::array<std::int64_t, 3> strides{};
    std::vector<std::uint8_t> bytes;
  };
  auto storage = std::make_unique<Storage>();
  storage->shape = {height, width, 3};
  storage->strides = {static_cast<std::int64_t>(width) * 6, 6, 1};
  storage->bytes = std::move(wide);
  storage->managed.dl_tensor = {.data = storage->bytes.data(),
                                .device = {kDLCPU, 0},
                                .ndim = 3,
                                .dtype = DLDataType{kDLUInt, 8, 1},
                                .shape = storage->shape.data(),
                                .strides = storage->strides.data(),
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
  render_depth_scene(make_depth_scene(width, height, frame), format, pixels.data(), pitch);
  return pixels;
}

holoscan::schema::ImageT scene_image(int width, int height, std::uint32_t frame, PixelFormat format,
                                     Memory memory = Memory::kHost,
                                     std::optional<ImageEncoding> label = std::nullopt) {
  holoscan::schema::ImageT image;
  image.header = std::make_shared<holoscan::schema::HeaderT>();
  image.header->frame_id = "camera";
  image.width = width;
  image.height = height;
  image.encoding = label.value_or(encoding_of(format));
  image.data = image_tensor(scene_pixels(width, height, frame, format),
                            {height, width, channels(format)}, memory);
  return image;
}

// A camera for width x height images.
hm::CameraInfo test_camera(int width, int height) {
  hm::CameraInfo info;
  info.header.frame_id = "camera";
  info.width = static_cast<std::uint32_t>(width);
  info.height = static_cast<std::uint32_t>(height);
  info.state = hm::CameraState::kUndistorted;
  info.fx = 0.75 * width;
  info.fy = 0.75 * width + 1.0;
  info.cx = width / 2.0 - 0.25;
  info.cy = height / 2.0 + 0.5;
  return info;
}

template <typename T>
void add_field(ht::TestFrame& frame, std::string name, T value, std::chrono::nanoseconds time,
               std::uint64_t frame_id) {
  auto sample = ht::sample_at(std::move(value), time);
  sample.metadata.frame_id = frame_id;
  std::vector<ht::TestSample<T>> samples;
  samples.push_back(std::move(sample));
  frame.fields.push_back({std::move(name), ht::staged_samples<T>(std::move(samples))});
}

MonoDepthConfig stub_config(int input_width = kInputWidth, int input_height = kInputHeight) {
  return MonoDepthConfig{.model = write_stub_model(g_directory, input_width, input_height),
                         .input_width = input_width,
                         .input_height = input_height,
                         .fp16 = false};
}

// The harness stages the images in GPU memory, `max_bytes` for the largest.
void stage_on_gpu(holoscan::testing::OperatorTestHarness<MonoDepthOp>& h, std::uint64_t max_bytes) {
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

// The image outputs `config` declares.
std::vector<const char*> outputs_of(const MonoDepthConfig& config) {
  std::vector<const char*> names{"depth"};
  if (gives_metric_depth(config)) names.push_back("metric_depth");
  if (config.colorize) names.push_back("colorized");
  return names;
}

void declare_outputs(holoscan::testing::OperatorTestHarness<MonoDepthOp>& h,
                     const MonoDepthConfig& config) {
  for (const char* name : outputs_of(config)) {
    h.output_device(name, holoscan::DeviceId{0});
    // Room for every frame of a run, and no more: the capture's capacity sizes
    // the output's GPU pool.
    h.capture_snapshot<holoscan::schema::ImageT>(name, kCaptureCapacity);
  }
  if (gives_metric_depth(config)) {
    h.capture<hm::schema::CameraInfoT>("depth_camera_info", kCaptureCapacity);
  }
}

template <typename T>
std::vector<T> values(const holoscan::testing::TensorSnapshot& snapshot) {
  std::vector<T> out(snapshot.bytes().size() / sizeof(T));
  std::memcpy(out.data(), snapshot.bytes().data(), out.size() * sizeof(T));
  return out;
}

// The largest absolute difference, relative to max(1, |expected|).
float max_error(const std::vector<float>& actual, const std::vector<float>& expected) {
  if (actual.size() != expected.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float worst = 0.0F;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float error = std::fabs(actual[i] - expected[i]) / std::max(1.0F, std::fabs(expected[i]));
    worst = std::max(worst, std::isfinite(error) ? error : std::numeric_limits<float>::infinity());
  }
  return worst;
}

// Checks one frame's outputs against the references for its image.
void check_frame(const MonoDepthConfig& config, const std::vector<std::uint8_t>& pixels,
                 PixelFormat format, int width, int height,
                 const std::vector<const ImageSnapshot*>& frame, float tolerance) {
  const std::string what = std::to_string(width) + "x" + std::to_string(height);
  const ImageFit fit = make_fit(width, height, config.input_width, config.input_height,
                                config.input_width, config.input_height, config.resize_mode);
  const std::vector<float> input =
      reference_preprocess(pixels.data(), static_cast<std::size_t>(width) * channels(format),
                           format, fit, config.normalization);
  const std::vector<float> output = reference_stub(input, config.input_width, config.input_height);
  const InverseDepthAffine* affine =
      config.metric_from_relative ? &*config.metric_from_relative : nullptr;
  const ReferenceDepth expected = reference_postprocess(output, fit, config.model_output, affine);
  const std::vector<std::int64_t> plane{height, width};

  std::size_t port = 0;
  for (const ImageSnapshot* output : frame) {
    const holoscan::schema::ImageT descriptor = output->metadata();
    VERIFY_MSG(descriptor.header && descriptor.header->frame_id == "camera",
               what + ": the image's header is not forwarded");
  }
  const ImageSnapshot& depth = *frame[port++];
  const auto& depth_tensor = depth.tensor(holoscan::schema::ImageTensorFields::data);
  VERIFY(depth.metadata().encoding == holoscan::schema::ImageEncoding_MONO32F);
  VERIFY(depth.metadata().width == width && depth.metadata().height == height);
  VERIFY(depth_tensor.shape() == plane && depth_tensor.dtype().code == kDLFloat);
  const std::vector<float> inverse = values<float>(depth_tensor);
  const float error = max_error(inverse, expected.inverse);
  VERIFY_MSG(error <= tolerance, what + ": inverse depth off by " + std::to_string(error));
  if (gives_metric_depth(config)) {
    const ImageSnapshot& metric = *frame[port++];
    const auto& metric_tensor = metric.tensor(holoscan::schema::ImageTensorFields::data);
    VERIFY(metric.metadata().encoding == holoscan::schema::ImageEncoding_MONO32F);
    VERIFY(metric_tensor.shape() == plane);
    const float metric_error = max_error(values<float>(metric_tensor), expected.metric);
    VERIFY_MSG(metric_error <= tolerance,
               what + ": metric depth off by " + std::to_string(metric_error));
  }
  if (config.colorize) {
    const ImageSnapshot& color = *frame[port++];
    const auto& color_tensor = color.tensor(holoscan::schema::ImageTensorFields::data);
    VERIFY(color.metadata().encoding == holoscan::schema::ImageEncoding_RGBA8);
    VERIFY(color_tensor.shape() == (std::vector<std::int64_t>{height, width, 4}));
    // The colorization of the operator's own depth, bit for bit.
    const std::array<float, 2> range = config.color_range.value_or(reference_range(inverse));
    VERIFY_MSG(
        values<std::uint8_t>(color_tensor) == reference_colorize(inverse, range[0], range[1]),
        what + ": colorized depth differs");
  }
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
void check_frames(MonoDepthConfig config, PixelFormat format, Memory memory,
                  const std::vector<Image>& images, float tolerance = 2e-5F) {
  // The engine is built first, as an application does, rather than in start()
  // under the harness's watchdog.
  prepare_engine(config);
  auto h = holoscan::testing::create_operator_test<MonoDepthOp>(config);
  const bool metric = gives_metric_depth(config);
  std::uint64_t max_bytes = 0;
  for (std::uint32_t f = 0; f < images.size(); ++f) {
    const auto [width, height] = images[f];
    // A metric configuration's camera info, sent before each image.
    ht::TestFrame frame;
    if (metric) {
      hm::CameraInfo info = test_camera(width, height);
      info.header.capture_timestamp_ns = capture_time(f).count();
      add_field(frame, "camera_info", hm::to_message(info), capture_time(f), 1U + f);
    }
    add_field(frame, "image", scene_image(width, height, f * 3U, format, memory), capture_time(f),
              1U + f);
    h.push_frame(std::move(frame));
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
  std::vector<std::vector<ImageSnapshot>> ports;
  for (const char* name : outputs_of(config)) {
    ports.push_back(result.snapshots<holoscan::schema::ImageT>(name));
    VERIFY_MSG(ports.back().size() == images.size(),
               std::string(name) + ": " + std::to_string(ports.back().size()) + " frames");
  }
  if (metric) {
    // Each frame's camera, with the frame's capture time and id.
    const auto infos = result.samples<hm::schema::CameraInfoT>("depth_camera_info");
    VERIFY_MSG(infos.size() == images.size(),
               "depth_camera_info: " + std::to_string(infos.size()) + " frames");
    for (std::uint32_t f = 0; f < infos.size() && f < images.size(); ++f) {
      VERIFY(infos[f].metadata.capture_timestamp_ns == capture_time(f).count());
      VERIFY(infos[f].metadata.frame_id == 1U + f);
      hm::CameraInfo want = test_camera(images[f].width, images[f].height);
      want.header.capture_timestamp_ns = capture_time(f).count();
      VERIFY_MSG(hm::from_message(infos[f].data) == want,
                 "frame " + std::to_string(f) + ": depth_camera_info is not the image's camera");
    }
  }
  for (std::uint32_t f = 0; f < images.size(); ++f) {
    std::vector<const ImageSnapshot*> frame;
    for (const auto& port : ports) {
      if (f >= port.size()) return;
      frame.push_back(&port[f]);
      VERIFY(port[f].sample_metadata().capture_timestamp_ns == capture_time(f).count());
      VERIFY(port[f].sample_metadata().frame_id == 1U + f);
    }
    const auto [width, height] = images[f];
    check_frame(config, scene_pixels(width, height, f * 3U, format), format, width, height, frame,
                tolerance);
  }
}

// A run with `image` fails, saying `message`; a metric configuration gets
// `camera` first, or the image's camera.
void check_run_fails(MonoDepthConfig config, holoscan::schema::ImageT image,
                     const std::string& message,
                     std::optional<hm::CameraInfo> camera = std::nullopt) {
  auto h = holoscan::testing::create_operator_test<MonoDepthOp>(config);
  if (gives_metric_depth(config)) {
    hm::CameraInfo info = camera.value_or(test_camera(image.width, image.height));
    info.header.capture_timestamp_ns = capture_time(0).count();
    ht::TestFrame frame;
    add_field(frame, "camera_info", hm::to_message(info), capture_time(0), 1U);
    add_field(frame, "image", std::move(image), capture_time(0), 1U);
    h.push_frame(std::move(frame));
  } else {
    h.push("image", std::move(image));
  }
  declare_outputs(h, config);
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), "expected a failure: " + message);
  VERIFY_MSG(result.describe_failure().find(message) != std::string::npos,
             "expected '" + message + "' in: " + result.describe_failure());
}

// A run with `image` gives no output: the SDK refuses the image before it
// reaches the operator (the harness's input source fails, or the harness
// refuses to stage it), or the operator refuses it.
void check_run_refused(MonoDepthConfig config, holoscan::schema::ImageT image,
                       const std::string& what) {
  try {
    auto h = holoscan::testing::create_operator_test<MonoDepthOp>(config);
    h.push("image", std::move(image));
    declare_outputs(h, config);
    const auto result = h.run();
    VERIFY_MSG(!result.ok(), what + ": expected the image to be refused");
    VERIFY(result.snapshots<holoscan::schema::ImageT>("depth").empty());
  } catch (const std::exception& error) {
    std::cout << "  " << what << ": refused before the run: " << error.what() << '\n';
  }
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
  MonoDepthConfig small = stub_config();
  small.max_width = 300;
  check_run_fails(small, scene_image(320, 240, 0, PixelFormat::kRgb8), "max_width x max_height");
  // A descriptor whose size is not its tensor's, 16-bit samples labeled RGB8,
  // and pixels that are not packed within a row (every other pixel): the
  // SDK refuses each before it reaches the operator, which checks them too
  // for producers outside the harness.
  holoscan::schema::ImageT mislabeled = scene_image(320, 240, 0, PixelFormat::kRgb8);
  mislabeled.width = 330;
  check_run_refused(stub_config(), mislabeled, "a descriptor of another size");
  holoscan::schema::ImageT wide = scene_image(320, 240, 0, PixelFormat::kRgb8);
  wide.data =
      uint16_tensor(std::vector<std::uint8_t>(std::size_t{240} * 320 * 3 * 2, 7U), {240, 320, 3});
  check_run_refused(stub_config(), wide, "16-bit samples");
  holoscan::schema::ImageT strided = scene_image(160, 120, 0, PixelFormat::kRgb8);
  strided.data = strided_tensor(scene_pixels(320, 120, 0, PixelFormat::kRgb8), 160, 120);
  check_run_refused(stub_config(), strided, "pixels not packed in a row");
}

// Metric configurations and their camera info: no output before the first
// camera info, then each frame with the newest one; a camera info for images
// of another size fails the run, naming both sizes.
void check_camera_info() {
  MonoDepthConfig config = stub_config();
  config.metric_from_relative = InverseDepthAffine{.scale = 0.4F, .shift = 0.1F};
  prepare_engine(config);
  constexpr int kWidth = 320;
  constexpr int kHeight = 240;
  auto h = holoscan::testing::create_operator_test<MonoDepthOp>(config);
  // Frame 0 before any camera info; 1 with it; 2 without (keeps 1's).
  for (std::uint32_t f = 0; f < 3; ++f) {
    ht::TestFrame frame;
    if (f == 1) {
      hm::CameraInfo info = test_camera(kWidth, kHeight);
      info.header.capture_timestamp_ns = capture_time(f).count();
      add_field(frame, "camera_info", hm::to_message(info), capture_time(f), 1U + f);
    }
    add_field(frame, "image", scene_image(kWidth, kHeight, f, PixelFormat::kRgb8), capture_time(f),
              1U + f);
    h.push_frame(std::move(frame));
  }
  declare_outputs(h, config);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (result.ok()) {
    const auto depths = result.snapshots<holoscan::schema::ImageT>("depth");
    const auto infos = result.samples<hm::schema::CameraInfoT>("depth_camera_info");
    VERIFY_MSG(depths.size() == 2 && infos.size() == 2,
               std::to_string(depths.size()) + " depths and " + std::to_string(infos.size()) +
                   " camera infos out of 3 frames, the first without camera info");
    for (std::size_t i = 0; i < depths.size() && i < infos.size(); ++i) {
      const std::int64_t time = capture_time(static_cast<std::uint32_t>(i + 1)).count();
      VERIFY(depths[i].sample_metadata().frame_id == i + 2U);
      VERIFY(infos[i].metadata.capture_timestamp_ns == time);
      VERIFY(hm::from_message(infos[i].data).header.capture_timestamp_ns == time);
    }
  }
  check_run_fails(config, scene_image(kWidth, kHeight, 0, PixelFormat::kRgb8),
                  "the camera info is for 640x480 images, and the image is 320x240",
                  test_camera(640, 480));
}

void check_rejects_models() {
  // A model exported for another input size.
  MonoDepthConfig other_size = stub_config();
  other_size.model = write_stub_model(g_directory, 364, 364);
  check_run_fails(other_size, scene_image(320, 240, 0, PixelFormat::kRgb8),
                  "export it for this input size");
  MonoDepthConfig missing = stub_config();
  missing.model = g_directory / "no_such_model.onnx";
  check_run_fails(missing, scene_image(320, 240, 0, PixelFormat::kRgb8), "cannot load");
}

void check_rejects_settings() {
  const auto rejects = [](auto edit) {
    MonoDepthConfig config{.model = "model.onnx"};
    edit(config);
    return test::throws<std::invalid_argument>([&] { MonoDepthOp op(config); });
  };
  const float nan = std::numeric_limits<float>::quiet_NaN();
  VERIFY(rejects([](auto& c) { c.model.clear(); }));
  VERIFY(rejects([](auto& c) { c.input_width = 0; }));
  VERIFY(rejects([](auto& c) { c.input_height = 4097; }));
  VERIFY(rejects([](auto& c) { c.normalization.std[1] = 0.0F; }));
  VERIFY(rejects([nan](auto& c) { c.normalization.mean[2] = nan; }));
  VERIFY(rejects([](auto& c) {
    c.model_output = ModelOutput::kMetricDepth;
    c.metric_from_relative = InverseDepthAffine{};
  }));
  VERIFY(rejects([](auto& c) { c.metric_from_relative = InverseDepthAffine{.scale = 0.0F}; }));
  VERIFY(rejects([nan](auto& c) { c.metric_from_relative = InverseDepthAffine{.shift = nan}; }));
  VERIFY(rejects([](auto& c) {
    c.colorize = true;
    c.color_range = std::array<float, 2>{2.0F, 1.0F};
  }));
  // A color range without the colorized output.
  VERIFY(rejects([](auto& c) { c.color_range = std::array<float, 2>{1.0F, 2.0F}; }));
  VERIFY(rejects([](auto& c) {
    c.metric_from_relative = InverseDepthAffine{};
    c.camera_info_input.max_age = std::chrono::nanoseconds{0};
  }));
  VERIFY(rejects([](auto& c) { c.max_width = 0; }));
  VERIFY(rejects([](auto& c) { c.max_height = 20000; }));
  VERIFY(rejects([](auto& c) { c.build_workspace_bytes = 1024; }));
  VERIFY(!rejects([](auto&) {}));
  VERIFY(!gives_metric_depth(MonoDepthConfig{}));
  VERIFY(gives_metric_depth(MonoDepthConfig{.model_output = ModelOutput::kMetricDepth}));
  VERIFY(gives_metric_depth(MonoDepthConfig{.metric_from_relative = InverseDepthAffine{}}));
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
  using md::Memory;
  using md::PixelFormat;
  const auto colorized = [] {
    md::MonoDepthConfig config = md::stub_config();
    config.colorize = true;
    return config;
  };
  return md::test::run({
      {"RGB8 from the GPU, wide, colorized",
       [&] {
         md::check_frames(colorized(), PixelFormat::kRgb8, Memory::kDevice,
                          {{1280, 720}, {1280, 720}});
       }},
      {"BGR8 from the GPU, tall",
       [] {
         md::check_frames(md::stub_config(), PixelFormat::kBgr8, Memory::kDevice, {{333, 1000}});
       }},
      {"RGBA8 from host memory, odd padding",
       [] {
         md::check_frames(md::stub_config(), PixelFormat::kRgba8, Memory::kHost, {{1000, 333}});
       }},
      {"BGRA8 from pinned memory, stretched into a wide input",
       [] {
         md::MonoDepthConfig config = md::stub_config(392, 224);
         config.resize_mode = md::ResizeMode::kStretch;
         md::check_frames(config, PixelFormat::kBgra8, Memory::kPinned, {{640, 480}});
       }},
      {"small image, upscaled, fixed color range",
       [&] {
         md::MonoDepthConfig config = colorized();
         config.color_range = std::array<float, 2>{1.0F, 4.0F};
         md::check_frames(config, PixelFormat::kRgb8, Memory::kHost, {{64, 48}});
       }},
      {"metric depth from an affine",
       [] {
         md::MonoDepthConfig config = md::stub_config();
         config.metric_from_relative = md::InverseDepthAffine{.scale = 0.4F, .shift = 0.1F};
         md::check_frames(config, PixelFormat::kRgb8, Memory::kDevice, {{640, 360}});
       }},
      {"metric model",
       [&] {
         md::MonoDepthConfig config = colorized();
         config.model_output = md::ModelOutput::kMetricDepth;
         md::check_frames(config, PixelFormat::kBgr8, Memory::kDevice, {{640, 360}});
       }},
      {"image size changes",
       [&] {
         md::check_frames(colorized(), PixelFormat::kRgba8, Memory::kDevice,
                          {{1280, 720}, {333, 500}, {1920, 1080}, {32, 32}});
       }},
      {"FP16 engine",
       [] {
         md::MonoDepthConfig config = md::stub_config();
         config.fp16 = true;
         md::check_frames(config, PixelFormat::kRgb8, Memory::kDevice, {{640, 480}}, 5e-3F);
       }},
      {"inference without a CUDA graph",
       [] {
         md::MonoDepthConfig config = md::stub_config();
         config.cuda_graphs = false;
         md::check_frames(config, PixelFormat::kRgb8, Memory::kDevice, {{640, 480}, {320, 240}});
       }},
      {"camera info for metric depth", md::check_camera_info},
      {"rejects unusable images", md::check_rejects_images},
      {"rejects unusable models", md::check_rejects_models},
      {"rejects unusable settings", md::check_rejects_settings},
  });
}
