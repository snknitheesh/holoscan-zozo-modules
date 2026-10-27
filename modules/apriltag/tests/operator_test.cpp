// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The operators on their own, through the SDK's operator test harness: images
// of each supported encoding and layout in, detections with the scene's ids,
// corners and poses out, for every compiled backend; a change of image size
// and non-default options; unsupported images rejected; the detector off the
// legacy default CUDA stream; the reporter's counts; and the overlay's
// Holoviz layers shaped, drawn and stamped as Holoviz expects.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/apriltag/apriltag_detector_op.hpp>
#include <holoscan/apriltag/synthetic_scene.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#if HOLOSCAN_APRILTAG_WITH_OVERLAY
#include <holoscan/apriltag/apriltag_overlay_op.hpp>
#include <holoscan/visualization/holoviz_op.hpp>
#endif
#include "test_support.hpp"

namespace holoscan::apriltag {
namespace {

constexpr int kWidth = 640;
constexpr int kHeight = 480;

// A host tensor owning `bytes`, with `shape` and, unless empty, `strides` (in
// elements).
std::shared_ptr<holoscan::Tensor> host_tensor(std::vector<std::uint8_t> bytes,
                                              std::vector<std::int64_t> shape,
                                              std::vector<std::int64_t> strides = {}) {
  struct Storage {
    DLManagedTensor managed{};
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
    std::vector<std::uint8_t> bytes;
  };
  auto storage = std::make_unique<Storage>();
  storage->shape = std::move(shape);
  storage->strides = std::move(strides);
  storage->bytes = std::move(bytes);
  storage->managed.dl_tensor = {
      .data = storage->bytes.data(),
      .device = {kDLCPU, 0},
      .ndim = static_cast<std::int32_t>(storage->shape.size()),
      .dtype = {kDLUInt, 8, 1},
      .shape = storage->shape.data(),
      .strides = storage->strides.empty() ? nullptr : storage->strides.data(),
      .byte_offset = 0};
  storage->managed.manager_ctx = storage.get();
  storage->managed.deleter = [](DLManagedTensor* value) {
    delete static_cast<Storage*>(value->manager_ctx);
  };
  return std::make_shared<holoscan::Tensor>(&storage.release()->managed);
}

// `scene` as an image tensor; with `rank2`, MONO8 as [height, width], not
// [height, width, 1]. (The harness stages only contiguous tensors.)
holoscan::schema::ImageT scene_image(const SyntheticScene& scene,
                                     holoscan::schema::ImageEncoding encoding, bool rank2 = false) {
  using holoscan::schema::ImageEncoding_BGR8;
  using holoscan::schema::ImageEncoding_BGRA8;
  using holoscan::schema::ImageEncoding_MONO8;
  using holoscan::schema::ImageEncoding_RGBA8;
  const bool alpha = encoding == ImageEncoding_RGBA8 || encoding == ImageEncoding_BGRA8;
  const bool bgr = encoding == ImageEncoding_BGR8 || encoding == ImageEncoding_BGRA8;
  const int channels = encoding == ImageEncoding_MONO8 ? 1 : alpha ? 4 : 3;
  const auto rgb = render_scene(scene, channels == 1 ? 1 : 3);
  const int source_channels = channels == 1 ? 1 : 3;
  const std::size_t row = static_cast<std::size_t>(scene.width) * channels;
  std::vector<std::uint8_t> pixels(row * scene.height, 0);
  for (int y = 0; y < scene.height; ++y) {
    for (int x = 0; x < scene.width; ++x) {
      const std::uint8_t* in =
          rgb.data() + (static_cast<std::size_t>(y) * scene.width + x) * source_channels;
      std::uint8_t* out = pixels.data() + y * row + static_cast<std::size_t>(x) * channels;
      if (channels == 1) {
        out[0] = in[0];
        continue;
      }
      out[0] = in[bgr ? 2 : 0];
      out[1] = in[1];
      out[2] = in[bgr ? 0 : 2];
      if (alpha) out[3] = 255;
    }
  }
  std::vector<std::int64_t> shape{scene.height, scene.width};
  std::vector<std::int64_t> strides{static_cast<std::int64_t>(row)};
  if (!rank2) {
    shape.push_back(channels);
    strides.push_back(channels);
  }
  strides.push_back(1);
  holoscan::schema::ImageT image;
  image.width = scene.width;
  image.height = scene.height;
  image.encoding = encoding;
  image.data = host_tensor(std::move(pixels), std::move(shape), std::move(strides));
  return image;
}

// Checks one message's detections against `scene`'s tags, corners and poses.
void check_message(const schema::AprilTagDetectionsT& message, const SyntheticScene& scene,
                   bool poses) {
  VERIFY(message.image_width == scene.width && message.image_height == scene.height);
  VERIFY(message.family == "tag36h11");
  VERIFY(message.dropped == 0 && !message.truncated);
  VERIFY_MSG(message.detections.size() == scene.tags.size(),
             "frame " + std::to_string(message.frame) + ": " +
                 std::to_string(message.detections.size()) + " tags");
  std::vector<Detection> found;
  for (const schema::TagDetection& tag : message.detections) found.push_back(from_message(tag));
  const TruthCounts counts = match_truth(scene, found, [poses](const TruthMatch& match) {
    for (const double error : match.corner_errors) VERIFY(error < 1.0);
    VERIFY(match.detection.pose.has_value() == poses);
    if (!match.detection.pose) return;
    const auto& t = match.tag.pose.translation;
    const auto& p = match.detection.pose->translation;
    VERIFY(std::hypot(p[0] - t[0], p[1] - t[1], p[2] - t[2]) < 0.02 * t[2]);
  });
  VERIFY_MSG(counts.extra == 0, std::to_string(counts.extra) + " unexpected or repeated tags");
}

// Runs a detector over `images`: one message per image, or none if the run fails.
std::vector<schema::AprilTagDetectionsT> run_detector(
    const AprilTagDetectorConfig& config, const std::vector<holoscan::schema::ImageT>& images) {
  auto h = holoscan::testing::create_operator_test<AprilTagDetectorOp>(config);
  for (const auto& image : images) h.push("image", image);
  h.capture<schema::AprilTagDetectionsT>("detections");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return {};
  auto messages = result.values<schema::AprilTagDetectionsT>("detections");
  VERIFY(messages.size() == images.size());
  return messages;
}

void check_detector(Backend backend, holoscan::schema::ImageEncoding encoding, bool rank2 = false) {
  constexpr std::uint32_t kFrames = 3;
  const SyntheticScene first = make_scene("tag36h11", 0, kWidth, kHeight);
  std::vector<holoscan::schema::ImageT> images;
  for (std::uint32_t frame = 0; frame < kFrames; ++frame) {
    images.push_back(scene_image(make_scene("tag36h11", frame, kWidth, kHeight), encoding, rank2));
  }
  const auto messages = run_detector(
      {.backend = backend, .intrinsics = first.intrinsics, .tag_size = first.tag_size}, images);
  for (std::size_t i = 0; i < messages.size(); ++i) {
    VERIFY(messages[i].frame == i);
    // VPI reports the hardware it chose: vpi-pva or vpi-cpu.
    VERIFY(messages[i].backend.starts_with(backend_name(backend)));
    check_message(messages[i],
                  make_scene("tag36h11", static_cast<std::uint32_t>(i), kWidth, kHeight), true);
  }
}

// The backend is rebuilt when the image size changes, and detects at both.
void check_size_change(Backend backend) {
  const std::array<SyntheticScene, 3> scenes{make_scene("tag36h11", 0, kWidth, kHeight),
                                             make_scene("tag36h11", 1, 800, 600),
                                             make_scene("tag36h11", 2, kWidth, kHeight)};
  std::vector<holoscan::schema::ImageT> images;
  for (const SyntheticScene& scene : scenes) {
    images.push_back(scene_image(scene, holoscan::schema::ImageEncoding_MONO8));
  }
  const auto messages = run_detector({.backend = backend}, images);
  for (std::size_t i = 0; i < messages.size() && i < scenes.size(); ++i) {
    check_message(messages[i], scenes[i], false);
  }
}

// Options other than the defaults reach the backend and still find every tag.
void check_options(Backend backend) {
  AprilTagDetectorConfig config{.backend = backend};
  config.options.decimate = 1;
  config.options.refine_edges = false;
  config.options.max_hamming = 0;
  config.options.decode_sharpening = 0.0;
  const SyntheticScene scene = make_scene("tag36h11", 3, kWidth, kHeight);
  const auto messages =
      run_detector(config, {scene_image(scene, holoscan::schema::ImageEncoding_MONO8)});
  if (messages.size() != 1) return;
  check_message(messages.front(), scene, false);
  for (const schema::TagDetection& tag : messages.front().detections) {
    VERIFY(tag.hamming() == 0);
  }
}

// A frame over a backend's capacity says so in its message: one-pixel stripes
// hold more boundary points than the CUDA backend keeps.
void check_detector_truncated() {
  constexpr int kDecimate = 2;
  std::vector<std::uint8_t> stripes(static_cast<std::size_t>(kWidth) * kHeight);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      stripes[static_cast<std::size_t>(y) * kWidth + x] = (x / kDecimate) % 2 ? 255 : 0;
    }
  }
  holoscan::schema::ImageT image;
  image.width = kWidth;
  image.height = kHeight;
  image.encoding = holoscan::schema::ImageEncoding_MONO8;
  image.data = host_tensor(std::move(stripes), {kHeight, kWidth}, {kWidth, 1});
  AprilTagDetectorConfig config{.backend = Backend::kCuda};
  config.options.decimate = kDecimate;
  const auto messages = run_detector(config, {image});
  if (messages.size() != 1) return;
  VERIFY(messages.front().truncated);
  VERIFY(messages.front().detections.empty() && messages.front().dropped == 0);
}

// Detections carry their image's capture time, frame identity and device sequence.
void check_detector_provenance() {
  constexpr std::chrono::nanoseconds kCaptured = std::chrono::milliseconds(250);
  constexpr std::uint64_t kFrameId = 42;
  constexpr std::uint64_t kSequence = 17;
  auto h = holoscan::testing::create_operator_test<AprilTagDetectorOp>(
      AprilTagDetectorConfig{.backend = Backend::kCpu});
  holoscan::testing::TestSample<holoscan::schema::ImageT> sample{
      .data = scene_image(make_scene("tag36h11", 0, kWidth, kHeight),
                          holoscan::schema::ImageEncoding_MONO8),
      .capture_coordinate = kCaptured};
  sample.metadata.frame_id = kFrameId;
  sample.data.header = std::make_shared<holoscan::schema::HeaderT>();
  sample.data.header->device_sequence = kSequence;
  h.push_samples<holoscan::schema::ImageT>("image", {std::move(sample)});
  h.capture<schema::AprilTagDetectionsT>("detections");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto samples = result.samples<schema::AprilTagDetectionsT>("detections");
  VERIFY(samples.size() == 1);
  if (samples.size() != 1) return;
  const holoscan::SampleMetadata& metadata = samples.front().metadata;
  VERIFY(metadata.source_clock_id.valid() && metadata.capture_timestamp_ns == kCaptured.count());
  VERIFY_MSG(metadata.frame_id == kFrameId,
             "frame id " + std::to_string(metadata.frame_id) + " forwarded");
  VERIFY_MSG(
      samples.front().data.device_sequence == kSequence,
      "device sequence " + std::to_string(samples.front().data.device_sequence) + " forwarded");
}

// How long the unrelated stream below is held.
constexpr std::chrono::seconds kHoldFor{5};

// With host input the runtime gives the detector no CUDA stream. Its CUDA work
// must then not go to the legacy default stream, which waits for every other
// blocking stream in the process: here, one held by a host function while the
// detector handles its second frame (the first allocates, which synchronizes
// the device anyway).
void check_detector_stream_with_host_input() {
  using holoscan::testing::frame;
  using holoscan::testing::input;
  using std::chrono::milliseconds;
  auto h = holoscan::testing::create_operator_test<AprilTagDetectorOp>(
      AprilTagDetectorConfig{.backend = Backend::kCuda});
  const SyntheticScene scene = make_scene("tag36h11", 0, kWidth, kHeight);
  for (int i = 0; i < 2; ++i) {
    h.push_frame(frame(input("image", scene_image(scene, holoscan::schema::ImageEncoding_RGB8))));
  }
  h.capture<schema::AprilTagDetectionsT>("detections");
  auto running = h.run_async({.frames = holoscan::testing::FrameRelease::kManual});
  VERIFY(running.release_frame(milliseconds(30000)).has_value());
  VERIFY(running.wait_for_output("detections", 1, milliseconds(30000)));

  struct Hold {
    std::atomic<bool> released{false};
  } hold;
  cudaStream_t other = nullptr;
  VERIFY(cudaStreamCreate(&other) == cudaSuccess);
  VERIFY(cudaLaunchHostFunc(
             other,
             [](void* data) {
               auto* held = static_cast<Hold*>(data);
               const auto until = std::chrono::steady_clock::now() + kHoldFor;
               while (!held->released.load() && std::chrono::steady_clock::now() < until) {
                 std::this_thread::sleep_for(milliseconds(1));
               }
             },
             &hold) == cudaSuccess);
  const auto start = std::chrono::steady_clock::now();
  VERIFY(running.release_frame(milliseconds(30000)).has_value());
  VERIFY(running.wait_for_output("detections", 2, milliseconds(30000)));
  const auto elapsed = std::chrono::steady_clock::now() - start;
  hold.released = true;
  cudaStreamSynchronize(other);
  cudaStreamDestroy(other);

  const auto result = running.finish();
  VERIFY_MSG(result.ok(), result.describe_failure());
  VERIFY_MSG(elapsed < kHoldFor / 2,
             "the detector waited " +
                 std::to_string(std::chrono::duration_cast<milliseconds>(elapsed).count()) +
                 " ms for an unrelated stream");
  if (result.ok()) {
    for (const auto& message : result.values<schema::AprilTagDetectionsT>("detections")) {
      check_message(message, scene, false);
    }
  }
}

void check_rejects_unsupported_encoding() {
  auto h = holoscan::testing::create_operator_test<AprilTagDetectorOp>(
      AprilTagDetectorConfig{.backend = Backend::kCpu});
  holoscan::schema::ImageT image =
      scene_image(make_scene("tag36h11", 0, kWidth, kHeight), holoscan::schema::ImageEncoding_RGB8);
  image.encoding = holoscan::schema::ImageEncoding_YUYV;
  h.push("image", image);
  h.capture<schema::AprilTagDetectionsT>("detections");
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), "a YUYV image is rejected");
  VERIFY(result.describe_failure().find("unsupported image encoding") != std::string::npos);
}

void check_rejects_bad_config() {
  VERIFY(test::throws<std::invalid_argument>([] {
    AprilTagDetectorConfig config;
    config.options.family = "tag99h99";
    AprilTagDetectorOp op(config);
  }));
  VERIFY(test::throws<std::invalid_argument>([] {
    AprilTagDetectorOp op(AprilTagDetectorConfig{.intrinsics = CameraIntrinsics{900, 900, 320, 240},
                                                 .tag_size = 0.0});
  }));
  VERIFY(test::throws<std::invalid_argument>([] {
    AprilTagDetectorConfig config;
    config.options.decimate = 0;
    AprilTagDetectorOp op(config);
  }));
}

#if HOLOSCAN_APRILTAG_WITH_OVERLAY
// One tag, id 7, in a 200x100 image; with a pose when `pose` is set.
schema::AprilTagDetectionsT one_tag_message(std::uint64_t frame, bool pose) {
  schema::AprilTagDetectionsT message;
  message.frame = frame;
  message.image_width = 200;
  message.image_height = 100;
  message.backend = "cuda";
  message.family = "tag36h11";
  message.detect_time_us = 1500.0F;
  const std::array<schema::Point2f, 4> corners{schema::Point2f(20, 80), schema::Point2f(80, 80),
                                               schema::Point2f(80, 20), schema::Point2f(20, 20)};
  std::array<float, 9> rotation{};
  std::array<float, 3> translation{};
  if (pose) {
    rotation = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    translation = {0.25F, -0.5F, 2.0F};
  }
  message.detections.push_back(schema::TagDetection(
      7, 0, 50.0F, schema::Point2f(50, 50), ::flatbuffers::span<const schema::Point2f, 4>(corners),
      pose, ::flatbuffers::span<const float, 9>(rotation),
      ::flatbuffers::span<const float, 3>(translation), 0.0F));
  return message;
}

// One tag message through the overlay; its axes image and labels, or nothing if it failed.
struct OverlayOutput {
  holoscan::testing::SchemaSnapshot<holoscan::schema::ImageT> axes;
  holoscan::testing::TensorSnapshot labels;
};
std::optional<OverlayOutput> run_overlay(
    const AprilTagOverlayConfig& config,
    holoscan::testing::TestSample<schema::AprilTagDetectionsT> sample) {
  auto h = holoscan::testing::create_operator_test<AprilTagOverlayOp>(config);
  h.push_samples<schema::AprilTagDetectionsT>("detections", {std::move(sample)});
  h.capture_snapshot<holoscan::schema::ImageT>("axes");
  h.capture_snapshot<holoscan::Tensor>("labels");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return std::nullopt;
  return OverlayOutput{result.snapshot<holoscan::schema::ImageT>("axes"),
                       result.snapshot<holoscan::Tensor>("labels")};
}

// The RGBA pixel at (x, y) of a 200x100 axes image.
std::array<std::uint8_t, 4> pixel(const holoscan::testing::TensorSnapshot& image, int x, int y) {
  const auto* p = reinterpret_cast<const std::uint8_t*>(image.bytes().data()) + (y * 200 + x) * 4;
  return {p[0], p[1], p[2], p[3]};
}

void check_overlay() {
  // The frame's provenance, which the overlay's layers must carry. (The harness
  // activates the overlay at the capture time, so the frame id is what tells
  // forwarded provenance from a fresh stamp.)
  constexpr std::chrono::nanoseconds kCaptured = std::chrono::milliseconds(1234);
  constexpr std::uint64_t kFrameId = 9;
  holoscan::testing::TestSample<schema::AprilTagDetectionsT> sample{
      .data = one_tag_message(0, false), .capture_coordinate = kCaptured};
  sample.metadata.frame_id = kFrameId;
  const auto out = run_overlay({.width = 200, .height = 100}, std::move(sample));
  if (!out) return;
  for (const auto* metadata : {&out->axes.sample_metadata(), &out->labels.sample_metadata()}) {
    VERIFY_MSG(metadata->source_clock_id.valid() &&
                   metadata->capture_timestamp_ns == kCaptured.count() &&
                   metadata->frame_id == kFrameId,
               "overlay layers carry the frame's provenance, not capture " +
                   std::to_string(metadata->capture_timestamp_ns) + " ns, frame " +
                   std::to_string(metadata->frame_id));
  }
  VERIFY(out->axes.metadata().encoding == holoscan::schema::ImageEncoding_RGBA8);
  const auto& image = out->axes.tensor(holoscan::schema::ImageTensorFields::data);
  VERIFY(image.shape() == (std::vector<std::int64_t>{100, 200, 4}));
  // Without a pose: X from the center (50, 50) toward (80, 50), Y toward (50, 80).
  VERIFY((pixel(image, 65, 50) == std::array<std::uint8_t, 4>{255, 0, 0, 255}));
  VERIFY((pixel(image, 50, 65) == std::array<std::uint8_t, 4>{0, 255, 0, 255}));
  VERIFY((pixel(image, 10, 10) == std::array<std::uint8_t, 4>{0, 0, 0, 0}));

  VERIFY(out->labels.shape() ==
         (std::vector<std::int64_t>{
             static_cast<std::int64_t>(kMaxDetectionsPerFrame),
             holoscan::visualization::kTextRunHeaderBytes + kOverlayLabelBytes}));
  const auto* first_text = reinterpret_cast<const char*>(out->labels.bytes().data()) +
                           holoscan::visualization::kTextRunHeaderBytes;
  VERIFY(std::string(first_text, 4) == "id 7");
}

// With a pose and intrinsics the Z axis is drawn too, in blue.
void check_overlay_pose() {
  const auto out = run_overlay({.width = 200,
                                .height = 100,
                                .intrinsics = CameraIntrinsics{400, 400, 50, 150},
                                .tag_size = 0.2},
                               {.data = one_tag_message(0, true)});
  if (!out) return;
  const auto& image = out->axes.tensor(holoscan::schema::ImageTensorFields::data);
  VERIFY((pixel(image, 110, 50) == std::array<std::uint8_t, 4>{255, 0, 0, 255}));
  VERIFY((pixel(image, 100, 62) == std::array<std::uint8_t, 4>{0, 255, 0, 255}));
  VERIFY((pixel(image, 97, 54) == std::array<std::uint8_t, 4>{0, 0, 255, 255}));
}

// A frame without tags clears the axes: a fully transparent image.
void check_overlay_empty() {
  schema::AprilTagDetectionsT message = one_tag_message(0, false);
  message.detections.clear();
  const auto out = run_overlay({.width = 200, .height = 100}, {.data = message});
  if (!out) return;
  const auto& bytes = out->axes.tensor(holoscan::schema::ImageTensorFields::data).bytes();
  VERIFY(bytes.size() == 200U * 100U * 4U);
  VERIFY(std::all_of(bytes.begin(), bytes.end(), [](std::byte b) { return b == std::byte{0}; }));
}

// Detections without a capture time still reach Holoviz, which selects layers
// by capture time: the overlay stamps them itself.
void check_overlay_unstamped() {
  auto h = holoscan::testing::create_operator_test<AprilTagOverlayOp>(
      AprilTagOverlayConfig{.width = 200, .height = 100});
  h.push("detections", one_tag_message(0, false));
  h.capture_snapshot<holoscan::schema::ImageT>("axes");
  h.capture_snapshot<holoscan::Tensor>("labels");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  VERIFY_MSG(
      result.snapshot<holoscan::schema::ImageT>("axes").sample_metadata().source_clock_id.valid(),
      "unstamped detections give stamped layers");
}
#endif

}  // namespace
}  // namespace holoscan::apriltag

int main() {
  using holoscan::apriltag::Backend;
  using holoscan::apriltag::test::TestCase;
  std::vector<TestCase> cases;
  for (Backend run : holoscan::apriltag::available_backends()) {
    const std::string name(holoscan::apriltag::backend_name(run));
    for (const auto encoding :
         {holoscan::schema::ImageEncoding_MONO8, holoscan::schema::ImageEncoding_RGB8,
          holoscan::schema::ImageEncoding_BGR8, holoscan::schema::ImageEncoding_RGBA8,
          holoscan::schema::ImageEncoding_BGRA8}) {
      cases.push_back({"detector " + name + " " + holoscan::schema::EnumNameImageEncoding(encoding),
                       [=] { holoscan::apriltag::check_detector(run, encoding); }});
    }
    cases.push_back({"detector " + name + " MONO8 as [height, width]", [=] {
                       holoscan::apriltag::check_detector(run,
                                                          holoscan::schema::ImageEncoding_MONO8,
                                                          /*rank2=*/true);
                     }});
    cases.push_back({"detector " + name + " image size change",
                     [=] { holoscan::apriltag::check_size_change(run); }});
  }
  for (Backend backend : {Backend::kCpu, Backend::kCuda}) {
    cases.push_back({"detector " + std::string(holoscan::apriltag::backend_name(backend)) +
                         " non-default options",
                     [=] { holoscan::apriltag::check_options(backend); }});
  }
  cases.push_back({"detector rejects unsupported encodings",
                   holoscan::apriltag::check_rejects_unsupported_encoding});
  cases.push_back(
      {"detector rejects bad configuration", holoscan::apriltag::check_rejects_bad_config});
  cases.push_back(
      {"detector reports a truncated frame", holoscan::apriltag::check_detector_truncated});
  cases.push_back({"detector forwards provenance", holoscan::apriltag::check_detector_provenance});
  cases.push_back({"detector keeps off the legacy default stream with host input",
                   holoscan::apriltag::check_detector_stream_with_host_input});
#if HOLOSCAN_APRILTAG_WITH_OVERLAY
  cases.push_back({"overlay layers", holoscan::apriltag::check_overlay});
  cases.push_back({"overlay axes with a pose", holoscan::apriltag::check_overlay_pose});
  cases.push_back({"overlay clears axes without tags", holoscan::apriltag::check_overlay_empty});
  cases.push_back(
      {"overlay stamps unstamped detections", holoscan::apriltag::check_overlay_unstamped});
#endif
  return holoscan::apriltag::test::run(cases);
}
