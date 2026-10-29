// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The operators on their own, through the SDK's operator test harness: ArucoDetectorOp and
// CharucoDetectorOp on images of each encoding and layout, from host and device memory, on each
// backend, checked against ground truth; a change of image size; the auto backend's choice; a
// host/device copy logged once, and only when there is one; provenance; unsupported images and
// configurations rejected; and the overlay's HoloViz layers.

#include <cuda_runtime_api.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/fiducials/aruco_detector_op.hpp>
#include <holoscan/fiducials/charuco_detector_op.hpp>
#include <holoscan/fiducials/detector_backend.hpp>
#include <holoscan/fiducials/fiducial_messages.hpp>
#include <holoscan/fiducials/synthetic_scene.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#if HOLOSCAN_FIDUCIALS_WITH_OVERLAY
#include <holoscan/fiducials/fiducial_overlay_op.hpp>
#include <holoscan/visualization/holoviz_op.hpp>
#endif
#include "test_support.hpp"

namespace holoscan::fiducials {
namespace {

constexpr int kWidth = 800;
constexpr int kHeight = 600;
constexpr const char* kDictionary = "DICT_5X5_100";
// The board the ChArUco tests show.
CharucoBoardConfig test_board() {
  return {.squares_x = 7,
          .squares_y = 5,
          .square_length = 0.04,
          .marker_length = 0.03,
          .dictionary = "DICT_4X4_50"};
}

// A host tensor owning `bytes`, with `shape` and `strides` (in elements).
std::shared_ptr<holoscan::Tensor> host_tensor(std::vector<std::uint8_t> bytes,
                                              std::vector<std::int64_t> shape,
                                              std::vector<std::int64_t> strides) {
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
  storage->managed.dl_tensor = {.data = storage->bytes.data(),
                                .device = {kDLCPU, 0},
                                .ndim = static_cast<std::int32_t>(storage->shape.size()),
                                .dtype = {kDLUInt, 8, 1},
                                .shape = storage->shape.data(),
                                .strides = storage->strides.data(),
                                .byte_offset = 0};
  storage->managed.manager_ctx = storage.get();
  storage->managed.deleter = [](DLManagedTensor* value) {
    delete static_cast<Storage*>(value->manager_ctx);
  };
  return std::make_shared<holoscan::Tensor>(&storage.release()->managed);
}

// A copy of `host` in CUDA device memory on GPU 0, ready to read when this returns.
std::shared_ptr<holoscan::Tensor> device_tensor(const holoscan::Tensor& host) {
  auto copy = host.copy_to(DLDevice{kDLCUDA, 0});
  VERIFY_MSG(copy.has_value(), "copy to the device: " + (copy ? "" : copy.error().message));
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
  return copy ? std::make_shared<holoscan::Tensor>(std::move(*copy)) : nullptr;
}

// `scene` as an image of `encoding`; with `rank2`, MONO8 as [height, width].
holoscan::schema::ImageT scene_image(const SyntheticScene& scene,
                                     holoscan::schema::ImageEncoding encoding, bool rank2 = false) {
  using holoscan::schema::ImageEncoding_BGR8;
  using holoscan::schema::ImageEncoding_BGRA8;
  using holoscan::schema::ImageEncoding_MONO8;
  using holoscan::schema::ImageEncoding_RGBA8;
  const bool alpha = encoding == ImageEncoding_RGBA8 || encoding == ImageEncoding_BGRA8;
  const bool bgr = encoding == ImageEncoding_BGR8 || encoding == ImageEncoding_BGRA8;
  const int channels = encoding == ImageEncoding_MONO8 ? 1 : alpha ? 4 : 3;
  const auto gray = render_scene(scene, 1);
  const std::size_t row = static_cast<std::size_t>(scene.width) * channels;
  std::vector<std::uint8_t> pixels(row * scene.height, 255);
  for (int y = 0; y < scene.height; ++y) {
    for (int x = 0; x < scene.width; ++x) {
      const std::uint8_t g = gray[static_cast<std::size_t>(y) * scene.width + x];
      std::uint8_t* out = pixels.data() + y * row + static_cast<std::size_t>(x) * channels;
      // Tinted, so the channel order matters to the gray value; luma keeps the contrast.
      const std::uint8_t tinted[3] = {g, static_cast<std::uint8_t>(g * 0.9), g};
      for (int c = 0; c < std::min(channels, 3); ++c) out[c] = tinted[bgr ? 2 - c : c];
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

void check_markers(const schema::FiducialDetectionsT& message, const SyntheticScene& scene,
                   bool poses) {
  VERIFY(message.image_width == scene.width && message.image_height == scene.height);
  VERIFY(message.dictionary == scene.dictionary && !message.board);
  VERIFY(message.dropped == 0 && !message.truncated);
  const std::vector<Marker> found = markers_of(message);
  const TruthCounts counts = match_markers(scene, found, [poses](const MarkerMatch& match) {
    for (const double error : match.corner_errors) VERIFY(error < 0.3);
    VERIFY(match.found.pose.has_value() == poses);
    if (match.translation_error) VERIFY(*match.translation_error < 0.01);
  });
  VERIFY_MSG(counts.missed == 0 && counts.extra == 0,
             "frame " + std::to_string(message.frame) + ": missed " +
                 std::to_string(counts.missed) + ", extra " + std::to_string(counts.extra));
}

std::vector<schema::FiducialDetectionsT> run_aruco(
    const ArucoDetectorConfig& config, const std::vector<holoscan::schema::ImageT>& images,
    bool device_input = false) {
  auto h = holoscan::testing::create_operator_test<ArucoDetectorOp>(config);
  if (device_input) h.input_source_device("image", holoscan::DeviceId{0});
  for (const auto& image : images) h.push("image", image);
  h.capture<schema::FiducialDetectionsT>("detections");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return {};
  auto messages = result.values<schema::FiducialDetectionsT>("detections");
  VERIFY(messages.size() == images.size());
  return messages;
}

ArucoDetectorConfig aruco_config(Backend backend, const SyntheticScene& scene) {
  return ArucoDetectorConfig{.backend = backend,
                             .options = DetectorOptions{.dictionary = scene.dictionary},
                             .intrinsics = scene.intrinsics,
                             .marker_length = scene.markers.front().length};
}

void check_aruco(Backend backend, holoscan::schema::ImageEncoding encoding, bool rank2,
                 bool device_input) {
  std::vector<SyntheticScene> scenes;
  std::vector<holoscan::schema::ImageT> images;
  for (std::uint32_t frame = 0; frame < 3; ++frame) {
    scenes.push_back(make_marker_scene(kDictionary, frame * 10, kWidth, kHeight));
    images.push_back(scene_image(scenes.back(), encoding, rank2));
    if (device_input) images.back().data = device_tensor(*images.back().data);
  }
  const auto messages = run_aruco(aruco_config(backend, scenes[0]), images, device_input);
  for (std::size_t i = 0; i < messages.size(); ++i) {
    VERIFY(messages[i].frame == i);
    VERIFY(messages[i].backend == backend_name(backend));
    check_markers(messages[i], scenes[i], true);
  }
}

// The backend is rebuilt when the image size changes, and detects at both.
void check_size_change(Backend backend) {
  const std::array<SyntheticScene, 3> scenes{make_marker_scene(kDictionary, 0, kWidth, kHeight),
                                             make_marker_scene(kDictionary, 1, 1280, 720),
                                             make_marker_scene(kDictionary, 2, kWidth, kHeight)};
  std::vector<holoscan::schema::ImageT> images;
  for (const SyntheticScene& scene : scenes) {
    images.push_back(scene_image(scene, holoscan::schema::ImageEncoding_MONO8));
  }
  const auto messages = run_aruco(
      ArucoDetectorConfig{.backend = backend, .options = {.dictionary = kDictionary}}, images);
  for (std::size_t i = 0; i < messages.size() && i < scenes.size(); ++i) {
    check_markers(messages[i], scenes[i], false);
  }
}

void check_charuco(Backend backend, bool device_input) {
  std::vector<SyntheticScene> scenes;
  std::vector<holoscan::schema::ImageT> images;
  for (std::uint32_t frame = 0; frame < 3; ++frame) {
    scenes.push_back(make_board_scene(test_board(), frame * 12, kWidth, kHeight));
    images.push_back(scene_image(scenes.back(), holoscan::schema::ImageEncoding_RGB8));
    if (device_input) images.back().data = device_tensor(*images.back().data);
  }
  auto h = holoscan::testing::create_operator_test<CharucoDetectorOp>(CharucoDetectorConfig{
      .backend = backend, .board = test_board(), .intrinsics = scenes[0].intrinsics});
  if (device_input) h.input_source_device("image", holoscan::DeviceId{0});
  for (const auto& image : images) h.push("image", image);
  h.capture<schema::FiducialDetectionsT>("detections");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto messages = result.values<schema::FiducialDetectionsT>("detections");
  VERIFY(messages.size() == scenes.size());
  const CharucoBoard board(test_board());
  for (std::size_t i = 0; i < messages.size() && i < scenes.size(); ++i) {
    const auto& message = messages[i];
    VERIFY(message.backend == backend_name(backend) &&
           message.dictionary == test_board().dictionary);
    VERIFY(message.board != nullptr);
    if (!message.board) continue;
    const schema::CharucoViewT& view = *message.board;
    VERIFY(view.squares_x == 7 && view.squares_y == 5 && !view.legacy_pattern &&
           view.first_marker_id == 0 && std::abs(view.square_length - 0.04F) < 1e-7F &&
           std::abs(view.marker_length - 0.03F) < 1e-7F);
    VERIFY(view.corners_total == board.corner_count() &&
           view.markers_total == board.marker_count());
    VERIFY(view.markers_found == board.marker_count() &&
           message.markers.size() == static_cast<std::size_t>(board.marker_count()));
    const auto truth = board_truth(scenes[i]);
    VERIFY(view.corners.size() == truth.size());
    for (std::size_t c = 0; c < view.corners.size() && c < truth.size(); ++c) {
      const schema::CharucoCorner& corner = view.corners[c];
      VERIFY(corner.id() == truth[c].id);
      VERIFY(std::hypot(corner.image().x() - truth[c].image[0],
                        corner.image().y() - truth[c].image[1]) < 0.3);
      VERIFY(std::abs(corner.board().x() - truth[c].board[0]) < 1e-6 &&
             std::abs(corner.board().y() - truth[c].board[1]) < 1e-6 && corner.board().z() == 0.0F);
    }
    VERIFY(view.has_pose && view.pose);
    if (view.pose) {
      const Pose pose = from_message(*view.pose);
      const auto& t = scenes[i].board->pose.translation;
      VERIFY(std::hypot(pose.translation[0] - t[0], pose.translation[1] - t[1],
                        pose.translation[2] - t[2]) < 0.003 * t[2]);
      VERIFY(rotation_error_deg(pose.rotation, scenes[i].board->pose.rotation) < 0.3);
      VERIFY(pose.reprojection_error < 0.2);
    }
    VERIFY(view.coverage > 0.02F && view.coverage < 0.6F);
    // The message reads back as the observation.
    const auto observation = board_of(message);
    VERIFY(observation && observation->corners.size() == view.corners.size() &&
           observation->quality.corners_found == static_cast<int>(view.corners.size()) &&
           observation->pose.has_value());
  }
}

// A board's view is reported even with nothing of the board in sight.
void check_charuco_empty() {
  SyntheticScene scene = make_marker_scene("DICT_4X4_50", 0, kWidth, kHeight);
  scene.markers.clear();
  auto h = holoscan::testing::create_operator_test<CharucoDetectorOp>(
      CharucoDetectorConfig{.backend = Backend::kCpu, .board = test_board()});
  h.push("image", scene_image(scene, holoscan::schema::ImageEncoding_MONO8));
  h.capture<schema::FiducialDetectionsT>("detections");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto messages = result.values<schema::FiducialDetectionsT>("detections");
  VERIFY(messages.size() == 1 && messages[0].board && messages[0].board->corners.empty() &&
         !messages[0].board->has_pose && messages[0].board->markers_found == 0);
}

// Detections carry their image's capture time, frame identity and device sequence.
void check_provenance() {
  constexpr std::chrono::nanoseconds kCaptured = std::chrono::milliseconds(250);
  constexpr std::uint64_t kFrameId = 42;
  constexpr std::uint64_t kSequence = 17;
  auto h = holoscan::testing::create_operator_test<ArucoDetectorOp>(
      ArucoDetectorConfig{.backend = Backend::kCpu, .options = {.dictionary = kDictionary}});
  holoscan::testing::TestSample<holoscan::schema::ImageT> sample{
      .data = scene_image(make_marker_scene(kDictionary, 0, kWidth, kHeight),
                          holoscan::schema::ImageEncoding_MONO8),
      .capture_coordinate = kCaptured};
  sample.metadata.frame_id = kFrameId;
  sample.data.header = std::make_shared<holoscan::schema::HeaderT>();
  sample.data.header->device_sequence = kSequence;
  h.push_samples<holoscan::schema::ImageT>("image", {std::move(sample)});
  h.capture<schema::FiducialDetectionsT>("detections");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto samples = result.samples<schema::FiducialDetectionsT>("detections");
  VERIFY(samples.size() == 1);
  if (samples.size() != 1) return;
  const holoscan::SampleMetadata& metadata = samples.front().metadata;
  VERIFY(metadata.source_clock_id.valid() && metadata.capture_timestamp_ns == kCaptured.count());
  VERIFY(metadata.frame_id == kFrameId);
  VERIFY(samples.front().data.device_sequence == kSequence);
}

// What the process writes to stdout and stderr, where the SDK logs, while `body` runs.
std::string capture_output(const std::function<void()>& body) {
  std::cout.flush();
  std::cerr.flush();
  std::fflush(nullptr);
  FILE* file = std::tmpfile();
  VERIFY(file != nullptr);
  if (file == nullptr) return {};
  const int saved_out = dup(STDOUT_FILENO);
  const int saved_err = dup(STDERR_FILENO);
  dup2(fileno(file), STDOUT_FILENO);
  dup2(fileno(file), STDERR_FILENO);
  struct Restore {
    int out;
    int err;
    ~Restore() {
      std::cout.flush();
      std::cerr.flush();
      std::fflush(nullptr);
      dup2(out, STDOUT_FILENO);
      dup2(err, STDERR_FILENO);
      close(out);
      close(err);
    }
  };
  {
    const Restore restore{saved_out, saved_err};
    body();
  }
  std::string output;
  std::rewind(file);
  char buffer[4096];
  for (std::size_t n; (n = std::fread(buffer, 1, sizeof(buffer), file)) > 0;) {
    output.append(buffer, n);
  }
  std::fclose(file);
  return output;
}

std::size_t count_of(const std::string& text, std::string_view needle) {
  std::size_t count = 0;
  for (std::size_t at = text.find(needle); at != std::string::npos;
       at = text.find(needle, at + needle.size())) {
    ++count;
  }
  return count;
}

// A frame whose gray image must cross between host and device memory is logged once, naming
// the backend, the direction and the size; frames read where they are log nothing.
void check_cross_copy_logged_once(Backend backend, bool device_input, std::size_t expected) {
  holoscan::set_log_level(holoscan::LogLevel::INFO);
  const SyntheticScene scene = make_marker_scene(kDictionary, 0, kWidth, kHeight);
  std::vector<holoscan::schema::ImageT> images;
  for (int i = 0; i < 3; ++i) {
    images.push_back(scene_image(scene, holoscan::schema::ImageEncoding_RGB8));
    if (device_input) images.back().data = device_tensor(*images.back().data);
  }
  std::vector<schema::FiducialDetectionsT> messages;
  const std::string output = capture_output([&] {
    messages =
        run_aruco(ArucoDetectorConfig{.backend = backend, .options = {.dictionary = kDictionary}},
                  images, device_input);
  });
  for (const auto& message : messages) check_markers(message, scene, false);
  const std::size_t logged = count_of(output, "gray image is copied");
  VERIFY_MSG(logged == expected,
             "the copy was logged " + std::to_string(logged) + " times:\n" + output);
  if (expected == 0 || logged != 1) return;
  const std::string direction = device_input ? "copied device to host" : "copied host to device";
  const std::string size = std::to_string(kWidth * kHeight) + " bytes";
  const std::string advice = device_input ? "use the cuda backend or send host images"
                                          : "send CUDA device images or use the cpu backend";
  for (const std::string& part :
       {"ArucoDetectorOp (" + std::string(backend_name(backend)) + " backend)", direction, size,
        advice}) {
    VERIFY_MSG(output.find(part) != std::string::npos, "the copy log names '" + part + "'");
  }
}

// kAuto chooses at start, logs the choice once, and reports it; until then it reports kAuto.
void check_auto_backend() {
  holoscan::set_log_level(holoscan::LogLevel::INFO);
  VERIFY(ArucoDetectorOp({.backend = Backend::kCpu}).resolved_backend() == Backend::kCpu);
  const Backend expected = resolve_backend(Backend::kAuto);
  {
    CharucoDetectorOp op({.backend = Backend::kAuto, .board = test_board()});
    VERIFY(op.resolved_backend() == Backend::kAuto);
    const std::string output = capture_output([&] {
      op.start();
      op.start();
    });
    VERIFY(op.resolved_backend() == expected);
    VERIFY_MSG(count_of(output, "CharucoDetectorOp: backend auto chose " +
                                    std::string(backend_name(expected))) == 1,
               "the choice is logged once:\n" + output);
    op.stop();
  }
  const SyntheticScene scene = make_marker_scene(kDictionary, 0, kWidth, kHeight);
  std::vector<schema::FiducialDetectionsT> messages;
  const std::string output = capture_output([&] {
    messages = run_aruco({.backend = Backend::kAuto, .options = {.dictionary = kDictionary}},
                         {scene_image(scene, holoscan::schema::ImageEncoding_MONO8)});
  });
  VERIFY_MSG(count_of(output, "backend auto chose") == 1, "one choice logged:\n" + output);
  for (const auto& message : messages) {
    VERIFY(message.backend == backend_name(expected));
    check_markers(message, scene, false);
  }
}

void check_rejects_unsupported_encoding() {
  auto h = holoscan::testing::create_operator_test<ArucoDetectorOp>(
      ArucoDetectorConfig{.backend = Backend::kCpu});
  holoscan::schema::ImageT image = scene_image(make_marker_scene(kDictionary, 0, kWidth, kHeight),
                                               holoscan::schema::ImageEncoding_RGB8);
  image.encoding = holoscan::schema::ImageEncoding_YUYV;
  h.push("image", image);
  h.capture<schema::FiducialDetectionsT>("detections");
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), "a YUYV image is rejected");
  VERIFY(result.describe_failure().find("unsupported image encoding") != std::string::npos);
}

void check_rejects_bad_config() {
  VERIFY(test::throws<std::invalid_argument>(
      [] { ArucoDetectorOp op(ArucoDetectorConfig{.options = {.dictionary = "DICT_9X9_50"}}); }));
  VERIFY(test::throws<std::invalid_argument>([] {
    ArucoDetectorOp op(ArucoDetectorConfig{.intrinsics = CameraIntrinsics{900, 900, 320, 240, {}},
                                           .marker_length = 0.0});
  }));
  VERIFY(test::throws<std::invalid_argument>([] {
    ArucoDetectorOp op(ArucoDetectorConfig{.intrinsics = CameraIntrinsics{-1, 900, 320, 240, {}},
                                           .marker_length = 0.1});
  }));
  VERIFY(test::throws<std::invalid_argument>(
      [] { CharucoDetectorOp op(CharucoDetectorConfig{.board = {.square_length = 0.02}}); }));
  VERIFY(test::throws<std::invalid_argument>(
      [] { CharucoDetectorOp op(CharucoDetectorConfig{.charuco = {.min_markers = 0}}); }));
  // The board's dictionary is used, whatever options.dictionary says.
  VERIFY(!test::throws<std::invalid_argument>([] {
    CharucoDetectorOp op(CharucoDetectorConfig{.options = {.dictionary = "DICT_9X9_50"}});
  }));
}

#if HOLOSCAN_FIDUCIALS_WITH_OVERLAY
// Marker 7 in a 200x100 image, and with `board`, one chessboard corner; with poses when
// `pose` is set.
schema::FiducialDetectionsT overlay_message(bool board, bool pose) {
  schema::FiducialDetectionsT message;
  message.image_width = 200;
  message.image_height = 100;
  message.backend = "cuda";
  message.dictionary = "DICT_4X4_50";
  Marker marker{.id = 7, .hamming = 0, .corners = {{{20, 20}, {80, 20}, {80, 80}, {20, 80}}}};
  if (pose) {
    marker.pose = Pose{.rotation = {1, 0, 0, 0, 1, 0, 0, 0, 1}, .translation = {0.0, 0.0, 2.0}};
  }
  message.markers.push_back(to_message(marker));
  if (board) {
    auto view = std::make_unique<schema::CharucoViewT>();
    view->squares_x = 5;
    view->squares_y = 7;
    view->square_length = 0.04F;
    view->corners_total = 24;
    view->corners.emplace_back(3, schema::Point2f(150, 50), schema::Point3f(0.08F, 0.04F, 0));
    view->has_pose = pose;
    if (pose) {
      view->pose = std::make_unique<schema::Pose>(to_message(
          Pose{.rotation = {1, 0, 0, 0, 1, 0, 0, 0, 1}, .translation = {0.0, 0.0, 2.0}}));
    }
    message.board = std::move(view);
  }
  return message;
}

struct OverlayOutput {
  holoscan::testing::SchemaSnapshot<holoscan::schema::ImageT> overlay;
  holoscan::testing::TensorSnapshot labels;
};

std::optional<OverlayOutput> run_overlay(
    const FiducialOverlayConfig& config,
    holoscan::testing::TestSample<schema::FiducialDetectionsT> sample) {
  auto h = holoscan::testing::create_operator_test<FiducialOverlayOp>(config);
  h.push_samples<schema::FiducialDetectionsT>("detections", {std::move(sample)});
  h.capture_snapshot<holoscan::schema::ImageT>("overlay");
  h.capture_snapshot<holoscan::Tensor>("labels");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return std::nullopt;
  return OverlayOutput{result.snapshot<holoscan::schema::ImageT>("overlay"),
                       result.snapshot<holoscan::Tensor>("labels")};
}

std::array<std::uint8_t, 4> pixel(const holoscan::testing::TensorSnapshot& image, int x, int y) {
  const auto* p = reinterpret_cast<const std::uint8_t*>(image.bytes().data()) + (y * 200 + x) * 4;
  return {p[0], p[1], p[2], p[3]};
}

std::string label_text(const holoscan::testing::TensorSnapshot& labels, std::size_t row) {
  const auto* p = reinterpret_cast<const char*>(labels.bytes().data()) +
                  row * (holoscan::visualization::kTextRunHeaderBytes + kOverlayLabelBytes) +
                  holoscan::visualization::kTextRunHeaderBytes;
  return std::string(p, strnlen(p, kOverlayLabelBytes));
}

void check_overlay() {
  constexpr std::chrono::nanoseconds kCaptured = std::chrono::milliseconds(1234);
  constexpr std::uint64_t kFrameId = 9;
  holoscan::testing::TestSample<schema::FiducialDetectionsT> sample{
      .data = overlay_message(true, false), .capture_coordinate = kCaptured};
  sample.metadata.frame_id = kFrameId;
  const auto out = run_overlay({.width = 200, .height = 100}, std::move(sample));
  if (!out) return;
  for (const auto* metadata : {&out->overlay.sample_metadata(), &out->labels.sample_metadata()}) {
    VERIFY(metadata->source_clock_id.valid() &&
           metadata->capture_timestamp_ns == kCaptured.count() && metadata->frame_id == kFrameId);
  }
  VERIFY(out->overlay.metadata().encoding == holoscan::schema::ImageEncoding_RGBA8);
  const auto& image = out->overlay.tensor(holoscan::schema::ImageTensorFields::data);
  VERIFY(image.shape() == (std::vector<std::int64_t>{100, 200, 4}));
  VERIFY((pixel(image, 50, 80) == std::array<std::uint8_t, 4>{255, 255, 0, 255}));   // Outline.
  VERIFY((pixel(image, 23, 23) == std::array<std::uint8_t, 4>{255, 0, 0, 255}));     // Corner 0.
  VERIFY((pixel(image, 146, 50) == std::array<std::uint8_t, 4>{255, 0, 255, 255}));  // Corner.
  VERIFY((pixel(image, 50, 50) == std::array<std::uint8_t, 4>{0, 0, 0, 0}));
  VERIFY(
      out->labels.shape() ==
      (std::vector<std::int64_t>{
          kOverlayLabelRows, holoscan::visualization::kTextRunHeaderBytes + kOverlayLabelBytes}));
  VERIFY(label_text(out->labels, 0) == "id 7");
  VERIFY(label_text(out->labels, kOverlayLabelRows - 1) == "corners 1/24");
}

// With poses and intrinsics, the axes are drawn: x to the right in red, y down in green.
void check_overlay_pose() {
  const auto out = run_overlay({.width = 200,
                                .height = 100,
                                .intrinsics = CameraIntrinsics{400, 400, 100, 50, {}},
                                .marker_length = 0.4},
                               {.data = overlay_message(true, true)});
  if (!out) return;
  const auto& image = out->overlay.tensor(holoscan::schema::ImageTensorFields::data);
  // The marker's axes, 0.2 m from (100, 50): x to (140, 50), y to (100, 90).
  VERIFY((pixel(image, 120, 50) == std::array<std::uint8_t, 4>{255, 0, 0, 255}));
  VERIFY((pixel(image, 100, 85) == std::array<std::uint8_t, 4>{0, 255, 0, 255}));
}

void check_overlay_empty() {
  schema::FiducialDetectionsT message = overlay_message(false, false);
  message.markers.clear();
  const auto out = run_overlay({.width = 200, .height = 100}, {.data = std::move(message)});
  if (!out) return;
  const auto& bytes = out->overlay.tensor(holoscan::schema::ImageTensorFields::data).bytes();
  VERIFY(std::all_of(bytes.begin(), bytes.end(), [](std::byte b) { return b == std::byte{0}; }));
  VERIFY(label_text(out->labels, 0).empty() &&
         label_text(out->labels, kOverlayLabelRows - 1).empty());
}

void check_overlay_unstamped() {
  auto h = holoscan::testing::create_operator_test<FiducialOverlayOp>(
      FiducialOverlayConfig{.width = 200, .height = 100});
  h.push("detections", overlay_message(false, false));
  h.capture_snapshot<holoscan::schema::ImageT>("overlay");
  h.capture_snapshot<holoscan::Tensor>("labels");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  VERIFY(result.snapshot<holoscan::schema::ImageT>("overlay")
             .sample_metadata()
             .source_clock_id.valid());
}

void check_overlay_too_large() {
  auto h = holoscan::testing::create_operator_test<FiducialOverlayOp>(
      FiducialOverlayConfig{.width = 100, .height = 100});
  h.push("detections", overlay_message(false, false));
  h.capture_snapshot<holoscan::schema::ImageT>("overlay");
  h.capture_snapshot<holoscan::Tensor>("labels");
  VERIFY(!h.run().ok());
  VERIFY(test::throws<std::invalid_argument>(
      [] { FiducialOverlayOp op({.width = 0, .height = 10}); }));
}
#endif

}  // namespace
}  // namespace holoscan::fiducials

int main() {
  namespace f = holoscan::fiducials;
  using holoscan::fiducials::Backend;
  std::vector<f::test::TestCase> cases;
  for (const Backend backend : f::available_backends()) {
    const std::string name(f::backend_name(backend));
    for (const auto encoding :
         {holoscan::schema::ImageEncoding_MONO8, holoscan::schema::ImageEncoding_RGB8,
          holoscan::schema::ImageEncoding_BGR8, holoscan::schema::ImageEncoding_RGBA8,
          holoscan::schema::ImageEncoding_BGRA8}) {
      cases.push_back({"aruco " + name + " " + holoscan::schema::EnumNameImageEncoding(encoding),
                       [=] { f::check_aruco(backend, encoding, false, false); }});
    }
    cases.push_back({"aruco " + name + " MONO8 as [height, width]", [=] {
                       f::check_aruco(backend, holoscan::schema::ImageEncoding_MONO8, true, false);
                     }});
    cases.push_back({"aruco " + name + " device images", [=] {
                       f::check_aruco(backend, holoscan::schema::ImageEncoding_RGBA8, false, true);
                     }});
    cases.push_back(
        {"aruco " + name + " image size change", [=] { f::check_size_change(backend); }});
    cases.push_back(
        {"charuco " + name + " host images", [=] { f::check_charuco(backend, false); }});
    cases.push_back(
        {"charuco " + name + " device images", [=] { f::check_charuco(backend, true); }});
  }
  // Host images reach cpu in place and cuda through a copy; device images the reverse.
  for (const auto& [backend, device_input, logs] :
       {std::tuple{Backend::kCpu, false, 0}, std::tuple{Backend::kCuda, false, 1},
        std::tuple{Backend::kCpu, true, 1}, std::tuple{Backend::kCuda, true, 0}}) {
    const std::string name =
        std::string(f::backend_name(backend)) + (device_input ? " device" : " host");
    cases.push_back(
        {"aruco " + name + " images: copy logged " + (logs == 0 ? "never" : "once"), [=] {
           f::check_cross_copy_logged_once(backend, device_input, static_cast<std::size_t>(logs));
         }});
  }
  cases.push_back({"charuco reports an empty view", f::check_charuco_empty});
  cases.push_back({"auto backend", f::check_auto_backend});
  cases.push_back({"provenance forwarded", f::check_provenance});
  cases.push_back({"unsupported encodings rejected", f::check_rejects_unsupported_encoding});
  cases.push_back({"bad configurations rejected", f::check_rejects_bad_config});
#if HOLOSCAN_FIDUCIALS_WITH_OVERLAY
  cases.push_back({"overlay layers", f::check_overlay});
  cases.push_back({"overlay axes with poses", f::check_overlay_pose});
  cases.push_back({"overlay clears without detections", f::check_overlay_empty});
  cases.push_back({"overlay stamps unstamped detections", f::check_overlay_unstamped});
  cases.push_back({"overlay rejects images over its size", f::check_overlay_too_large});
#endif
  return f::test::run(cases);
}
