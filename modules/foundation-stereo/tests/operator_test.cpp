// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// FoundationStereoOp on its own, through the SDK's operator test harness, with
// the stub model, so through TensorRT without downloaded weights: each frame's
// disparity against the CPU reference of the whole chain (resize, normalize,
// the stub, resize back and rescale, invalidate), for every pixel format, in
// GPU memory and pinned, pageable and imported pinned host memory, smaller
// and larger than the model, with
// FP32 and FP16 engines; the depth against fx * B / d and its camera info;
// the left image's capture time and frame id on the outputs; the camera info
// input, at its own rate and with every frame; and images, camera infos,
// settings and models it must reject.
//
// Usage: foundation_stereo_operator_test DIR (a folder for the stub models and engines)

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/foundation_stereo/camera_info.hpp>
#include <holoscan/foundation_stereo/depth.hpp>
#include <holoscan/foundation_stereo/foundation_stereo_op.hpp>
#include <holoscan/foundation_stereo/reference.hpp>
#include <holoscan/foundation_stereo/stub_model.hpp>
#include <holoscan/foundation_stereo/synthetic_scene.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/messages/camera_info.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/messages/synthetic_calibration.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "stereo_inputs.hpp"
#include "test_support.hpp"

namespace holoscan::foundation_stereo {
namespace {

namespace ht = holoscan::testing;
namespace hm = holoscan::messages;
using holoscan::schema::ImageEncoding;
using test::capture_time;
using test::Eye;
using test::format_of;
using test::Memory;
using test::scene_info;
using test::scene_pair;
using test::stereo_frame;
using test::values_of;

std::filesystem::path g_directory;

// The stub model's usual input size: not the images', so both resizes show.
constexpr int kModelWidth = 160;
constexpr int kModelHeight = 96;
// A model larger than some images, which it then upscales.
constexpr int kLargeModelWidth = 416;
constexpr int kLargeModelHeight = 256;
constexpr std::size_t kCaptures = 8;

// The operator's disparity for a pair, by the CPU references, before
// invalidation.
std::vector<double> expected_raw(const std::array<Eye, 2>& pair, int mw, int mh) {
  const int width = pair[0].image.width;
  const int height = pair[0].image.height;
  const PixelFormat format = format_of(pair[0].image.encoding);
  const std::size_t pitch = static_cast<std::size_t>(width) * channels(format);
  const auto left =
      reference::preprocess(pair[0].pixels.data(), pitch, width, height, format, mw, mh);
  const auto right =
      reference::preprocess(pair[1].pixels.data(), pitch, width, height, format, mw, mh);
  return reference::rescale(reference::stub(left, right, mw, mh), mw, mh, width, height);
}

FoundationStereoConfig stub_config(bool fp16 = false, int mw = kModelWidth, int mh = kModelHeight) {
  FoundationStereoConfig config;
  config.model = write_stub_model(g_directory, mw, mh);
  config.fp16 = fp16;
  config.max_width = 1280;
  config.max_height = 720;
  return config;
}

// Compares a disparity with the reference, pixel by pixel: within
// `tolerance`, relative to the larger of 1 and the value.
void check_disparity(const std::vector<float>& disparity, const std::vector<double>& raw, int width,
                     double tolerance, bool invalidate, const std::string& what) {
  std::size_t wrong = 0;
  std::size_t valid = 0;
  double worst = 0;
  for (std::size_t i = 0; i < raw.size() && i < disparity.size(); ++i) {
    const int x = static_cast<int>(i % static_cast<std::size_t>(width));
    const double limit = tolerance * std::max(1.0, std::abs(raw[i]));
    if (reference::near_limit(raw[i], x, limit)) continue;
    const double expected = reference::valid(raw[i], x, invalidate);
    const double error = std::abs(disparity[i] - expected);
    worst = std::max(worst, error);
    wrong += error > limit;
    valid += disparity[i] > 0;
  }
  VERIFY_MSG(disparity.size() == raw.size(), what + ": output size");
  VERIFY_MSG(wrong == 0, what + ": " + std::to_string(wrong) + " pixels off the reference, worst " +
                             std::to_string(worst) + " px");
  VERIFY_MSG(valid > raw.size() / 4, what + ": too few valid pixels");
}

// The depth's camera info, one per frame, before its depth: the rectified
// left camera at the images' size, with the frame's capture time.
void check_depth_camera_info(const std::vector<holoscan::Sample<hm::schema::CameraInfoT>>& infos,
                             const hm::StereoCameraInfo& pair, std::size_t frames,
                             const std::string& what) {
  VERIFY_MSG(infos.size() == frames, what + ": a depth camera info with every frame");
  for (std::size_t frame = 0; frame < infos.size(); ++frame) {
    const std::int64_t time = capture_time(static_cast<std::uint32_t>(frame)).count();
    const hm::CameraInfo info = hm::from_message(infos[frame].data);
    VERIFY_MSG(info.header.capture_timestamp_ns == time &&
                   infos[frame].metadata.capture_timestamp_ns == time,
               what + ": the depth camera info carries its frame's capture time");
    VERIFY_MSG(info.header.frame_id == "left" && info.state == hm::CameraState::kRectified &&
                   info.distortion_model == hm::DistortionModel::kNone,
               what + ": the depth camera info is the rectified left camera");
    VERIFY_MSG(info.width == pair.left.width && info.height == pair.left.height &&
                   info.fx == pair.left.fx && info.fy == pair.left.fy && info.cx == pair.left.cx &&
                   info.cy == pair.left.cy,
               what + ": the depth camera info is the left camera at the images' size");
  }
}

// Two frames of the scene through the stub, with depth: each output against
// the reference, with the left image's capture time and frame id.
void check_chain(ImageEncoding encoding, Memory memory, int width, int height, bool fp16 = false,
                 bool invalidate = true, int mw = kModelWidth, int mh = kModelHeight) {
  const std::string what = std::string(holoscan::schema::EnumNameImageEncoding(encoding)) + " " +
                           std::to_string(width) + "x" + std::to_string(height) +
                           (memory == Memory::kDevice     ? " device"
                            : memory == Memory::kPinned   ? " imported pinned"
                            : memory == Memory::kPageable ? " pageable"
                                                          : " pinned pool") +
                           (fp16 ? " fp16" : "");
  FoundationStereoConfig config = stub_config(fp16, mw, mh);
  config.invalidate_out_of_view = invalidate;
  const hm::StereoCameraInfo info = scene_camera_info(make_scene(0, width, height));
  config.depth = true;
  config.depth_camera_info = true;
  config.stats = std::make_shared<FoundationStereoStats>();
  auto h = ht::create_operator_test<FoundationStereoOp>(config);
  std::vector<std::vector<double>> expected;
  for (std::uint32_t frame = 0; frame < 2; ++frame) {
    auto pair = scene_pair(make_scene(frame * 5U, width, height), encoding, memory);
    expected.push_back(expected_raw(pair, mw, mh));
    h.push_frame(
        stereo_frame(capture_time(frame), std::move(pair), hm::to_message(info), 7U + frame));
  }
  if (memory == Memory::kDevice) {
    h.input_source_device("left", holoscan::DeviceId{0});
    h.input_source_device("right", holoscan::DeviceId{0});
  }
  for (const char* port : {"disparity", "depth"}) {
    h.output_device(port, holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::schema::ImageT>(port, kCaptures);
  }
  h.capture<hm::schema::CameraInfoT>("depth_camera_info", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), what + ": " + result.describe_failure());
  if (!result.ok()) return;
  const auto disparities = result.snapshots<holoscan::schema::ImageT>("disparity");
  const auto depths = result.snapshots<holoscan::schema::ImageT>("depth");
  VERIFY_MSG(disparities.size() == 2 && depths.size() == 2, what + ": two frames out");
  VERIFY(config.stats->frames == 2 && config.stats->model_width == mw &&
         config.stats->model_height == mh);
  // The first frame is timed on its own: it also captures the CUDA graph.
  VERIFY_MSG(config.stats->timed_frames == 1 && config.stats->gpu_ns_total > 0 &&
                 config.stats->gpu_ns_first > 0,
             what + ": both frames timed");
  check_depth_camera_info(result.samples<hm::schema::CameraInfoT>("depth_camera_info"), info, 2,
                          what);
  const auto fb = static_cast<float>(hm::fx_baseline(info));
  // FP16 holds the stub's 0..255 values to about 0.1.
  const double tolerance = fp16 ? 1e-2 : 1e-4;
  for (std::size_t frame = 0; frame < disparities.size() && frame < depths.size(); ++frame) {
    const auto& image = disparities[frame];
    VERIFY(image.metadata().encoding == holoscan::schema::ImageEncoding_MONO32F);
    VERIFY(image.metadata().width == width && image.metadata().height == height);
    const auto& tensor = image.tensor(holoscan::schema::ImageTensorFields::data);
    VERIFY(tensor.shape() == (std::vector<std::int64_t>{height, width}));
    for (const auto* snapshot : {&image, &depths[frame]}) {
      VERIFY_MSG(snapshot->sample_metadata().capture_timestamp_ns ==
                         capture_time(static_cast<std::uint32_t>(frame)).count() &&
                     snapshot->sample_metadata().frame_id == 7U + frame,
                 what + ": the outputs keep the left image's capture time and frame id");
    }
    const auto disparity = values_of<float>(tensor);
    check_disparity(disparity, expected[frame], width, tolerance, invalidate,
                    what + " frame " + std::to_string(frame));
    const auto depth =
        values_of<float>(depths[frame].tensor(holoscan::schema::ImageTensorFields::data));
    std::size_t wrong = 0;
    for (std::size_t i = 0; i < disparity.size() && i < depth.size(); ++i) {
      wrong += depth[i] != depth_from_disparity(disparity[i], fb, 0.0F);
    }
    VERIFY_MSG(wrong == 0, what + ": " + std::to_string(wrong) + " depths off fx * B / d");
  }
}

// Each output frame's disparity and depth, in frame order, and the frames
// they came from (their capture times).
struct DepthRun {
  std::vector<std::vector<float>> disparity;
  std::vector<std::vector<float>> depth;
  std::vector<std::int64_t> times;
};
DepthRun run_depth(
    hm::CameraInfoTiming timing,
    const std::function<std::optional<hm::schema::StereoCameraInfoT>(std::uint32_t)>& sent) {
  constexpr int kWidth = 320;
  constexpr int kHeight = 240;
  FoundationStereoConfig config = stub_config();
  config.camera_info_input.timing = timing;
  config.depth = true;
  auto h = ht::create_operator_test<FoundationStereoOp>(config);
  for (std::uint32_t frame = 0; frame < 4; ++frame) {
    h.push_frame(stereo_frame(capture_time(frame),
                              scene_pair(make_scene(frame, kWidth, kHeight),
                                         holoscan::schema::ImageEncoding_MONO8, Memory::kHost),
                              sent(frame)));
  }
  for (const char* port : {"disparity", "depth"}) {
    h.output_device(port, holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::schema::ImageT>(port, kCaptures);
  }
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  DepthRun run;
  if (!result.ok()) return run;
  for (const auto& image : result.snapshots<holoscan::schema::ImageT>("disparity")) {
    run.disparity.push_back(
        values_of<float>(image.tensor(holoscan::schema::ImageTensorFields::data)));
    run.times.push_back(image.sample_metadata().capture_timestamp_ns);
  }
  for (const auto& image : result.snapshots<holoscan::schema::ImageT>("depth")) {
    run.depth.push_back(values_of<float>(image.tensor(holoscan::schema::ImageTensorFields::data)));
  }
  return run;
}

// Whether every depth pixel is fx * baseline / disparity of its disparity.
bool depth_uses(const std::vector<float>& disparity, const std::vector<float>& depth, double fx,
                double baseline) {
  const auto fb = static_cast<float>(fx * baseline);
  std::size_t wrong = disparity.size() == depth.size() ? 0 : 1;
  for (std::size_t p = 0; p < disparity.size() && p < depth.size(); ++p) {
    wrong += depth[p] != depth_from_disparity(disparity[p], fb, 0.0F);
  }
  return wrong == 0;
}

// The camera info input: frames before the first camera info are skipped;
// each frame uses the newest camera info, so a calibration that changes
// (frame 2's has twice the baseline, so twice the depth) applies from its
// frame on, although frame 3 comes without one, as from CameraInfoSourceOp
// at its own rate; and with kEveryFrame, each frame waits for its own.
void check_camera_info_input() {
  const hm::StereoCameraInfo info = scene_camera_info(make_scene(0, 320, 240));
  hm::StereoCameraInfo wide = info;
  wide.baseline *= 2.0;
  wide.right_from_left.translation[0] *= 2.0;
  const std::array<double, 4> baseline{0.0, info.baseline, wide.baseline, wide.baseline};
  DepthRun run = run_depth(hm::CameraInfoTiming::kLatest, [&](std::uint32_t frame) {
    std::optional<hm::schema::StereoCameraInfoT> sent;
    if (frame == 1) sent = hm::to_message(info);
    if (frame == 2) sent = hm::to_message(wide);
    return sent;
  });
  VERIFY_MSG(run.times.size() == 3 && run.depth.size() == 3,
             std::to_string(run.times.size()) + " frames out of 4, the first without camera info");
  for (std::size_t i = 0; i < run.times.size() && i < run.depth.size(); ++i) {
    const auto frame = static_cast<std::uint32_t>(i + 1);
    VERIFY(run.times[i] == capture_time(frame).count());
    VERIFY_MSG(depth_uses(run.disparity[i], run.depth[i], info.left.fx, baseline[frame]),
               "kLatest: frame " + std::to_string(frame) + " uses the newest camera info");
  }
  run = run_depth(hm::CameraInfoTiming::kEveryFrame, [&](std::uint32_t frame) {
    return std::optional<hm::schema::StereoCameraInfoT>(hm::to_message(frame >= 2 ? wide : info));
  });
  VERIFY_MSG(run.times.size() == 4 && run.depth.size() == 4, "kEveryFrame: every frame out");
  for (std::size_t i = 0; i < run.times.size() && i < run.depth.size(); ++i) {
    VERIFY_MSG(depth_uses(run.disparity[i], run.depth[i], info.left.fx,
                          i >= 2 ? wide.baseline : info.baseline),
               "kEveryFrame: frame " + std::to_string(i) + " uses its own camera info");
  }
}

// A camera info whose calibration is the same but whose frame differs: the
// depth's camera info follows it from that frame on.
void check_depth_camera_info_frame() {
  FoundationStereoConfig config = stub_config();
  config.camera_info_input.timing = hm::CameraInfoTiming::kEveryFrame;
  config.depth = true;
  config.depth_camera_info = true;
  const std::array<std::string, 3> frames{"left", "left_moved", "left_moved"};
  auto h = ht::create_operator_test<FoundationStereoOp>(config);
  for (std::uint32_t frame = 0; frame < frames.size(); ++frame) {
    const SyntheticScene scene = make_scene(frame, 320, 240);
    hm::StereoCameraInfo info = scene_camera_info(scene);
    info.header.frame_id = frames[frame];
    info.left.header.frame_id = frames[frame];
    h.push_frame(
        stereo_frame(capture_time(frame),
                     scene_pair(scene, holoscan::schema::ImageEncoding_MONO8, Memory::kHost),
                     hm::to_message(info)));
  }
  for (const char* port : {"disparity", "depth"}) {
    h.output_device(port, holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::schema::ImageT>(port, kCaptures);
  }
  h.capture<hm::schema::CameraInfoT>("depth_camera_info", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto infos = result.samples<hm::schema::CameraInfoT>("depth_camera_info");
  VERIFY_MSG(infos.size() == frames.size(), "a depth camera info with every frame");
  for (std::size_t frame = 0; frame < infos.size() && frame < frames.size(); ++frame) {
    const std::string frame_id = hm::from_message(infos[frame].data).header.frame_id;
    VERIFY_MSG(frame_id == frames[frame], "frame " + std::to_string(frame) +
                                              ": the depth camera info is in '" + frames[frame] +
                                              "', not '" + frame_id + "'");
  }
}

// A run of one pair and its camera info, edited by `edit`, fails, saying
// `message`.
using Edit = std::function<void(std::array<Eye, 2>&, hm::schema::StereoCameraInfoT&)>;
void check_run_fails(const std::string& message, FoundationStereoConfig config, const Edit& edit) {
  auto h = ht::create_operator_test<FoundationStereoOp>(config);
  const SyntheticScene scene = make_scene(0, 320, 240);
  auto pair = scene_pair(scene, holoscan::schema::ImageEncoding_RGB8, Memory::kHost);
  hm::schema::StereoCameraInfoT info = scene_info(scene);
  edit(pair, info);
  h.push_frame(stereo_frame(capture_time(0), std::move(pair), std::move(info)));
  h.output_device("disparity", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("disparity", 1);
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), "expected a failure: " + message);
  VERIFY_MSG(result.describe_failure().find(message) != std::string::npos,
             "expected '" + message + "' in: " + result.describe_failure());
}

void check_rejects_inputs() {
  using Info = hm::schema::StereoCameraInfoT;
  const Edit none = [](std::array<Eye, 2>&, Info&) {};
  check_run_fails("unsupported left image encoding MONO16", stub_config(),
                  [](std::array<Eye, 2>& pair, Info&) {
                    pair[0].image.encoding = holoscan::schema::ImageEncoding_MONO16;
                  });
  check_run_fails(
      "the left image is 320x240 and the right 160x120", stub_config(),
      [](std::array<Eye, 2>& pair, Info&) {
        pair[1] = std::move(scene_pair(make_scene(0, 160, 120),
                                       holoscan::schema::ImageEncoding_RGB8, Memory::kHost)[1]);
      });
  FoundationStereoConfig small = stub_config();
  small.max_width = 200;
  check_run_fails("exceeds the configured 200x720", small, none);
  check_run_fails(
      "the stereo camera info is for 640x480 (left)", stub_config(),
      [](std::array<Eye, 2>&, Info& info) { info = scene_info(make_scene(0, 640, 480)); });
  // A raw pair: the images are not rectified.
  check_run_fails("must be a rectified pair", stub_config(), [](std::array<Eye, 2>&, Info& info) {
    info =
        hm::to_message(hm::synthetic_stereo_camera_info(hm::DistortionModel::kPlumbBob, 320, 240));
  });
  // Not valid: a rectified pair without a baseline.
  check_run_fails("baseline", stub_config(),
                  [](std::array<Eye, 2>&, Info& info) { info.baseline = 0.0; });
}

void check_rejects_models() {
  const Edit none = [](std::array<Eye, 2>&, hm::schema::StereoCameraInfoT&) {};
  FoundationStereoConfig missing = stub_config();
  missing.model = g_directory / "no_such_model.onnx";
  check_run_fails("cannot load", missing, none);
  // Not an engine.
  const auto garbage = g_directory / "garbage.engine";
  std::ofstream(garbage) << "not a TensorRT engine";
  FoundationStereoConfig not_engine = stub_config();
  not_engine.model = garbage;
  check_run_fails("cannot load", not_engine, none);
}

// A model replaced under the same name: its cached engine is deleted and
// built again from the new content, not loaded.
void check_replaced_model() {
  const std::filesystem::path folder = g_directory / "replaced";
  std::filesystem::create_directories(folder);
  const std::filesystem::path model = folder / "model.onnx";
  std::ofstream(model, std::ios::binary) << stub_model_onnx(kModelWidth, kModelHeight);
  FoundationStereoConfig config = stub_config();
  config.model = model;
  prepare_engine(config);
  const auto engines = [&] {
    std::vector<std::filesystem::path> found;
    for (const auto& entry : std::filesystem::directory_iterator(folder)) {
      if (entry.path().filename().string().find(".engine") != std::string::npos) {
        found.push_back(entry.path());
      }
    }
    return found;
  };
  const auto first = engines();
  VERIFY_MSG(first.size() == 1, "one engine cached beside the model");
  // Another model's bytes under the same name; the engine of the first would
  // still be named after it.
  std::ofstream(model, std::ios::binary | std::ios::trunc)
      << stub_model_onnx(kLargeModelWidth, kLargeModelHeight);
  const auto before = first.empty() ? std::filesystem::file_time_type{}
                                    : std::filesystem::last_write_time(first[0]);
  config.stats = std::make_shared<FoundationStereoStats>();
  auto h = ht::create_operator_test<FoundationStereoOp>(config);
  const SyntheticScene scene = make_scene(0, 320, 240);
  h.push_frame(stereo_frame(capture_time(0),
                            scene_pair(scene, holoscan::schema::ImageEncoding_RGB8, Memory::kHost),
                            scene_info(scene)));
  h.output_device("disparity", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("disparity", 1);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  VERIFY_MSG(config.stats->model_width == kLargeModelWidth &&
                 config.stats->model_height == kLargeModelHeight,
             "the replaced model is loaded, not the engine of the old one");
  const auto second = engines();
  VERIFY_MSG(second.size() == 1 && (first.empty() || second[0] != first[0] ||
                                    std::filesystem::last_write_time(second[0]) != before),
             "the old engine is replaced");
}

void check_rejects_settings() {
  const auto rejects = [](auto edit) {
    FoundationStereoConfig config{.model = "model.onnx"};
    edit(config);
    return test::throws<std::invalid_argument>([&] { FoundationStereoOp op(config); });
  };
  VERIFY(rejects([](auto& c) { c.model.clear(); }));
  VERIFY(rejects([](auto& c) { c.depth_camera_info = true; }));
  VERIFY(rejects([](auto& c) { c.camera_info_input.max_age = std::chrono::nanoseconds::zero(); }));
  VERIFY(rejects([](auto& c) { c.max_width = 0; }));
  VERIFY(rejects([](auto& c) { c.max_height = 20000; }));
  VERIFY(rejects([](auto& c) { c.build_workspace_bytes = 1024; }));
  VERIFY(!rejects([](auto& c) { c.depth = true; }));
  VERIFY(!rejects([](auto& c) {
    c.depth = true;
    c.depth_camera_info = true;
  }));
  VERIFY(!rejects([](auto&) {}));
}

}  // namespace
}  // namespace holoscan::foundation_stereo

int main(int argc, char** argv) {
  namespace fs = holoscan::foundation_stereo;
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " DIR\n";
    return 2;
  }
  fs::g_directory = argv[1];
  std::filesystem::create_directories(fs::g_directory);
  using fs::test::Memory;
  const auto chain = [](holoscan::schema::ImageEncoding encoding, Memory memory, int width,
                        int height, bool fp16 = false, bool invalidate = true,
                        int mw = fs::kModelWidth, int mh = fs::kModelHeight) {
    return [=] { fs::check_chain(encoding, memory, width, height, fp16, invalidate, mw, mh); };
  };
  return fs::test::run({
      {"MONO8 from the GPU, 1280x720",
       chain(holoscan::schema::ImageEncoding_MONO8, Memory::kDevice, 1280, 720)},
      {"RGB8 from pinned pool memory, 640x480",
       chain(holoscan::schema::ImageEncoding_RGB8, Memory::kHost, 640, 480)},
      {"RGB8 from pageable memory, 640x480",
       chain(holoscan::schema::ImageEncoding_RGB8, Memory::kPageable, 640, 480)},
      {"BGR8 from imported pinned memory, odd size",
       chain(holoscan::schema::ImageEncoding_BGR8, Memory::kPinned, 333, 251)},
      {"RGBA8 from the GPU, smaller than the model",
       chain(holoscan::schema::ImageEncoding_RGBA8, Memory::kDevice, 200, 150, false, true,
             fs::kLargeModelWidth, fs::kLargeModelHeight)},
      {"BGRA8 from pinned pool memory, out-of-view pixels kept",
       chain(holoscan::schema::ImageEncoding_BGRA8, Memory::kHost, 640, 480, false, false)},
      {"RGB8 with an FP16 engine",
       chain(holoscan::schema::ImageEncoding_RGB8, Memory::kDevice, 640, 480, true)},
      {"camera info input", fs::check_camera_info_input},
      {"depth camera info frame", fs::check_depth_camera_info_frame},
      {"rejects unusable inputs", fs::check_rejects_inputs},
      {"rejects unusable models", fs::check_rejects_models},
      {"replaced model", fs::check_replaced_model},
      {"rejects unusable settings", fs::check_rejects_settings},
  });
}
