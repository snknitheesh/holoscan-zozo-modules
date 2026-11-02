// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// FoundationStereoOp on its own, through the SDK's operator test harness, with
// the stub model, so through TensorRT without downloaded weights: each frame's
// disparity against the CPU reference of the whole chain (resize, normalize,
// the stub, resize back and rescale, invalidate), for every pixel format, in
// GPU, host and pinned host memory, smaller and larger than the model, with
// FP32 and FP16 engines; the depth against fx * B / d; the left image's
// capture time and frame id on the outputs; per-frame geometry; and images,
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
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/foundation_stereo/foundation_stereo_op.hpp>
#include <holoscan/foundation_stereo/reference.hpp>
#include <holoscan/foundation_stereo/stub_model.hpp>
#include <holoscan/foundation_stereo/synthetic_scene.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "stereo_inputs.hpp"
#include "test_support.hpp"

namespace holoscan::foundation_stereo {
namespace {

namespace ht = holoscan::testing;
using holoscan::schema::ImageEncoding;
using test::capture_time;
using test::Eye;
using test::format_of;
using test::Memory;
using test::scene_pair;
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

// Two frames of the scene through the stub, with depth: each output against
// the reference, with the left image's capture time and frame id.
void check_chain(ImageEncoding encoding, Memory memory, int width, int height, bool fp16 = false,
                 bool invalidate = true, int mw = kModelWidth, int mh = kModelHeight) {
  const std::string what = std::string(holoscan::schema::EnumNameImageEncoding(encoding)) + " " +
                           std::to_string(width) + "x" + std::to_string(height) +
                           (memory == Memory::kDevice   ? " device"
                            : memory == Memory::kPinned ? " pinned"
                                                        : " host") +
                           (fp16 ? " fp16" : "");
  FoundationStereoConfig config = stub_config(fp16, mw, mh);
  config.invalidate_out_of_view = invalidate;
  const SyntheticScene first = make_scene(0, width, height);
  config.geometry = scene_geometry(first);
  config.depth = true;
  config.stats = std::make_shared<FoundationStereoStats>();
  auto h = ht::create_operator_test<FoundationStereoOp>(config);
  std::array<std::vector<ht::TestSample<holoscan::schema::ImageT>>, 2> samples;
  std::vector<std::vector<double>> expected;
  for (std::uint32_t frame = 0; frame < 2; ++frame) {
    auto pair = scene_pair(make_scene(frame * 5U, width, height), encoding, memory);
    expected.push_back(expected_raw(pair, mw, mh));
    for (int side = 0; side < 2; ++side) {
      auto sample = ht::sample_at(std::move(pair[side].image), capture_time(frame));
      sample.metadata.frame_id = 7U + frame;
      samples[side].push_back(std::move(sample));
    }
  }
  h.push_samples("left", std::move(samples[0]));
  h.push_samples("right", std::move(samples[1]));
  if (memory == Memory::kDevice) {
    h.input_source_device("left", holoscan::DeviceId{0});
    h.input_source_device("right", holoscan::DeviceId{0});
  }
  for (const char* port : {"disparity", "depth"}) {
    h.output_device(port, holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::schema::ImageT>(port, kCaptures);
  }
  const auto result = h.run();
  VERIFY_MSG(result.ok(), what + ": " + result.describe_failure());
  if (!result.ok()) return;
  const auto disparities = result.snapshots<holoscan::schema::ImageT>("disparity");
  const auto depths = result.snapshots<holoscan::schema::ImageT>("depth");
  VERIFY_MSG(disparities.size() == 2 && depths.size() == 2, what + ": two frames out");
  VERIFY(config.stats->frames == 2 && config.stats->model_width == mw &&
         config.stats->model_height == mh);
  VERIFY_MSG(config.stats->timed_frames == 2 && config.stats->gpu_ns_total > 0,
             what + ": both frames timed");
  const float fb = static_cast<float>(config.geometry->fx * config.geometry->baseline);
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

// The geometry with every frame: frame 1's has twice the baseline, so twice
// the depth; a frame whose geometry has another capture time gives nothing.
void check_geometry_input() {
  FoundationStereoConfig config = stub_config();
  config.geometry_input = true;
  config.depth = true;
  auto h = ht::create_operator_test<FoundationStereoOp>(config);
  const int width = 320;
  const int height = 240;
  const StereoGeometry geometry = scene_geometry(make_scene(0, width, height));
  for (std::uint32_t frame = 0; frame < 3; ++frame) {
    auto pair = scene_pair(make_scene(frame, width, height), holoscan::schema::ImageEncoding_RGB8,
                           Memory::kHost);
    StereoGeometry g = geometry;
    if (frame == 1) g.baseline *= 2.0;
    const auto geometry_time = capture_time(frame) + std::chrono::milliseconds(frame == 2 ? 1 : 0);
    h.push_frame(ht::frame_at(capture_time(frame), ht::input("left", std::move(pair[0].image)),
                              ht::input("right", std::move(pair[1].image))));
    h.push_frame(ht::frame_at(geometry_time, ht::input("geometry", to_message(g))));
  }
  for (const char* port : {"disparity", "depth"}) {
    h.output_device(port, holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::schema::ImageT>(port, kCaptures);
  }
  const auto result = h.run();
  // Frame 2 never completes; the run may end by noticing that.
  const auto disparities = result.snapshots<holoscan::schema::ImageT>("disparity");
  const auto depths = result.snapshots<holoscan::schema::ImageT>("depth");
  VERIFY_MSG(disparities.size() == 2 && depths.size() == 2,
             "frames 0 and 1 out, frame 2 not: " + std::to_string(disparities.size()));
  for (std::size_t i = 0; i < disparities.size() && i < depths.size(); ++i) {
    const auto captured = disparities[i].sample_metadata().capture_timestamp_ns;
    const bool second = captured == capture_time(1).count();
    VERIFY(captured == capture_time(0).count() || second);
    const auto fb = static_cast<float>(geometry.fx * geometry.baseline * (second ? 2.0 : 1.0));
    const auto d =
        values_of<float>(disparities[i].tensor(holoscan::schema::ImageTensorFields::data));
    const auto z = values_of<float>(depths[i].tensor(holoscan::schema::ImageTensorFields::data));
    std::size_t wrong = 0;
    for (std::size_t p = 0; p < d.size() && p < z.size(); ++p) {
      wrong += z[p] != depth_from_disparity(d[p], fb, 0.0F);
    }
    VERIFY_MSG(wrong == 0, "frame " + std::to_string(i) + " uses its own geometry");
  }
}

// A run of one pair, edited by `edit`, fails, saying `message`.
template <typename Edit>
void check_run_fails(const std::string& message, FoundationStereoConfig config, Edit edit,
                     std::optional<StereoGeometry> geometry = std::nullopt) {
  auto h = ht::create_operator_test<FoundationStereoOp>(config);
  auto pair =
      scene_pair(make_scene(0, 320, 240), holoscan::schema::ImageEncoding_RGB8, Memory::kHost);
  edit(pair);
  if (geometry) {
    h.push_frame(ht::frame_at(capture_time(0), ht::input("left", std::move(pair[0].image)),
                              ht::input("right", std::move(pair[1].image)),
                              ht::input("geometry", to_message(*geometry))));
  } else {
    h.push_frame(ht::frame_at(capture_time(0), ht::input("left", std::move(pair[0].image)),
                              ht::input("right", std::move(pair[1].image))));
  }
  h.output_device("disparity", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("disparity", 1);
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), "expected a failure: " + message);
  VERIFY_MSG(result.describe_failure().find(message) != std::string::npos,
             "expected '" + message + "' in: " + result.describe_failure());
}

void check_rejects_inputs() {
  const auto none = [](std::array<Eye, 2>&) {};
  check_run_fails("unsupported left image encoding MONO16", stub_config(),
                  [](std::array<Eye, 2>& pair) {
                    pair[0].image.encoding = holoscan::schema::ImageEncoding_MONO16;
                  });
  check_run_fails(
      "the left image is 320x240 and the right 160x120", stub_config(),
      [](std::array<Eye, 2>& pair) {
        pair[1] = std::move(scene_pair(make_scene(0, 160, 120),
                                       holoscan::schema::ImageEncoding_RGB8, Memory::kHost)[1]);
      });
  FoundationStereoConfig small = stub_config();
  small.max_width = 200;
  check_run_fails("exceeds the configured 200x720", small, none);
  FoundationStereoConfig per_frame = stub_config();
  per_frame.geometry_input = true;
  StereoGeometry wrong_size = scene_geometry(make_scene(0, 640, 480));
  check_run_fails("the geometry is for 640x480 images", per_frame, none, wrong_size);
  StereoGeometry no_baseline = scene_geometry(make_scene(0, 320, 240));
  no_baseline.baseline = 0;
  check_run_fails("baseline must be finite and positive", per_frame, none, no_baseline);
}

void check_rejects_models() {
  const auto none = [](std::array<Eye, 2>&) {};
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

void check_rejects_settings() {
  const auto rejects = [](auto edit) {
    FoundationStereoConfig config{.model = "model.onnx"};
    edit(config);
    return test::throws<std::invalid_argument>([&] { FoundationStereoOp op(config); });
  };
  const StereoGeometry geometry{.fx = 100, .baseline = 0.1, .width = 10, .height = 10};
  VERIFY(rejects([](auto& c) { c.model.clear(); }));
  VERIFY(rejects([](auto& c) { c.depth = true; }));
  VERIFY(rejects([&](auto& c) {
    c.geometry = geometry;
    c.geometry_input = true;
  }));
  VERIFY(rejects([&](auto& c) {
    c.geometry = geometry;
    c.geometry->fx = -1;
  }));
  VERIFY(rejects([](auto& c) { c.max_width = 0; }));
  VERIFY(rejects([](auto& c) { c.max_height = 20000; }));
  VERIFY(rejects([](auto& c) { c.build_workspace_bytes = 1024; }));
  VERIFY(!rejects([&](auto& c) {
    c.geometry = geometry;
    c.depth = true;
  }));
  VERIFY(!rejects([](auto& c) {
    c.geometry_input = true;
    c.depth = true;
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
      {"RGB8 from host memory, 640x480",
       chain(holoscan::schema::ImageEncoding_RGB8, Memory::kHost, 640, 480)},
      {"BGR8 from pinned memory, odd size",
       chain(holoscan::schema::ImageEncoding_BGR8, Memory::kPinned, 333, 251)},
      {"RGBA8 from the GPU, smaller than the model",
       chain(holoscan::schema::ImageEncoding_RGBA8, Memory::kDevice, 200, 150, false, true,
             fs::kLargeModelWidth, fs::kLargeModelHeight)},
      {"BGRA8 from host memory, out-of-view pixels kept",
       chain(holoscan::schema::ImageEncoding_BGRA8, Memory::kHost, 640, 480, false, false)},
      {"RGB8 with an FP16 engine",
       chain(holoscan::schema::ImageEncoding_RGB8, Memory::kDevice, 640, 480, true)},
      {"geometry input", fs::check_geometry_input},
      {"rejects unusable inputs", fs::check_rejects_inputs},
      {"rejects unusable models", fs::check_rejects_models},
      {"rejects unusable settings", fs::check_rejects_settings},
  });
}
