// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// StereoDisparityOp on its own, through the SDK's operator test harness, on
// every backend this platform has: the synthetic scene's disparity against
// its truth, from MONO8 and color images in host and device memory; the depth
// and confidence outputs; per-frame geometry, and frames whose geometry does
// not match their capture time; the auto backend's choice and the one-time
// copy log; and unusable inputs and configurations rejected.

#include <cuda_runtime_api.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#include <holoscan/logger/logger.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/stereo_disparity/platform.hpp>
#include <holoscan/stereo_disparity/stereo_disparity_op.hpp>
#include <holoscan/stereo_disparity/synthetic_scene.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "test_support.hpp"

namespace holoscan::stereo_disparity {
namespace {

namespace ht = holoscan::testing;
using holoscan::schema::ImageEncoding;
using test::throws;

constexpr int kWidth = 640;
constexpr int kHeight = 480;
// Outputs a test captures at most. The harness backs each captured output with
// device memory for every record it may hold, so keep it small.
constexpr std::size_t kCaptures = 8;

// The capture time of frame `frame` in the tests' frames.
std::chrono::nanoseconds capture_time(std::uint32_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

bool on_jetson() {
  const char* platform = std::getenv("HOLOSCAN_CI_PLATFORM");
  return platform != nullptr && std::string_view(platform) == "jetson";
}

// A tensor of `shape` holding `bytes`, in host or device memory.
holoscan::Tensor make_tensor(const std::vector<std::int64_t>& shape, DLDataType dtype,
                             const void* bytes, std::size_t size, bool device) {
  auto tensor = holoscan::make_tensor(
      shape, dtype, std::nullopt,
      device ? holoscan::MemoryStorageType::kDevice : holoscan::MemoryStorageType::kHost,
      std::make_shared<holoscan::UnboundedAllocator>());
  if (!tensor) throw std::runtime_error(tensor.error().message);
  auto destination = tensor->mutable_data();
  if (!destination) throw std::runtime_error(destination.error().message);
  if (cudaMemcpy(*destination, bytes, size, cudaMemcpyDefault) != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed");
  }
  return std::move(*tensor);
}

// Frame `frame`'s left and right images in `encoding` (MONO8, RGB8, BGR8,
// RGBA8 or BGRA8), in host or device memory.
std::array<holoscan::schema::ImageT, 2> scene_pair(const SyntheticScene& scene,
                                                   ImageEncoding encoding, bool device) {
  const std::size_t pixels = static_cast<std::size_t>(scene.width) * scene.height;
  const bool mono = encoding == holoscan::schema::ImageEncoding_MONO8;
  const bool bgr = encoding == holoscan::schema::ImageEncoding_BGR8 ||
                   encoding == holoscan::schema::ImageEncoding_BGRA8;
  const int channels = mono ? 1
                       : encoding == holoscan::schema::ImageEncoding_RGBA8 ||
                               encoding == holoscan::schema::ImageEncoding_BGRA8
                           ? 4
                           : 3;
  const int rendered = mono ? 1 : 3;
  std::uint8_t* gpu = nullptr;
  if (cudaMalloc(&gpu, 2 * pixels * rendered) != cudaSuccess) {
    throw std::runtime_error("cudaMalloc failed");
  }
  std::vector<std::uint8_t> rgb(2 * pixels * rendered);
  const cudaError_t status = render_scene(scene, rendered, gpu, gpu + pixels * rendered, nullptr);
  const cudaError_t copied = status == cudaSuccess
                                 ? cudaMemcpy(rgb.data(), gpu, rgb.size(), cudaMemcpyDeviceToHost)
                                 : status;
  cudaFree(gpu);
  if (copied != cudaSuccess) throw std::runtime_error("render_scene failed");

  std::array<holoscan::schema::ImageT, 2> pair;
  for (int side = 0; side < 2; ++side) {
    std::vector<std::uint8_t> bytes(pixels * channels);
    const std::uint8_t* from = rgb.data() + side * pixels * rendered;
    for (std::size_t i = 0; i < pixels; ++i) {
      for (int c = 0; c < channels; ++c) {
        const int source = c == 3 ? -1 : bgr ? 2 - c : c;
        bytes[i * channels + c] = source < 0 ? 255 : from[i * rendered + source];
      }
    }
    std::vector<std::int64_t> shape{scene.height, scene.width};
    if (!mono) shape.push_back(channels);
    pair[side].width = scene.width;
    pair[side].height = scene.height;
    pair[side].encoding = encoding;
    pair[side].data = std::make_shared<holoscan::Tensor>(
        make_tensor(shape, {kDLUInt, 8, 1}, bytes.data(), bytes.size(), device));
  }
  return pair;
}

template <typename T>
std::vector<T> values_of(const ht::TensorSnapshot& tensor) {
  std::vector<T> out(tensor.bytes().size() / sizeof(T));
  std::memcpy(out.data(), tensor.bytes().data(), out.size() * sizeof(T));
  return out;
}

// How a disparity map compares with the scene's truth, over the pixels both
// cameras see.
struct Accuracy {
  std::size_t truth{};  // Pixels both cameras see.
  double valid{};       // Share of them with a disparity.
  double within_1px{};  // Share of those within 1 px of the truth.
  double mae{};         // Their mean absolute error, pixels.
};

Accuracy accuracy(const std::vector<float>& disparity, const SyntheticScene& scene) {
  Accuracy result;
  std::size_t valid = 0;
  std::size_t within = 0;
  double error = 0.0;
  for (int v = 0; v < scene.height; ++v) {
    for (int u = 0; u < scene.width; ++u) {
      const float truth = true_disparity(scene, static_cast<float>(u), static_cast<float>(v));
      if (truth < 0.0F) continue;
      ++result.truth;
      const float d = disparity[static_cast<std::size_t>(v) * scene.width + u];
      if (d <= 0.0F) continue;
      ++valid;
      const double e = std::abs(static_cast<double>(d) - truth);
      within += e <= 1.0;
      error += e;
    }
  }
  result.valid = result.truth == 0 ? 0.0 : static_cast<double>(valid) / result.truth;
  result.within_1px = valid == 0 ? 0.0 : static_cast<double>(within) / valid;
  result.mae = valid == 0 ? 0.0 : error / valid;
  return result;
}

// What every backend must reach on the synthetic scene, measured with VPI
// 4.1.4 (cuda on an RTX 5090, ofa and pva on Jetson Orin) and set with margin.
struct Bounds {
  double valid;
  double within_1px;
  double mae;
};
Bounds bounds_for(Backend backend) {
  switch (backend) {
    case Backend::kOfa:
      return {0.90, 0.90, 0.5};
    case Backend::kPva:
      return {0.85, 0.90, 0.5};
    default:
      return {0.90, 0.95, 0.35};
  }
}

void check_accuracy(const std::vector<float>& disparity, const SyntheticScene& scene,
                    Backend backend, const std::string& what) {
  const Accuracy a = accuracy(disparity, scene);
  const Bounds b = bounds_for(backend);
  std::cout << "  " << what << ": valid=" << a.valid << " within_1px=" << a.within_1px
            << " mae_px=" << a.mae << '\n';
  VERIFY_MSG(a.truth > static_cast<std::size_t>(scene.width) * scene.height / 2,
             what + ": most pixels have a truth");
  VERIFY_MSG(a.valid >= b.valid, what + ": valid share " + std::to_string(a.valid));
  VERIFY_MSG(a.within_1px >= b.within_1px,
             what + ": share within 1 px " + std::to_string(a.within_1px));
  VERIFY_MSG(a.mae <= b.mae, what + ": mean error " + std::to_string(a.mae) + " px");
}

StereoDisparityConfig base_config(Backend backend) {
  StereoDisparityConfig config;
  config.backend = backend;
  config.max_width = kWidth;
  config.max_height = kHeight;
  return config;
}

// Two frames of the scene in `encoding`, through `backend`, with depth (and a
// confidence map where the backend has one): each output against the truth
// and the others, with the left image's capture time and frame id.
void check_backend(Backend backend, ImageEncoding encoding, bool device) {
  const std::string what = std::string(backend_name(backend)) + " " +
                           holoscan::schema::EnumNameImageEncoding(encoding) +
                           (device ? " device" : " host");
  StereoDisparityConfig config = base_config(backend);
  const SyntheticScene first = make_scene(0, kWidth, kHeight);
  config.geometry = scene_geometry(first);
  config.depth = true;
  config.confidence = backend != Backend::kOfa;
  config.stats = std::make_shared<StereoDisparityStats>();
  auto h = ht::create_operator_test<StereoDisparityOp>(config);
  std::array<std::vector<ht::TestSample<holoscan::schema::ImageT>>, 2> samples;
  for (std::uint32_t frame = 0; frame < 2; ++frame) {
    auto pair = scene_pair(make_scene(frame, kWidth, kHeight), encoding, device);
    for (int side = 0; side < 2; ++side) {
      auto sample = ht::sample_at(std::move(pair[side]), capture_time(frame));
      sample.metadata.frame_id = 7U + frame;
      samples[side].push_back(std::move(sample));
    }
  }
  h.push_samples("left", std::move(samples[0]));
  h.push_samples("right", std::move(samples[1]));
  if (device) {
    h.input_source_device("left", holoscan::DeviceId{0});
    h.input_source_device("right", holoscan::DeviceId{0});
  }
  for (const char* port : {"disparity", "depth", "confidence"}) {
    if (std::string_view(port) == "confidence" && !config.confidence) continue;
    h.output_device(port, holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::schema::ImageT>(port, kCaptures);
  }
  const auto result = h.run();
  VERIFY_MSG(result.ok(), what + ": " + result.describe_failure());
  if (!result.ok()) return;
  const auto disparities = result.snapshots<holoscan::schema::ImageT>("disparity");
  const auto depths = result.snapshots<holoscan::schema::ImageT>("depth");
  VERIFY_MSG(disparities.size() == 2 && depths.size() == 2, what + ": two frames out");
  VERIFY(config.stats->frames == 2 && config.stats->backend == backend);
  const float fb = static_cast<float>(config.geometry->fx * config.geometry->baseline);
  for (std::size_t frame = 0; frame < disparities.size() && frame < depths.size(); ++frame) {
    const auto& image = disparities[frame];
    VERIFY(image.metadata().encoding == holoscan::schema::ImageEncoding_MONO32F);
    VERIFY(image.metadata().width == kWidth && image.metadata().height == kHeight);
    const auto& tensor = image.tensor(holoscan::schema::ImageTensorFields::data);
    VERIFY(tensor.shape() == (std::vector<std::int64_t>{kHeight, kWidth}));
    VERIFY_MSG(image.sample_metadata().capture_timestamp_ns ==
                       capture_time(static_cast<std::uint32_t>(frame)).count() &&
                   image.sample_metadata().frame_id == 7U + frame,
               what + ": the disparity keeps the left image's capture time and frame id");
    const auto disparity = values_of<float>(tensor);
    check_accuracy(disparity, make_scene(static_cast<std::uint32_t>(frame), kWidth, kHeight),
                   backend, what + " frame " + std::to_string(frame));
    // The depth is exactly the formula's, pixel by pixel.
    const auto depth =
        values_of<float>(depths[frame].tensor(holoscan::schema::ImageTensorFields::data));
    std::size_t wrong = 0;
    for (std::size_t i = 0; i < disparity.size() && i < depth.size(); ++i) {
      wrong += depth[i] != depth_from_disparity(disparity[i], fb, 0.0F);
    }
    VERIFY_MSG(wrong == 0, what + ": " + std::to_string(wrong) + " depths off the formula");
  }
  if (config.confidence) {
    const auto confidences = result.snapshots<holoscan::schema::ImageT>("confidence");
    VERIFY(confidences.size() == 2);
    if (confidences.empty()) return;
    VERIFY(confidences[0].metadata().encoding == holoscan::schema::ImageEncoding_MONO16);
    const auto confidence =
        values_of<std::uint16_t>(confidences[0].tensor(holoscan::schema::ImageTensorFields::data));
    const auto high = std::count_if(confidence.begin(), confidence.end(),
                                    [](std::uint16_t c) { return c > 32767; });
    VERIFY_MSG(high > static_cast<std::ptrdiff_t>(confidence.size() / 2),
               what + ": most of the scene is matched confidently");
  }
}

void check_every_backend() {
  const std::vector<Backend> backends = available_backends();
  if (on_jetson()) {
    VERIFY_MSG(std::find(backends.begin(), backends.end(), Backend::kOfa) != backends.end(),
               "a Jetson has the ofa backend");
  }
  for (Backend backend : backends) {
    check_backend(backend, holoscan::schema::ImageEncoding_MONO8, true);
    check_backend(backend, holoscan::schema::ImageEncoding_RGB8, false);
  }
  // Every color layout reads as the same gray.
  check_backend(Backend::kCuda, holoscan::schema::ImageEncoding_MONO8, false);
  check_backend(Backend::kCuda, holoscan::schema::ImageEncoding_BGR8, true);
  check_backend(Backend::kCuda, holoscan::schema::ImageEncoding_RGBA8, false);
  check_backend(Backend::kCuda, holoscan::schema::ImageEncoding_BGRA8, true);
}

// What the process writes to stdout and stderr, where the SDK logs, while
// `body` runs.
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

// Runs three frames of `config` from host or device MONO8 images; how many
// disparity maps came out, or nothing if the run failed.
std::optional<std::size_t> run_frames(const StereoDisparityConfig& config, bool device,
                                      std::string* failure = nullptr) {
  auto h = ht::create_operator_test<StereoDisparityOp>(config);
  for (std::uint32_t frame = 0; frame < 3; ++frame) {
    auto [left, right] = scene_pair(make_scene(frame, kWidth, kHeight),
                                    holoscan::schema::ImageEncoding_MONO8, device);
    h.push_frame(ht::frame_at(capture_time(frame), ht::input("left", std::move(left)),
                              ht::input("right", std::move(right))));
  }
  if (device) {
    h.input_source_device("left", holoscan::DeviceId{0});
    h.input_source_device("right", holoscan::DeviceId{0});
  }
  h.output_device("disparity", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("disparity", kCaptures);
  const auto result = h.run();
  if (!result.ok()) {
    if (failure != nullptr) *failure = result.describe_failure();
    return std::nullopt;
  }
  return result.snapshots<holoscan::schema::ImageT>("disparity").size();
}

// Host images are copied to the device, which is logged once, naming the
// backend, the direction and the size; device images log nothing.
void check_copy_log() {
  holoscan::set_log_level(holoscan::LogLevel::INFO);
  std::optional<std::size_t> frames;
  std::string output =
      capture_output([&] { frames = run_frames(base_config(Backend::kCuda), false); });
  VERIFY(frames == 3U);
  VERIFY_MSG(count_of(output, "copied host to device") == 1, "logged once:\n" + output);
  const std::string bytes = std::to_string(2 * kWidth * kHeight) + " bytes";
  for (const std::string& part :
       {std::string("(cuda backend)"), bytes, std::string("send CUDA device images")}) {
    VERIFY_MSG(output.find(part) != std::string::npos, "the copy log names '" + part + "'");
  }
  output = capture_output([&] { frames = run_frames(base_config(Backend::kCuda), true); });
  VERIFY(frames == 3U);
  VERIFY_MSG(count_of(output, "copied host to device") == 0, "nothing logged:\n" + output);
}

// kAuto chooses on the first frame and logs its choice once: ofa on a Jetson,
// cuda on x86_64; cuda wherever a confidence map is asked for.
void check_auto() {
  holoscan::set_log_level(holoscan::LogLevel::INFO);
  VERIFY(StereoDisparityOp(base_config(Backend::kCuda)).resolved_backend() == Backend::kCuda);
  VERIFY(StereoDisparityOp(base_config(Backend::kAuto)).resolved_backend() == Backend::kAuto);
  const Backend expected =
      resolve_auto(probe_platform(), DisparityParams{}, false, kWidth, kHeight).backend;
  if (std::getenv("HOLOSCAN_CI_PLATFORM") != nullptr) {
    VERIFY_MSG(expected == (on_jetson() ? Backend::kOfa : Backend::kCuda),
               "auto chooses " + std::string(backend_name(expected)) + " on this CI runner");
  }
  StereoDisparityConfig config = base_config(Backend::kAuto);
  config.stats = std::make_shared<StereoDisparityStats>();
  std::optional<std::size_t> frames;
  std::string output = capture_output([&] { frames = run_frames(config, true); });
  VERIFY(frames == 3U);
  VERIFY(config.stats->backend == expected);
  VERIFY_MSG(count_of(output, "backend auto chose " + std::string(backend_name(expected))) == 1,
             "the choice is logged once:\n" + output);
  config.confidence = true;
  config.stats = std::make_shared<StereoDisparityStats>();
  auto h = ht::create_operator_test<StereoDisparityOp>(config);
  auto [left, right] =
      scene_pair(make_scene(0, kWidth, kHeight), holoscan::schema::ImageEncoding_MONO8, false);
  h.push_frame(ht::frame_at(capture_time(0), ht::input("left", std::move(left)),
                            ht::input("right", std::move(right))));
  h.output_device("disparity", holoscan::DeviceId{0});
  h.output_device("confidence", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("disparity", kCaptures);
  h.capture_snapshot<holoscan::schema::ImageT>("confidence", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  VERIFY(config.stats->backend == Backend::kCuda);
}

// The geometry with every frame: frame 1's has twice the baseline, so twice
// the depth; a frame whose geometry carries another capture time gives nothing,
// while the frames around it still do.
void check_geometry_input() {
  StereoDisparityConfig config = base_config(Backend::kCuda);
  config.geometry_input = true;
  config.depth = true;
  auto h = ht::create_operator_test<StereoDisparityOp>(config);
  const StereoGeometry geometry = scene_geometry(make_scene(0, kWidth, kHeight));
  for (std::uint32_t frame = 0; frame < 4; ++frame) {
    StereoGeometry g = geometry;
    if (frame == 1) g.baseline *= 2.0;
    auto [left, right] = scene_pair(make_scene(frame, kWidth, kHeight),
                                    holoscan::schema::ImageEncoding_MONO8, false);
    // Frame 2's geometry is a millisecond late.
    const auto geometry_time = capture_time(frame) + std::chrono::milliseconds(frame == 2 ? 1 : 0);
    h.push_frame(ht::frame_at(capture_time(frame), ht::input("left", std::move(left)),
                              ht::input("right", std::move(right))));
    h.push_frame(ht::frame_at(geometry_time, ht::input("geometry", to_message(g))));
  }
  for (const char* port : {"disparity", "depth"}) {
    h.output_device(port, holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::schema::ImageT>(port, kCaptures);
  }
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto disparities = result.snapshots<holoscan::schema::ImageT>("disparity");
  const auto depths = result.snapshots<holoscan::schema::ImageT>("depth");
  VERIFY_MSG(disparities.size() == 3 && depths.size() == 3,
             std::to_string(disparities.size()) + " frames from 3 matched ones");
  const std::array<std::uint32_t, 3> matched{0U, 1U, 3U};
  for (std::size_t i = 0; i < disparities.size() && i < depths.size(); ++i) {
    VERIFY(depths[i].sample_metadata().capture_timestamp_ns == capture_time(matched[i]).count());
    const double baseline = geometry.baseline * (matched[i] == 1 ? 2.0 : 1.0);
    const auto fb = static_cast<float>(geometry.fx * baseline);
    const auto disparity =
        values_of<float>(disparities[i].tensor(holoscan::schema::ImageTensorFields::data));
    const auto depth =
        values_of<float>(depths[i].tensor(holoscan::schema::ImageTensorFields::data));
    std::size_t wrong = 0;
    for (std::size_t p = 0; p < disparity.size() && p < depth.size(); ++p) {
      wrong += depth[p] != depth_from_disparity(disparity[p], fb, 0.0F);
    }
    VERIFY_MSG(wrong == 0, "frame " + std::to_string(matched[i]) + " uses its own geometry");
  }
}

// A frame that cannot be matched fails the run, saying why.
void check_rejected_frame(
    const std::string& expected,
    const std::function<void(StereoDisparityConfig&, std::array<holoscan::schema::ImageT, 2>&)>&
        change,
    std::optional<StereoGeometry> geometry = std::nullopt) {
  StereoDisparityConfig config = base_config(Backend::kCuda);
  auto pair =
      scene_pair(make_scene(0, kWidth, kHeight), holoscan::schema::ImageEncoding_MONO8, false);
  change(config, pair);
  auto h = ht::create_operator_test<StereoDisparityOp>(config);
  if (geometry) {
    h.push_frame(ht::frame_at(capture_time(0), ht::input("left", pair[0]),
                              ht::input("right", pair[1]),
                              ht::input("geometry", to_message(*geometry))));
  } else {
    h.push_frame(
        ht::frame_at(capture_time(0), ht::input("left", pair[0]), ht::input("right", pair[1])));
  }
  h.output_device("disparity", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("disparity", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), "rejected: " + expected);
  VERIFY_MSG(result.describe_failure().find(expected) != std::string::npos,
             "the failure says '" + expected + "':\n" + result.describe_failure());
}

void check_rejected_frames() {
  using Pair = std::array<holoscan::schema::ImageT, 2>;
  check_rejected_frame("the right 320x240", [](StereoDisparityConfig&, Pair& pair) {
    pair[1] = scene_pair(make_scene(0, 320, 240), holoscan::schema::ImageEncoding_MONO8, false)[0];
  });
  check_rejected_frame("unsupported left image encoding YUYV", [](StereoDisparityConfig&,
                                                                  Pair& pair) {
    pair = scene_pair(make_scene(0, kWidth, kHeight), holoscan::schema::ImageEncoding_RGB8, false);
    pair[0].encoding = holoscan::schema::ImageEncoding_YUYV;
  });
  check_rejected_frame("exceeds the configured 320x240", [](StereoDisparityConfig& config, Pair&) {
    config.max_width = 320;
    config.max_height = 240;
  });
  StereoGeometry wrong_size = scene_geometry(make_scene(0, kWidth, kHeight));
  wrong_size.width = 1280;
  check_rejected_frame(
      "the geometry is for 1280x480 images",
      [](StereoDisparityConfig& config, Pair&) { config.geometry_input = true; }, wrong_size);
  StereoGeometry no_baseline = scene_geometry(make_scene(0, kWidth, kHeight));
  no_baseline.baseline = 0.0;
  check_rejected_frame(
      "baseline must be finite and positive",
      [](StereoDisparityConfig& config, Pair&) { config.geometry_input = true; }, no_baseline);
}

void check_rejected_configs() {
  const auto rejects = [](const std::string& expected, StereoDisparityConfig config) {
    try {
      StereoDisparityOp op(std::move(config));
      VERIFY_MSG(false, "rejected: " + expected);
    } catch (const std::invalid_argument& error) {
      VERIFY_MSG(std::string(error.what()).find(expected) != std::string::npos,
                 std::string("the error says '") + expected + "': " + error.what());
    }
  };
  StereoDisparityConfig config = base_config(Backend::kCuda);
  config.depth = true;
  rejects("the depth output needs the geometry", config);
  config.geometry = scene_geometry(make_scene(0, kWidth, kHeight));
  config.geometry_input = true;
  rejects("not both", config);
  config = base_config(Backend::kCuda);
  config.geometry = StereoGeometry{.fx = 100, .baseline = 0.1, .width = 0, .height = 10};
  rejects("width and height must be 1..16384", config);
  config = base_config(Backend::kCuda);
  config.params.max_disparity = 300;
  rejects("max_disparity must be 1..256", config);
  config = base_config(Backend::kOfa);
  config.confidence = true;
  rejects("no confidence output", config);
  config = base_config(Backend::kCuda);
  config.max_width = 0;
  rejects("max_width", config);
  // ofa and pva need the hardware.
  const std::vector<Backend> backends = available_backends();
  for (Backend backend : {Backend::kOfa, Backend::kPva}) {
    if (std::find(backends.begin(), backends.end(), backend) == backends.end()) {
      rejects("VPI reports none on this platform", base_config(backend));
    } else {
      VERIFY(!throws<std::invalid_argument>([&] { StereoDisparityOp op(base_config(backend)); }));
    }
  }
}

}  // namespace
}  // namespace holoscan::stereo_disparity

int main() {
  using namespace holoscan::stereo_disparity;  // NOLINT(build/namespaces)
  return test::run({
      {"every backend against the truth", check_every_backend},
      {"copy log", check_copy_log},
      {"auto backend", check_auto},
      {"geometry input", check_geometry_input},
      {"rejected frames", check_rejected_frames},
      {"rejected configurations", check_rejected_configs},
  });
}
