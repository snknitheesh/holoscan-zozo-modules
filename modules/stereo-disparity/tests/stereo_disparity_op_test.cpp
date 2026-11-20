// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// StereoDisparityOp on its own, through the SDK's operator test harness, on
// every backend this platform has: the synthetic scene's disparity against
// its truth, from MONO8 and color images in host and device memory, at full
// size and downscaled, also with rows that are not aligned; the depth,
// depth camera info and confidence outputs; the camera info input taken as
// the newest one or with every frame, frames before the first one skipped,
// and a calibration that changes; the auto backend's choice and the one-time
// copy log; and unusable inputs, camera info, configurations and platform
// limits rejected.

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

#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/messages/synthetic_calibration.hpp>
#include <holoscan/stereo_disparity/camera_info.hpp>
#include <holoscan/stereo_disparity/depth.hpp>
#include <holoscan/stereo_disparity/platform.hpp>
#include <holoscan/stereo_disparity/stereo_disparity_op.hpp>
#include <holoscan/stereo_disparity/synthetic_scene.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "test_support.hpp"

namespace holoscan::stereo_disparity {
namespace {

namespace ht = holoscan::testing;
namespace hm = holoscan::messages;
using holoscan::schema::ImageEncoding;
using test::throws;
using Pair = std::array<holoscan::schema::ImageT, 2>;

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

// The scene's camera info as the operator's input takes it.
hm::schema::StereoCameraInfoT scene_info(const SyntheticScene& scene) {
  return hm::to_message(scene_camera_info(scene));
}

template <typename T>
void add_field(ht::TestFrame& frame, std::string name, T value, std::chrono::nanoseconds time,
               std::uint64_t frame_id) {
  auto sample = ht::sample_at(std::move(value), time);
  if (frame_id != 0) sample.metadata.frame_id = frame_id;
  std::vector<ht::TestSample<T>> samples;
  samples.push_back(std::move(sample));
  frame.fields.push_back({std::move(name), ht::staged_samples<T>(std::move(samples))});
}

// One stereo frame at `time`: its camera info first, when given, as a
// producer sending it with every pair does, then the left and right images;
// all with frame id `frame_id` when it is not 0.
ht::TestFrame stereo_frame(std::chrono::nanoseconds time, Pair pair,
                           std::optional<hm::schema::StereoCameraInfoT> info,
                           std::uint64_t frame_id = 0) {
  ht::TestFrame frame;
  if (info) add_field(frame, "camera_info", std::move(*info), time, frame_id);
  add_field(frame, "left", std::move(pair[0]), time, frame_id);
  add_field(frame, "right", std::move(pair[1]), time, frame_id);
  return frame;
}

template <typename T>
std::vector<T> values_of(const ht::TensorSnapshot& tensor) {
  std::vector<T> out(tensor.bytes().size() / sizeof(T));
  std::memcpy(out.data(), tensor.bytes().data(), out.size() * sizeof(T));
  return out;
}

// How a disparity map compares with the scene's truth, over the pixels both
// cameras see. Output pixel (u, v) of a map downscaled by `downscale` covers
// input pixels u * downscale .. u * downscale + downscale - 1 (and so for v):
// its truth is the scene's at their center.
struct Accuracy {
  std::size_t truth{};  // Pixels both cameras see.
  double valid{};       // Share of them with a disparity.
  double within_1px{};  // Share of those within 1 px of the truth.
  double median{};      // Their median absolute error, pixels.
  double mae{};         // Their mean absolute error, pixels.
  double whole{};       // Share of those that are whole pixels.
};

Accuracy accuracy(const std::vector<float>& disparity, const SyntheticScene& scene,
                  int downscale = 1) {
  Accuracy result;
  std::size_t valid = 0;
  std::size_t within = 0;
  std::size_t whole = 0;
  double error = 0.0;
  std::vector<float> errors;
  const int width = output_side(scene.width, downscale);
  const int height = output_side(scene.height, downscale);
  const float center = static_cast<float>(downscale - 1) / 2.0F;
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      const float truth = true_disparity(scene, static_cast<float>(u * downscale) + center,
                                         static_cast<float>(v * downscale) + center);
      if (truth < 0.0F) continue;
      ++result.truth;
      const float d = disparity[static_cast<std::size_t>(v) * width + u];
      if (d <= 0.0F) continue;
      ++valid;
      const double e = std::abs(static_cast<double>(d) - truth);
      within += e <= 1.0;
      whole += d == std::floor(d);
      error += e;
      errors.push_back(static_cast<float>(e));
    }
  }
  result.valid = result.truth == 0 ? 0.0 : static_cast<double>(valid) / result.truth;
  result.within_1px = valid == 0 ? 0.0 : static_cast<double>(within) / valid;
  result.mae = valid == 0 ? 0.0 : error / valid;
  result.whole = valid == 0 ? 0.0 : static_cast<double>(whole) / valid;
  if (!errors.empty()) {
    const auto middle = errors.begin() + static_cast<std::ptrdiff_t>(errors.size() / 2);
    std::nth_element(errors.begin(), middle, errors.end());
    result.median = *middle;
  }
  return result;
}

// What every backend must reach on the synthetic scene, measured with VPI
// 4.1.4 (cuda on an RTX 5090, Orin and Thor; ofa and pva on Orin and Thor) and
// set with margin. The median, not the mean: Thor's pva leaves a few pixels
// far off, which move the mean (up to 1.6 px) but not the median. Thor's ofa
// and pva give whole-pixel disparities, which are up to 0.5 px off: the
// scene's background, most of the image, is 6.4 px away at 640 px wide, so
// their median error there is 0.4 px.
struct Bounds {
  double valid;
  double within_1px;
  double median;
};
Bounds bounds_for(Backend backend) {
  switch (backend) {
    case Backend::kOfa:
      return {0.90, 0.90, 0.5};
    case Backend::kPva:
      return {0.85, 0.90, 0.5};
    default:
      return {0.90, 0.95, 0.3};
  }
}

void check_accuracy(const std::vector<float>& disparity, const SyntheticScene& scene,
                    Backend backend, const std::string& what, int downscale = 1) {
  const Accuracy a = accuracy(disparity, scene, downscale);
  const Bounds b = bounds_for(backend);
  std::cout << "  " << what << ": valid=" << a.valid << " within_1px=" << a.within_1px
            << " median_px=" << a.median << " mae_px=" << a.mae << " whole_px=" << a.whole << '\n';
  VERIFY_MSG(a.truth > static_cast<std::size_t>(output_side(scene.width, downscale)) *
                           output_side(scene.height, downscale) / 2,
             what + ": most pixels have a truth");
  VERIFY_MSG(a.valid >= b.valid, what + ": valid share " + std::to_string(a.valid));
  VERIFY_MSG(a.within_1px >= b.within_1px,
             what + ": share within 1 px " + std::to_string(a.within_1px));
  VERIFY_MSG(a.median <= b.median, what + ": median error " + std::to_string(a.median) + " px");
}

// The depth's camera info, one per frame, before its depth: the left camera
// at the output's size, its intrinsics in output pixels, with the frame's
// capture time.
void check_depth_camera_info(const std::vector<holoscan::Sample<hm::schema::CameraInfoT>>& infos,
                             const hm::StereoCameraInfo& pair, int downscale, std::size_t frames,
                             const std::string& what) {
  VERIFY_MSG(infos.size() == frames, what + ": a depth camera info with every frame");
  const double shift = (downscale - 1) / 2.0;
  for (std::size_t frame = 0; frame < infos.size(); ++frame) {
    const std::int64_t time = capture_time(static_cast<std::uint32_t>(frame)).count();
    const hm::CameraInfo info = hm::from_message(infos[frame].data);
    VERIFY_MSG(info.header.capture_timestamp_ns == time &&
                   infos[frame].metadata.capture_timestamp_ns == time,
               what + ": the depth camera info carries its frame's capture time");
    VERIFY_MSG(info.header.frame_id == "left" && info.state == hm::CameraState::kRectified &&
                   info.distortion_model == hm::DistortionModel::kNone,
               what + ": the depth camera info is the rectified left camera");
    VERIFY_MSG(info.width == static_cast<std::uint32_t>(output_side(kWidth, downscale)) &&
                   info.height == static_cast<std::uint32_t>(output_side(kHeight, downscale)),
               what + ": the depth camera info has the output's size");
    VERIFY_MSG(info.fx == pair.left.fx / downscale && info.fy == pair.left.fy / downscale &&
                   info.cx == (pair.left.cx - shift) / downscale &&
                   info.cy == (pair.left.cy - shift) / downscale &&
                   info.binning_x == static_cast<std::uint32_t>(downscale),
               what + ": the depth camera info's intrinsics are in output pixels");
  }
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
// and the others, at the downscaled size, with the left image's capture time
// and frame id. `passes` unset: the platform's default.
void check_backend(Backend backend, ImageEncoding encoding, bool device, int downscale = 1,
                   std::optional<int> passes = std::nullopt) {
  const std::string what =
      std::string(backend_name(backend)) + " " + holoscan::schema::EnumNameImageEncoding(encoding) +
      (device ? " device" : " host") + " downscale " + std::to_string(downscale) +
      (passes ? " passes " + std::to_string(*passes) : std::string());
  const int out_width = output_side(kWidth, downscale);
  const int out_height = output_side(kHeight, downscale);
  StereoDisparityConfig config = base_config(backend);
  config.params.downscale = downscale;
  config.params.num_passes = passes;
  const hm::StereoCameraInfo info = scene_camera_info(make_scene(0, kWidth, kHeight));
  config.depth = true;
  config.depth_camera_info = true;
  config.confidence = backend != Backend::kOfa;
  config.stats = std::make_shared<StereoDisparityStats>();
  auto h = ht::create_operator_test<StereoDisparityOp>(config);
  for (std::uint32_t frame = 0; frame < 2; ++frame) {
    h.push_frame(stereo_frame(capture_time(frame),
                              scene_pair(make_scene(frame, kWidth, kHeight), encoding, device),
                              hm::to_message(info), 7U + frame));
  }
  if (device) {
    h.input_source_device("left", holoscan::DeviceId{0});
    h.input_source_device("right", holoscan::DeviceId{0});
  }
  for (const char* port : {"disparity", "depth", "confidence"}) {
    if (std::string_view(port) == "confidence" && !config.confidence) continue;
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
  VERIFY(config.stats->frames == 2 && config.stats->backend == backend);
  check_depth_camera_info(result.samples<hm::schema::CameraInfoT>("depth_camera_info"), info,
                          downscale, 2, what);
  const auto fb = static_cast<float>(hm::fx_baseline(info));
  for (std::size_t frame = 0; frame < disparities.size() && frame < depths.size(); ++frame) {
    const auto& image = disparities[frame];
    VERIFY(image.metadata().encoding == holoscan::schema::ImageEncoding_MONO32F);
    VERIFY(image.metadata().width == out_width && image.metadata().height == out_height);
    const auto& tensor = image.tensor(holoscan::schema::ImageTensorFields::data);
    VERIFY(tensor.shape() == (std::vector<std::int64_t>{out_height, out_width}));
    VERIFY_MSG(image.sample_metadata().capture_timestamp_ns ==
                       capture_time(static_cast<std::uint32_t>(frame)).count() &&
                   image.sample_metadata().frame_id == 7U + frame,
               what + ": the disparity keeps the left image's capture time and frame id");
    const auto disparity = values_of<float>(tensor);
    check_accuracy(disparity, make_scene(static_cast<std::uint32_t>(frame), kWidth, kHeight),
                   backend, what + " frame " + std::to_string(frame), downscale);
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

// Rows the GPU does not align (650 px, so MONO8 rows of 650 bytes and BGR8
// of 1950), which VPI reads in place: two frames of each against the truth.
void check_unaligned(Backend backend, ImageEncoding encoding) {
  constexpr int kOddWidth = 650;
  const std::string what = std::string(backend_name(backend)) + " 650 px " +
                           holoscan::schema::EnumNameImageEncoding(encoding);
  StereoDisparityConfig config = base_config(backend);
  config.max_width = kOddWidth;
  auto h = ht::create_operator_test<StereoDisparityOp>(config);
  for (std::uint32_t frame = 0; frame < 2; ++frame) {
    const SyntheticScene scene = make_scene(frame, kOddWidth, kHeight);
    h.push_frame(
        stereo_frame(capture_time(frame), scene_pair(scene, encoding, true), scene_info(scene)));
  }
  h.input_source_device("left", holoscan::DeviceId{0});
  h.input_source_device("right", holoscan::DeviceId{0});
  h.output_device("disparity", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("disparity", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), what + ": " + result.describe_failure());
  if (!result.ok()) return;
  const auto disparities = result.snapshots<holoscan::schema::ImageT>("disparity");
  VERIFY_MSG(disparities.size() == 2, what + ": every frame out");
  for (std::size_t frame = 0; frame < disparities.size(); ++frame) {
    check_accuracy(
        values_of<float>(disparities[frame].tensor(holoscan::schema::ImageTensorFields::data)),
        make_scene(static_cast<std::uint32_t>(frame), kOddWidth, kHeight), backend,
        what + " frame " + std::to_string(frame));
  }
}

void check_every_backend() {
  const PlatformFacts facts = probe_platform();
  const std::vector<Backend> backends = available_backends(facts);
  if (on_jetson()) {
    VERIFY_MSG(std::find(backends.begin(), backends.end(), Backend::kOfa) != backends.end(),
               "a Jetson has the ofa backend");
  }
  for (Backend backend : backends) {
    check_backend(backend, holoscan::schema::ImageEncoding_MONO8, true);
    check_backend(backend, holoscan::schema::ImageEncoding_RGB8, false);
    if (backend != Backend::kCuda) {
      for (int downscale : {2, 4, 8}) {
        // Thor refuses 8 (checked by check_platform_limits()).
        if (facts.thor && downscale == 8) continue;
        check_backend(backend, holoscan::schema::ImageEncoding_MONO8, true, downscale);
      }
      check_backend(backend, holoscan::schema::ImageEncoding_BGRA8, true, 2);
      // 3 passes, Thor's default, also where the default is 1 (Orin).
      check_backend(backend, holoscan::schema::ImageEncoding_MONO8, true, 1, 3);
    }
  }
  for (Backend backend : backends) {
    check_unaligned(backend, holoscan::schema::ImageEncoding_MONO8);
    check_unaligned(backend, holoscan::schema::ImageEncoding_BGR8);
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
    const SyntheticScene scene = make_scene(frame, kWidth, kHeight);
    h.push_frame(stereo_frame(capture_time(frame),
                              scene_pair(scene, holoscan::schema::ImageEncoding_MONO8, device),
                              scene_info(scene)));
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
  const SyntheticScene scene = make_scene(0, kWidth, kHeight);
  h.push_frame(stereo_frame(capture_time(0),
                            scene_pair(scene, holoscan::schema::ImageEncoding_MONO8, false),
                            scene_info(scene)));
  h.output_device("disparity", holoscan::DeviceId{0});
  h.output_device("confidence", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("disparity", kCaptures);
  h.capture_snapshot<holoscan::schema::ImageT>("confidence", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  VERIFY(config.stats->backend == Backend::kCuda);
}

// Runs four frames of the scene through cuda with depth, with the camera info
// as `timing` takes it: `sent(frame)` is the camera info sent with frame
// `frame` (before its images), or nothing. Returns the run's depth maps, in
// frame order, and the frames they came from (their capture times).
struct DepthRun {
  std::vector<std::vector<float>> disparity;
  std::vector<std::vector<float>> depth;
  std::vector<std::int64_t> times;
};
DepthRun run_depth(
    hm::CameraInfoTiming timing,
    const std::function<std::optional<hm::schema::StereoCameraInfoT>(std::uint32_t)>& sent) {
  StereoDisparityConfig config = base_config(Backend::kCuda);
  config.camera_info_input.timing = timing;
  config.depth = true;
  auto h = ht::create_operator_test<StereoDisparityOp>(config);
  for (std::uint32_t frame = 0; frame < 4; ++frame) {
    h.push_frame(stereo_frame(capture_time(frame),
                              scene_pair(make_scene(frame, kWidth, kHeight),
                                         holoscan::schema::ImageEncoding_MONO8, false),
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
// frame on; and with kEveryFrame, each frame waits for its own.
void check_camera_info_input() {
  const hm::StereoCameraInfo info = scene_camera_info(make_scene(0, kWidth, kHeight));
  hm::StereoCameraInfo wide = info;
  wide.baseline *= 2.0;
  wide.right_from_left.translation[0] *= 2.0;
  const std::array<double, 4> baseline{0.0, info.baseline, wide.baseline, wide.baseline};
  // kLatest: none with frame 0, the scene's with frame 1, the wider one with
  // frame 2, none with frame 3, which keeps using frame 2's.
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
  // kEveryFrame: every frame with its own camera info.
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
  StereoDisparityConfig config = base_config(Backend::kCuda);
  config.camera_info_input.timing = hm::CameraInfoTiming::kEveryFrame;
  config.depth = true;
  config.depth_camera_info = true;
  const std::array<std::string, 3> frames{"left", "left_moved", "left_moved"};
  auto h = ht::create_operator_test<StereoDisparityOp>(config);
  for (std::uint32_t frame = 0; frame < frames.size(); ++frame) {
    const SyntheticScene scene = make_scene(frame, kWidth, kHeight);
    hm::StereoCameraInfo info = scene_camera_info(scene);
    info.header.frame_id = frames[frame];
    info.left.header.frame_id = frames[frame];
    h.push_frame(stereo_frame(capture_time(frame),
                              scene_pair(scene, holoscan::schema::ImageEncoding_MONO8, false),
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

// A frame that cannot be matched, or whose camera info cannot be used, fails
// the run, saying why.
void check_rejected_frame(const std::string& expected,
                          const std::function<void(StereoDisparityConfig&, Pair&,
                                                   hm::schema::StereoCameraInfoT&)>& change) {
  StereoDisparityConfig config = base_config(Backend::kCuda);
  const SyntheticScene scene = make_scene(0, kWidth, kHeight);
  Pair pair = scene_pair(scene, holoscan::schema::ImageEncoding_MONO8, false);
  hm::schema::StereoCameraInfoT info = scene_info(scene);
  change(config, pair, info);
  auto h = ht::create_operator_test<StereoDisparityOp>(config);
  h.push_frame(stereo_frame(capture_time(0), std::move(pair), std::move(info)));
  h.output_device("disparity", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("disparity", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), "rejected: " + expected);
  VERIFY_MSG(result.describe_failure().find(expected) != std::string::npos,
             "the failure says '" + expected + "':\n" + result.describe_failure());
}

void check_rejected_frames() {
  using Info = hm::schema::StereoCameraInfoT;
  check_rejected_frame("the right 320x240", [](StereoDisparityConfig&, Pair& pair, Info&) {
    pair[1] = scene_pair(make_scene(0, 320, 240), holoscan::schema::ImageEncoding_MONO8, false)[0];
  });
  check_rejected_frame("unsupported left image encoding YUYV", [](StereoDisparityConfig&,
                                                                  Pair& pair, Info&) {
    pair = scene_pair(make_scene(0, kWidth, kHeight), holoscan::schema::ImageEncoding_RGB8, false);
    pair[0].encoding = holoscan::schema::ImageEncoding_YUYV;
  });
  check_rejected_frame("exceeds the configured 320x240",
                       [](StereoDisparityConfig& config, Pair&, Info&) {
                         config.max_width = 320;
                         config.max_height = 240;
                       });
  check_rejected_frame("the stereo camera info is for 1280x480 (left)",
                       [](StereoDisparityConfig&, Pair&, Info& info) {
                         info = scene_info(make_scene(0, 1280, kHeight));
                       });
  // A raw pair: the images are not rectified.
  check_rejected_frame("must be a rectified pair", [](StereoDisparityConfig&, Pair&, Info& info) {
    info = hm::to_message(
        hm::synthetic_stereo_camera_info(hm::DistortionModel::kPlumbBob, kWidth, kHeight));
  });
  // Not valid: a rectified pair without a baseline.
  check_rejected_frame("baseline",
                       [](StereoDisparityConfig&, Pair&, Info& info) { info.baseline = 0.0; });
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
  config.depth_camera_info = true;
  rejects("goes with the depth output", config);
  config = base_config(Backend::kCuda);
  config.camera_info_input.max_age = std::chrono::nanoseconds::zero();
  rejects("max_age must be > 0", config);
  config = base_config(Backend::kCuda);
  config.params.max_disparity = 300;
  rejects("max_disparity must be 1..256", config);
  config = base_config(Backend::kOfa);
  config.confidence = true;
  rejects("no confidence output", config);
  config = base_config(Backend::kCuda);
  config.max_width = 0;
  rejects("max_width", config);
  config = base_config(Backend::kCuda);
  config.params.downscale = 2;
  rejects("downscale needs the ofa or pva backend", config);
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

// For compiling a graph around the operator, never run: the pair and its
// camera info, and a sink for the disparity.
class PairStubOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(left, "left");
    spec.output(right, "right");
    spec.output(camera_info, "camera_info");
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::seconds{1}});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }
  holoscan::Output<holoscan::schema::ImageT> left;
  holoscan::Output<holoscan::schema::ImageT> right;
  holoscan::Output<hm::schema::StereoCameraInfoT> camera_info;
};

class DisparitySinkStubOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(disparity, "disparity"); }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{disparity});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }
  holoscan::Input<holoscan::schema::ImageT> disparity;
};

// With downscale, the output pool's slots are sized for the downscaled
// output, not for the input.
void check_output_bounds() {
  const std::vector<Backend> backends = available_backends();
  for (Backend backend : {Backend::kOfa, Backend::kPva}) {
    if (std::find(backends.begin(), backends.end(), backend) == backends.end()) continue;
    StereoDisparityConfig config = base_config(backend);
    config.params.downscale = 4;
    config.max_width = 1280;
    config.max_height = 720;
    holoscan::Graph graph{"bounds"};
    auto source = graph.op<PairStubOp>("source");
    auto stereo = graph.op<StereoDisparityOp>("stereo", config);
    auto sink = graph.op<DisparitySinkStubOp>("sink");
    graph.add_flow(source->left, stereo->left);
    graph.add_flow(source->right, stereo->right);
    graph.add_flow(source->camera_info, stereo->camera_info);
    graph.add_flow(stereo->disparity, sink->disparity);
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
    holoscan::CompileOptions options;
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "stereo", .output_port = "disparity", .device = holoscan::DeviceId{0}});
    const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
    VERIFY_MSG(plan.ok(), holoscan::format_diagnostics(plan.diagnostics(), plan.graph()));
    if (!plan.ok()) continue;
    const std::uint64_t expected = static_cast<std::uint64_t>(output_side(1280, 4)) *
                                   static_cast<std::uint64_t>(output_side(720, 4)) * sizeof(float);
    std::size_t outputs = 0;
    for (const holoscan::TensorAllocationPlanRecord& allocation : plan.tensor_allocations()) {
      if (allocation.memory_kind != holoscan::MemoryKind::kCudaDevice) continue;
      ++outputs;
      VERIFY_MSG(allocation.bounds.max_byte_span <= expected,
                 std::string(backend_name(backend)) + " downscale 4: a slot of " +
                     std::to_string(allocation.bounds.max_byte_span) + " bytes for a " +
                     std::to_string(expected) + "-byte output");
    }
    VERIFY_MSG(outputs == 1, "one device output, the disparity");
  }
}

// The platform's own limits. On Jetson Thor, ofa and pva refuse 1 and 2
// passes and downscale 8; on Orin, they run them. pva with max_disparity 256
// and a confidence map refuses a width VPI 4.1.4 cannot pad (848), and runs
// one it can (896).
void check_platform_limits() {
  const PlatformFacts facts = probe_platform();
  if (on_jetson()) {
    cudaDeviceProp device{};
    VERIFY(cudaGetDeviceProperties(&device, 0) == cudaSuccess);
    const bool named_thor = std::string_view(device.name).find("Thor") != std::string_view::npos;
    VERIFY_MSG(facts.thor == named_thor, std::string("Thor detected for GPU ") + device.name);
  }
  const std::vector<Backend> backends = available_backends(facts);
  for (Backend backend : {Backend::kOfa, Backend::kPva}) {
    if (std::find(backends.begin(), backends.end(), backend) == backends.end()) continue;
    StereoDisparityConfig one_pass = base_config(backend);
    one_pass.params.num_passes = 1;
    StereoDisparityConfig two_passes = base_config(backend);
    two_passes.params.num_passes = 2;
    StereoDisparityConfig eighth = base_config(backend);
    eighth.params.downscale = 8;
    for (const StereoDisparityConfig& config : {one_pass, two_passes, eighth}) {
      if (facts.thor) {
        VERIFY(throws<std::invalid_argument>([&] { StereoDisparityOp op(config); }));
      } else {
        VERIFY_MSG(
            run_frames(config, true) == 3U,
            std::string(backend_name(backend)) + " runs 1 and 2 passes and downscale 8 off Thor");
      }
    }
  }
  if (std::find(backends.begin(), backends.end(), Backend::kPva) == backends.end()) return;
  for (const int width : {848, 896}) {
    StereoDisparityConfig config = base_config(Backend::kPva);
    config.params.max_disparity = 256;
    config.confidence = true;
    config.max_width = width;
    auto h = ht::create_operator_test<StereoDisparityOp>(config);
    const SyntheticScene scene = make_scene(0, width, kHeight);
    h.push_frame(stereo_frame(capture_time(0),
                              scene_pair(scene, holoscan::schema::ImageEncoding_MONO8, true),
                              scene_info(scene)));
    h.input_source_device("left", holoscan::DeviceId{0});
    h.input_source_device("right", holoscan::DeviceId{0});
    for (const char* port : {"disparity", "confidence"}) {
      h.output_device(port, holoscan::DeviceId{0});
      h.capture_snapshot<holoscan::schema::ImageT>(port, kCaptures);
    }
    const auto result = h.run();
    if (width == 848) {
      VERIFY_MSG(!result.ok() &&
                     result.describe_failure().find("remainder modulo 256 is 0 or at least 128") !=
                         std::string::npos,
                 "848 px refused on pva:\n" + result.describe_failure());
    } else {
      VERIFY_MSG(result.ok(), "896 px runs on pva: " + result.describe_failure());
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
      {"camera info input", check_camera_info_input},
      {"depth camera info frame", check_depth_camera_info_frame},
      {"rejected frames", check_rejected_frames},
      {"rejected configurations", check_rejected_configs},
      {"platform limits", check_platform_limits},
      {"output bounds", check_output_bounds},
  });
}
