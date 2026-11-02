// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// MonoDepthOp with a real Depth Anything V2 model, which tools/prepare_model.py
// exports; skipped (exit 77) when the model is absent, as it is in CI.
//
// - The synthetic scene, whose true depth is known: the predicted relative
//   inverse depth must rank the pixels as the truth does (Spearman's
//   correlation at least kMinSpearman) and the regions in their order (the
//   near panel before the sphere, the sphere before the far wall). With an
//   FP16 and an FP32 engine, letterboxed and stretched.
// - When HOLOSCAN_TEST_DATA holds the recorded stereo-camera clips, real
//   RGB frames of a RealSense D585 against its own depth, reprojected into
//   the RGB camera: the rank correlation, and the error left once one scale
//   and shift per frame map the inverse depth onto the camera's, which is the
//   best a relative model can do.
//
// Usage: mono_depth_real_model_test MODEL.onnx

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/mono_depth/mono_depth_op.hpp>
#include <holoscan/mono_depth/synthetic_scene.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "test_support.hpp"

namespace holoscan::mono_depth {
namespace {

constexpr double kMinSpearman = 0.85;
constexpr int kSkip = 77;

std::filesystem::path g_model;

// Packed RGB8 pixels in host memory as an image.
holoscan::schema::ImageT host_image(std::vector<std::uint8_t> rgb, int width, int height) {
  struct Storage {
    DLManagedTensor managed{};
    std::array<std::int64_t, 3> shape{};
    std::vector<std::uint8_t> bytes;
  };
  auto storage = std::make_unique<Storage>();
  storage->shape = {height, width, 3};
  storage->bytes = std::move(rgb);
  storage->managed.dl_tensor = {.data = storage->bytes.data(),
                                .device = {kDLCPU, 0},
                                .ndim = 3,
                                .dtype = {kDLUInt, 8, 1},
                                .shape = storage->shape.data(),
                                .strides = nullptr,
                                .byte_offset = 0};
  storage->managed.manager_ctx = storage.get();
  storage->managed.deleter = [](DLManagedTensor* tensor) {
    delete static_cast<Storage*>(tensor->manager_ctx);
  };
  holoscan::schema::ImageT image;
  image.width = width;
  image.height = height;
  image.encoding = holoscan::schema::ImageEncoding_RGB8;
  image.data = std::make_shared<holoscan::Tensor>(&storage.release()->managed);
  return image;
}

struct RgbFrame {
  std::vector<std::uint8_t> rgb;
  int width{};
  int height{};
};

// Each frame's inverse depth through MonoDepthOp with `config`, or nothing if
// the run failed.
std::optional<std::vector<std::vector<float>>> predict(const MonoDepthConfig& config,
                                                       const std::vector<RgbFrame>& frames) {
  // The engine is built first: building it in start(), which takes minutes on
  // a Jetson, would outlast the harness's watchdog.
  prepare_engine(config);
  auto h = holoscan::testing::create_operator_test<MonoDepthOp>(config);
  for (const RgbFrame& frame : frames) {
    h.push("image", host_image(frame.rgb, frame.width, frame.height));
  }
  h.output_device("depth", holoscan::DeviceId{0});
  // Room for every frame, and no more: the capture's capacity sizes the
  // output's GPU pool.
  h.capture_snapshot<holoscan::schema::ImageT>("depth", frames.size());
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) {
    return std::nullopt;
  }
  std::vector<std::vector<float>> depths;
  for (const auto& snapshot : result.snapshots<holoscan::schema::ImageT>("depth")) {
    const auto bytes = snapshot.tensor(holoscan::schema::ImageTensorFields::data).bytes();
    std::vector<float> depth(bytes.size() / sizeof(float));
    std::memcpy(depth.data(), bytes.data(), depth.size() * sizeof(float));
    depths.push_back(std::move(depth));
  }
  VERIFY(depths.size() == frames.size());
  return depths;
}

MonoDepthConfig model_config(bool fp16, ResizeMode mode) {
  return MonoDepthConfig{.model = g_model, .resize_mode = mode, .fp16 = fp16};
}

void check_synthetic(bool fp16, ResizeMode mode) {
  constexpr int kWidth = 1280;
  constexpr int kHeight = 720;
  const std::vector<std::uint32_t> frame_numbers{0, 7, 14, 21};
  std::vector<RgbFrame> frames;
  for (const std::uint32_t f : frame_numbers) {
    RgbFrame frame{std::vector<std::uint8_t>(kWidth * kHeight * 3), kWidth, kHeight};
    render_depth_scene(make_depth_scene(kWidth, kHeight, f), PixelFormat::kRgb8, frame.rgb.data(),
                       kWidth * 3);
    frames.push_back(std::move(frame));
  }
  const auto depths = predict(model_config(fp16, mode), frames);
  if (!depths) return;
  for (std::size_t i = 0; i < depths->size() && i < frame_numbers.size(); ++i) {
    const OrderingResult result =
        check_ordering(make_depth_scene(kWidth, kHeight, frame_numbers[i]), (*depths)[i]);
    std::cout << std::fixed << std::setprecision(4) << "  frame " << frame_numbers[i]
              << ": spearman=" << result.spearman << " median panel=" << result.median[2]
              << " sphere=" << result.median[3] << " wall=" << result.median[1]
              << " floor=" << result.median[0] << '\n';
    VERIFY_MSG(result.spearman >= kMinSpearman,
               "Spearman's correlation " + std::to_string(result.spearman));
    VERIFY_MSG(result.ordered, "the regions are not ranked panel, sphere, wall");
  }
}

// The recorded clips: a test-local reader of their format.

std::string read_text(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path.string());
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// A binary PGM or PPM: its header, then `channels` samples per pixel of
// `bytes` bytes each, big-endian.
std::vector<std::uint16_t> read_pnm(const std::filesystem::path& path, const char* magic,
                                    int& width, int& height) {
  const std::string text = read_text(path);
  std::istringstream in(text);
  std::string kind;
  int maxval = 0;
  in >> kind >> width >> height >> maxval;
  in.get();
  if (kind != magic || width < 1 || height < 1 || (maxval != 255 && maxval != 65535)) {
    throw std::runtime_error(path.string() + ": not a supported " + magic + " file");
  }
  const int channels = kind == "P6" ? 3 : 1;
  const int bytes = maxval > 255 ? 2 : 1;
  const auto offset = static_cast<std::size_t>(in.tellg());
  const std::size_t count = static_cast<std::size_t>(width) * height * channels;
  if (text.size() < offset + count * bytes) throw std::runtime_error(path.string() + ": truncated");
  std::vector<std::uint16_t> samples(count);
  const auto* data = reinterpret_cast<const unsigned char*>(text.data() + offset);
  for (std::size_t i = 0; i < count; ++i) {
    samples[i] =
        bytes == 2 ? static_cast<std::uint16_t>(data[2 * i] << 8U | data[2 * i + 1]) : data[i];
  }
  return samples;
}

// The numbers in the JSON array or value after `"key":` within the object
// `"section": {`, as the recorder writes its calibration.
std::vector<double> json_numbers(const std::string& json, const std::string& section,
                                 const std::string& key) {
  std::size_t at = json.find("\"" + section + "\": {");
  if (at == std::string::npos) throw std::runtime_error("calibration: no " + section);
  at = json.find("\"" + key + "\":", at);
  if (at == std::string::npos) throw std::runtime_error("calibration: no " + section + "." + key);
  at = json.find(':', at) + 1;
  std::vector<double> numbers;
  const bool array = json.find_first_not_of(" \n", at) != std::string::npos &&
                     json[json.find_first_not_of(" \n", at)] == '[';
  const std::size_t end = array ? json.find(']', at) : json.find_first_of(",}\n", at);
  std::istringstream in(json.substr(at, end - at));
  std::string token;
  while (std::getline(in, token, ',')) {
    token.erase(std::remove_if(token.begin(), token.end(),
                               [](char c) { return c == '[' || c == ' ' || c == '\n'; }),
                token.end());
    if (!token.empty()) numbers.push_back(std::stod(token));
  }
  return numbers;
}

struct Recorded {
  RgbFrame frame;
  // Meters in the RGB camera's pixels, 0 where the camera has no depth.
  std::vector<float> depth;
};

// Frame `index` of `clip`: its RGB image, and its depth (in the left
// infrared camera) reprojected into the RGB camera, nearest point first.
Recorded read_recorded(const std::filesystem::path& clip, int index) {
  const std::string json = read_text(clip / "calibration.json");
  char name[16];
  std::snprintf(name, sizeof(name), "%06d", index);
  Recorded result;
  int width = 0;
  int height = 0;
  const auto rgb = read_pnm(clip / "rgb" / (std::string(name) + ".ppm"), "P6", width, height);
  result.frame = {std::vector<std::uint8_t>(rgb.begin(), rgb.end()), width, height};
  int depth_width = 0;
  int depth_height = 0;
  const auto raw =
      read_pnm(clip / "depth" / (std::string(name) + ".pgm"), "P5", depth_width, depth_height);
  const double scale = json_numbers(json, "depth", "scale").at(0);
  const auto d = [&](const char* key) { return json_numbers(json, "depth", key).at(0); };
  const auto c = [&](const char* key) { return json_numbers(json, "rgb", key).at(0); };
  const std::vector<double> rotation = json_numbers(json, "rgb", "rotation");
  const std::vector<double> translation = json_numbers(json, "rgb", "translation");
  if (rotation.size() != 9 || translation.size() != 3) {
    throw std::runtime_error("calibration: rgb.from_left is not a rotation and a translation");
  }
  // The depth is in the left camera's frame (its from_left is the identity).
  const double fx = d("fx"), fy = d("fy"), cx = d("cx"), cy = d("cy");
  const double rfx = c("fx"), rfy = c("fy"), rcx = c("cx"), rcy = c("cy");
  result.depth.assign(static_cast<std::size_t>(width) * height, 0.0F);
  for (int v = 0; v < depth_height; ++v) {
    for (int u = 0; u < depth_width; ++u) {
      const std::uint16_t units = raw[static_cast<std::size_t>(v) * depth_width + u];
      if (units == 0) continue;
      const double z = units * scale;
      const double p[3] = {(u - cx) / fx * z, (v - cy) / fy * z, z};
      double q[3];
      for (int r = 0; r < 3; ++r) {
        q[r] = rotation[3 * r] * p[0] + rotation[3 * r + 1] * p[1] + rotation[3 * r + 2] * p[2] +
               translation[r];
      }
      if (q[2] <= 0.0) continue;
      const auto x = static_cast<int>(std::lround(rfx * q[0] / q[2] + rcx));
      const auto y = static_cast<int>(std::lround(rfy * q[1] / q[2] + rcy));
      if (x < 0 || y < 0 || x >= width || y >= height) continue;
      float& slot = result.depth[static_cast<std::size_t>(y) * width + x];
      if (slot == 0.0F || q[2] < slot) slot = static_cast<float>(q[2]);
    }
  }
  return result;
}

void check_recorded() {
  const char* root = std::getenv("HOLOSCAN_TEST_DATA");
  const std::filesystem::path clip =
      root == nullptr ? std::filesystem::path{}
                      : std::filesystem::path(root) / "stereo-camera" / "d585_720p_emitter_on";
  if (root == nullptr || *root == '\0' || !std::filesystem::is_directory(clip)) {
    std::cout << "  skipped: HOLOSCAN_TEST_DATA has no stereo-camera/d585_720p_emitter_on\n";
    return;
  }
  const std::vector<int> indices{0, 20, 40, 59};
  std::vector<Recorded> recorded;
  std::vector<RgbFrame> frames;
  for (const int index : indices) {
    recorded.push_back(read_recorded(clip, index));
    frames.push_back(recorded.back().frame);
  }
  const auto depths = predict(model_config(true, ResizeMode::kLetterbox), frames);
  if (!depths) return;
  for (std::size_t i = 0; i < depths->size() && i < recorded.size(); ++i) {
    const std::vector<float>& predicted = (*depths)[i];
    std::vector<float> a;
    std::vector<float> b;
    for (std::size_t p = 0; p < predicted.size(); p += 3) {
      if (recorded[i].depth[p] > 0.0F) {
        a.push_back(predicted[p]);
        b.push_back(1.0F / recorded[i].depth[p]);
      }
    }
    // The least-squares scale and shift of the inverse depth, and the
    // relative error of the depth it then gives.
    double sa = 0, sb = 0, saa = 0, sab = 0;
    for (std::size_t k = 0; k < a.size(); ++k) {
      sa += a[k];
      sb += b[k];
      saa += static_cast<double>(a[k]) * a[k];
      sab += static_cast<double>(a[k]) * b[k];
    }
    const double n = static_cast<double>(a.size());
    const double scale = (n * sab - sa * sb) / (n * saa - sa * sa);
    const double shift = (sb - scale * sa) / n;
    double absolute_relative = 0.0;
    std::size_t within = 0;
    std::size_t counted = 0;
    for (std::size_t k = 0; k < a.size(); ++k) {
      const double inverse = scale * a[k] + shift;
      if (inverse <= 0) continue;
      const double z = 1.0 / inverse;
      const double truth = 1.0 / b[k];
      absolute_relative += std::fabs(z - truth) / truth;
      within += std::max(z / truth, truth / z) < 1.25 ? 1U : 0U;
      ++counted;
    }
    const double rho = spearman(a, b);
    std::cout << std::fixed << std::setprecision(4) << "  recorded frame " << indices[i]
              << ": pixels=" << a.size() << " spearman=" << rho
              << " aligned AbsRel=" << absolute_relative / std::max<std::size_t>(counted, 1)
              << " delta<1.25=" << static_cast<double>(within) / std::max<std::size_t>(counted, 1)
              << " (1/z = " << scale << " * d + " << shift << ")\n";
    VERIFY_MSG(a.size() > 10000, "too few pixels with depth");
    VERIFY_MSG(rho >= 0.5, "Spearman's correlation " + std::to_string(rho));
  }
}

}  // namespace
}  // namespace holoscan::mono_depth

int main(int argc, char** argv) {
  namespace md = holoscan::mono_depth;
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " MODEL.onnx\n";
    return 2;
  }
  md::g_model = argv[1];
  std::error_code error;
  if (!std::filesystem::is_regular_file(md::g_model, error)) {
    std::cout << "skipped: " << md::g_model.string()
              << " is missing; export it with tools/prepare_model.py (see the README)\n";
    return md::kSkip;
  }
  using md::ResizeMode;
  return md::test::run({
      {"synthetic scene, FP16, letterboxed",
       [] { md::check_synthetic(true, ResizeMode::kLetterbox); }},
      {"synthetic scene, FP32, letterboxed",
       [] { md::check_synthetic(false, ResizeMode::kLetterbox); }},
      {"synthetic scene, FP16, stretched", [] { md::check_synthetic(true, ResizeMode::kStretch); }},
      {"recorded RealSense frames", md::check_recorded},
  });
}
