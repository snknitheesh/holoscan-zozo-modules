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
// - With --recorded, the local recordings: for each clip in
//   $HOLOSCAN_TEST_DATA/stereo-camera/<clip>/ with an rgb.yaml and a
//   depth.yaml (ROS camera_info files, read with holoscan-messages), RGB
//   frames against the clip's own depth, reprojected into the RGB camera: the
//   rank correlation, and the error left once one scale and shift per frame
//   map the inverse depth onto the depth's, which is the best a relative
//   model can do. The camera_info files hold no transform between the depth
//   and the RGB camera, so the depth is reprojected as if both had one
//   optical center; the distance between them shifts a point by fx * b / z
//   pixels (b their distance), which blurs depth edges but keeps the ranking.
//   A clip with depth on fewer than kMinDepthFraction of its pixels is left
//   out. Skipped (exit 77) when HOLOSCAN_TEST_DATA is not set; failed when it
//   names no clip with enough depth.
//
// Usage: mono_depth_real_model_test MODEL.onnx [--recorded --depth-scale M]
//   (M: meters per unit of the recorded 16-bit depth)

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
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/mono_depth/mono_depth_op.hpp>
#include <holoscan/mono_depth/synthetic_scene.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "test_support.hpp"

namespace holoscan::mono_depth {
namespace {

namespace hm = holoscan::messages;

constexpr double kMinSpearman = 0.85;
constexpr double kMinDepthFraction = 0.05;
constexpr int kSkip = 77;

std::filesystem::path g_model;
double g_depth_scale = 0.0;

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
  // Jetson Thor and Orin, would outlast the harness's watchdog.
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

struct Recorded {
  RgbFrame frame;
  // Meters in the RGB camera's pixels, 0 where the camera has no depth.
  std::vector<float> depth;
  // The fraction of the depth image's pixels with a depth.
  double depth_fraction{};
};

// The RGB frame `rgb_file` of a clip and the depth image `depth_file` taken
// with it, reprojected into the RGB camera, nearest point first.
Recorded read_recorded(const hm::CameraInfo& rgb_camera, const hm::CameraInfo& depth_camera,
                       const std::filesystem::path& rgb_file,
                       const std::filesystem::path& depth_file) {
  Recorded result;
  int width = 0;
  int height = 0;
  const auto rgb = read_pnm(rgb_file, "P6", width, height);
  if (width != static_cast<int>(rgb_camera.width) ||
      height != static_cast<int>(rgb_camera.height)) {
    throw std::runtime_error(rgb_file.string() + ": not of rgb.yaml's size");
  }
  result.frame = {std::vector<std::uint8_t>(rgb.begin(), rgb.end()), width, height};
  int depth_width = 0;
  int depth_height = 0;
  const auto raw = read_pnm(depth_file, "P5", depth_width, depth_height);
  if (depth_width != static_cast<int>(depth_camera.width) ||
      depth_height != static_cast<int>(depth_camera.height)) {
    throw std::runtime_error(depth_file.string() + ": not of depth.yaml's size");
  }
  const double fx = depth_camera.fx, fy = depth_camera.fy;
  const double cx = depth_camera.cx, cy = depth_camera.cy;
  const double rfx = rgb_camera.fx, rfy = rgb_camera.fy;
  const double rcx = rgb_camera.cx, rcy = rgb_camera.cy;
  result.depth.assign(static_cast<std::size_t>(width) * height, 0.0F);
  std::size_t valid = 0;
  for (int v = 0; v < depth_height; ++v) {
    for (int u = 0; u < depth_width; ++u) {
      const std::uint16_t units = raw[static_cast<std::size_t>(v) * depth_width + u];
      if (units == 0) continue;
      ++valid;
      const double z = units * g_depth_scale;
      const double x = (u - cx) / fx * z;
      const double y = (v - cy) / fy * z;
      const auto px = static_cast<int>(std::lround(rfx * x / z + rcx));
      const auto py = static_cast<int>(std::lround(rfy * y / z + rcy));
      if (px < 0 || py < 0 || px >= width || py >= height) continue;
      float& slot = result.depth[static_cast<std::size_t>(py) * width + px];
      if (slot == 0.0F || z < slot) slot = static_cast<float>(z);
    }
  }
  result.depth_fraction =
      static_cast<double>(valid) / (static_cast<double>(depth_width) * depth_height);
  return result;
}

// The clips under $HOLOSCAN_TEST_DATA/stereo-camera with an rgb.yaml and a
// depth.yaml, in name order.
std::vector<std::filesystem::path> recorded_clips(const char* root) {
  std::vector<std::filesystem::path> clips;
  const std::filesystem::path cameras = std::filesystem::path(root) / "stereo-camera";
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator(cameras, error)) {
    if (std::filesystem::is_regular_file(entry.path() / "rgb.yaml") &&
        std::filesystem::is_regular_file(entry.path() / "depth.yaml")) {
      clips.push_back(entry.path());
    }
  }
  std::sort(clips.begin(), clips.end());
  return clips;
}

// The files of a clip's `folder` with `extension`, in name order.
std::vector<std::filesystem::path> clip_files(const std::filesystem::path& folder,
                                              const std::string& extension) {
  std::vector<std::filesystem::path> files;
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator(folder, error)) {
    if (entry.path().extension() == extension) files.push_back(entry.path());
  }
  std::sort(files.begin(), files.end());
  return files;
}

std::size_t g_clips_checked = 0;

// Clips are named by their number: their folders may carry product names.
void check_clip(const std::filesystem::path& clip, std::size_t number) {
  const std::string name = "clip " + std::to_string(number);
  const hm::CameraInfo rgb_camera = hm::read_camera_info_yaml(clip / "rgb.yaml");
  const hm::CameraInfo depth_camera = hm::read_camera_info_yaml(clip / "depth.yaml");
  const std::vector<std::filesystem::path> rgb_files = clip_files(clip / "rgb", ".ppm");
  VERIFY_MSG(!rgb_files.empty(), name + ": no rgb/*.ppm");
  if (rgb_files.empty()) return;
  // Four frames spread over the clip, each with the depth image of its name.
  std::vector<Recorded> recorded;
  std::vector<RgbFrame> frames;
  std::vector<std::string> names;
  for (const double at : {0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0}) {
    const auto index = static_cast<std::size_t>(at * static_cast<double>(rgb_files.size() - 1));
    const std::filesystem::path& rgb_file = rgb_files[index];
    const std::filesystem::path depth_file = clip / "depth" / (rgb_file.stem().string() + ".pgm");
    recorded.push_back(read_recorded(rgb_camera, depth_camera, rgb_file, depth_file));
    if (recorded.back().depth_fraction < kMinDepthFraction) {
      std::cout << "  " << name << ": left out, depth on "
                << static_cast<int>(100.0 * recorded.back().depth_fraction) << " % of the pixels\n";
      return;
    }
    frames.push_back(recorded.back().frame);
    names.push_back(rgb_file.stem().string());
  }
  const auto depths = predict(model_config(true, ResizeMode::kLetterbox), frames);
  if (!depths) return;
  ++g_clips_checked;
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
    std::cout << std::fixed << std::setprecision(4) << "  " << name << ", frame " << names[i]
              << ": pixels=" << a.size() << " spearman=" << rho
              << " aligned AbsRel=" << absolute_relative / std::max<std::size_t>(counted, 1)
              << " delta<1.25=" << static_cast<double>(within) / std::max<std::size_t>(counted, 1)
              << " (1/z = " << scale << " * d + " << shift << ")\n";
    VERIFY_MSG(a.size() > 10000, name + ": too few pixels with depth");
    VERIFY_MSG(rho >= 0.5, name + ": Spearman's correlation " + std::to_string(rho));
  }
}

}  // namespace
}  // namespace holoscan::mono_depth

int main(int argc, char** argv) {
  namespace md = holoscan::mono_depth;
  const std::string usage =
      std::string("usage: ") + argv[0] + " MODEL.onnx [--recorded --depth-scale METERS_PER_UNIT]\n";
  // An empty model path may arrive as no argument at all.
  bool recorded = false;
  std::optional<std::filesystem::path> model;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--recorded") {
      recorded = true;
    } else if (option == "--depth-scale" && i + 1 < argc) {
      md::g_depth_scale = std::strtod(argv[++i], nullptr);
    } else if (!model && option.rfind("--", 0) != 0) {
      model = option;
    } else {
      std::cerr << usage;
      return 2;
    }
  }
  if (recorded && !(md::g_depth_scale > 0.0)) {
    std::cerr << usage;
    return 2;
  }
  md::g_model = model.value_or(std::filesystem::path{});
  std::error_code error;
  if (md::g_model.empty() || !std::filesystem::is_regular_file(md::g_model, error)) {
    std::cout << "skipped: no model at '" << md::g_model.string()
              << "'; export it with tools/prepare_model.py and give its path in "
                 "HOLOSCAN_MONO_DEPTH_MODEL (see the README)\n";
    return md::kSkip;
  }
  using md::ResizeMode;
  if (!recorded) {
    return md::test::run({
        {"synthetic scene, FP16, letterboxed",
         [] { md::check_synthetic(true, ResizeMode::kLetterbox); }},
        {"synthetic scene, FP32, letterboxed",
         [] { md::check_synthetic(false, ResizeMode::kLetterbox); }},
        {"synthetic scene, FP16, stretched",
         [] { md::check_synthetic(true, ResizeMode::kStretch); }},
    });
  }
  const char* root = std::getenv("HOLOSCAN_TEST_DATA");
  if (root == nullptr || *root == '\0') {
    std::cout << "skipped: HOLOSCAN_TEST_DATA is not set\n";
    return md::kSkip;
  }
  const std::vector<std::filesystem::path> clips = md::recorded_clips(root);
  if (clips.empty()) {
    std::cerr << "no stereo-camera/<clip>/ with rgb.yaml and depth.yaml in " << root << '\n';
    return 1;
  }
  std::vector<md::test::TestCase> cases;
  for (std::size_t i = 0; i < clips.size(); ++i) {
    cases.push_back(
        {"recorded clip " + std::to_string(i), [clip = clips[i], i] { md::check_clip(clip, i); }});
  }
  cases.push_back({"a clip with depth was checked", [] { VERIFY(md::g_clips_checked > 0); }});
  return md::test::run(cases);
}
