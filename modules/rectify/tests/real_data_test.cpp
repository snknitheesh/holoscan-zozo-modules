// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Real recordings: RealSense infrared pairs that the camera has already
// rectified (identity rotation, no distortion, equal intrinsics). Rectifying
// them again must change nothing: the geometry keeps the camera's, the maps
// are the identity, and StereoRectifyOp gives back the images it got.
// Reads $HOLOSCAN_TEST_DATA/stereo-camera/<clip>/; skipped (exit 77) when the
// variable is unset.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/rectify/camera_model.hpp>
#include <holoscan/rectify/rectification.hpp>
#include <holoscan/rectify/stereo_rectify_op.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "tensor_support.hpp"
#include "test_support.hpp"

namespace holoscan::rectify {
namespace {

namespace fs = std::filesystem;

constexpr int kSkip = 77;
constexpr int kFrames = 3;
// Records each capture keeps. The harness sizes its output pools by it, so a
// small number keeps the GPU memory of a test small.
constexpr std::size_t kCaptures = 8;

struct Gray {
  int width{};
  int height{};
  std::vector<std::uint8_t> pixels;
};

// An 8-bit binary PGM (P5), as the recordings hold the infrared eyes.
Gray read_pgm(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error(path.string() + ": cannot open");
  std::string magic;
  file >> magic;
  int values[3] = {};
  for (int& value : values) {
    file >> std::ws;
    while (file.peek() == '#') {
      std::string comment;
      std::getline(file, comment);
      file >> std::ws;
    }
    file >> value;
  }
  file.get();  // The single whitespace before the pixels.
  if (magic != "P5" || values[2] != 255 || values[0] < 1 || values[1] < 1) {
    throw std::runtime_error(path.string() + ": not an 8-bit binary PGM");
  }
  Gray image{values[0], values[1], {}};
  image.pixels.resize(static_cast<std::size_t>(image.width) * image.height);
  file.read(reinterpret_cast<char*>(image.pixels.data()),
            static_cast<std::streamsize>(image.pixels.size()));
  if (!file) throw std::runtime_error(path.string() + ": truncated");
  return image;
}

// The left and right paths of the first frames, from index.csv.
std::vector<std::pair<fs::path, fs::path>> first_pairs(const fs::path& clip) {
  std::ifstream index(clip / "index.csv");
  std::string line;
  std::getline(index, line);  // Header.
  std::vector<std::pair<fs::path, fs::path>> pairs;
  while (pairs.size() < kFrames && std::getline(index, line)) {
    std::vector<std::string> fields;
    std::stringstream row(line);
    std::string field;
    while (std::getline(row, field, ',')) fields.push_back(field);
    if (fields.size() >= 5) pairs.emplace_back(clip / fields[3], clip / fields[4]);
  }
  return pairs;
}

void check_clip(const fs::path& clip) {
  const std::string name = clip.filename().string();
  const StereoCalibration calibration = read_stereo_calibration(clip / "calibration.json");
  const StereoRectification r = compute_stereo_rectification(calibration);
  const CameraModel& left = calibration.left;
  VERIFY_MSG(r.width == left.width && r.height == left.height, name + ": size changed");
  VERIFY_MSG(std::abs(r.fx - left.fy) < 1e-9, name + ": focal length changed");
  VERIFY_MSG(std::abs(r.cx_left - left.cx) < 1e-6 && std::abs(r.cy - left.cy) < 1e-6,
             name + ": principal point changed");
  VERIFY_MSG(std::abs(r.baseline - std::abs(calibration.right_from_left.translation[0])) < 1e-12,
             name + ": baseline changed");
  for (const RemapModel& model : {left_remap(calibration, r), right_remap(calibration, r)}) {
    const RemapTable table = build_remap_table(model, r.width, r.height);
    double worst = 0.0;
    for (int v = 0; v < r.height; ++v) {
      for (int u = 0; u < r.width; ++u) {
        const std::size_t i = (static_cast<std::size_t>(v) * r.width + u) * 2;
        worst = std::max({worst, std::abs(static_cast<double>(table.xy[i]) - u),
                          std::abs(static_cast<double>(table.xy[i + 1]) - v)});
      }
    }
    VERIFY_MSG(worst < 1e-3, name + ": the map moves a pixel by " + std::to_string(worst));
  }

  // The images through StereoRectifyOp, from host memory, on the platform's auto backend.
  auto h = holoscan::testing::create_operator_test<StereoRectifyOp>(
      StereoRectifyConfig{.calibration = calibration, .backend = Backend::kAuto});
  std::vector<Gray> lefts;
  std::vector<Gray> rights;
  std::uint32_t frame = 0;
  for (const auto& [left_path, right_path] : first_pairs(clip)) {
    lefts.push_back(read_pgm(left_path));
    rights.push_back(read_pgm(right_path));
    const auto at = std::chrono::milliseconds(33 * (frame++ + 1));
    h.push_frame(holoscan::testing::frame_at(
        at,
        holoscan::testing::input(
            "left", test::image(lefts.back().pixels, lefts.back().width, lefts.back().height, 1,
                                holoscan::schema::ImageEncoding_MONO8, false)),
        holoscan::testing::input(
            "right", test::image(rights.back().pixels, rights.back().width, rights.back().height, 1,
                                 holoscan::schema::ImageEncoding_MONO8, false))));
  }
  VERIFY_MSG(lefts.size() == kFrames, name + ": too few frames in index.csv");
  h.output_device("rectified_left", holoscan::DeviceId{0});
  h.output_device("rectified_right", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("rectified_left", kCaptures);
  h.capture_snapshot<holoscan::schema::ImageT>("rectified_right", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), name + ": " + result.describe_failure());
  if (!result.ok()) return;
  const auto out_left = result.snapshots<holoscan::schema::ImageT>("rectified_left");
  const auto out_right = result.snapshots<holoscan::schema::ImageT>("rectified_right");
  VERIFY(out_left.size() == lefts.size() && out_right.size() == rights.size());
  int worst = 0;
  for (std::size_t f = 0; f < std::min(out_left.size(), lefts.size()); ++f) {
    for (const auto& [out, in] :
         {std::pair{&out_left[f], &lefts[f]}, std::pair{&out_right[f], &rights[f]}}) {
      const auto pixels = test::pixels<std::uint8_t>(*out);
      VERIFY(pixels.size() == in->pixels.size());
      for (std::size_t i = 0; i < std::min(pixels.size(), in->pixels.size()); ++i) {
        worst = std::max(worst, std::abs(static_cast<int>(pixels[i]) - in->pixels[i]));
      }
    }
  }
  std::cout << "  " << name << ": " << r.width << "x" << r.height << ", fx " << r.fx
            << ", baseline " << r.baseline << " m, largest pixel change " << worst << '\n';
  VERIFY_MSG(worst <= 1, name + ": rectifying changed the images");
}

}  // namespace
}  // namespace holoscan::rectify

int main() {
  namespace rc = holoscan::rectify;
  namespace fs = std::filesystem;
  const char* root = std::getenv("HOLOSCAN_TEST_DATA");
  if (root == nullptr || *root == '\0') {
    std::cout << "HOLOSCAN_TEST_DATA is not set; skipping the real-data tests\n";
    return rc::kSkip;
  }
  const fs::path base = fs::path(root) / "stereo-camera";
  std::vector<fs::path> clips;
  if (fs::is_directory(base)) {
    for (const auto& entry : fs::directory_iterator(base)) {
      if (fs::is_regular_file(entry.path() / "calibration.json")) clips.push_back(entry.path());
    }
  }
  std::sort(clips.begin(), clips.end());
  if (clips.empty()) {
    std::cout << base << " holds no recordings; skipping the real-data tests\n";
    return rc::kSkip;
  }
  std::vector<rc::test::TestCase> cases;
  for (const fs::path& clip : clips) {
    cases.push_back({"recording " + clip.filename().string(), [clip] { rc::check_clip(clip); }});
  }
  return rc::test::run(cases);
}
