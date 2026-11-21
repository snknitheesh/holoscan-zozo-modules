// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The module's host-only pieces, without the SDK, CUDA or a GPU: the stub
// model file and its formula, and the engines of a replaced one; the camera
// info rules, the depth's camera and the depth formula;
// and the CPU references that the GPU tests compare against, on cases worked
// out by hand.
//
// Usage: foundation_stereo_host_test DIR (a scratch folder for the files)

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/foundation_stereo/camera_info.hpp>
#include <holoscan/foundation_stereo/depth.hpp>
#include <holoscan/foundation_stereo/reference.hpp>
#include <holoscan/foundation_stereo/stub_model.hpp>
#include <holoscan/messages/camera_info.hpp>
#include <holoscan/messages/synthetic_calibration.hpp>

#include "test_support.hpp"

namespace holoscan::foundation_stereo {
namespace {

std::filesystem::path g_directory;

std::string read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void check_stub_model_file() {
  VERIFY(stub_model_onnx(160, 96) == stub_model_onnx(160, 96));
  VERIFY(stub_model_onnx(160, 96) != stub_model_onnx(96, 160));
  VERIFY(test::throws<std::invalid_argument>([] { (void)stub_model_onnx(4, 96); }));
  VERIFY(test::throws<std::invalid_argument>([] { (void)stub_model_onnx(160, 5000); }));
  const std::string model = stub_model_onnx(160, 96);
  for (const char* name : {"left_image", "right_image", "disparity", "ReduceSum"}) {
    VERIFY_MSG(model.find(name) != std::string::npos, std::string("the model names ") + name);
  }
  const auto folder = g_directory / "stub";
  std::filesystem::remove_all(folder);
  std::filesystem::create_directories(folder);
  const auto path = write_stub_model(folder, 160, 96);
  VERIFY(path == folder / "foundation_stereo_stub_160x96.onnx");
  VERIFY(read_file(path) == model);
  // An unchanged file is left alone, so an engine cached beside it stays
  // valid; a changed one is replaced.
  const auto written = std::filesystem::last_write_time(path);
  VERIFY(write_stub_model(folder, 160, 96) == path &&
         std::filesystem::last_write_time(path) == written);
  // A changed file is replaced, and the engines cached for it go with it;
  // another model's are kept.
  std::ofstream(path, std::ios::binary) << "corrupt";
  const auto stale = folder / "foundation_stereo_stub_160x96.GPU.trt.10.cfg0.engine.fp16";
  const auto other = folder / "foundation_stereo_stub_96x160.GPU.trt.10.cfg0.engine.fp16";
  std::ofstream(stale) << "engine";
  std::ofstream(other) << "engine";
  VERIFY(write_stub_model(folder, 160, 96) == path && read_file(path) == model);
  VERIFY(!std::filesystem::exists(stale) && std::filesystem::exists(other));
  VERIFY(test::throws<std::runtime_error>(
      [] { (void)write_stub_model(g_directory / "no" / "such" / "folder", 160, 96); }));
}

void check_stub_formula() {
  const float gray[3] = {128.0F, 128.0F, 128.0F};
  // 128 * (1/8 + 1/16 + 1/32) - 64 / 64.
  VERIFY(stub_disparity(gray, 64.0F) == 28.0F - 1.0F);
  const float red[3] = {255.0F, 0.0F, 0.0F};
  VERIFY(stub_disparity(red, 255.0F) == 255.0F / 8 - 255.0F / 64);
  const float black[3] = {0.0F, 0.0F, 0.0F};
  VERIFY(stub_disparity(black, 64.0F) < 0.0F);
}

// A rectified 1280x720 pair with a 95 mm baseline and 2.5 px of doffs.
messages::StereoCameraInfo rectified_pair() {
  messages::CameraInfo left;
  left.header.frame_id = "left";
  left.width = 1280;
  left.height = 720;
  left.state = messages::CameraState::kRectified;
  left.fx = 337.0;
  left.fy = 337.0;
  left.cx = 640.5;
  left.cy = 360.5;
  messages::CameraInfo right = left;
  right.header.frame_id = "right";
  right.cx = 643.0;
  messages::StereoCameraInfo pair;
  pair.header.frame_id = "left";
  pair.left = left;
  pair.right = right;
  pair.right_from_left.translation = {-0.095, 0.0, 0.0};
  pair.baseline = 0.095;
  pair.doffs = 2.5;
  messages::validate(pair);
  return pair;
}

void check_camera_info() {
  const messages::StereoCameraInfo pair = rectified_pair();
  VERIFY(!camera_info_problem(pair, 1280, 720));
  const auto size = camera_info_problem(pair, 640, 480);
  VERIFY(size && size->find("is for 1280x720 (left)") != std::string::npos);
  const auto raw = camera_info_problem(
      messages::synthetic_stereo_camera_info(messages::DistortionModel::kPlumbBob, 1280, 720), 1280,
      720);
  VERIFY(raw && raw->find("must be a rectified pair") != std::string::npos);
  // The depth's camera: the left camera, with the pair's frame id when it has
  // none, and no capture time.
  messages::StereoCameraInfo stamped = pair;
  stamped.left.header.frame_id.clear();
  stamped.left.header.capture_timestamp_ns = 42;
  const messages::CameraInfo depth = depth_camera_info(stamped);
  VERIFY(depth.width == 1280 && depth.height == 720 && depth.fx == 337.0 && depth.cx == 640.5);
  VERIFY(depth.state == messages::CameraState::kRectified);
  VERIFY(depth.header.frame_id == "left" && depth.header.capture_timestamp_ns == 0);
  VERIFY(test::throws<std::invalid_argument>([] {
    (void)depth_camera_info(
        messages::synthetic_stereo_camera_info(messages::DistortionModel::kPlumbBob, 640, 480));
  }));
  // The pair's depth formula, as the kernels use it.
  VERIFY(messages::fx_baseline(pair) == 337.0 * 0.095);
  // z = fx * B / (d + doffs); 0 for a disparity of 0 or less.
  VERIFY(depth_from_disparity(32.0F, 32.0F, 0.0F) == 1.0F);
  VERIFY(depth_from_disparity(30.0F, 32.0F, 2.0F) == 1.0F);
  VERIFY(depth_from_disparity(0.0F, 32.0F, 0.0F) == 0.0F);
  VERIFY(depth_from_disparity(-3.0F, 32.0F, 0.0F) == 0.0F);
  VERIFY(depth_from_disparity(1.0F, 32.0F, -2.0F) == 0.0F);
  VERIFY(depth_from_disparity(std::nanf(""), 32.0F, 0.0F) == 0.0F);
}

void check_references() {
  // A 2x1 RGB image to 4x1: half-pixel centers sample at -0.25 (clamped to
  // 0), 0.25, 0.75 and 1.25 (clamped to 1).
  const std::vector<std::uint8_t> pixels{0, 10, 20, 100, 110, 120};
  const auto out = reference::preprocess(pixels.data(), 6, 2, 1, PixelFormat::kRgb8, 4, 1);
  const double red[4] = {0, 25, 75, 100};
  for (int x = 0; x < 4; ++x) {
    VERIFY(std::abs(out[x] - (red[x] - kImageMean[0]) / kImageStd[0]) < 1e-6);
  }
  // BGR swaps the first and last channels.
  const auto bgr = reference::preprocess(pixels.data(), 6, 2, 1, PixelFormat::kBgr8, 4, 1);
  VERIFY(std::abs(bgr[0] - (20 - kImageMean[0]) / kImageStd[0]) < 1e-6);
  // A 2-wide disparity of 1 and 3 to 4 wide: values times 2, interpolated.
  const auto scaled = reference::rescale({1.0F, 3.0F}, 2, 1, 4, 1);
  const double expected[4] = {2, 3, 5, 6};
  for (int x = 0; x < 4; ++x) VERIFY(std::abs(scaled[x] - expected[x]) < 1e-12);
  VERIFY(reference::valid(2, 1, true) == 0);
  VERIFY(reference::valid(2, 1, false) == 2);
  VERIFY(reference::valid(-1, 5, false) == 0);
  VERIFY(reference::valid(5, 5, true) == 5);
}

}  // namespace
}  // namespace holoscan::foundation_stereo

int main(int argc, char** argv) {
  namespace fs = holoscan::foundation_stereo;
  if (argc != 2) return 2;
  fs::g_directory = argv[1];
  std::filesystem::create_directories(fs::g_directory);
  return fs::test::run({
      {"stub model file", fs::check_stub_model_file},
      {"stub formula", fs::check_stub_formula},
      {"camera info and depth", fs::check_camera_info},
      {"CPU references", fs::check_references},
  });
}
