// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// UndistortOp and StereoRectifyOp through the SDK's operator test harness,
// calibrated from camera_info files written for the synthetic pair: distorted
// pairs from host and device memory, rectified and measured against the known
// geometry; the camera info sent with every pair and image, stamped with its
// capture time; color and 16-bit images; the capture time and header carried
// to the outputs; auto's choice; and the files, configurations and images the
// operators refuse.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/rectify/remapper.hpp>
#include <holoscan/rectify/stereo_rectify_op.hpp>
#include <holoscan/rectify/synthetic_stereo.hpp>
#include <holoscan/rectify/undistort_op.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "tensor_support.hpp"
#include "test_support.hpp"

namespace holoscan::rectify {
namespace {

using std::chrono_literals::operator""ms;
using holoscan::schema::ImageEncoding_MONO16;
using holoscan::schema::ImageEncoding_MONO8;
using holoscan::schema::ImageEncoding_RGB8;
using ImageSnapshot = holoscan::testing::Snapshot<holoscan::schema::ImageT>;

constexpr int kWidth = 640;
constexpr int kHeight = 480;
// Records each capture keeps. The harness sizes its output pools by it, so a
// small number keeps the GPU memory of a test small.
constexpr std::size_t kCaptures = 8;

std::chrono::nanoseconds capture_time(std::uint32_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

const SyntheticStereo& scene() {
  static const SyntheticStereo s = synthetic_stereo(kWidth, kHeight);
  return s;
}

// The synthetic pair's camera_info files, written once for every test.
struct Files {
  test::TempDir dir;
  std::filesystem::path left = dir / "left.yaml";
  std::filesystem::path right = dir / "right.yaml";

  Files() { messages::write_stereo_camera_info_yaml(scene().calibration, left, right); }
};

const Files& files() {
  static const Files f;
  return f;
}

// The rectification the operator computes from the files.
const StereoRectification& rectification() {
  static const StereoRectification r = compute_stereo_rectification(
      messages::read_stereo_camera_info_yaml(files().left, files().right));
  return r;
}

StereoRectifyConfig stereo_config(Backend backend) {
  return StereoRectifyConfig{.left_camera_info_path = files().left,
                             .right_camera_info_path = files().right,
                             .backend = backend};
}

UndistortConfig undistort_config(Interpolation interpolation = Interpolation::kLinear) {
  return UndistortConfig{.camera_info_path = files().left, .interpolation = interpolation};
}

// Frames 0..frames-1 of the synthetic pair, rectified by StereoRectifyOp,
// checked against the geometry.
void check_stereo(Backend backend, bool device, std::uint32_t frames) {
  const std::string what =
      std::string(backend_name(backend)) + (device ? " device input" : " host input");
  auto h = holoscan::testing::create_operator_test<StereoRectifyOp>(stereo_config(backend));
  const SyntheticView left = left_view(scene());
  const SyntheticView right = right_view(scene());
  for (std::uint32_t f = 0; f < frames; ++f) {
    h.push_frame(holoscan::testing::frame_at(
        capture_time(f),
        holoscan::testing::input("left", test::image(left.render(f), kWidth, kHeight, 1,
                                                     ImageEncoding_MONO8, device, f + 1)),
        holoscan::testing::input("right", test::image(right.render(f), kWidth, kHeight, 1,
                                                      ImageEncoding_MONO8, device, f + 1))));
  }
  if (device) {
    h.input_source_device("left", holoscan::DeviceId{0});
    h.input_source_device("right", holoscan::DeviceId{0});
  }
  h.output_device("rectified_left", holoscan::DeviceId{0});
  h.output_device("rectified_right", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("rectified_left", kCaptures);
  h.capture_snapshot<holoscan::schema::ImageT>("rectified_right", kCaptures);
  h.capture<messages::schema::StereoCameraInfoT>("camera_info", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), what + ": " + result.describe_failure());
  if (!result.ok()) return;
  const auto lefts = result.snapshots<holoscan::schema::ImageT>("rectified_left");
  const auto rights = result.snapshots<holoscan::schema::ImageT>("rectified_right");
  const auto infos = result.samples<messages::schema::StereoCameraInfoT>("camera_info");
  VERIFY(lefts.size() == frames && rights.size() == frames && infos.size() == frames);
  if (lefts.size() != frames || rights.size() != frames) return;
  const StereoRectification& r = rectification();
  for (std::uint32_t f = 0; f < frames; ++f) {
    const holoscan::schema::ImageT descriptor = lefts[f].metadata();
    VERIFY(descriptor.width == r.width && descriptor.height == r.height);
    VERIFY(descriptor.encoding == ImageEncoding_MONO8);
    VERIFY(descriptor.header && descriptor.header->device_sequence == f + 1 &&
           descriptor.header->frame_id == "camera");
    VERIFY(lefts[f].sample_metadata().capture_timestamp_ns == capture_time(f).count());
    VERIFY(rights[f].sample_metadata().capture_timestamp_ns == capture_time(f).count());
    const auto l = test::pixels<std::uint8_t>(lefts[f]);
    const auto rr = test::pixels<std::uint8_t>(rights[f]);
    const AlignmentReport report = check_rectified(scene(), r, l.data(), rr.data(), r.width, f);
    std::cout << "  " << what << " frame " << f << ": " << report.dots << " dots, row error max "
              << report.max_row_error << " px, position error max " << report.max_position_error
              << " px\n";
    VERIFY(report.dots > 50 && report.missing == 0);
    VERIFY_MSG(report.max_row_error < 0.1, what + ": rows differ");
    VERIFY_MSG(report.max_position_error < 0.15, what + ": dots misplaced");
  }
  // One camera info per pair: the rectified pair, stamped with the pair's capture time.
  for (std::size_t f = 0; f < infos.size(); ++f) {
    const std::int64_t at = capture_time(static_cast<std::uint32_t>(f)).count();
    VERIFY(infos[f].metadata.capture_timestamp_ns == at);
    const messages::StereoCameraInfo info = messages::from_message(infos[f].data);
    VERIFY_MSG(messages::same_calibration(info, r.camera_info),
               what + ": the camera info is not the rectified pair");
    VERIFY(info.header.capture_timestamp_ns == at && info.left.header.capture_timestamp_ns == at &&
           info.right.header.capture_timestamp_ns == at);
    VERIFY(info.header.frame_id == "left" && info.right.header.frame_id == "right");
  }
}

void test_stereo_cuda() {
  check_stereo(Backend::kCuda, false, 2);
  check_stereo(Backend::kCuda, true, 2);
}

void test_stereo_vpi() {
  for (const Backend backend : available_backends()) {
    if (backend == Backend::kCuda) continue;
    if (try_backend(backend, PixelFormat::kMono8, Interpolation::kLinear, kWidth, kHeight)) {
      continue;
    }
    check_stereo(backend, true, 2);
  }
  check_stereo(Backend::kAuto, true, 1);
}

// UndistortOp on color and on 16-bit depth, from host memory.
void test_undistort_formats() {
  const messages::CameraInfo camera = messages::read_camera_info_yaml(files().left);
  const Undistortion u = compute_undistortion(camera);
  const int out_width = static_cast<int>(u.output.width);
  const int out_height = static_cast<int>(u.output.height);
  const SyntheticView view(camera, messages::RigidTransform{}, scene().plane);
  const std::vector<std::uint8_t> gray = view.render(1);
  // RGB8: three copies of the gray image, which each must come out as the gray one does.
  {
    std::vector<std::uint8_t> rgb(gray.size() * 3);
    for (std::size_t i = 0; i < gray.size(); ++i) {
      rgb[3 * i] = gray[i];
      rgb[3 * i + 1] = gray[i];
      rgb[3 * i + 2] = gray[i];
    }
    auto h = holoscan::testing::create_operator_test<UndistortOp>(undistort_config());
    h.push_at("image", test::image(rgb, kWidth, kHeight, 3, ImageEncoding_RGB8, false), 100ms);
    h.output_device("undistorted", holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::schema::ImageT>("undistorted", kCaptures);
    h.capture<messages::schema::CameraInfoT>("camera_info", kCaptures);
    const auto result = h.run();
    VERIFY_MSG(result.ok(), result.describe_failure());
    if (result.ok()) {
      const auto out = result.snapshot<holoscan::schema::ImageT>("undistorted");
      VERIFY(out.metadata().encoding == ImageEncoding_RGB8);
      VERIFY((out.tensor(holoscan::schema::ImageTensorFields::data).shape() ==
              std::vector<std::int64_t>{out_height, out_width, 3}));
      // One camera info, the undistorted camera, stamped with the image's capture time.
      const auto infos = result.samples<messages::schema::CameraInfoT>("camera_info");
      VERIFY(infos.size() == 1);
      if (infos.size() == 1) {
        const std::int64_t at = std::chrono::nanoseconds(100ms).count();
        VERIFY(infos.front().metadata.capture_timestamp_ns == at);
        const messages::CameraInfo info = messages::from_message(infos.front().data);
        VERIFY(messages::same_calibration(info, u.output));
        VERIFY(info.state == messages::CameraState::kUndistorted);
        VERIFY(info.header.capture_timestamp_ns == at && info.header.frame_id == "left");
      }
      const auto pixels = test::pixels<std::uint8_t>(out);
      std::vector<std::uint8_t> green(pixels.size() / 3);
      for (std::size_t i = 0; i < green.size(); ++i) green[i] = pixels[3 * i + 1];
      const AlignmentReport report =
          check_undistorted(scene().plane, u, green.data(), u.output.width, 1);
      std::cout << "  RGB8: " << report.dots << " dots, position error max "
                << report.max_position_error << " px\n";
      VERIFY(report.dots > 50 && report.max_position_error < 0.15);
    }
  }
  // MONO16 depth, nearest: every output value is one of the input's.
  {
    std::vector<std::uint16_t> depth(gray.size());
    for (std::size_t i = 0; i < gray.size(); ++i)
      depth[i] = static_cast<std::uint16_t>(gray[i] * 37);
    auto h = holoscan::testing::create_operator_test<UndistortOp>(
        undistort_config(Interpolation::kNearest));
    h.push_at("image", test::image(depth, kWidth, kHeight, 1, ImageEncoding_MONO16, true), 100ms);
    h.input_source_device("image", holoscan::DeviceId{0});
    h.output_device("undistorted", holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::schema::ImageT>("undistorted", kCaptures);
    h.capture<messages::schema::CameraInfoT>("camera_info", kCaptures);
    const auto result = h.run();
    VERIFY_MSG(result.ok(), result.describe_failure());
    if (result.ok()) {
      const auto out = result.snapshot<holoscan::schema::ImageT>("undistorted");
      VERIFY(out.metadata().encoding == ImageEncoding_MONO16);
      const auto values = test::pixels<std::uint16_t>(out);
      int blended = 0;
      for (const std::uint16_t v : values) blended += v % 37 != 0;
      VERIFY_MSG(blended == 0, std::to_string(blended) + " depths are not input depths");
    }
  }
}

// Inputs the operators refuse fail the run with a clear error.
void check_refused(holoscan::schema::ImageT image, const std::string& what,
                   const std::string& expected) {
  auto h = holoscan::testing::create_operator_test<UndistortOp>(undistort_config());
  h.push_at("image", std::move(image), 100ms);
  h.output_device("undistorted", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("undistorted", kCaptures);
  h.capture<messages::schema::CameraInfoT>("camera_info", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), what + ": accepted");
  if (!result.ok()) {
    VERIFY_MSG(result.describe_failure().find(expected) != std::string::npos,
               what + ": " + result.describe_failure());
  }
}

void test_refusals() {
  const std::vector<std::uint8_t> gray(static_cast<std::size_t>(kWidth) * kHeight, 9);
  check_refused(
      test::image(std::vector<std::uint8_t>(320 * 240), 320, 240, 1, ImageEncoding_MONO8, false),
      "wrong size", "the camera 640x480");
  check_refused(test::image(std::vector<std::uint16_t>(gray.size()), kWidth, kHeight, 1,
                            ImageEncoding_MONO16, false),
                "blended depth", "nearest");
  {
    const std::vector<float> meters(gray.size(), 1.5F);
    holoscan::schema::ImageT image;
    image.width = kWidth;
    image.height = kHeight;
    image.encoding = holoscan::schema::ImageEncoding_MONO32F;
    image.data = std::make_shared<holoscan::Tensor>(
        test::make({kHeight, kWidth}, {kDLFloat, 32, 1}, meters.data(), meters.size() * 4, false));
    check_refused(image, "unsupported encoding", "unsupported encoding MONO32F");
  }
  // Configurations and files that cannot work are refused at construction.
  const auto refused = [](const std::string& what, const auto& make) {
    VERIFY_MSG(test::throws<std::invalid_argument>(make), what);
  };
  refused("negative alpha", [] {
    UndistortConfig config = undistort_config();
    config.options.alpha = -1;
    UndistortOp op(config);
  });
  refused("no file", [] { UndistortOp op(UndistortConfig{}); });
  refused("no right file",
          [] { StereoRectifyOp op(StereoRectifyConfig{.left_camera_info_path = files().left}); });
  const test::TempDir dir;
  refused("a missing file", [&] {
    StereoRectifyOp op(StereoRectifyConfig{.left_camera_info_path = files().left,
                                           .right_camera_info_path = dir / "missing.yaml"});
  });
  // A camera of the pair that this module cannot remap: one with skew.
  messages::StereoCameraInfo skewed = scene().calibration;
  skewed.left.skew = 0.5;
  messages::write_stereo_camera_info_yaml(skewed, dir / "left.yaml", dir / "right.yaml");
  refused("skew", [&] {
    StereoRectifyOp op(StereoRectifyConfig{.left_camera_info_path = dir / "left.yaml",
                                           .right_camera_info_path = dir / "right.yaml"});
  });
  if (!HOLOSCAN_RECTIFY_TEST_WITH_VPI) {
    refused("vpi-vic without VPI", [] {
      UndistortConfig config = undistort_config();
      config.backend = Backend::kVpiVic;
      UndistortOp op(config);
    });
  }
}

}  // namespace
}  // namespace holoscan::rectify

int main() {
  namespace rc = holoscan::rectify;
  return rc::test::run({
      {"StereoRectifyOp, cuda backend", rc::test_stereo_cuda},
      {"StereoRectifyOp, VPI backends and auto", rc::test_stereo_vpi},
      {"UndistortOp, RGB8 and MONO16", rc::test_undistort_formats},
      {"refusals", rc::test_refusals},
  });
}
