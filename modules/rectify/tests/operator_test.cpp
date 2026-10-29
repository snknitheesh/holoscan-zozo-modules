// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// UndistortOp and StereoRectifyOp through the SDK's operator test harness:
// synthetic distorted pairs from host and device memory, rectified and
// measured against the known geometry; the geometry message; color and 16-bit
// images; the capture time and header carried to the outputs; auto's choice;
// and images the operators refuse.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

// Frames 0..frames-1 of the synthetic pair, rectified by StereoRectifyOp,
// checked against the geometry.
void check_stereo(Backend backend, bool device, std::uint32_t frames) {
  const std::string what =
      std::string(backend_name(backend)) + (device ? " device input" : " host input");
  StereoRectifyConfig config{
      .calibration = scene().calibration, .backend = backend, .geometry_output = true};
  auto h = holoscan::testing::create_operator_test<StereoRectifyOp>(config);
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
  h.capture<schema::StereoGeometryT>("geometry", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), what + ": " + result.describe_failure());
  if (!result.ok()) return;
  const auto lefts = result.snapshots<holoscan::schema::ImageT>("rectified_left");
  const auto rights = result.snapshots<holoscan::schema::ImageT>("rectified_right");
  const auto geometries = result.values<schema::StereoGeometryT>("geometry");
  VERIFY(lefts.size() == frames && rights.size() == frames && geometries.size() == frames);
  if (lefts.size() != frames || rights.size() != frames) return;
  const StereoRectification r = compute_stereo_rectification(scene().calibration);
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
  // The geometry message says what the rectification computed.
  if (!geometries.empty()) {
    const schema::StereoGeometryT& g = geometries.front();
    VERIFY(g.width == r.width && g.height == r.height);
    VERIFY(g.fx == r.fx && g.cx_left == r.cx_left && g.cx_right == r.cx_right && g.cy == r.cy);
    VERIFY(g.baseline == r.baseline && g.doffs == r.doffs);
    VERIFY(g.disparity_to_depth && g.disparity_to_depth->m()->Get(11) == r.fx);
    VERIFY(g.valid_left && g.valid_left->width() == r.valid_left.width);
    VERIFY(g.rect_from_right && g.rect_from_right->m()->Get(4) == r.rect_from_right[4]);
    if (backend != Backend::kAuto) VERIFY(g.backend == backend_name(backend));
    if (backend == Backend::kAuto) VERIFY(!g.backend.empty());
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
  const CameraModel camera = scene().calibration.left;
  const Undistortion u = compute_undistortion(camera);
  const SyntheticView view(camera, RigidTransform{}, scene().plane);
  const std::vector<std::uint8_t> gray = view.render(1);
  // RGB8: three copies of the gray image, which each must come out as the gray one does.
  {
    std::vector<std::uint8_t> rgb(gray.size() * 3);
    for (std::size_t i = 0; i < gray.size(); ++i) {
      rgb[3 * i] = gray[i];
      rgb[3 * i + 1] = gray[i];
      rgb[3 * i + 2] = gray[i];
    }
    auto h =
        holoscan::testing::create_operator_test<UndistortOp>(UndistortConfig{.camera = camera});
    h.push_at("image", test::image(rgb, kWidth, kHeight, 3, ImageEncoding_RGB8, false), 100ms);
    h.output_device("undistorted", holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::schema::ImageT>("undistorted", kCaptures);
    const auto result = h.run();
    VERIFY_MSG(result.ok(), result.describe_failure());
    if (result.ok()) {
      const auto out = result.snapshot<holoscan::schema::ImageT>("undistorted");
      VERIFY(out.metadata().encoding == ImageEncoding_RGB8);
      VERIFY((out.tensor(holoscan::schema::ImageTensorFields::data).shape() ==
              std::vector<std::int64_t>{u.output.height, u.output.width, 3}));
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
        UndistortConfig{.camera = camera, .interpolation = Interpolation::kNearest});
    h.push_at("image", test::image(depth, kWidth, kHeight, 1, ImageEncoding_MONO16, true), 100ms);
    h.input_source_device("image", holoscan::DeviceId{0});
    h.output_device("undistorted", holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::schema::ImageT>("undistorted", kCaptures);
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
void check_refused(UndistortConfig config, holoscan::schema::ImageT image, const std::string& what,
                   const std::string& expected) {
  auto h = holoscan::testing::create_operator_test<UndistortOp>(std::move(config));
  h.push_at("image", std::move(image), 100ms);
  h.output_device("undistorted", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("undistorted", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), what + ": accepted");
  if (!result.ok()) {
    VERIFY_MSG(result.describe_failure().find(expected) != std::string::npos,
               what + ": " + result.describe_failure());
  }
}

void test_refusals() {
  const CameraModel camera = scene().calibration.left;
  const std::vector<std::uint8_t> gray(static_cast<std::size_t>(kWidth) * kHeight, 9);
  check_refused(
      UndistortConfig{.camera = camera},
      test::image(std::vector<std::uint8_t>(320 * 240), 320, 240, 1, ImageEncoding_MONO8, false),
      "wrong size", "the camera 640x480");
  check_refused(UndistortConfig{.camera = camera},
                test::image(std::vector<std::uint16_t>(gray.size()), kWidth, kHeight, 1,
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
    check_refused(UndistortConfig{.camera = camera}, image, "unsupported encoding",
                  "unsupported encoding MONO32F");
  }
  // Configurations that cannot work are refused at construction.
  VERIFY(test::throws<std::invalid_argument>(
      [&] { UndistortOp op(UndistortConfig{.camera = camera, .options = {.alpha = -1}}); }));
  StereoCalibration swapped = scene().calibration;
  swapped.right_from_left.translation[0] = 0.1;
  VERIFY(test::throws<std::invalid_argument>(
      [&] { StereoRectifyOp op(StereoRectifyConfig{.calibration = swapped}); }));
  if (!HOLOSCAN_RECTIFY_TEST_WITH_VPI) {
    VERIFY(test::throws<std::invalid_argument>(
        [&] { UndistortOp op(UndistortConfig{.camera = camera, .backend = Backend::kVpiCuda}); }));
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
