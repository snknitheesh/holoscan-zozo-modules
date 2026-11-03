// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// CameraInfo and StereoCameraInfo without the SDK: what validate() accepts
// and refuses (naming the field), the stereo helpers, same_calibration() and
// the synthetic calibration.

#include "holoscan/messages/camera_info.hpp"

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "holoscan/messages/synthetic_calibration.hpp"
#include "test_support.hpp"

namespace holoscan::messages {
namespace {

constexpr std::array<DistortionModel, 4> all_models() {
  return {DistortionModel::kNone, DistortionModel::kPlumbBob, DistortionModel::kRationalPolynomial,
          DistortionModel::kEquidistant};
}

CameraInfo raw_camera() { return synthetic_stereo_camera_info(DistortionModel::kPlumbBob).left; }

// A rectified pair: 0.1 m baseline, the right principal point 4 px further.
StereoCameraInfo rectified_pair() {
  StereoCameraInfo pair;
  pair.header.frame_id = "left";
  for (CameraInfo* camera : {&pair.left, &pair.right}) {
    camera->width = 640;
    camera->height = 480;
    camera->state = CameraState::kRectified;
    camera->fx = 400.0;
    camera->fy = 400.0;
    camera->cx = 319.5;
    camera->cy = 239.5;
  }
  pair.right.cx = 323.5;
  pair.right_from_left.translation = {-0.1, 0.0, 0.0};
  pair.baseline = 0.1;
  pair.doffs = 4.0;
  return pair;
}

template <typename Change>
bool camera_refused(Change change, std::string_view text) {
  CameraInfo info = raw_camera();
  change(info);
  return test::refuses([&] { validate(info); }, text);
}

template <typename Change>
bool pair_refused(StereoCameraInfo pair, Change change, std::string_view text) {
  change(pair);
  return test::refuses([&] { validate(pair); }, text);
}

void test_names() {
  for (const DistortionModel model : all_models()) {
    VERIFY(parse_distortion_model(distortion_model_name(model)) == model);
  }
  VERIFY(parse_distortion_model("") == DistortionModel::kNone);
  VERIFY(!parse_distortion_model("fisheye"));
  VERIFY(coefficient_count(DistortionModel::kNone) == 0);
  VERIFY(coefficient_count(DistortionModel::kPlumbBob) == 5);
  VERIFY(coefficient_count(DistortionModel::kRationalPolynomial) == 8);
  VERIFY(coefficient_count(DistortionModel::kEquidistant) == 4);
  VERIFY(camera_state_name(CameraState::kRaw) == "raw");
  VERIFY(camera_state_name(CameraState::kUndistorted) == "undistorted");
  VERIFY(camera_state_name(CameraState::kRectified) == "rectified");
}

void test_synthetic_pairs_are_valid() {
  for (const DistortionModel model : all_models()) {
    const StereoCameraInfo pair = synthetic_stereo_camera_info(model, 640, 480);
    validate(pair);
    VERIFY(pair.left.width == 640 && pair.right.height == 480);
    VERIFY(pair.left.coefficients.size() == coefficient_count(model));
    VERIFY(pair.left.state ==
           (model == DistortionModel::kNone ? CameraState::kUndistorted : CameraState::kRaw));
    VERIFY(pair.left.rectification && pair.right.rectification);
  }
  validate(rectified_pair());
}

void test_camera_refusals() {
  VERIFY(camera_refused([](CameraInfo& c) { c.width = 0; }, "camera_info.width"));
  VERIFY(camera_refused([](CameraInfo& c) { c.fx = -1.0; }, "camera_info.fx"));
  VERIFY(camera_refused([](CameraInfo& c) { c.fy = 0.0; }, "camera_info.fy"));
  VERIFY(camera_refused([](CameraInfo& c) { c.cx = std::nan(""); }, "camera_info.cx"));
  VERIFY(camera_refused([](CameraInfo& c) { c.skew = INFINITY; }, "camera_info.skew"));
  VERIFY(camera_refused([](CameraInfo& c) { c.coefficients.pop_back(); },
                        "camera_info.coefficients has 4 values; the plumb_bob model has 5"));
  VERIFY(camera_refused([](CameraInfo& c) { c.coefficients[2] = std::nan(""); },
                        "camera_info.coefficients[2]"));
  VERIFY(camera_refused([](CameraInfo& c) { c.coefficients.assign(5, 0.0); },
                        "camera_info.state is raw, but every distortion coefficient is zero"));
  VERIFY(camera_refused([](CameraInfo& c) { c.state = CameraState::kUndistorted; },
                        "camera_info.state is undistorted, but a distortion coefficient"));
  VERIFY(camera_refused([](CameraInfo& c) { c.state = static_cast<CameraState>(7); },
                        "camera_info.state"));
  VERIFY(camera_refused([](CameraInfo& c) { c.distortion_model = static_cast<DistortionModel>(9); },
                        "camera_info.distortion_model"));
  VERIFY(camera_refused([](CameraInfo& c) { (*c.rectification)[0] = 1.5; },
                        "camera_info.rectification is not a rotation"));
  VERIFY(
      camera_refused([](CameraInfo& c) { c.rectification = Matrix3{-1, 0, 0, 0, 1, 0, 0, 0, 1}; },
                     "camera_info.rectification is not a rotation"));
  VERIFY(camera_refused([](CameraInfo& c) { c.binning_y = 0; }, "camera_info.binning_x"));
  VERIFY(camera_refused([](CameraInfo& c) { c.roi = RegionOfInterest{0, 0, 0, 10}; },
                        "camera_info.roi is empty"));
  VERIFY(camera_refused([](CameraInfo& c) { c.roi = RegionOfInterest{1200, 0, 100, 10}; },
                        "camera_info.roi reaches past the 1280x720 sensor"));
  VERIFY(camera_refused([](CameraInfo& c) { c.header.frame_id.assign(kMaxFrameIdBytes + 1, 'f'); },
                        "camera_info.header.frame_id has 257 bytes"));
  // Binning makes the sensor larger than the images; a ROI may cover it.
  CameraInfo binned = raw_camera();
  binned.binning_x = 2;
  binned.roi = RegionOfInterest{1280, 0, 1280, 720};
  validate(binned);
  binned.header.frame_id.assign(kMaxFrameIdBytes, 'f');
  validate(binned);
  // A rectified camera has no rotation left.
  StereoCameraInfo pair = rectified_pair();
  pair.left.rectification = kIdentity3;
  VERIFY(test::refuses([&] { validate(pair.left, "left"); }, "left.rectification must be absent"));
}

void test_pair_refusals() {
  const StereoCameraInfo raw = synthetic_stereo_camera_info(DistortionModel::kPlumbBob);
  VERIFY(pair_refused(
      raw, [](StereoCameraInfo& p) { p.right.fx = 0; }, "stereo_camera_info.right.fx"));
  VERIFY(pair_refused(
      raw, [](StereoCameraInfo& p) { p.right_from_left.rotation[1] += 0.1; },
      "stereo_camera_info.right_from_left.rotation is not a rotation"));
  VERIFY(pair_refused(
      raw, [](StereoCameraInfo& p) { p.right_from_left.translation = {0, 0, 0}; },
      "right_from_left.translation must be finite and not zero"));
  VERIFY(pair_refused(
      raw, [](StereoCameraInfo& p) { p.baseline = 0.12; },
      "baseline and doffs describe a rectified pair"));
  VERIFY(pair_refused(
      raw,
      [](StereoCameraInfo& p) { p.right_from_left.rotation = rotation_matrix({0, 1, 0}, 0.1); },
      "the rectifications do not agree with right_from_left"));
  VERIFY(pair_refused(
      raw, [](StereoCameraInfo& p) { p.right_from_left.translation[1] = 0.05; },
      "the rectifications do not agree with right_from_left"));
  // Without rectifications the extrinsics may be anything.
  StereoCameraInfo free_pair = raw;
  free_pair.left.rectification.reset();
  free_pair.right.rectification.reset();
  validate(free_pair);

  const StereoCameraInfo rectified = rectified_pair();
  VERIFY(pair_refused(
      rectified, [](StereoCameraInfo& p) { p.right.state = CameraState::kUndistorted; },
      "one camera is rectified and the other undistorted"));
  VERIFY(pair_refused(
      rectified, [](StereoCameraInfo& p) { p.right.fy = 401.0; }, "a rectified pair has one fy"));
  VERIFY(pair_refused(
      rectified, [](StereoCameraInfo& p) { p.right.cy = 240.0; }, "a rectified pair has one cy"));
  VERIFY(pair_refused(
      rectified, [](StereoCameraInfo& p) { p.left.skew = 0.5; }, "a rectified pair has no skew"));
  VERIFY(pair_refused(
      rectified, [](StereoCameraInfo& p) { p.baseline = -0.1; },
      "stereo_camera_info.baseline must be finite and > 0"));
  VERIFY(pair_refused(
      rectified, [](StereoCameraInfo& p) { p.right_from_left.translation[2] = 0.01; },
      "right_from_left.translation of a rectified pair must be (-0.10000000000000001, 0, 0)"));
  VERIFY(pair_refused(
      rectified,
      [](StereoCameraInfo& p) { p.right_from_left.rotation = rotation_matrix({0, 1, 0}, 0.01); },
      "right_from_left.rotation of a rectified pair must be the identity"));
  VERIFY(pair_refused(
      rectified, [](StereoCameraInfo& p) { p.doffs = 0.0; },
      "stereo_camera_info.doffs must be cx_right - cx_left = 4"));
}

void test_depth() {
  const StereoCameraInfo pair = rectified_pair();
  VERIFY(fx_baseline(pair) == 40.0);
  VERIFY(depth_from_disparity(pair, 36.0) == 1.0);
  VERIFY(depth_from_disparity(pair, 16.0) == 2.0);
  VERIFY(std::isnan(depth_from_disparity(pair, -4.0)));
  VERIFY(std::isnan(depth_from_disparity(pair, -10.0)));
  const StereoCameraInfo raw = synthetic_stereo_camera_info(DistortionModel::kPlumbBob);
  VERIFY(fx_baseline(raw) == 0.0);
  VERIFY(std::isnan(depth_from_disparity(raw, 10.0)));
}

void test_same_calibration() {
  StereoCameraInfo a = synthetic_stereo_camera_info(DistortionModel::kEquidistant);
  StereoCameraInfo b = a;
  b.header.capture_timestamp_ns = 99;
  b.left.header.device_sequence = 4;
  b.right.header.frame_id = "elsewhere";
  VERIFY(same_calibration(a, b));
  VERIFY(same_calibration(a.left, b.left));
  b.right.coefficients[3] += 1e-12;
  VERIFY(!same_calibration(a, b));
  VERIFY(!same_calibration(a.right, b.right));
  VERIFY(same_calibration(a.left, b.left));
}

void test_rotation_matrix() {
  const Matrix3 r = rotation_matrix({0, 0, 2}, M_PI / 2);
  const Matrix3 expected{0, -1, 0, 1, 0, 0, 0, 0, 1};
  for (std::size_t i = 0; i < 9; ++i) VERIFY(std::abs(r[i] - expected[i]) < 1e-15);
}

}  // namespace
}  // namespace holoscan::messages

int main() {
  namespace m = holoscan::messages;
  return holoscan::messages::test::run({
      {"model and state names", m::test_names},
      {"synthetic pairs are valid", m::test_synthetic_pairs_are_valid},
      {"camera refusals name the field", m::test_camera_refusals},
      {"stereo refusals name the field", m::test_pair_refusals},
      {"depth from disparity", m::test_depth},
      {"same calibration ignores headers", m::test_same_calibration},
      {"rotation matrix", m::test_rotation_matrix},
  });
}
