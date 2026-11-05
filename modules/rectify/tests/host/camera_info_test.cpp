// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The calibration as the operators get it, from ROS camera_info files written
// and read with holoscan-messages: a raw pair rectifies the same from its files
// as from its values; a pair that is rectified already maps onto itself; the
// rectified pair's camera info is a valid rectified pair that its own files
// give back; and the cameras this module refuses.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <holoscan/messages/camera_info.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/rectify/lens.hpp>
#include <holoscan/rectify/rectification.hpp>
#include <holoscan/rectify/synthetic_stereo.hpp>

#include "test_support.hpp"

namespace holoscan::rectify {
namespace {

using messages::CameraInfo;
using messages::CameraState;
using messages::DistortionModel;
using messages::StereoCameraInfo;

double largest_difference(const Matrix3& a, const Matrix3& b) {
  double worst = 0.0;
  for (int i = 0; i < 9; ++i) worst = std::max(worst, std::abs(a[i] - b[i]));
  return worst;
}

// The synthetic pair through its two files: what StereoRectifyOp reads.
void test_files_give_the_same_rectification() {
  for (const DistortionModel model :
       {DistortionModel::kPlumbBob, DistortionModel::kRationalPolynomial,
        DistortionModel::kEquidistant}) {
    const std::string name(messages::distortion_model_name(model));
    const SyntheticStereo scene = synthetic_stereo(640, 480, model);
    const test::TempDir dir;
    messages::write_stereo_camera_info_yaml(scene.calibration, dir / "left.yaml",
                                            dir / "right.yaml");
    const StereoCameraInfo read =
        messages::read_stereo_camera_info_yaml(dir / "left.yaml", dir / "right.yaml");
    const StereoRectification direct = compute_stereo_rectification(scene.calibration);
    const StereoRectification from_files = compute_stereo_rectification(read);
    VERIFY_MSG(read.left.state == CameraState::kRaw && read.left.distortion_model == model,
               name + ": the files describe another camera");
    VERIFY_MSG(from_files.width == direct.width && from_files.height == direct.height,
               name + ": size");
    VERIFY_MSG(std::abs(from_files.fx - direct.fx) < 1e-9 &&
                   std::abs(from_files.cx_left - direct.cx_left) < 1e-9 &&
                   std::abs(from_files.cy - direct.cy) < 1e-9,
               name + ": intrinsics differ");
    VERIFY_MSG(std::abs(from_files.baseline - direct.baseline) < 1e-12, name + ": baseline");
    VERIFY_MSG(largest_difference(from_files.rect_from_left, direct.rect_from_left) < 1e-12 &&
                   largest_difference(from_files.rect_from_right, direct.rect_from_right) < 1e-12,
               name + ": rotations differ");
  }
}

// Files as a camera writes them for images it has rectified already (the local
// recordings): K equal, no distortion, R the identity, the baseline in P.
void test_rectified_files_map_onto_themselves() {
  const char* left = R"(image_width: 1280
image_height: 720
camera_name: left
camera_matrix: {rows: 3, cols: 3, data: [336.98760986328125, 0, 674.053955078125, 0, 336.98760986328125, 355.81134033203125, 0, 0, 1]}
distortion_model: plumb_bob
distortion_coefficients: {rows: 1, cols: 5, data: [0, 0, 0, 0, 0]}
rectification_matrix: {rows: 3, cols: 3, data: [1, 0, 0, 0, 1, 0, 0, 0, 1]}
projection_matrix: {rows: 3, cols: 4, data: [336.98760986328125, 0, 674.053955078125, 0, 0, 336.98760986328125, 355.81134033203125, 0, 0, 0, 1, 0]}
)";
  const char* right = R"(image_width: 1280
image_height: 720
camera_name: right
camera_matrix: {rows: 3, cols: 3, data: [336.98760986328125, 0, 674.053955078125, 0, 336.98760986328125, 355.81134033203125, 0, 0, 1]}
distortion_model: plumb_bob
distortion_coefficients: {rows: 1, cols: 5, data: [0, 0, 0, 0, 0]}
rectification_matrix: {rows: 3, cols: 3, data: [1, 0, 0, 0, 1, 0, 0, 0, 1]}
projection_matrix: {rows: 3, cols: 4, data: [336.98760986328125, 0, 674.053955078125, -31.961013860134244, 0, 336.98760986328125, 355.81134033203125, 0, 0, 0, 1, 0]}
)";
  const StereoCameraInfo pair = messages::parse_stereo_camera_info_yaml(left, right);
  VERIFY(pair.left.state == CameraState::kRectified);
  const StereoRectification r = compute_stereo_rectification(pair);
  VERIFY(std::abs(r.fx - pair.left.fx) < 1e-9);
  VERIFY(std::abs(r.cx_left - pair.left.cx) < 1e-9 && std::abs(r.cy - pair.left.cy) < 1e-9);
  VERIFY(std::abs(r.baseline - pair.baseline) < 1e-12);
  VERIFY(largest_difference(r.rect_from_left, messages::kIdentity3) < 1e-15);
  // What the operator sends is what it got.
  VERIFY(std::abs(r.camera_info.left.cx - pair.left.cx) < 1e-9 &&
         std::abs(r.camera_info.baseline - pair.baseline) < 1e-12);
  VERIFY(r.camera_info.left.header.frame_id == "left" &&
         r.camera_info.right.header.frame_id == "right");
}

// The rectified pair's camera info is valid, and its files read back as a
// rectified pair with the same values.
void test_rectified_camera_info_round_trip() {
  const SyntheticStereo scene = synthetic_stereo(640, 480);
  for (const bool zero_disparity : {true, false}) {
    const StereoRectification r =
        compute_stereo_rectification(scene.calibration, {.zero_disparity = zero_disparity});
    const StereoCameraInfo& info = r.camera_info;
    messages::validate(info);
    VERIFY(info.left.distortion_model == DistortionModel::kNone);
    VERIFY(!info.left.rectification && !info.right.rectification);
    VERIFY(info.right_from_left.translation[0] == -r.baseline);
    const test::TempDir dir;
    messages::write_stereo_camera_info_yaml(info, dir / "left.yaml", dir / "right.yaml");
    const StereoCameraInfo back =
        messages::read_stereo_camera_info_yaml(dir / "left.yaml", dir / "right.yaml");
    VERIFY(back.left.state == CameraState::kRectified);
    VERIFY(back.left.fx == info.left.fx && back.right.cx == info.right.cx &&
           back.left.cy == info.left.cy && back.doffs == info.doffs);
    VERIFY(std::abs(back.baseline - info.baseline) < 1e-15);
  }
}

// What this module refuses beyond what messages::validate() does.
void test_refused_cameras() {
  const CameraInfo good = synthetic_stereo(640, 480).calibration.left;
  check_camera(good);
  const auto refuses = [](const CameraInfo& camera, const std::string& what) {
    VERIFY_MSG(test::throws<std::invalid_argument>([&] { check_camera(camera); }), what);
    VERIFY_MSG(test::throws<std::invalid_argument>([&] { (void)compute_undistortion(camera); }),
               what);
  };
  CameraInfo skewed = good;
  skewed.skew = 0.5;
  refuses(skewed, "skew");
  CameraInfo cropped = good;
  cropped.roi = messages::RegionOfInterest{.x = 0, .y = 0, .width = 320, .height = 240};
  refuses(cropped, "region of interest");
  CameraInfo binned = good;
  binned.binning_x = 2;
  refuses(binned, "binning");
  CameraInfo huge = good;
  huge.width = static_cast<std::uint32_t>(kMaxImageSide) + 1;
  refuses(huge, "too wide");
  CameraInfo invalid = good;
  invalid.fx = -1.0;
  refuses(invalid, "invalid for messages::validate()");

  // A pair of cameras of different sizes.
  StereoCameraInfo pair = synthetic_stereo(640, 480).calibration;
  pair.right.width = 648;
  VERIFY(test::throws<std::invalid_argument>([&] { (void)compute_stereo_rectification(pair); }));
}

}  // namespace
}  // namespace holoscan::rectify

int main() {
  namespace rc = holoscan::rectify;
  return rc::test::run({
      {"a raw pair rectifies the same from its files", rc::test_files_give_the_same_rectification},
      {"rectified files map onto themselves", rc::test_rectified_files_map_onto_themselves},
      {"the rectified camera info and its files", rc::test_rectified_camera_info_round_trip},
      {"cameras this module refuses", rc::test_refused_cameras},
  });
}
