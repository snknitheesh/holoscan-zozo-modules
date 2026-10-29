// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The calibration files: a recording's calibration.json, ROS camera_info YAML
// and a stereo pair recovered from two of them, and what each refuses.

#include <cmath>
#include <stdexcept>
#include <string>

#include <holoscan/rectify/camera_model.hpp>
#include <holoscan/rectify/rectification.hpp>

#include "test_support.hpp"

namespace holoscan::rectify {
namespace {

// A recording's calibration.json, as holoscan-stereo-camera writes it.
constexpr const char* kRecording = R"({
  "version": 1,
  "camera": {"name": "RealSense D585", "serial": "1", "settings": {"fps": "30"}},
  "rectified": true,
  "left": {"width": 1280, "height": 720, "fx": 336.98, "fy": 336.98, "cx": 674.05, "cy": 355.81,
           "distortion": {"model": "brown_conrady", "coefficients": [0, 0, 0, 0, 0]}},
  "right": {"width": 1280, "height": 720, "fx": 336.98, "fy": 336.98, "cx": 674.05, "cy": 355.81,
            "distortion": {"model": "brown_conrady", "coefficients": [0, 0, 0, 0, 0]}},
  "right_from_left": {"rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                      "translation": [-0.094843290746212, 0, 0]},
  "rgb": {"width": 1280, "height": 720, "fx": 331.4, "fy": 331.6, "cx": 702.4, "cy": 382.2,
          "distortion": {"model": "brown_conrady", "coefficients": [-0.0005, 0.0007, -0.0001, -0.0002, -0.0001]},
          "from_left": {"rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1], "translation": [-0.0476, 0, 0]}}
})";

// ROS camera_calibration's output for a stereo pair (left, right).
constexpr const char* kLeftInfo = R"(image_width: 640
image_height: 480
camera_name: narrow_stereo/left
camera_matrix:
  rows: 3
  cols: 3
  data: [430.2, 0, 306.7, 0, 430.9, 237.3, 0, 0, 1]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [-0.36, 0.11, 0.0012, -0.0009, 0]
rectification_matrix:
  rows: 3
  cols: 3
  data: [0.9998, 0.0051, -0.0189,
         -0.0050, 0.99998, 0.0040,
         0.01892, -0.00391, 0.99981]
projection_matrix:
  rows: 3
  cols: 4
  data: [420.5, 0, 318.2, 0, 0, 420.5, 240.1, 0, 0, 0, 1, 0]
)";

constexpr const char* kRightInfo = R"(# A comment.
image_width: 640
image_height: 480
camera_name: "narrow_stereo/right"
camera_matrix:
  rows: 3
  cols: 3
  data:
    - 431.0
    - 0
    - 310.1
    - 0
    - 431.5
    - 241.0
    - 0
    - 0
    - 1
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [-0.35, 0.10, -0.0008, 0.0011, 0]
rectification_matrix:
  rows: 3
  cols: 3
  data: [0.99993, -0.0108, -0.0049, 0.0108, 0.99994, 0.0039, 0.00494, -0.00385, 0.99998]
projection_matrix:
  rows: 3
  cols: 4
  data: [420.5, 0, 318.2, -50.46, 0, 420.5, 240.1, 0, 0, 0, 1, 0]
)";

void test_recording() {
  const StereoCalibration c = parse_stereo_calibration(kRecording);
  VERIFY(c.left.width == 1280 && c.left.height == 720);
  VERIFY(c.left.distortion_model == DistortionModel::kBrownConrady);
  VERIFY(c.left.distortion.size() == 5);
  VERIFY(std::abs(c.right.fx - 336.98) < 1e-12);
  VERIFY(std::abs(c.right_from_left.translation[0] + 0.094843290746212) < 1e-15);
}

void test_recording_refusals() {
  const auto refuses = [](std::string json, const std::string& what) {
    VERIFY_MSG(test::throws<std::invalid_argument>([&] { (void)parse_stereo_calibration(json); }),
               what);
  };
  std::string text = kRecording;
  const auto replaced = [&text](const std::string& from, const std::string& to) {
    std::string copy = text;
    copy.replace(copy.find(from), from.size(), to);
    return copy;
  };
  refuses(replaced("\"brown_conrady\"", "\"modified_brown_conrady\""), "unsupported model");
  refuses(replaced("[0, 0, 0, 0, 0]", "[0, 0, 0]"), "wrong coefficient count");
  refuses(replaced("\"fx\": 336.98", "\"fx\": -1"), "negative fx");
  refuses(replaced("[1, 0, 0, 0, 1, 0, 0, 0, 1]", "[1, 0, 0, 0, 1, 0, 0, 0, 2]"), "not a rotation");
  refuses(replaced("[-0.094843290746212, 0, 0]", "[0, 0, 0]"), "no baseline");
  refuses(replaced("\"version\": 1", "\"version\": 2"), "version");
  refuses(text.substr(0, 40), "truncated JSON");
  refuses("[]", "not an object");
  // Four Brown-Conrady coefficients mean k3 = 0.
  const StereoCalibration four =
      parse_stereo_calibration(replaced("[0, 0, 0, 0, 0]", "[0.1, 0, 0, 0]"));
  VERIFY(four.left.distortion.size() == 5 && four.left.distortion[0] == 0.1);
}

void test_camera_info() {
  const CameraInfo left = parse_camera_info(kLeftInfo);
  VERIFY(left.camera_name == "narrow_stereo/left");
  VERIFY(left.camera.width == 640 && left.camera.height == 480);
  VERIFY(left.camera.fx == 430.2 && left.camera.cy == 237.3);
  VERIFY(left.camera.distortion_model == DistortionModel::kBrownConrady);
  VERIFY(left.camera.distortion.size() == 5 && left.camera.distortion[0] == -0.36);
  VERIFY(left.rectification[4] == 0.99998 && left.rectification[6] == 0.01892);
  VERIFY(left.projection[0] == 420.5);
  const CameraInfo right = parse_camera_info(kRightInfo);
  VERIFY(right.camera_name == "narrow_stereo/right");
  VERIFY(right.camera.cx == 310.1 && right.camera.fy == 431.5);
  VERIFY(right.projection[3] == -50.46);

  std::string skewed = kLeftInfo;
  skewed.replace(skewed.find("430.2, 0,"), 9, "430.2, 1,");
  VERIFY(test::throws<std::invalid_argument>([&] { (void)parse_camera_info(skewed); }));
  std::string unknown = kLeftInfo;
  unknown.replace(unknown.find("plumb_bob"), 9, "unknown_model");
  VERIFY(test::throws<std::invalid_argument>([&] { (void)parse_camera_info(unknown); }));
  VERIFY(
      test::throws<std::invalid_argument>([] { (void)parse_camera_info("image_width: 640\n"); }));
}

// The pair recovered from two camera_info files rectifies back to their R and P.
void test_stereo_from_camera_info() {
  // Make consistent files: rectify a known pair and write what ROS would.
  StereoCalibration truth;
  truth.left = parse_camera_info(kLeftInfo).camera;
  truth.right = parse_camera_info(kRightInfo).camera;
  truth.right_from_left.rotation = axis_angle({0.2, 1.0, -0.3}, 0.02);
  truth.right_from_left.translation = {-0.12, 0.002, 0.001};
  const StereoRectification r = compute_stereo_rectification(truth);
  CameraInfo left{
      .camera = truth.left, .rectification = r.rect_from_left, .projection = r.projection_left};
  CameraInfo right{
      .camera = truth.right, .rectification = r.rect_from_right, .projection = r.projection_right};
  const StereoCalibration back = stereo_from_camera_info(left, right);
  double worst = 0.0;
  for (int i = 0; i < 9; ++i) {
    worst = std::max(
        worst, std::abs(back.right_from_left.rotation[i] - truth.right_from_left.rotation[i]));
  }
  VERIFY_MSG(worst < 1e-12, "rotation differs by " + std::to_string(worst));
  for (int i = 0; i < 3; ++i) {
    VERIFY(std::abs(back.right_from_left.translation[i] - truth.right_from_left.translation[i]) <
           1e-12);
  }
  // A right camera without a baseline in its projection is refused.
  right.projection[3] = 0.0;
  VERIFY(test::throws<std::invalid_argument>([&] { (void)stereo_from_camera_info(left, right); }));
}

void test_model_names() {
  VERIFY(parse_distortion_model("plumb_bob") == DistortionModel::kBrownConrady);
  VERIFY(parse_distortion_model("equidistant") == DistortionModel::kKannalaBrandt4);
  VERIFY(parse_distortion_model("rational_polynomial") == DistortionModel::kRationalPolynomial);
  VERIFY(parse_distortion_model("none") == DistortionModel::kNone);
  VERIFY(!parse_distortion_model("ftheta"));
  VERIFY(distortion_model_name(DistortionModel::kKannalaBrandt4) == "kannala_brandt4");
}

}  // namespace
}  // namespace holoscan::rectify

int main() {
  namespace rc = holoscan::rectify;
  return rc::test::run({
      {"recording calibration.json", rc::test_recording},
      {"calibration.json refusals", rc::test_recording_refusals},
      {"camera_info YAML", rc::test_camera_info},
      {"stereo pair from camera_info", rc::test_stereo_from_camera_info},
      {"distortion model names", rc::test_model_names},
  });
}
