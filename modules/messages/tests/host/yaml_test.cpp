// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The camera_info YAML files without the SDK: the YAML forms the reader
// takes and refuses; ROS files read with every lens model; write and read
// round trips of single cameras and stereo pairs; the stereo extrinsics
// recovered from R and P against known values; and the refusals, each
// naming what is wrong.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/messages/camera_info.hpp"
#include "holoscan/messages/camera_info_yaml.hpp"
#include "holoscan/messages/synthetic_calibration.hpp"
#include "ros_camera_info_text.hpp"
#include "test_support.hpp"
#include "yaml_reader.hpp"

namespace holoscan::messages {
namespace {

namespace fs = std::filesystem;
using detail::parse_yaml;
using detail::YamlNode;

constexpr std::array<DistortionModel, 4> all_models() {
  return {DistortionModel::kNone, DistortionModel::kPlumbBob, DistortionModel::kRationalPolynomial,
          DistortionModel::kEquidistant};
}

// A ROS mono calibration, as camera_calibration writes it.
constexpr const char* kMonoYaml = R"(image_width: 640
image_height: 480
camera_name: narrow_stereo
camera_matrix:
  rows: 3
  cols: 3
  data: [ 412.34, 0., 318.72, 0., 413.01, 241.37, 0., 0., 1. ]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [ -0.31, 0.11, 0.0012, -0.0004, 0. ]
rectification_matrix:
  rows: 3
  cols: 3
  data: [ 1., 0., 0., 0., 1., 0., 0., 0., 1. ]
projection_matrix:
  rows: 3
  cols: 4
  data: [ 330.1, 0., 317.2, 0., 0., 372.4, 240.9, 0., 0., 0., 1., 0. ]
)";

std::string replace(std::string text, const std::string& from, const std::string& to) {
  const auto at = text.find(from);
  if (at == std::string::npos) throw std::logic_error("fixture lacks '" + from + "'");
  return text.replace(at, from.size(), to);
}

bool mono_refused(const std::string& from, const std::string& to, std::string_view message) {
  const std::string text = replace(kMonoYaml, from, to);
  return test::refuses([&] { (void)parse_camera_info_yaml(text); }, message);
}

bool close(double a, double b, double tolerance) {
  return std::abs(a - b) <= tolerance * std::max(1.0, std::abs(b));
}

bool close(const Matrix3& a, const Matrix3& b, double tolerance) {
  for (std::size_t i = 0; i < 9; ++i) {
    if (std::abs(a[i] - b[i]) > tolerance) return false;
  }
  return true;
}

bool close(const Vector3& a, const Vector3& b, double tolerance) {
  for (std::size_t i = 0; i < 3; ++i) {
    if (std::abs(a[i] - b[i]) > tolerance) return false;
  }
  return true;
}

// Two stereo infos alike up to the rounding of R and P through a file.
bool same_pair(const StereoCameraInfo& a, const StereoCameraInfo& b) {
  const bool rectifications =
      a.left.rectification.has_value() == b.left.rectification.has_value() &&
      a.right.rectification.has_value() == b.right.rectification.has_value() &&
      (!a.left.rectification || close(*a.left.rectification, *b.left.rectification, 0.0)) &&
      (!a.right.rectification || close(*a.right.rectification, *b.right.rectification, 0.0));
  CameraInfo left = a.left;
  CameraInfo right = a.right;
  left.rectification = b.left.rectification;
  right.rectification = b.right.rectification;
  return rectifications && left == b.left && right == b.right && a.header == b.header &&
         close(a.right_from_left.rotation, b.right_from_left.rotation, 1e-12) &&
         close(a.right_from_left.translation, b.right_from_left.translation, 1e-12) &&
         close(a.baseline, b.baseline, 1e-12) && close(a.doffs, b.doffs, 1e-12);
}

void test_yaml_forms() {
  // Block, flow over several lines, a block list, a flow mapping, quotes,
  // comments, a directive and the document markers.
  const YamlNode root = parse_yaml(R"(%YAML 1.2
---
# a comment
a: 1   # after a value
b: "two # not a comment"
c: 'it''s'
d: [1, 2,
    3]
e:
  - 4
  - -5.5
f: {rows: 1, cols: 2, data: [6, 7]}
g:
  h: plain text
  i:
...
)");
  VERIFY(root.kind == YamlNode::Kind::kMapping && root.entries.size() == 7);
  VERIFY(root.find("a")->scalar == "1");
  VERIFY(root.find("b")->scalar == "two # not a comment" && root.find("b")->quoted);
  VERIFY(root.find("c")->scalar == "it's");
  VERIFY(root.find("d")->items.size() == 3 && root.find("d")->items[2].scalar == "3");
  VERIFY(root.find("e")->kind == YamlNode::Kind::kSequence &&
         root.find("e")->items[1].scalar == "-5.5");
  VERIFY(root.find("f")->find("data")->items[1].scalar == "7");
  VERIFY(root.find("g")->find("h")->scalar == "plain text");
  VERIFY(root.find("g")->find("i")->scalar.empty());
  // A list at its key's own indentation, as some writers put it.
  const YamlNode same_indent = parse_yaml("data:\n- 1\n- 2\nnext: 3\n");
  VERIFY(same_indent.find("data")->items.size() == 2 && same_indent.find("next")->scalar == "3");
  // A UTF-8 byte order mark before the first key, as some editors write.
  const YamlNode marked = parse_yaml(
      "\xEF\xBB\xBF"
      "a: 1\nb: 2\n");
  VERIFY(marked.kind == YamlNode::Kind::kMapping && marked.find("a") != nullptr &&
         marked.find("a")->scalar == "1");
}

void test_yaml_refusals() {
  const auto refused = [](const char* text, std::string_view message) {
    return test::refuses([&] { (void)parse_yaml(text); }, message);
  };
  VERIFY(refused("a: 1\n\tb: 2\n", "line 2: indentation must be spaces"));
  VERIFY(refused("a: 1\na: 2\n", "line 2: the key 'a' is given twice"));
  VERIFY(refused("a: &anchor 1\n", "line 1: '&anchor 1': anchors, aliases, tags"));
  VERIFY(refused("a: !!float 1\n", "anchors, aliases, tags and block scalars"));
  VERIFY(refused("a: |\n  text\n", "anchors, aliases, tags and block scalars"));
  VERIFY(refused("a: [1, 2\nb: 3\n", "a [ or { is never closed"));
  VERIFY(refused("a: 1]\n", "line 1: a ] or } closes nothing"));
  VERIFY(refused("a: 1\n---\nb: 2\n", "line 2: only one document is supported"));
  VERIFY(refused("a: 1\n...\nb: 2\n", "line 3: only one document is supported"));
  VERIFY(refused("a: 1\n   b: 2\n", "line 2: unexpected indentation"));
  VERIFY(refused("a: \"open\n", "line 1: a quoted scalar must end"));
  VERIFY(refused("just text\n", "line 1: expected 'key: value'"));
  VERIFY(refused("a: b: c\n", "a mapping on one line needs braces"));
  VERIFY(refused("a:\n  - b: 1\n", "a mapping inside a list item is not supported"));
  VERIFY(refused("a: [1,, 2]\n", "a value is missing"));
  VERIFY(refused("a: {x 1}\n", "expected key: value"));
}

void test_read_ros_mono() {
  const CameraInfo info = parse_camera_info_yaml(kMonoYaml);
  VERIFY(info.width == 640 && info.height == 480);
  VERIFY(info.header.frame_id == "narrow_stereo");
  VERIFY(info.fx == 412.34 && info.fy == 413.01 && info.cx == 318.72 && info.cy == 241.37);
  VERIFY(info.skew == 0.0);
  VERIFY(info.state == CameraState::kRaw);
  VERIFY(info.distortion_model == DistortionModel::kPlumbBob);
  VERIFY((info.coefficients == std::vector<double>{-0.31, 0.11, 0.0012, -0.0004, 0.0}));
  VERIFY(!info.rectification);  // The identity is no rectification.
  VERIFY(!info.roi && info.binning_x == 1 && info.binning_y == 1);
  // No distortion: undistorted, whatever P says.
  const CameraInfo pinhole = parse_camera_info_yaml(
      replace(kMonoYaml, "[ -0.31, 0.11, 0.0012, -0.0004, 0. ]", "[0, 0, 0, 0, 0]"));
  VERIFY(pinhole.state == CameraState::kUndistorted);
  VERIFY(pinhole.distortion_model == DistortionModel::kPlumbBob);
  // Only the required keys.
  const CameraInfo minimal = parse_camera_info_yaml(
      "image_width: 8\nimage_height: 6\n"
      "camera_matrix: {rows: 3, cols: 3, data: [5, 0.25, 3.5, 0, 6, 2.5, 0, 0, 1]}\n");
  VERIFY(minimal.state == CameraState::kUndistorted &&
         minimal.distortion_model == DistortionModel::kNone);
  VERIFY(minimal.skew == 0.25 && minimal.coefficients.empty() && minimal.header.frame_id.empty());
  // "none" and "" are no distortion; a block list and a + sign are numbers.
  const CameraInfo none = parse_camera_info_yaml(
      "image_width: 8\nimage_height: 6\ndistortion_model: none\n"
      "distortion_coefficients: {rows: 1, cols: 0, data: []}\n"
      "camera_matrix:\n  rows: 3\n  cols: 3\n  data:\n    - +5\n    - 0\n    - 3.5\n    - 0\n"
      "    - 6e0\n    - 2.5\n    - 0\n    - 0\n    - 1\n");
  VERIFY(none.distortion_model == DistortionModel::kNone && none.fx == 5.0 && none.fy == 6.0);
}

void test_mono_round_trips() {
  for (const DistortionModel model : all_models()) {
    for (const bool rectification : {false, true}) {
      CameraInfo info = synthetic_stereo_camera_info(model).right;
      if (!rectification) info.rectification.reset();
      info.skew = 0.125;
      const CameraInfo back = parse_camera_info_yaml(format_camera_info_yaml(info));
      VERIFY_MSG(back == info, std::string(distortion_model_name(model)) +
                                   (rectification ? " with" : " without") + " rectification");
    }
  }
  // The format has no region of interest or binning: such a camera is
  // refused rather than written as the whole sensor.
  CameraInfo cropped = synthetic_stereo_camera_info(DistortionModel::kPlumbBob).left;
  cropped.roi = RegionOfInterest{16, 8, 640, 480};
  VERIFY(test::refuses([&] { (void)format_camera_info_yaml(cropped); },
                       "camera_info: a camera_info file has no region of interest or binning"));
  CameraInfo binned = synthetic_stereo_camera_info(DistortionModel::kPlumbBob).left;
  binned.binning_y = 2;
  VERIFY(test::refuses([&] { (void)format_camera_info_yaml(binned); }, "or binning"));
  StereoCameraInfo binned_pair = synthetic_stereo_camera_info(DistortionModel::kPlumbBob);
  binned_pair.right.binning_x = 2;
  VERIFY(test::refuses([&] { (void)format_stereo_camera_info_yaml(binned_pair); },
                       "stereo_camera_info.right: a camera_info file has no region of interest"));
  // A camera whose frame id needs quotes keeps it.
  CameraInfo quoted = synthetic_stereo_camera_info(DistortionModel::kPlumbBob).left;
  quoted.header.frame_id = "cam #1: \"left\"\\";
  VERIFY(parse_camera_info_yaml(format_camera_info_yaml(quoted)) == quoted);
  quoted.header.frame_id = "";
  VERIFY(parse_camera_info_yaml(format_camera_info_yaml(quoted)) == quoted);
}

void test_read_ros_stereo() {
  for (const DistortionModel model : all_models()) {
    const StereoCameraInfo truth = synthetic_stereo_camera_info(model);
    const auto [left_text, right_text] = test::ros_stereo_camera_info(truth, 900.0, 640.5, 359.5);
    const StereoCameraInfo read = parse_stereo_camera_info_yaml(left_text, right_text);
    const std::string what(distortion_model_name(model));
    VERIFY_MSG(read.left == truth.left && read.right == truth.right, what + ": cameras");
    VERIFY_MSG(read.header.frame_id == "left", what + ": frame");
    // R = R_r^T R_l and t = -R_r^T (B, 0, 0), against the known extrinsics.
    VERIFY_MSG(close(read.right_from_left.rotation, truth.right_from_left.rotation, 1e-12),
               what + ": rotation");
    VERIFY_MSG(close(read.right_from_left.translation, truth.right_from_left.translation, 1e-9),
               what + ": translation");
    VERIFY_MSG(read.baseline == 0.0 && read.doffs == 0.0, what + ": not rectified");
  }
}

void test_read_ros_calibrator_precision() {
  // As the calibrator writes them: K and P with 5 decimals, D with 6, R with
  // 8, whose rounding leaves R up to ~1e-8 from a rotation. Each R reads back
  // as the rotation nearest to it, and the extrinsics to that rounding.
  for (const DistortionModel model : all_models()) {
    const StereoCameraInfo truth = synthetic_stereo_camera_info(model);
    const auto [left_text, right_text] =
        test::ros_stereo_camera_info(truth, 900.0, 640.5, 359.5, test::RosPrecision::kCalibrator);
    const std::string what(distortion_model_name(model));
    StereoCameraInfo read;
    try {
      read = parse_stereo_camera_info_yaml(left_text, right_text);
    } catch (const std::invalid_argument& error) {
      VERIFY_MSG(false, what + ": refused: " + error.what());
      continue;
    }
    for (const auto& [got, want] :
         {std::pair{&read.left, &truth.left}, std::pair{&read.right, &truth.right}}) {
      CameraInfo expected = test::ros_calibrator_rounded(*want);
      expected.rectification = got->rectification;
      VERIFY_MSG(*got == expected, what + ": K and D as written");
      VERIFY_MSG(got->rectification && close(*got->rectification, *want->rectification, 1e-7),
                 what + ": R to its 8 decimals");
    }
    VERIFY_MSG(close(read.right_from_left.rotation, truth.right_from_left.rotation, 1e-7),
               what + ": rotation");
    VERIFY_MSG(close(read.right_from_left.translation, truth.right_from_left.translation, 1e-7),
               what + ": translation");
  }
  // A matrix further from a rotation than that rounding is still refused.
  VERIFY(mono_refused("data: [ 1., 0., 0., 0., 1., 0., 0., 0., 1. ]",
                      "data: [ 1., 0.00001, 0., 0., 1., 0., 0., 0., 1. ]",
                      "camera_info.rectification is not a rotation"));
}

void test_stereo_round_trips() {
  for (const DistortionModel model : all_models()) {
    const StereoCameraInfo pair = synthetic_stereo_camera_info(model);
    const auto [left, right] = format_stereo_camera_info_yaml(pair);
    VERIFY_MSG(same_pair(parse_stereo_camera_info_yaml(left, right), pair),
               std::string(distortion_model_name(model)));
  }
  // A rectified pair, read back as one.
  StereoCameraInfo rectified;
  rectified.header.frame_id = "left";
  for (CameraInfo* camera : {&rectified.left, &rectified.right}) {
    camera->width = 1280;
    camera->height = 720;
    camera->state = CameraState::kRectified;
    camera->fx = 701.25;
    camera->fy = 701.25;
    camera->cx = 641.5;
    camera->cy = 361.75;
  }
  rectified.left.header.frame_id = "left";
  rectified.right.header.frame_id = "right";
  rectified.right.cx = 652.0;
  rectified.baseline = 0.0634;
  rectified.doffs = 652.0 - 641.5;
  rectified.right_from_left.translation = {-0.0634, 0.0, 0.0};
  const auto [left, right] = format_stereo_camera_info_yaml(rectified);
  const StereoCameraInfo back = parse_stereo_camera_info_yaml(left, right);
  VERIFY(back.left.state == CameraState::kRectified && back.right.state == CameraState::kRectified);
  VERIFY(same_pair(back, rectified));
  VERIFY(close(depth_from_disparity(back, 20.0), 701.25 * 0.0634 / 30.5, 1e-12));
  // A pair whose extrinsics the rectifications do not hold cannot be written.
  StereoCameraInfo free_pair = synthetic_stereo_camera_info(DistortionModel::kPlumbBob);
  free_pair.left.rectification.reset();
  free_pair.right.rectification.reset();
  VERIFY(test::refuses([&] { (void)format_stereo_camera_info_yaml(free_pair); },
                       "holds the extrinsics only as the rectifying rotations"));
}

void test_refusals() {
  VERIFY(mono_refused("image_width: 640\n", "", "image_width is missing"));
  VERIFY(mono_refused("image_height: 480", "image_height: 0", "image_height must be > 0"));
  VERIFY(mono_refused("image_height: 480", "image_height: 4.5",
                      "image_height (line 2) must be a whole number"));
  VERIFY(
      mono_refused("camera_matrix:\n  rows: 3\n  cols: 3\n  data: [ 412.34, 0., 318.72, 0., "
                   "413.01, 241.37, 0., 0., 1. ]\n",
                   "", "camera_matrix is missing"));
  VERIFY(mono_refused("  rows: 3\n  cols: 3\n  data: [ 412.34",
                      "  rows: 3\n  cols: 4\n  data: [ 412.34",
                      "camera_matrix must be 3x3, not 3x4"));
  VERIFY(
      mono_refused("0., 0., 1. ]", "0., 0. ]", "camera_matrix is 3x3 but its data has 8 values"));
  VERIFY(mono_refused("0., 413.01", "1., 413.01",
                      "camera_matrix must be [fx, skew, cx, 0, fy, cy, 0, 0, 1]"));
  VERIFY(mono_refused("412.34,", "abc,",
                      "camera_matrix.data[0] (line 7) holds 'abc', not a finite number"));
  VERIFY(mono_refused("412.34,", ".nan,", "not a finite number"));
  VERIFY(mono_refused("[ -0.31,", "[ +-0.31,", "holds '+-0.31', not a finite number"));
  VERIFY(mono_refused("[ -0.31,", "[ ++0.31,", "holds '++0.31', not a finite number"));
  VERIFY(mono_refused("  rows: 3\n  cols: 3\n  data: [ 412.34",
                      "  rows: 3\n  cols: 3\n  depth: 2\n  data: [ 412.34",
                      "camera_matrix.depth (line 7) is not a matrix key"));
  VERIFY(mono_refused("  cols: 5\n  data: [ -0.31, 0.11, 0.0012, -0.0004, 0. ]",
                      "  cols: 4\n  data: [ -0.31, 0.11, 0.0012, -0.0004 ]",
                      "distortion_coefficients has 4 values, but the plumb_bob model has 5"));
  VERIFY(mono_refused("distortion_model: plumb_bob", "distortion_model: fisheye",
                      "distortion_model 'fisheye' is not supported"));
  VERIFY(mono_refused("distortion_model: plumb_bob\n", "",
                      "the none model has 0 (distortion_model is missing)"));
  VERIFY(mono_refused("data: [ 1., 0., 0., 0., 1., 0., 0., 0., 1. ]",
                      "data: [ 1., 0.2, 0., 0., 1., 0., 0., 0., 1. ]",
                      "camera_info.rectification is not a rotation"));
  VERIFY(mono_refused("0., 0., 1., 0. ]", "0., 0., 1., 5. ]",
                      "projection_matrix must be [fx', s', cx', Tx, 0, fy', cy', Ty, 0, 0, 1, 0]"));
  VERIFY(mono_refused("camera_name: narrow_stereo", "camera_nmae: narrow_stereo",
                      "'camera_nmae' (line 3) is not a camera_info key"));
  VERIFY(mono_refused("image_width: 640", "image_width: [640]",
                      "image_width (line 1) must be a number"));
  VERIFY(mono_refused("image_width: 640", "image_width: \"640\"",
                      "image_width (line 1) must be a number"));
  VERIFY(mono_refused("image_width: 640", "image_width: 640\n\tx: 1",
                      "line 2: indentation must be spaces"));
  // A stereo pair needs both projections, and a baseline in the right one.
  const StereoCameraInfo truth = synthetic_stereo_camera_info(DistortionModel::kPlumbBob);
  auto [left, right] = test::ros_stereo_camera_info(truth, 900.0, 640.5, 359.5);
  const std::string no_projection = right.substr(0, right.find("projection_matrix"));
  VERIFY(test::refuses([&] { (void)parse_stereo_camera_info_yaml(left, no_projection); },
                       "right: projection_matrix is missing"));
  const auto [same_left, same_right] = test::ros_stereo_camera_info(truth, 900.0, 640.5, 359.5);
  const std::string swapped = same_left;
  VERIFY(test::refuses([&] { (void)parse_stereo_camera_info_yaml(same_right, swapped); },
                       "right: projection_matrix carries no baseline"));
  VERIFY(test::refuses([&] { (void)parse_stereo_camera_info_yaml(left, "image_width: 1\n"); },
                       "right camera_info: image_height is missing"));
}

void test_files() {
  const test::TemporaryFolder folder("messages_yaml_test");
  const fs::path& dir = folder.path();
  const StereoCameraInfo pair = synthetic_stereo_camera_info(DistortionModel::kRationalPolynomial);
  write_camera_info_yaml(pair.left, dir / "mono.yaml");
  VERIFY(read_camera_info_yaml(dir / "mono.yaml") == pair.left);
  write_stereo_camera_info_yaml(pair, dir / "left.yaml", dir / "right.yaml");
  VERIFY(same_pair(read_stereo_camera_info_yaml(dir / "left.yaml", dir / "right.yaml"), pair));
  // Errors start with the file.
  const fs::path missing = dir / "missing.yaml";
  VERIFY(test::refuses([&] { (void)read_camera_info_yaml(missing); },
                       missing.string() + ": cannot read"));
  {
    std::ofstream bad(dir / "bad.yaml");
    bad << "image_width: 1\nimage_height: 1\n";
  }
  VERIFY(test::refuses([&] { (void)read_camera_info_yaml(dir / "bad.yaml"); },
                       (dir / "bad.yaml").string() + ": camera_matrix is missing"));
  VERIFY(test::refuses(
      [&] { (void)read_stereo_camera_info_yaml(dir / "left.yaml", dir / "bad.yaml"); },
      (dir / "bad.yaml").string() + ": camera_matrix is missing"));
  {
    std::ofstream big(dir / "big.yaml");
    big << std::string(kMaxCameraInfoFileBytes + 1, '#');
  }
  VERIFY(test::refuses([&] { (void)read_camera_info_yaml(dir / "big.yaml"); },
                       "more than a camera_info file has"));
  VERIFY(test::refuses([&] { write_camera_info_yaml(pair.left, dir / "no" / "such" / "dir.yaml"); },
                       "cannot open for writing"));
  CameraInfo invalid = pair.left;
  invalid.fx = 0;
  VERIFY(test::refuses([&] { write_camera_info_yaml(invalid, dir / "invalid.yaml"); },
                       "camera_info.fx"));
}

}  // namespace
}  // namespace holoscan::messages

int main() {
  namespace m = holoscan::messages;
  return holoscan::messages::test::run({
      {"YAML forms", m::test_yaml_forms},
      {"YAML refusals give the line", m::test_yaml_refusals},
      {"read a ROS mono calibration", m::test_read_ros_mono},
      {"mono round trips, every model", m::test_mono_round_trips},
      {"read ROS stereo calibrations: extrinsics recovered", m::test_read_ros_stereo},
      {"read ROS stereo calibrations at the calibrator's precision",
       m::test_read_ros_calibrator_precision},
      {"stereo round trips, every model and rectified", m::test_stereo_round_trips},
      {"refusals name what is wrong", m::test_refusals},
      {"files", m::test_files},
  });
}
