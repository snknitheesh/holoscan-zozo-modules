// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The calibration file: a real RealSense calibration and the synthetic one read
// back exactly from what format_calibration() writes; every field the reader
// requires, every check validate() makes, and JSON that is not JSON.

#include <cmath>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/stereo_camera/calibration.hpp>
#include <holoscan/stereo_camera/synthetic.hpp>

#include "host_support.hpp"
#include "json.hpp"
#include "test_support.hpp"

namespace holoscan::stereo_camera {
namespace {

using test::contains;
using test::invalid_message;

// A D585's calibration at 1280x720, as librealsense reported it.
StereoCalibration realsense_like() {
  StereoCalibration c;
  c.left = CameraModel{.width = 1280,
                       .height = 720,
                       .fx = 336.98760986328125,
                       .fy = 336.98760986328125,
                       .cx = 674.053955078125,
                       .cy = 355.81134033203125,
                       .distortion_model = DistortionModel::kBrownConrady,
                       .distortion = {0, 0, 0, 0, 0}};
  c.right = c.left;
  c.right_from_left.translation = {-0.094843290746212, 0, 0};
  c.rectified = true;
  c.rgb = AuxiliaryCamera{
      .model = CameraModel{.width = 1280,
                           .height = 720,
                           .fx = 331.381103515625,
                           .fy = 331.5785827636719,
                           .cx = 702.3963012695312,
                           .cy = 382.2177429199219,
                           .distortion_model = DistortionModel::kBrownConrady,
                           .distortion = {-0.0005273931310512125, 0.000670437584631145,
                                          -0.0001487435947638005, -0.00016126202535815537,
                                          -0.00012041200534440577}},
      .from_left = RigidTransform{
          .rotation = {0.9999869465827942, 0.004099610261619091, -0.003044846700504422,
                       -0.004095201846212149, 0.9999905824661255, 0.001452688593417406,
                       0.0030507734045386314, -0.0014402003725990653, 0.9999943375587463},
          .translation = {-0.047618623822927475, 0.0003647664561867714, -9.136702283285558e-05}}};
  c.depth = DepthCamera{.model = c.left, .scale = 0.00045200000749900937};
  c.camera = CameraInfo{.name = "RealSense D585",
                        .serial = "541522320091",
                        .firmware = "7.58.40947.13653",
                        .settings = {{"emitter_enabled", "true"}, {"fps", "30"}}};
  return c;
}

bool same_camera(const CameraModel& a, const CameraModel& b) {
  return a.width == b.width && a.height == b.height && a.fx == b.fx && a.fy == b.fy &&
         a.cx == b.cx && a.cy == b.cy && a.distortion_model == b.distortion_model &&
         a.distortion == b.distortion;
}

bool same_transform(const RigidTransform& a, const RigidTransform& b) {
  return a.rotation == b.rotation && a.translation == b.translation;
}

bool same(const StereoCalibration& a, const StereoCalibration& b) {
  return same_camera(a.left, b.left) && same_camera(a.right, b.right) &&
         same_transform(a.right_from_left, b.right_from_left) && a.rectified == b.rectified &&
         a.rgb.has_value() == b.rgb.has_value() &&
         (!a.rgb || (same_camera(a.rgb->model, b.rgb->model) &&
                     same_transform(a.rgb->from_left, b.rgb->from_left))) &&
         a.depth.has_value() == b.depth.has_value() &&
         (!a.depth || (same_camera(a.depth->model, b.depth->model) &&
                       same_transform(a.depth->from_left, b.depth->from_left) &&
                       a.depth->scale == b.depth->scale)) &&
         a.camera.name == b.camera.name && a.camera.serial == b.camera.serial &&
         a.camera.firmware == b.camera.firmware && a.camera.settings == b.camera.settings;
}

void check_round_trip() {
  for (const StereoCalibration& original :
       {realsense_like(), synthetic_calibration(320, 240, false, false),
        synthetic_calibration(640, 480, true, true)}) {
    const std::string text = format_calibration(original);
    const StereoCalibration back = parse_calibration(text);
    VERIFY_MSG(same(original, back), "round trip changed the calibration:\n" + text);
    VERIFY(format_calibration(back) == text);
  }
  const StereoCalibration c = realsense_like();
  VERIFY(std::abs(c.baseline() - 0.094843290746212) < 1e-15);
  VERIFY(std::abs(c.depth_from_disparity() - 336.98760986328125 * 0.094843290746212) < 1e-9);
}

void check_file_round_trip(const std::filesystem::path& work) {
  const auto folder = test::fresh_folder(work, "calibration_file");
  const StereoCalibration original = realsense_like();
  write_calibration(folder / "c.json", original);
  VERIFY(same(read_calibration(folder / "c.json"), original));
  const std::string missing = invalid_message([&] { (void)read_calibration(folder / "no.json"); });
  VERIFY_MSG(contains(missing, "no.json: cannot open"), missing);
  test::write_text(folder / "bad.json", "{\"version\": 1,\n  \"left\": }");
  const std::string bad = invalid_message([&] { (void)read_calibration(folder / "bad.json"); });
  VERIFY_MSG(contains(bad, "bad.json: line 2, column 11"), bad);
}

// The minimal valid file, with `edit` applied to its text.
std::string minimal(const std::string& extra = "") {
  return R"({
  "version": 1,
  "rectified": false,
  "left": {"width": 640, "height": 480, "fx": 500, "fy": 500, "cx": 319.5, "cy": 239.5},
  "right": {"width": 640, "height": 480, "fx": 500, "fy": 500, "cx": 319.5, "cy": 239.5},
  "right_from_left": {"rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1], "translation": [-0.12, 0, 0]})" +
         extra + "\n}";
}

std::string replace(std::string text, const std::string& from, const std::string& to) {
  const std::size_t at = text.find(from);
  if (at == std::string::npos) throw std::logic_error("no " + from);
  return text.replace(at, from.size(), to);
}

void check_minimal_and_unknown_keys() {
  const StereoCalibration c = parse_calibration(minimal(R"(,
  "future": {"anything": [1, "two", null, true]})"));
  VERIFY(c.left.distortion_model == DistortionModel::kNone && c.left.distortion.empty());
  VERIFY(!c.rgb && !c.depth && !c.rectified);
  VERIFY(c.camera.name.empty() && c.camera.settings.empty());
  VERIFY(std::abs(c.baseline() - 0.12) < 1e-15);
}

void check_rejections() {
  const std::vector<std::pair<std::string, std::string>> cases{
      {minimal(", \"version\": 2"), "duplicate member \"version\""},
      {replace(minimal(), "\"version\": 1", "\"version\": 2"), "reads version 1, not 2"},
      {replace(minimal(), "\"version\": 1,", ""), "missing \"version\""},
      {replace(minimal(), "\"rectified\": false,", ""), "missing \"rectified\""},
      {replace(minimal(), "\"rectified\": false", "\"rectified\": 0"),
       "rectified: expected a boolean, not a number"},
      {replace(minimal(), "\"width\": 640, \"height\": 480, \"fx\": 500",
               "\"width\": 640.5, \"height\": 480, \"fx\": 500"),
       "left.width: expected a whole number"},
      {replace(minimal(), "\"fx\": 500, \"fy\": 500, \"cx\": 319.5",
               "\"fx\": -1, \"fy\": 500, "
               "\"cx\": 319.5"),
       "left: fx and fy must be finite and positive"},
      {replace(minimal(),
               "\"width\": 640, \"height\": 480, \"fx\": 500, \"fy\": 500, \"cx\": "
               "319.5, \"cy\": 239.5}",
               "\"width\": 0, \"height\": 480, \"fx\": 500, \"fy\": 500, \"cx\": 319.5, \"cy\": "
               "239.5}"),
       "left: width and height must be 1..16384"},
      {replace(minimal(), "[-0.12, 0, 0]", "[0, 0, 0]"),
       "right_from_left: the baseline must not be zero"},
      {replace(minimal(), "[-0.12, 0, 0]", "[-0.12, 0]"),
       "right_from_left.translation: expected 3 numbers"},
      {replace(minimal(), "[1, 0, 0, 0, 1, 0, 0, 0, 1]", "[1, 0, 0, 0, 1, 0, 0, 0, -1]"),
       "rotation must have determinant 1"},
      {replace(minimal(), "[1, 0, 0, 0, 1, 0, 0, 0, 1]", "[2, 0, 0, 0, 1, 0, 0, 0, 1]"),
       "rotation is not orthonormal"},
      {replace(minimal(), "\"cy\": 239.5}", "\"cy\": 239.5, \"distortion\": {\"model\": \"zz\"}}"),
       "left.distortion.model: unknown distortion model \"zz\""},
      {replace(minimal(), "\"cy\": 239.5}",
               "\"cy\": 239.5, \"distortion\": {\"model\": \"kannala_brandt4\", "
               "\"coefficients\": [1, 2]}}"),
       "kannala_brandt4 takes 4 coefficients, not 2"},
      {replace(replace(minimal(), "\"rectified\": false", "\"rectified\": true"), "\"cy\": 239.5}",
               "\"cy\": 239.5, \"distortion\": {\"model\": \"brown_conrady\", "
               "\"coefficients\": [0.1, 0, 0, 0, 0]}}"),
       "rectified: a rectified pair has no distortion"},
      {replace(minimal(), "\"right\": {\"width\": 640", "\"right\": {\"width\": 320"),
       "the eyes must have the same size"},
      {minimal(R"(, "depth": {"width": 640, "height": 480, "fx": 500, "fy": 500, "cx": 1,
         "cy": 1, "from_left": {"rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
         "translation": [0, 0, 0]}, "scale": 0})"),
       "depth.scale: must be finite and positive"},
      {minimal(R"(, "rgb": {"width": 640, "height": 480, "fx": 500, "fy": 500, "cx": 1,
         "cy": 1})"),
       "rgb: missing \"from_left\""},
      {"[1, 2]", "calibration: expected an object, not an array"},
      {"", "line 1, column 1: unexpected character"},
      {minimal() + " x", "unexpected text after the value"},
      {"{\"version\": 1,}", "expected a member name"},
      {"{\"a\": \"\\q\"}", "bad escape"},
      {"{\"a\": 1e999}", "number out of range"},
      {"{\"a\": 01}", "expected '}'"},
      {"{\"a\": \"x", "unterminated string"},
      {std::string(70, '[') + std::string(70, ']'), "nested deeper than 64"},
  };
  for (const auto& [text, expected] : cases) {
    const std::string message = invalid_message([&] { (void)parse_calibration(text); });
    VERIFY_MSG(contains(message, expected),
               "expected \"" + expected + "\", got \"" + message + "\" for:\n" + text);
  }
}

void check_json_strings_and_numbers() {
  const json::Value value = json::parse(R"({"s": "a\"b\\c\/\n\u00e9\ud83d\ude00", "n": -1.5e-3})");
  VERIFY(value.find("s")->string == "a\"b\\c/\n\xC3\xA9\xF0\x9F\x98\x80");
  VERIFY(value.find("n")->number == -1.5e-3);
  VERIFY(json::parse(json::quote("tab\tnl\n\x01quote\"")).string == "tab\tnl\n\x01quote\"");
  for (const double x : {0.1, 1.0 / 3.0, 336.98760986328125, -9.136702283285558e-05, 1e300}) {
    VERIFY(json::parse(json::number(x)).number == x);
  }
  VERIFY(json::number(-0.0) == "0");
  VERIFY(test::throws<std::invalid_argument>([] { (void)json::number(std::nan("")); }));
}

void check_model_names() {
  for (const auto model : {DistortionModel::kNone, DistortionModel::kBrownConrady,
                           DistortionModel::kModifiedBrownConrady,
                           DistortionModel::kInverseBrownConrady, DistortionModel::kKannalaBrandt4,
                           DistortionModel::kFTheta, DistortionModel::kRationalPolynomial}) {
    VERIFY(parse_distortion_model(distortion_model_name(model)) == model);
  }
  VERIFY(!parse_distortion_model("plumb_bob"));
  VERIFY(distortion_coefficient_count(DistortionModel::kRationalPolynomial) == 8);
}

}  // namespace
}  // namespace holoscan::stereo_camera

int main(int argc, char** argv) {
  namespace sc = holoscan::stereo_camera;
  const std::filesystem::path work = argc > 1 ? argv[1] : std::filesystem::current_path();
  return sc::test::run({
      {"calibration round trip", sc::check_round_trip},
      {"calibration file round trip", [&] { sc::check_file_round_trip(work); }},
      {"minimal file, unknown keys ignored", sc::check_minimal_and_unknown_keys},
      {"rejected files", sc::check_rejections},
      {"JSON strings and numbers", sc::check_json_strings_and_numbers},
      {"distortion model names", sc::check_model_names},
  });
}
