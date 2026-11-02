// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The rig calibration without a GPU: distortion models, mounting poses, the
// projection and its inverse, the lens' valid range, and rig files, a
// recording's calibration.json included.

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/stitching/camera_rig.hpp>

#include "test_support.hpp"

namespace holoscan::stitching {
namespace {

std::filesystem::path g_dir;

Camera pinhole(DistortionModel model, std::vector<double> coefficients) {
  Camera c;
  c.name = "camera";
  c.width = 1280;
  c.height = 720;
  c.fx = 600.0;
  c.fy = 610.0;
  c.cx = 641.0;
  c.cy = 358.0;
  c.distortion_model = model;
  c.distortion = std::move(coefficients);
  return c;
}

std::array<double, 3> to_camera(const Camera& c, const std::array<double, 3>& p) {
  const auto& r = c.camera_from_rig.rotation;
  const auto& t = c.camera_from_rig.translation;
  return {r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0],
          r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1],
          r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2]};
}

void model_names() {
  VERIFY(parse_distortion_model("plumb_bob") == DistortionModel::kBrownConrady);
  VERIFY(parse_distortion_model("equidistant") == DistortionModel::kKannalaBrandt4);
  VERIFY(parse_distortion_model("fisheye") == DistortionModel::kKannalaBrandt4);
  VERIFY(!parse_distortion_model("modified_brown_conrady"));
  for (const auto model :
       {DistortionModel::kNone, DistortionModel::kBrownConrady,
        DistortionModel::kRationalPolynomial, DistortionModel::kKannalaBrandt4}) {
    VERIFY(parse_distortion_model(distortion_model_name(model)) == model);
  }
  VERIFY(distortion_coefficient_count(DistortionModel::kRationalPolynomial) == 8);
}

void mounting() {
  Camera c = pinhole(DistortionModel::kNone, {});
  // Upright, looking forward: +x is the image centre, +z (up) above it, +y
  // (left) to the left.
  c.camera_from_rig = mount({0.0, 0.0, 0.0}, 0.0, 0.0);
  const auto ahead = project_to_pixel(c, to_camera(c, {5.0, 0.0, 0.0}));
  VERIFY(ahead && std::abs(ahead->x - c.cx) < 1e-9 && std::abs(ahead->y - c.cy) < 1e-9);
  const auto up = project_to_pixel(c, to_camera(c, {5.0, 0.0, 1.0}));
  VERIFY(up && up->y < c.cy && std::abs(up->x - c.cx) < 1e-9);
  const auto left = project_to_pixel(c, to_camera(c, {5.0, 1.0, 0.0}));
  VERIFY(left && left->x < c.cx);
  VERIFY(!project_to_pixel(c, to_camera(c, {-5.0, 0.0, 0.0})));
  // Turned left by 90 degrees, it looks along +y; tilted down, at the ground.
  c.camera_from_rig = mount({1.0, 2.0, 3.0}, 90.0, 0.0);
  const auto side = project_to_pixel(c, to_camera(c, {1.0, 7.0, 3.0}));
  VERIFY(side && std::abs(side->x - c.cx) < 1e-6 && std::abs(side->y - c.cy) < 1e-6);
  c.camera_from_rig = mount({0.0, 0.0, 2.0}, 0.0, -45.0);
  const auto ground = project_to_pixel(c, to_camera(c, {2.0, 0.0, 0.0}));
  VERIFY(ground && std::abs(ground->x - c.cx) < 1e-6 && std::abs(ground->y - c.cy) < 1e-6);
  // Rolled clockwise by 90 degrees (seen from behind), its top points to the
  // rig's right, so up is to the image's left.
  c.camera_from_rig = mount({0.0, 0.0, 0.0}, 0.0, 0.0, 90.0);
  const auto rolled = project_to_pixel(c, to_camera(c, {5.0, 0.0, 1.0}));
  VERIFY(rolled && rolled->x < c.cx - 50.0 && std::abs(rolled->y - c.cy) < 1e-6);
  // The pose is a rigid transform: inverse and compose agree.
  const RigidTransform a = mount({1.0, -2.0, 0.5}, 33.0, -12.0, 4.0);
  const RigidTransform identity = compose(a, inverse(a));
  for (int i = 0; i < 9; ++i) {
    VERIFY(std::abs(identity.rotation[i] - (i % 4 == 0 ? 1.0 : 0.0)) < 1e-12);
  }
  for (const double t : identity.translation) VERIFY(std::abs(t) < 1e-12);
}

void projection_round_trips() {
  const std::vector<Camera> cameras{
      pinhole(DistortionModel::kNone, {}),
      pinhole(DistortionModel::kBrownConrady, {-0.28, 0.07, 0.001, -0.0005, -0.006}),
      pinhole(DistortionModel::kRationalPolynomial,
              {0.4, -0.1, 0.0005, 0.0002, 0.01, 0.75, -0.05, 0.03}),
      pinhole(DistortionModel::kKannalaBrandt4, {0.02, -0.005, 0.001, -0.0001}),
  };
  for (const Camera& c : cameras) {
    const std::string name(distortion_model_name(c.distortion_model));
    int checked = 0;
    for (double y = -0.5; y <= c.height - 0.5; y += 37.3) {
      for (double x = -0.5; x <= c.width - 0.5; x += 41.7) {
        const auto ray = pixel_ray(c, {x, y});
        if (!ray) continue;
        VERIFY_MSG(std::abs(std::hypot((*ray)[0], (*ray)[1], (*ray)[2]) - 1.0) < 1e-12, name);
        const auto back = project_to_pixel(c, {3.0 * (*ray)[0], 3.0 * (*ray)[1], 3.0 * (*ray)[2]});
        VERIFY_MSG(back && std::abs(back->x - x) < 1e-6 && std::abs(back->y - y) < 1e-6,
                   name + " round trip at " + std::to_string(x) + ", " + std::to_string(y));
        ++checked;
      }
    }
    VERIFY_MSG(checked > 300, name + ": " + std::to_string(checked) + " pixels");
  }
  // The fisheye sees behind its image plane.
  const Camera& fisheye = cameras[3];
  const double theta = 100.0 * std::numbers::pi / 180.0;
  const auto wide = project_to_pixel(fisheye, {std::sin(theta), 0.0, std::cos(theta)});
  VERIFY(wide && wide->x > fisheye.cx + 900.0);
  // Brown-Conrady by its formula.
  const Point2 d = distort(cameras[1], {0.3, -0.2});
  const double r2 = 0.13;
  const double radial = 1.0 - 0.28 * r2 + 0.07 * r2 * r2 - 0.006 * r2 * r2 * r2;
  VERIFY(std::abs(d.x - (0.3 * radial + 2 * 0.001 * 0.3 * -0.2 - 0.0005 * (r2 + 2 * 0.09))) <
         1e-15);
  VERIFY(std::abs(d.y - (-0.2 * radial + 0.001 * (r2 + 2 * 0.04) - 2 * 0.0005 * 0.3 * -0.2)) <
         1e-15);
}

void valid_range() {
  VERIFY(std::abs(max_valid_angle(pinhole(DistortionModel::kNone, {})) -
                  (std::numbers::pi / 2.0 - 1e-3)) < 2e-3);
  // Strong barrel distortion folds back: past that angle nothing projects.
  const Camera folding = pinhole(DistortionModel::kBrownConrady, {-0.5, 0.0, 0.0, 0.0, 0.0});
  const double fold = max_valid_angle(folding);
  // d/dr (r - 0.5 r^3) = 0 at r = sqrt(2/3).
  VERIFY(std::abs(fold - std::atan(std::sqrt(2.0 / 3.0))) < 2e-3);
  const double beyond = std::tan(fold + 0.01);
  VERIFY(!project_to_pixel(folding, {beyond, 0.0, 1.0}));
  VERIFY(project_to_pixel(folding, {std::tan(fold - 0.01), 0.0, 1.0}).has_value());
  const Camera fisheye = pinhole(DistortionModel::kKannalaBrandt4, {-0.1, 0.0, 0.0, 0.0});
  // d/dtheta (theta - 0.1 theta^3) = 0 at theta = sqrt(10/3).
  VERIFY(std::abs(max_valid_angle(fisheye) - std::sqrt(10.0 / 3.0)) < 2e-3);
  Camera bad = pinhole(DistortionModel::kKannalaBrandt4, {-1e6, 0.0, 0.0, 0.0});
  VERIFY(test::throws<std::invalid_argument>([&] { validate(bad); }));
}

CameraRig two_cameras() {
  CameraRig rig;
  Camera a = pinhole(DistortionModel::kKannalaBrandt4, {0.02, -0.005, 0.001, -0.0001});
  a.name = "front \"wide\"";
  a.camera_from_rig = mount({2.0, 0.0, 0.8}, 0.0, -30.0);
  Camera b = pinhole(DistortionModel::kRationalPolynomial,
                     {0.4, -0.1, 0.0005, 0.0002, 0.01, 0.75, -0.05, 0.03});
  b.name = "rear";
  b.camera_from_rig = mount({-2.0, 0.1, 0.8}, 181.0, -31.0, 2.0);
  rig.cameras = {a, b};
  return rig;
}

bool same(const CameraRig& a, const CameraRig& b) {
  if (a.cameras.size() != b.cameras.size()) return false;
  for (std::size_t i = 0; i < a.cameras.size(); ++i) {
    const Camera& x = a.cameras[i];
    const Camera& y = b.cameras[i];
    if (x.name != y.name || x.width != y.width || x.height != y.height || x.fx != y.fx ||
        x.fy != y.fy || x.cx != y.cx || x.cy != y.cy || x.distortion_model != y.distortion_model ||
        x.distortion != y.distortion || x.camera_from_rig.rotation != y.camera_from_rig.rotation ||
        x.camera_from_rig.translation != y.camera_from_rig.translation) {
      return false;
    }
  }
  return true;
}

void rig_files() {
  const CameraRig rig = two_cameras();
  VERIFY(same(parse_rig(to_json(rig)), rig));
  const std::filesystem::path file = g_dir / "rig.json";
  std::filesystem::create_directories(g_dir);
  std::ofstream(file) << to_json(rig);
  VERIFY(same(read_rig(file), rig));
  VERIFY(test::throws<std::invalid_argument>([&] { (void)read_rig(g_dir / "missing.json"); }));

  // rig_from_camera is the inverse; both, or neither, is an error.
  const std::string base =
      R"({"version": 1, "cameras": [{"name": "a", "width": 64, "height": 48, "fx": 50, "fy": 50,
          "cx": 31.5, "cy": 23.5, "distortion": {"model": "plumb_bob",
          "coefficients": [0.1, 0.01, 0, 0]}, )";
  const std::string pose =
      R"({"rotation": [0, -1, 0, 1, 0, 0, 0, 0, 1], "translation": [1, 2, 3]})";
  const CameraRig from_camera = parse_rig(base + R"("rig_from_camera": )" + pose + "}]}");
  const CameraRig from_rig = parse_rig(base + R"("camera_from_rig": )" + pose + "}]}");
  const RigidTransform back = inverse(from_camera.cameras[0].camera_from_rig);
  VERIFY(back.rotation == from_rig.cameras[0].camera_from_rig.rotation);
  VERIFY(back.translation == from_rig.cameras[0].camera_from_rig.translation);
  // Four Brown-Conrady coefficients mean k3 = 0.
  VERIFY(from_rig.cameras[0].distortion.size() == 5 && from_rig.cameras[0].distortion[4] == 0.0);

  const auto fails = [](const std::string& text, const std::string& what) {
    try {
      (void)parse_rig(text);
    } catch (const std::invalid_argument& error) {
      const bool named = std::string(error.what()).find(what) != std::string::npos;
      VERIFY_MSG(named, "'" + what + "' not in: " + error.what());
      return;
    }
    VERIFY_MSG(false, "accepted, expected: " + what);
  };
  fails(base.substr(0, base.rfind(',')) + "}]}", "needs camera_from_rig or rig_from_camera");
  fails(base + R"("rig_from_camera": )" + pose + R"(, "camera_from_rig": )" + pose + "}]}",
        "not both");
  fails(R"({"version": 2, "cameras": []})", "version must be 1");
  fails(R"({"cameras": []})", "a rig has 1..8 cameras");
  fails(R"({"cameras": {}})", "cameras must be an array");
  fails(base + R"("camera_from_rig": {"rotation": [1, 0, 0, 0, 1, 0, 0, 0, 2],
        "translation": [0, 0, 0]}}]})",
        "not orthonormal");
  fails(base + R"("camera_from_rig": {"rotation": [1, 0, 0, 0, 1, 0, 0, 0, -1],
        "translation": [0, 0, 0]}}]})",
        "reflection");
  std::string wrong_model = base + R"("camera_from_rig": )" + pose + "}]}";
  wrong_model.replace(wrong_model.find("plumb_bob"), 9, "kannala_brandt4");
  wrong_model.replace(wrong_model.find("[0.1, 0.01, 0, 0]"), 17, "[0.1, 0.01, 0]");
  fails(wrong_model, "takes 4 coefficients, not 3");
  std::string unknown = base + R"("camera_from_rig": )" + pose + "}]}";
  unknown.replace(unknown.find("plumb_bob"), 9, "modified_brown_conrady");
  fails(unknown, "is not supported");
  const std::string one = base + R"("camera_from_rig": )" + pose + "}";
  fails(R"({"cameras": [)" + one.substr(one.find("{\"name\"")) + ", " +
            one.substr(one.find("{\"name\"")) + "]}",
        "two cameras are named a");
  fails("{\"cameras\": [", "JSON");
}

void readme_example() {
  // The rig file of the README: the surround rig's front camera.
  const CameraRig rig = parse_rig(R"({
  "version": 1,
  "cameras": [
    {
      "name": "front", "width": 1280, "height": 720,
      "fx": 330.0, "fy": 330.0, "cx": 639.5, "cy": 359.5,
      "distortion": {"model": "kannala_brandt4", "coefficients": [0.02, -0.005, 0.001, -0.0001]},
      "camera_from_rig": {
        "rotation": [0, -1, 0, -0.5, 0, -0.8660254037844386, 0.8660254037844386, 0, -0.5],
        "translation": [0, 1.692820323027551, -1.3320508075688773]
      }
    }
  ]
})");
  const RigidTransform want = mount({2.0, 0.0, 0.8}, 0.0, -30.0);
  for (int i = 0; i < 9; ++i) {
    VERIFY(std::abs(rig.cameras[0].camera_from_rig.rotation[i] - want.rotation[i]) < 1e-12);
  }
  for (int i = 0; i < 3; ++i) {
    VERIFY(std::abs(rig.cameras[0].camera_from_rig.translation[i] - want.translation[i]) < 1e-12);
  }
}

void recording_calibration() {
  // A recording's calibration.json, as the stereo camera recordings write it.
  const std::string text = R"({
    "version": 1,
    "camera": {"name": "RealSense D585 Prototype", "serial": "1"},
    "rectified": true,
    "left": {"width": 848, "height": 480, "fx": 223.25, "fy": 223.25, "cx": 446.39,
             "cy": 237.05, "distortion": {"model": "brown_conrady",
             "coefficients": [0, 0, 0, 0, 0]}},
    "right": {"width": 848, "height": 480, "fx": 223.25, "fy": 223.25, "cx": 446.39,
              "cy": 237.05, "distortion": {"model": "brown_conrady",
              "coefficients": [0, 0, 0, 0, 0]}},
    "right_from_left": {"rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        "translation": [-0.0948, 0, 0]},
    "rgb": {"width": 848, "height": 480, "fx": 219.54, "fy": 219.67, "cx": 465.17, "cy": 254.55,
            "distortion": {"model": "brown_conrady",
                           "coefficients": [-0.0005, 0.0007, -0.0001, -0.0002, -0.0001]},
            "from_left": {"rotation": [0.9999869465827942, 0.004099610261619091,
                                       -0.003044846700504422, -0.004095201846212149,
                                       0.9999905824661255, 0.001452688593417406,
                                       0.0030507734045386314, -0.0014402003725990653,
                                       0.9999943375587463],
                          "translation": [-0.0476, 0.0004, -0.0001]}},
    "depth": {"width": 848, "height": 480, "fx": 223.25, "fy": 223.25, "cx": 446.39, "cy": 237.05}
  })";
  const CameraRig rig = parse_rig(text);
  VERIFY(rig.cameras.size() == 3);
  VERIFY(rig.cameras[0].name == "left" && rig.cameras[1].name == "right" &&
         rig.cameras[2].name == "rgb");
  VERIFY(rig.cameras[0].distortion.empty());  // All zero.
  VERIFY(rig.cameras[1].camera_from_rig.translation[0] == -0.0948);
  VERIFY(rig.cameras[2].distortion.size() == 5 && rig.cameras[2].distortion[0] == -0.0005);
  VERIFY(rig.cameras[2].camera_from_rig.translation[0] == -0.0476);
  // The float32 rotation is a rotation to the 1e-6 the check allows.
  VERIFY(rig.cameras[2].camera_from_rig.rotation[1] == 0.004099610261619091);
}

}  // namespace
}  // namespace holoscan::stitching

int main(int argc, char** argv) {
  namespace st = holoscan::stitching;
  st::g_dir = argc > 1 ? argv[1] : std::filesystem::temp_directory_path() / "stitching_calibration";
  return st::test::run({
      {"model_names", st::model_names},
      {"mounting", st::mounting},
      {"projection_round_trips", st::projection_round_trips},
      {"valid_range", st::valid_range},
      {"rig_files", st::rig_files},
      {"readme_example", st::readme_example},
      {"recording_calibration", st::recording_calibration},
  });
}
