// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The rig's cameras without a GPU: their calibration from ROS camera_info
// files (holoscan-messages' CameraInfo) and what the stitcher refuses,
// mounting poses, the projection and its inverse for every lens model, and
// the lens' valid range.

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/messages/camera_info.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/stitching/camera_rig.hpp>

#include "test_support.hpp"

namespace holoscan::stitching {
namespace {

namespace hm = holoscan::messages;

hm::CameraInfo pinhole(hm::DistortionModel model, std::vector<double> coefficients) {
  hm::CameraInfo c;
  c.header.frame_id = "camera";
  c.width = 1280;
  c.height = 720;
  c.fx = 600.0;
  c.fy = 610.0;
  c.cx = 641.0;
  c.cy = 358.0;
  c.distortion_model = model;
  c.coefficients = std::move(coefficients);
  c.coefficients.resize(hm::coefficient_count(model), 0.0);
  bool raw = false;
  for (const double k : c.coefficients) raw = raw || k != 0.0;
  c.state = raw ? hm::CameraState::kRaw : hm::CameraState::kUndistorted;
  return c;
}

std::array<double, 3> to_camera(const Camera& c, const std::array<double, 3>& p) {
  const auto& r = c.camera_from_rig.rotation;
  const auto& t = c.camera_from_rig.translation;
  return {r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0],
          r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1],
          r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2]};
}

// `body` throws std::invalid_argument naming `what`.
template <typename Body>
void refuses(Body&& body, const std::string& what) {
  try {
    body();
  } catch (const std::invalid_argument& error) {
    const bool named = std::string(error.what()).find(what) != std::string::npos;
    VERIFY_MSG(named, "'" + what + "' not in: " + error.what());
    return;
  }
  VERIFY_MSG(false, "accepted, expected: " + what);
}

void mounting() {
  Camera c{.info = pinhole(hm::DistortionModel::kNone, {})};
  // Upright, looking forward: +x is the image centre, +z (up) above it, +y
  // (left) to the left.
  c.camera_from_rig = mount({0.0, 0.0, 0.0}, 0.0, 0.0);
  const auto ahead = project_to_pixel(c.info, to_camera(c, {5.0, 0.0, 0.0}));
  VERIFY(ahead && std::abs(ahead->x - c.info.cx) < 1e-9 && std::abs(ahead->y - c.info.cy) < 1e-9);
  const auto up = project_to_pixel(c.info, to_camera(c, {5.0, 0.0, 1.0}));
  VERIFY(up && up->y < c.info.cy && std::abs(up->x - c.info.cx) < 1e-9);
  const auto left = project_to_pixel(c.info, to_camera(c, {5.0, 1.0, 0.0}));
  VERIFY(left && left->x < c.info.cx);
  VERIFY(!project_to_pixel(c.info, to_camera(c, {-5.0, 0.0, 0.0})));
  // Turned left by 90 degrees, it looks along +y; tilted down, at the ground.
  c.camera_from_rig = mount({1.0, 2.0, 3.0}, 90.0, 0.0);
  const auto side = project_to_pixel(c.info, to_camera(c, {1.0, 7.0, 3.0}));
  VERIFY(side && std::abs(side->x - c.info.cx) < 1e-6 && std::abs(side->y - c.info.cy) < 1e-6);
  c.camera_from_rig = mount({0.0, 0.0, 2.0}, 0.0, -45.0);
  const auto ground = project_to_pixel(c.info, to_camera(c, {2.0, 0.0, 0.0}));
  VERIFY(ground && std::abs(ground->x - c.info.cx) < 1e-6 &&
         std::abs(ground->y - c.info.cy) < 1e-6);
  // Rolled clockwise by 90 degrees (seen from behind), its top points to the
  // rig's right, so up is to the image's left.
  c.camera_from_rig = mount({0.0, 0.0, 0.0}, 0.0, 0.0, 90.0);
  const auto rolled = project_to_pixel(c.info, to_camera(c, {5.0, 0.0, 1.0}));
  VERIFY(rolled && rolled->x < c.info.cx - 50.0 && std::abs(rolled->y - c.info.cy) < 1e-6);
  // The pose is a rigid transform: inverse and compose agree.
  const RigidTransform a = mount({1.0, -2.0, 0.5}, 33.0, -12.0, 4.0);
  const RigidTransform identity = compose(a, inverse(a));
  for (int i = 0; i < 9; ++i) {
    VERIFY(std::abs(identity.rotation[i] - (i % 4 == 0 ? 1.0 : 0.0)) < 1e-12);
  }
  for (const double t : identity.translation) VERIFY(std::abs(t) < 1e-12);
}

void projection_round_trips() {
  const std::vector<hm::CameraInfo> cameras{
      pinhole(hm::DistortionModel::kNone, {}),
      pinhole(hm::DistortionModel::kPlumbBob, {-0.28, 0.07, 0.001, -0.0005, -0.006}),
      pinhole(hm::DistortionModel::kRationalPolynomial,
              {0.4, -0.1, 0.0005, 0.0002, 0.01, 0.75, -0.05, 0.03}),
      pinhole(hm::DistortionModel::kEquidistant, {0.02, -0.005, 0.001, -0.0001}),
  };
  for (const hm::CameraInfo& c : cameras) {
    const std::string name(hm::distortion_model_name(c.distortion_model));
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
  const hm::CameraInfo& fisheye = cameras[3];
  const double theta = 100.0 * std::numbers::pi / 180.0;
  const auto wide = project_to_pixel(fisheye, {std::sin(theta), 0.0, std::cos(theta)});
  VERIFY(wide && wide->x > fisheye.cx + 900.0);
  // plumb_bob by its formula.
  const Point2 d = distort(cameras[1], {0.3, -0.2});
  const double r2 = 0.13;
  const double radial = 1.0 - 0.28 * r2 + 0.07 * r2 * r2 - 0.006 * r2 * r2 * r2;
  VERIFY(std::abs(d.x - (0.3 * radial + 2 * 0.001 * 0.3 * -0.2 - 0.0005 * (r2 + 2 * 0.09))) <
         1e-15);
  VERIFY(std::abs(d.y - (-0.2 * radial + 0.001 * (r2 + 2 * 0.04) - 2 * 0.0005 * 0.3 * -0.2)) <
         1e-15);
}

void valid_range() {
  VERIFY(std::abs(max_valid_angle(pinhole(hm::DistortionModel::kNone, {})) -
                  (std::numbers::pi / 2.0 - 1e-3)) < 2e-3);
  // Strong barrel distortion folds back: past that angle nothing projects.
  const hm::CameraInfo folding =
      pinhole(hm::DistortionModel::kPlumbBob, {-0.5, 0.0, 0.0, 0.0, 0.0});
  const double fold = max_valid_angle(folding);
  // d/dr (r - 0.5 r^3) = 0 at r = sqrt(2/3).
  VERIFY(std::abs(fold - std::atan(std::sqrt(2.0 / 3.0))) < 2e-3);
  const double beyond = std::tan(fold + 0.01);
  VERIFY(!project_to_pixel(folding, {beyond, 0.0, 1.0}));
  VERIFY(project_to_pixel(folding, {std::tan(fold - 0.01), 0.0, 1.0}).has_value());
  const hm::CameraInfo fisheye = pinhole(hm::DistortionModel::kEquidistant, {-0.1, 0.0, 0.0, 0.0});
  // d/dtheta (theta - 0.1 theta^3) = 0 at theta = sqrt(10/3).
  VERIFY(std::abs(max_valid_angle(fisheye) - std::sqrt(10.0 / 3.0)) < 2e-3);
  const hm::CameraInfo bad = pinhole(hm::DistortionModel::kEquidistant, {-1e6, 0.0, 0.0, 0.0});
  refuses([&] { check_camera_info(bad); }, "folds back at the optical axis");
}

// A camera_info file as ROS's camera_calibration writes one, read with the
// messages module: the stitcher takes exactly what it describes.
void camera_info_files() {
  const hm::CameraInfo info = hm::parse_camera_info_yaml(R"(image_width: 1280
image_height: 720
camera_name: front
camera_matrix:
  rows: 3
  cols: 3
  data: [330.0, 0.0, 639.5, 0.0, 331.0, 359.5, 0.0, 0.0, 1.0]
distortion_model: equidistant
distortion_coefficients:
  rows: 1
  cols: 4
  data: [0.02, -0.005, 0.001, -0.0001]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
projection_matrix:
  rows: 3
  cols: 4
  data: [330.0, 0.0, 639.5, 0.0, 0.0, 331.0, 359.5, 0.0, 0.0, 0.0, 1.0, 0.0]
)");
  const Camera camera{.info = info, .camera_from_rig = mount({2.0, 0.0, 0.8}, 0.0, -30.0)};
  validate(camera, "front");
  VERIFY(camera_label(camera, 0) == "camera 0 (front)");
  VERIFY(info.distortion_model == hm::DistortionModel::kEquidistant);
  VERIFY(info.coefficients.size() == 4 && info.coefficients[3] == -0.0001);
  VERIFY(info.fy == 331.0 && info.cx == 639.5);
  // Every lens model the stitcher takes reads back from its file to the same
  // calibration and the same projection.
  for (const hm::CameraInfo& c :
       {pinhole(hm::DistortionModel::kNone, {}),
        pinhole(hm::DistortionModel::kPlumbBob, {-0.28, 0.07, 0.001, -0.0005, -0.006}),
        pinhole(hm::DistortionModel::kRationalPolynomial,
                {0.4, -0.1, 0.0005, 0.0002, 0.01, 0.75, -0.05, 0.03}),
        pinhole(hm::DistortionModel::kEquidistant, {0.02, -0.005, 0.001, -0.0001})}) {
    const hm::CameraInfo back = hm::parse_camera_info_yaml(hm::format_camera_info_yaml(c));
    VERIFY_MSG(hm::same_calibration(back, c),
               std::string(hm::distortion_model_name(c.distortion_model)));
    check_camera_info(back);
    const auto a = project_to_pixel(c, {0.3, -0.2, 1.0});
    const auto b = project_to_pixel(back, {0.3, -0.2, 1.0});
    VERIFY(a && b && a->x == b->x && a->y == b->y);
  }
}

void refused_cameras() {
  const hm::CameraInfo good = pinhole(hm::DistortionModel::kPlumbBob, {0.1, 0.01, 0.0, 0.0, 0.0});
  check_camera_info(good);
  hm::CameraInfo skewed = good;
  skewed.skew = 0.5;
  refuses([&] { check_camera_info(skewed, "left"); }, "left: a camera with skew is not supported");
  hm::CameraInfo cropped = good;
  cropped.roi = hm::RegionOfInterest{.x = 0, .y = 0, .width = 640, .height = 360};
  refuses([&] { check_camera_info(cropped); }, "a region of interest or binning");
  hm::CameraInfo binned = good;
  binned.binning_y = 2;
  refuses([&] { check_camera_info(binned); }, "a region of interest or binning");
  hm::CameraInfo huge = good;
  huge.width = static_cast<std::uint32_t>(kMaxImageSide) + 1U;
  refuses([&] { check_camera_info(huge); }, "at most 16384");
  // What messages::validate() refuses, the stitcher does too.
  hm::CameraInfo no_focal = good;
  no_focal.fx = 0.0;
  refuses([&] { check_camera_info(no_focal); }, "fx");

  // Poses.
  Camera camera{.info = good};
  camera.camera_from_rig.rotation = {1, 0, 0, 0, 1, 0, 0, 0, 2};
  refuses([&] { validate(camera); }, "not orthonormal");
  camera.camera_from_rig.rotation = {1, 0, 0, 0, 1, 0, 0, 0, -1};
  refuses([&] { validate(camera); }, "reflection");
  camera.camera_from_rig = RigidTransform{};
  camera.camera_from_rig.translation[1] = std::numeric_limits<double>::quiet_NaN();
  refuses([&] { validate(camera); }, "must be finite");

  // Rigs.
  refuses([] { validate(CameraRig{}); }, "a rig has 1..8 cameras, not 0");
  CameraRig nine;
  nine.cameras.assign(9, Camera{.info = good});
  refuses([&] { validate(nine); }, "not 9");
  CameraRig two;
  two.cameras = {Camera{.info = good}, Camera{.info = skewed}};
  refuses([&] { validate(two); }, "camera 1 (camera): a camera with skew");
}

}  // namespace
}  // namespace holoscan::stitching

int main() {
  namespace st = holoscan::stitching;
  return st::test::run({
      {"camera_info_files", st::camera_info_files},
      {"refused_cameras", st::refused_cameras},
      {"mounting", st::mounting},
      {"projection_round_trips", st::projection_round_trips},
      {"valid_range", st::valid_range},
  });
}
