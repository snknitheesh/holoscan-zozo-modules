// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// resized_camera_info() without the SDK: values worked out by hand for a
// letterbox, a stretch and a crop; every projection agreeing with where the
// sampling puts the source pixel; distortion, skew, rectification and the
// header carried; the region of interest recording a crop; and the refusals.

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include <holoscan/color_convert/camera_info.hpp>
#include <holoscan/color_convert/conversion.hpp>
#include <holoscan/messages/camera_info.hpp>

#include "test_support.hpp"

namespace holoscan::color_convert {
namespace {

namespace hm = holoscan::messages;

bool near(double a, double b, double tolerance = 1e-9) { return std::abs(a - b) <= tolerance; }

hm::CameraInfo camera(std::uint32_t width, std::uint32_t height, double f, double cx, double cy) {
  hm::CameraInfo info;
  info.header.frame_id = "camera";
  info.width = width;
  info.height = height;
  info.state = hm::CameraState::kUndistorted;
  info.fx = f;
  info.fy = f;
  info.cx = cx;
  info.cy = cy;
  return info;
}

// The worked example: a 1920x1080 camera, f = 1000, principal point at the
// centre, letterboxed into 640x640. The content is 640x360 at y = 140, a
// third of the size: fx' = 1000/3, cx' = (960.5)/3 - 0.5 = 319.6667,
// cy' = (540.5)/3 - 0.5 + 140 = 319.6667.
void check_letterbox() {
  const hm::CameraInfo info = camera(1920, 1080, 1000.0, 960.0, 540.0);
  const hm::CameraInfo out = resized_camera_info(info, 1920, 1080, 640, 640, PixelFormat::kRgb8,
                                                 ConversionParams{.letterbox = true});
  VERIFY(out.width == 640 && out.height == 640);
  VERIFY(near(out.fx, 1000.0 / 3.0) && near(out.fy, 1000.0 / 3.0));
  VERIFY(near(out.cx, 960.5 / 3.0 - 0.5));
  VERIFY(near(out.cy, 540.5 / 3.0 - 0.5 + 140.0));
  VERIFY(near(out.cx, 319.0 + 2.0 / 3.0) && near(out.cy, 319.0 + 2.0 / 3.0));
  VERIFY(out.state == hm::CameraState::kUndistorted && out.header.frame_id == "camera");
  VERIFY(!out.roi && out.binning_x == 1 && out.binning_y == 1);
}

// A stretch scales each axis by its own ratio.
void check_stretch() {
  const hm::CameraInfo info = camera(1280, 720, 900.0, 640.0, 360.0);
  const hm::CameraInfo out =
      resized_camera_info(info, 1280, 720, 320, 360, PixelFormat::kRgb8, ConversionParams{});
  VERIFY(near(out.fx, 900.0 / 4.0) && near(out.fy, 900.0 / 2.0));
  VERIFY(near(out.cx, 640.5 / 4.0 - 0.5) && near(out.cy, 360.5 / 2.0 - 0.5));
}

// Where the sampling reads output pixel (u, v) from, as the kernel computes
// it (kernel_math.hpp): the source coordinate under the pixel's centre.
struct Point {
  double x;
  double y;
};
Point sampled_from(const Mapping& m, double u, double v) {
  const double scale_x = static_cast<double>(m.source.width) / m.content.width;
  const double scale_y = static_cast<double>(m.source.height) / m.content.height;
  return {m.source.x + (u - m.content.x + 0.5) * scale_x - 0.5,
          m.source.y + (v - m.content.y + 0.5) * scale_y - 0.5};
}

// For crops, letterboxes, stretches and subsampled outputs: a point projected
// by the resized camera lands on the output pixel that samples the source
// pixel where the input camera projects it.
void check_projections() {
  hm::CameraInfo info = camera(1920, 1080, 1100.0, 955.25, 541.5);
  info.fy = 1090.0;
  info.skew = 0.75;
  struct Case {
    int width;
    int height;
    PixelFormat format;
    ConversionParams params;
  };
  const Case cases[] = {
      {640, 640, PixelFormat::kRgb8, ConversionParams{.letterbox = true}},
      {640, 640, PixelFormat::kNv12, ConversionParams{.letterbox = true}},
      {333, 201, PixelFormat::kRgb8, ConversionParams{}},
      {300, 300, PixelFormat::kI420,
       ConversionParams{.crop = Rect{400, 100, 1001, 700}, .letterbox = true}},
      {3840, 2160, PixelFormat::kRgba8, ConversionParams{.crop = Rect{0, 0, 1920, 1080}}},
  };
  for (const Case& c : cases) {
    const hm::CameraInfo out =
        resized_camera_info(info, 1920, 1080, c.width, c.height, c.format, c.params);
    const Mapping m = compute_mapping(1920, 1080, c.width, c.height, c.format, c.params);
    for (const auto& p : {Point{0.1, -0.2}, Point{-0.3, 0.25}, Point{0.0, 0.0}}) {
      const double z = 2.0;
      const double u = info.fx * p.x / z + info.skew * p.y / z + info.cx;
      const double v = info.fy * p.y / z + info.cy;
      const double u2 = out.fx * p.x / z + out.skew * p.y / z + out.cx;
      const double v2 = out.fy * p.y / z + out.cy;
      const Point back = sampled_from(m, u2, v2);
      VERIFY_MSG(near(back.x, u, 1e-6) && near(back.y, v, 1e-6),
                 std::to_string(c.width) + "x" + std::to_string(c.height) +
                     ": the projection lands where its source pixel is sampled");
    }
  }
}

// Distortion acts before the intrinsics: the model and coefficients stay, and
// so do the state and the rectification.
void check_carried() {
  hm::CameraInfo info = camera(848, 480, 420.0, 424.0, 240.0);
  info.state = hm::CameraState::kRaw;
  info.distortion_model = hm::DistortionModel::kPlumbBob;
  info.coefficients = {-0.05, 0.01, 0.001, -0.002, 0.0};
  info.rectification = hm::kIdentity3;
  info.rectification->at(0) = std::cos(0.01);
  info.rectification->at(1) = -std::sin(0.01);
  info.rectification->at(3) = std::sin(0.01);
  info.rectification->at(4) = std::cos(0.01);
  info.header.capture_timestamp_ns = 42;
  const hm::CameraInfo out =
      resized_camera_info(info, 848, 480, 424, 240, PixelFormat::kRgb8, ConversionParams{});
  VERIFY(out.state == hm::CameraState::kRaw);
  VERIFY(out.distortion_model == hm::DistortionModel::kPlumbBob);
  VERIFY(out.coefficients == info.coefficients && out.rectification == info.rectification);
  VERIFY(out.header == info.header);
}

// A crop narrows the region of interest, in sensor pixels.
void check_region_of_interest() {
  // No region of interest, binning 2: the crop's pixels are two sensor pixels.
  hm::CameraInfo binned = camera(960, 540, 500.0, 480.0, 270.0);
  binned.binning_x = 2;
  binned.binning_y = 2;
  hm::CameraInfo out = resized_camera_info(binned, 960, 540, 200, 100, PixelFormat::kRgb8,
                                           ConversionParams{.crop = Rect{100, 50, 400, 200}});
  VERIFY(out.roi && *out.roi == (hm::RegionOfInterest{200, 100, 800, 400}));
  VERIFY(out.binning_x == 2 && out.binning_y == 2);
  // A region of interest of 1280x720 sensor pixels read into 640x360 images,
  // at (100, 60): a crop at (10, 20) of 320x180 is 640x360 sensor pixels at
  // (120, 100).
  hm::CameraInfo roi = camera(640, 360, 400.0, 320.0, 180.0);
  roi.roi = hm::RegionOfInterest{100, 60, 1280, 720};
  out = resized_camera_info(roi, 640, 360, 320, 180, PixelFormat::kRgb8,
                            ConversionParams{.crop = Rect{10, 20, 320, 180}});
  VERIFY(out.roi && *out.roi == (hm::RegionOfInterest{120, 100, 640, 360}));
  // Without a crop the region of interest stays.
  out = resized_camera_info(roi, 640, 360, 320, 180, PixelFormat::kRgb8, ConversionParams{});
  VERIFY(out.roi && *out.roi == *roi.roi);
}

void check_refusals() {
  const hm::CameraInfo info = camera(640, 480, 500.0, 320.0, 240.0);
  VERIFY(camera_info_problem(info, 640, 480) == std::nullopt);
  const auto problem = camera_info_problem(info, 1280, 720);
  VERIFY(problem && problem->find("640x480") != std::string::npos &&
         problem->find("1280x720") != std::string::npos);
  VERIFY(test::throws<std::invalid_argument>([&] {
    (void)resized_camera_info(info, 1280, 720, 64, 64, PixelFormat::kRgb8, ConversionParams{});
  }));
  VERIFY(test::throws<std::invalid_argument>([&] {
    (void)resized_camera_info(info, 640, 480, 64, 64, PixelFormat::kRgb8,
                              ConversionParams{.crop = Rect{600, 0, 100, 100}});
  }));
}

}  // namespace
}  // namespace holoscan::color_convert

int main() {
  namespace cc = holoscan::color_convert;
  return cc::test::run({
      {"a letterbox's camera, worked by hand", cc::check_letterbox},
      {"a stretch's camera", cc::check_stretch},
      {"projections land where the sampling reads", cc::check_projections},
      {"distortion, rectification and header carried", cc::check_carried},
      {"a crop narrows the region of interest", cc::check_region_of_interest},
      {"refusals", cc::check_refusals},
  });
}
