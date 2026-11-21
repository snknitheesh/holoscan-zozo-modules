// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The rules every backend's parameters and sizes follow, what the auto backend
// chooses for each platform, what the operator needs of its camera info and
// the camera info it sends with the depth, and the depth formula: no SDK,
// CUDA, VPI or GPU.

#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "holoscan/messages/synthetic_calibration.hpp"
#include "holoscan/stereo_disparity/camera_info.hpp"
#include "holoscan/stereo_disparity/depth.hpp"
#include "holoscan/stereo_disparity/disparity_options.hpp"
#include "test_support.hpp"

namespace holoscan::stereo_disparity {
namespace {

using test::throws;

bool fine(Backend backend, const DisparityParams& params, bool confidence = false) {
  return !params_problem(backend, params, confidence).has_value();
}

// The problem's text contains `part`.
bool says(const std::optional<std::string>& problem, const std::string& part) {
  return problem && problem->find(part) != std::string::npos;
}

void check_names() {
  for (Backend backend : {Backend::kCuda, Backend::kOfa, Backend::kPva, Backend::kAuto}) {
    VERIFY(parse_backend(backend_name(backend)) == backend);
  }
  VERIFY(backend_name(Backend::kOfa) == "ofa");
  VERIFY(!parse_backend("cpu").has_value());
  VERIFY(!parse_backend("CUDA").has_value());
  VERIFY(!parse_backend("").has_value());
}

void check_defaults() {
  const DisparityParams defaults;
  VERIFY(defaults.max_disparity == 128);
  VERIFY(defaults.downscale == 1);
  // The passes: 1 on Orin and 3 on Thor unless set; x86_64 has no OFA.
  VERIFY(!defaults.num_passes.has_value());
  const PlatformFacts orin{.has_ofa = true, .has_vic = true, .has_pva = true};
  const PlatformFacts thor{.has_ofa = true, .has_vic = true, .has_pva = true, .thor = true};
  VERIFY(num_passes_for(defaults, orin) == 1);
  VERIFY(num_passes_for(defaults, thor) == 3);
  DisparityParams passes;
  passes.num_passes = 2;
  VERIFY(num_passes_for(passes, orin) == 2);
  passes.num_passes = 3;
  VERIFY(num_passes_for(passes, orin) == 3);
  // The diagonal paths: off on cuda and on on ofa unless set.
  VERIFY(!defaults.include_diagonals.has_value());
  VERIFY(!include_diagonals_for(Backend::kCuda, defaults));
  VERIFY(include_diagonals_for(Backend::kOfa, defaults));
  DisparityParams set;
  set.include_diagonals = true;
  VERIFY(include_diagonals_for(Backend::kCuda, set));
  set.include_diagonals = false;
  VERIFY(!include_diagonals_for(Backend::kOfa, set));
  // The defaults run everywhere, also with a confidence map except on ofa.
  for (Backend backend : {Backend::kCuda, Backend::kOfa, Backend::kPva, Backend::kAuto}) {
    VERIFY_MSG(fine(backend, defaults), std::string(backend_name(backend)) + " takes the defaults");
    VERIFY(!platform_problem(backend, defaults, orin).has_value());
    VERIFY(!platform_problem(backend, defaults, thor).has_value());
  }
  VERIFY(fine(Backend::kCuda, defaults, true));
  VERIFY(fine(Backend::kPva, defaults, true));
  VERIFY(says(params_problem(Backend::kOfa, defaults, true), "no confidence"));
}

void check_max_disparity() {
  DisparityParams params;
  for (int value : {1, 64, 65, 200, 256}) {
    params.max_disparity = value;
    VERIFY(fine(Backend::kCuda, params));
    VERIFY(fine(Backend::kAuto, params));
  }
  for (int value : {0, -1, 257}) {
    params.max_disparity = value;
    VERIFY(says(params_problem(Backend::kCuda, params, false), "1..256"));
  }
  for (Backend backend : {Backend::kOfa, Backend::kPva}) {
    for (int value : {128, 256}) {
      params.max_disparity = value;
      VERIFY(fine(backend, params));
    }
    for (int value : {64, 100, 255}) {
      params.max_disparity = value;
      VERIFY(says(params_problem(backend, params, false), "128 or 256"));
    }
  }
  params.max_disparity = 1;
  VERIFY(searched_disparity(Backend::kCuda, params) == 64);
  params.max_disparity = 65;
  VERIFY(searched_disparity(Backend::kCuda, params) == 128);
  params.max_disparity = 129;
  VERIFY(searched_disparity(Backend::kCuda, params) == 256);
  params.max_disparity = 256;
  VERIFY(searched_disparity(Backend::kOfa, params) == 256);
  params.max_disparity = 128;
  VERIFY(searched_disparity(Backend::kPva, params) == 128);
}

void check_penalties_and_the_rest() {
  DisparityParams params;
  params.p1 = 0;
  VERIFY(says(params_problem(Backend::kCuda, params, false), "1 <= p1 <= p2"));
  params.p1 = 10;
  params.p2 = 9;
  VERIFY(says(params_problem(Backend::kCuda, params, false), "1 <= p1 <= p2"));
  params.p1 = 3;
  params.p2 = 255;
  VERIFY(fine(Backend::kCuda, params));
  VERIFY(says(params_problem(Backend::kOfa, params, false), "at most 217"));
  params.p2 = 256;
  VERIFY(says(params_problem(Backend::kCuda, params, false), "at most 255"));
  params.p2 = 214;
  VERIFY(fine(Backend::kOfa, params));
  VERIFY(fine(Backend::kPva, params));
  params.confidence_type = ConfidenceType::kInference;
  VERIFY(says(params_problem(Backend::kPva, params, false), "at most 89"));
  params.p2 = 86;
  VERIFY(fine(Backend::kPva, params));
  VERIFY(says(params_problem(Backend::kCuda, params, false), "pva only"));
  VERIFY(says(params_problem(Backend::kOfa, params, false), "pva only"));

  params = {};
  params.confidence_threshold = 65536;
  VERIFY(says(params_problem(Backend::kCuda, params, false), "0..65535"));
  params = {};
  for (float value : {-1.0F, 0.0F, 0.5F, 1.0F}) {
    params.uniqueness = value;
    VERIFY(fine(Backend::kCuda, params));
  }
  for (float value : {-0.5F, 1.5F, std::numeric_limits<float>::quiet_NaN()}) {
    params.uniqueness = value;
    VERIFY(says(params_problem(Backend::kCuda, params, false), "uniqueness"));
  }
  params = {};
  params.p2_alpha = 3;
  VERIFY(says(params_problem(Backend::kOfa, params, false), "p2_alpha"));
  params = {};
  params.num_passes = 4;
  VERIFY(says(params_problem(Backend::kOfa, params, false), "num_passes"));
  params.num_passes = 0;
  VERIFY(says(params_problem(Backend::kPva, params, false), "num_passes"));
  params.num_passes = 1;
  VERIFY(fine(Backend::kOfa, params));
  params = {};
  params.window = 9;
  VERIFY(says(params_problem(Backend::kPva, params, false), "1, 3, 5 or 7"));
  VERIFY(fine(Backend::kCuda, params));  // cuda ignores the window.
  params.window = 0;
  VERIFY(says(params_problem(Backend::kCuda, params, false), "window"));
}

bool size_fine(Backend backend, const DisparityParams& params, int width, int height,
               bool confidence = false) {
  return !size_problem(backend, params, confidence, width, height).has_value();
}

void check_sizes() {
  const DisparityParams params;
  VERIFY(size_fine(Backend::kCuda, params, 1, 1));
  VERIFY(size_fine(Backend::kCuda, params, 16384, 16384));
  VERIFY(!size_fine(Backend::kCuda, params, 16385, 10));
  VERIFY(!size_fine(Backend::kCuda, params, 0, 10));
  VERIFY(size_fine(Backend::kOfa, params, 16, 16));
  VERIFY(says(size_problem(Backend::kOfa, params, false, 15, 100), "too small for ofa"));
  VERIFY(says(size_problem(Backend::kPva, params, false, 100, 100), "at least 128"));
  VERIFY(size_fine(Backend::kPva, params, 128, 16));
  DisparityParams wide;
  wide.max_disparity = 256;
  VERIFY(says(size_problem(Backend::kPva, wide, false, 200, 100), "at least 256"));
}

void check_downscale() {
  VERIFY(output_side(1280, 1) == 1280);
  VERIFY(output_side(1280, 2) == 640);
  VERIFY(output_side(1281, 2) == 641);
  VERIFY(output_side(1283, 4) == 321);
  VERIFY(output_side(720, 8) == 90);
  DisparityParams params;
  for (int factor : {1, 2, 4, 8}) {
    params.downscale = factor;
    VERIFY(fine(Backend::kOfa, params));
    VERIFY(fine(Backend::kPva, params));
  }
  for (int factor : {0, 3, 16, -2}) {
    params.downscale = factor;
    VERIFY(says(params_problem(Backend::kOfa, params, false), "1, 2, 4 or 8"));
  }
  params.downscale = 2;
  VERIFY(says(params_problem(Backend::kCuda, params, false), "ofa or pva"));
  VERIFY(says(params_problem(Backend::kAuto, params, false), "not auto"));
  params.downscale = 1;
  VERIFY(fine(Backend::kCuda, params));
  // ofa: sides of at least 16 once downscaled, rounded up.
  params.downscale = 8;
  VERIFY(size_fine(Backend::kOfa, params, 121, 121));
  VERIFY(says(size_problem(Backend::kOfa, params, false, 120, 640), "at least 16 once downscaled"));
  // pva: a width of at least max(64, max_disparity / downscale) * downscale.
  params.downscale = 2;
  VERIFY(size_fine(Backend::kPva, params, 128, 32));
  VERIFY(says(size_problem(Backend::kPva, params, false, 127, 32), "at least 128"));
  params.downscale = 4;
  VERIFY(size_fine(Backend::kPva, params, 256, 64));
  VERIFY(says(size_problem(Backend::kPva, params, false, 255, 64), "at least 256"));
  params.max_disparity = 256;
  params.downscale = 8;
  VERIFY(size_fine(Backend::kPva, params, 512, 128));
  VERIFY(says(size_problem(Backend::kPva, params, false, 511, 128), "at least 512"));
}

// pva, max_disparity 256, full size and a confidence map: VPI 4.1.4 fails a
// width whose remainder modulo 256 is 1..127 (measured at every width from 256
// to 1024 on Jetson Orin, and in steps of 16 to 2048).
void check_pva_padding() {
  DisparityParams params;
  params.max_disparity = 256;
  for (int width = 256; width <= 2048; ++width) {
    const int remainder = width % 256;
    const bool refused = remainder >= 1 && remainder <= 127;
    VERIFY_MSG(size_fine(Backend::kPva, params, width, 480, true) == !refused,
               std::to_string(width) + (refused ? " is refused" : " is allowed"));
  }
  VERIFY(says(size_problem(Backend::kPva, params, true, 848, 480),
              "remainder modulo 256 is 0 or at least 128"));
  // Each other setting runs at 848 wide, as measured.
  VERIFY(size_fine(Backend::kPva, params, 848, 480, false));
  params.downscale = 2;
  VERIFY(size_fine(Backend::kPva, params, 848, 480, true));
  params = {};
  VERIFY(size_fine(Backend::kPva, params, 848, 480, true));
  params.max_disparity = 256;
  VERIFY(size_fine(Backend::kOfa, params, 848, 480, false));
  VERIFY(size_fine(Backend::kCuda, params, 848, 480, true));
}

void check_platform() {
  const PlatformFacts orin{.has_ofa = true, .has_vic = true, .has_pva = true};
  const PlatformFacts thor{.has_ofa = true, .has_vic = true, .has_pva = true, .thor = true};
  DisparityParams params;
  for (Backend backend : {Backend::kCuda, Backend::kOfa, Backend::kPva}) {
    VERIFY(!platform_problem(backend, params, orin).has_value());
    VERIFY(!platform_problem(backend, params, thor).has_value());
  }
  for (int passes : {1, 2}) {
    params.num_passes = passes;
    VERIFY(!platform_problem(Backend::kOfa, params, orin).has_value());
    VERIFY(!platform_problem(Backend::kPva, params, orin).has_value());
    VERIFY(says(platform_problem(Backend::kOfa, params, thor), "Jetson Thor"));
    VERIFY(says(platform_problem(Backend::kPva, params, thor),
                "num_passes " + std::to_string(passes) + " on pva"));
    VERIFY(says(platform_problem(Backend::kPva, params, thor), "use 3"));
    // cuda reads no passes.
    VERIFY(!platform_problem(Backend::kCuda, params, thor).has_value());
  }
  params.num_passes = 3;
  VERIFY(!platform_problem(Backend::kOfa, params, thor).has_value());
  params.downscale = 8;
  VERIFY(!platform_problem(Backend::kOfa, params, orin).has_value());
  VERIFY(says(platform_problem(Backend::kOfa, params, thor), "refuses downscale 8"));
  VERIFY(says(platform_problem(Backend::kPva, params, thor), "use 1, 2 or 4"));
  params.downscale = 4;
  VERIFY(!platform_problem(Backend::kPva, params, thor).has_value());
}

void check_available_backends() {
  using Backends = std::vector<Backend>;
  VERIFY(available_backends(PlatformFacts{}) == Backends{Backend::kCuda});
  VERIFY(available_backends({.has_ofa = true}) == Backends{Backend::kCuda});
  VERIFY(available_backends({.has_ofa = true, .has_vic = true}) ==
         (Backends{Backend::kCuda, Backend::kOfa}));
  VERIFY(available_backends({.has_ofa = true, .has_vic = true, .has_pva = true}) ==
         (Backends{Backend::kCuda, Backend::kOfa, Backend::kPva}));
  VERIFY(available_backends({.has_vic = true, .has_pva = true}) == Backends{Backend::kCuda});
}

void check_auto() {
  const PlatformFacts jetson{.has_ofa = true, .has_vic = true, .has_pva = true};
  const DisparityParams defaults;
  // x86_64: no OFA, so cuda.
  AutoChoice choice = resolve_auto(PlatformFacts{}, defaults, false, 1280, 720);
  VERIFY(choice.backend == Backend::kCuda);
  VERIFY(choice.reason.find("no Optical Flow Accelerator") != std::string::npos);
  // Jetson: ofa when everything suits it.
  choice = resolve_auto(jetson, defaults, false, 1280, 720);
  VERIFY(choice.backend == Backend::kOfa);
  // ... but cuda for a confidence map, a max_disparity the OFA does not take,
  // penalties beyond its range, or an image too small for it.
  choice = resolve_auto(jetson, defaults, true, 1280, 720);
  VERIFY(choice.backend == Backend::kCuda);
  VERIFY(choice.reason.find("no confidence output") != std::string::npos);
  DisparityParams narrow;
  narrow.max_disparity = 64;
  VERIFY(resolve_auto(jetson, narrow, false, 1280, 720).backend == Backend::kCuda);
  DisparityParams steep;
  steep.p2 = 250;
  VERIFY(resolve_auto(jetson, steep, false, 1280, 720).backend == Backend::kCuda);
  choice = resolve_auto(jetson, defaults, false, 8, 720);
  VERIFY(choice.backend == Backend::kCuda);
  VERIFY(choice.reason.find("too small") != std::string::npos);
  // No VIC to convert with: no ofa.
  VERIFY(resolve_auto({.has_ofa = true}, defaults, false, 1280, 720).backend == Backend::kCuda);
  // One or two passes: ofa on Orin, cuda on Thor.
  PlatformFacts thor = jetson;
  thor.thor = true;
  for (int passes : {1, 2}) {
    DisparityParams fewer;
    fewer.num_passes = passes;
    VERIFY(resolve_auto(jetson, fewer, false, 1280, 720).backend == Backend::kOfa);
    choice = resolve_auto(thor, fewer, false, 1280, 720);
    VERIFY(choice.backend == Backend::kCuda);
    VERIFY(choice.reason.find("Jetson Thor") != std::string::npos);
  }
  VERIFY(resolve_auto(thor, defaults, false, 1280, 720).backend == Backend::kOfa);
}

// A rectified pair of `width` x `height` images, as a rectifier sends it.
messages::StereoCameraInfo rectified_pair(std::uint32_t width, std::uint32_t height) {
  messages::CameraInfo left;
  left.header.frame_id = "left";
  left.width = width;
  left.height = height;
  left.state = messages::CameraState::kRectified;
  left.fx = 337.0;
  left.fy = 337.0;
  left.cx = 320.25;
  left.cy = 240.75;
  messages::CameraInfo right = left;
  right.header.frame_id = "right";
  right.cx = 322.75;
  messages::StereoCameraInfo pair;
  pair.header.frame_id = "left";
  pair.left = left;
  pair.right = right;
  pair.right_from_left.translation = {-0.095, 0.0, 0.0};
  pair.baseline = 0.095;
  pair.doffs = 2.5;
  messages::validate(pair);
  return pair;
}

void check_camera_info_problem() {
  const messages::StereoCameraInfo pair = rectified_pair(641, 480);
  VERIFY(!camera_info_problem(pair, 641, 480).has_value());
  VERIFY(says(camera_info_problem(pair, 640, 480), "is for 641x480 (left) and 641x480 (right)"));
  VERIFY(says(camera_info_problem(pair, 641, 481), "the images are 641x481"));
  messages::StereoCameraInfo right_differs = pair;
  right_differs.right.width = 640;
  VERIFY(says(camera_info_problem(right_differs, 641, 480), "640x480 (right)"));
  const messages::StereoCameraInfo raw =
      messages::synthetic_stereo_camera_info(messages::DistortionModel::kPlumbBob, 641, 480);
  VERIFY(says(camera_info_problem(raw, 641, 480), "describes raw cameras"));
  VERIFY(says(camera_info_problem(raw, 641, 480), "must be a rectified pair"));
}

void check_depth_camera_info() {
  messages::StereoCameraInfo pair = rectified_pair(641, 480);
  pair.left.roi = messages::RegionOfInterest{.x = 4, .y = 2, .width = 630, .height = 470};
  pair.right.roi = pair.left.roi;
  const messages::CameraInfo full = depth_camera_info(pair, 1);
  messages::CameraInfo expected = pair.left;
  VERIFY(full == expected);
  for (int factor : {2, 4, 8}) {
    const std::string what = "downscale " + std::to_string(factor);
    const messages::CameraInfo depth = depth_camera_info(pair, factor);
    messages::validate(depth);
    VERIFY_MSG(depth.width == static_cast<std::uint32_t>((641 + factor - 1) / factor) &&
                   depth.height == static_cast<std::uint32_t>(480 / factor),
               what + ": the output's size, rounded up");
    VERIFY_MSG(depth.binning_x == static_cast<std::uint32_t>(factor) &&
                   depth.binning_y == static_cast<std::uint32_t>(factor) &&
                   depth.roi == pair.left.roi,
               what + ": binning by the factor, the same region");
    VERIFY_MSG(depth.state == messages::CameraState::kRectified &&
                   depth.header.frame_id == "left" && depth.header.capture_timestamp_ns == 0,
               what + ": the rectified left camera, not yet stamped");
    // A point projects to the output pixel whose input pixels it falls on:
    // u_out = (u_in - (factor - 1) / 2) / factor, for the input's projection.
    for (const auto& [x, y, z] :
         {std::array<double, 3>{0.3, -0.2, 1.5}, std::array<double, 3>{-1.0, 0.5, 4.0}}) {
      const double u_in = pair.left.fx * x / z + pair.left.cx;
      const double v_in = pair.left.fy * y / z + pair.left.cy;
      const double u_out = depth.fx * x / z + depth.cx;
      const double v_out = depth.fy * y / z + depth.cy;
      VERIFY_MSG(std::abs(u_out - (u_in - (factor - 1) / 2.0) / factor) < 1e-9 &&
                     std::abs(v_out - (v_in - (factor - 1) / 2.0) / factor) < 1e-9,
                 what + ": a point lands on the output pixel covering its input pixel");
    }
  }
  VERIFY(throws<std::invalid_argument>([&] { (void)depth_camera_info(pair, 3); }));
  const messages::StereoCameraInfo raw =
      messages::synthetic_stereo_camera_info(messages::DistortionModel::kPlumbBob, 641, 480);
  VERIFY(throws<std::invalid_argument>([&] { (void)depth_camera_info(raw, 1); }));
}

void check_depth() {
  // 337 px * 0.095 m = 32.015 px m.
  constexpr float kFb = 32.015F;
  VERIFY(depth_from_disparity(32.015F, kFb, 0.0F) == 1.0F);
  VERIFY(std::abs(depth_from_disparity(10.0F, kFb, 0.0F) - 3.2015F) < 1e-6F);
  VERIFY(std::abs(depth_from_disparity(10.0F, kFb, 2.0F) - kFb / 12.0F) < 1e-6F);
  VERIFY(depth_from_disparity(0.0F, kFb, 0.0F) == 0.0F);
  VERIFY(depth_from_disparity(-1.0F, kFb, 0.0F) == 0.0F);
  VERIFY(depth_from_disparity(NAN, kFb, 0.0F) == 0.0F);
  // A disparity no larger than -doffs is beyond infinity.
  VERIFY(depth_from_disparity(3.0F, kFb, -3.0F) == 0.0F);
  VERIFY(depth_from_disparity(3.0F, kFb, -2.0F) == kFb);
}

}  // namespace
}  // namespace holoscan::stereo_disparity

int main() {
  using namespace holoscan::stereo_disparity;  // NOLINT(build/namespaces)
  return test::run({
      {"backend names", check_names},
      {"defaults", check_defaults},
      {"max_disparity", check_max_disparity},
      {"penalties and the other parameters", check_penalties_and_the_rest},
      {"image sizes", check_sizes},
      {"downscale", check_downscale},
      {"pva padding", check_pva_padding},
      {"platform", check_platform},
      {"available backends", check_available_backends},
      {"auto backend", check_auto},
      {"camera info problems", check_camera_info_problem},
      {"depth camera info", check_depth_camera_info},
      {"depth from disparity", check_depth},
  });
}
