// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The rules every backend's parameters and sizes follow, what the auto backend
// chooses for each platform, and the geometry: no SDK, CUDA, VPI or GPU.

#include <cmath>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "holoscan/stereo_disparity/disparity_options.hpp"
#include "holoscan/stereo_disparity/stereo_geometry.hpp"
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
  // The defaults run everywhere, also with a confidence map except on ofa.
  for (Backend backend : {Backend::kCuda, Backend::kOfa, Backend::kPva, Backend::kAuto}) {
    VERIFY_MSG(fine(backend, defaults), std::string(backend_name(backend)) + " takes the defaults");
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
  params = {};
  params.window = 9;
  VERIFY(says(params_problem(Backend::kPva, params, false), "1, 3, 5 or 7"));
  VERIFY(fine(Backend::kCuda, params));  // cuda ignores the window.
  params.window = 0;
  VERIFY(says(params_problem(Backend::kCuda, params, false), "window"));
}

void check_sizes() {
  const DisparityParams params;
  VERIFY(!size_problem(Backend::kCuda, params, 1, 1).has_value());
  VERIFY(!size_problem(Backend::kCuda, params, 16384, 16384).has_value());
  VERIFY(size_problem(Backend::kCuda, params, 16385, 10).has_value());
  VERIFY(size_problem(Backend::kCuda, params, 0, 10).has_value());
  VERIFY(!size_problem(Backend::kOfa, params, 16, 16).has_value());
  VERIFY(says(size_problem(Backend::kOfa, params, 15, 100), "too small for ofa"));
  VERIFY(says(size_problem(Backend::kPva, params, 100, 100), "at least 128"));
  VERIFY(!size_problem(Backend::kPva, params, 128, 16).has_value());
  DisparityParams wide;
  wide.max_disparity = 256;
  VERIFY(says(size_problem(Backend::kPva, wide, 200, 100), "at least 256"));
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
}

void check_geometry() {
  const StereoGeometry good{.fx = 337.0, .baseline = 0.095, .width = 1280, .height = 720};
  validate_geometry(good);
  StereoGeometry with_point = good;
  with_point.cx = 640.0;
  with_point.cy = 360.0;
  with_point.doffs = 2.5;
  validate_geometry(with_point);
  using Change = void (*)(StereoGeometry&);
  for (Change bad : std::initializer_list<Change>{
           [](StereoGeometry& g) { g.fx = 0; }, [](StereoGeometry& g) { g.fx = NAN; },
           [](StereoGeometry& g) { g.baseline = -0.1; }, [](StereoGeometry& g) { g.width = 0; },
           [](StereoGeometry& g) { g.height = 16385; }, [](StereoGeometry& g) { g.cx = INFINITY; },
           [](StereoGeometry& g) { g.doffs = NAN; }}) {
    StereoGeometry geometry = good;
    bad(geometry);
    VERIFY(throws<std::invalid_argument>([&] { validate_geometry(geometry); }));
  }
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
      {"available backends", check_available_backends},
      {"auto backend", check_auto},
      {"geometry", check_geometry},
      {"depth from disparity", check_depth},
  });
}
