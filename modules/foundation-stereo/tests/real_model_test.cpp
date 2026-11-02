// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// FoundationStereoOp with a real Fast-FoundationStereo model, on the synthetic
// scene: textured planes at known depths, whose true disparity is known at
// every pixel both cameras see. The end-point error (EPE, the absolute
// disparity error in pixels) over those pixels must stay below the bounds
// below, in color and in gray, at two image sizes. Skipped (exit status 77)
// when the model file does not exist; tools/prepare_model.py writes it.
//
// Usage: foundation_stereo_real_model_test MODEL

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/foundation_stereo/foundation_stereo_op.hpp>
#include <holoscan/foundation_stereo/synthetic_scene.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "stereo_inputs.hpp"
#include "test_support.hpp"

namespace holoscan::foundation_stereo {
namespace {

namespace ht = holoscan::testing;
using test::capture_time;
using test::Memory;
using test::scene_pair;
using test::values_of;

std::filesystem::path g_model;

// What the model must reach on the synthetic scene, over the pixels both
// cameras see, set with margin over what the pinned C-Fast-FoundationStereo
// 480x736 model gives with an FP16 engine on an RTX 5090: valid 1.0, mean
// EPE 0.03 px (640x480) and 0.15 px (1280x720), median 0.02 px, 0.2 % and
// 0.9 % of the pixels more than 1 px off.
constexpr double kMinValid = 0.97;
constexpr double kMaxMeanEpe = 0.5;
constexpr double kMaxMedianEpe = 0.1;
constexpr double kMaxBad1px = 0.03;  // Share more than 1 px off.

void check_scene(holoscan::schema::ImageEncoding encoding, int width, int height) {
  const std::string what = std::string(holoscan::schema::EnumNameImageEncoding(encoding)) + " " +
                           std::to_string(width) + "x" + std::to_string(height);
  FoundationStereoConfig config;
  config.model = g_model;
  config.max_width = width;
  config.max_height = height;
  config.stats = std::make_shared<FoundationStereoStats>();
  auto h = ht::create_operator_test<FoundationStereoOp>(config);
  constexpr std::uint32_t kFrames = 3;
  for (std::uint32_t frame = 0; frame < kFrames; ++frame) {
    auto pair = scene_pair(make_scene(frame * 9U, width, height), encoding, Memory::kDevice);
    h.push_frame(ht::frame_at(capture_time(frame), ht::input("left", std::move(pair[0].image)),
                              ht::input("right", std::move(pair[1].image))));
  }
  h.input_source_device("left", holoscan::DeviceId{0});
  h.input_source_device("right", holoscan::DeviceId{0});
  h.output_device("disparity", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("disparity", kFrames);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), what + ": " + result.describe_failure());
  if (!result.ok()) return;
  const auto disparities = result.snapshots<holoscan::schema::ImageT>("disparity");
  VERIFY_MSG(disparities.size() == kFrames, what + ": every frame out");
  for (std::size_t frame = 0; frame < disparities.size(); ++frame) {
    const SyntheticScene scene = make_scene(static_cast<std::uint32_t>(frame) * 9U, width, height);
    const auto d =
        values_of<float>(disparities[frame].tensor(holoscan::schema::ImageTensorFields::data));
    std::vector<float> errors;
    std::size_t truth = 0;
    for (int v = 0; v < height; ++v) {
      for (int u = 0; u < width; ++u) {
        const float expected = true_disparity(scene, static_cast<float>(u), static_cast<float>(v));
        if (expected < 0.0F) continue;
        ++truth;
        const float got = d[static_cast<std::size_t>(v) * width + u];
        if (got > 0.0F) errors.push_back(std::abs(got - expected));
      }
    }
    const double valid = truth == 0 ? 0.0 : static_cast<double>(errors.size()) / truth;
    double mean = 0;
    for (const float e : errors) mean += e;
    mean = errors.empty() ? 0.0 : mean / static_cast<double>(errors.size());
    const auto bad = std::count_if(errors.begin(), errors.end(), [](float e) { return e > 1.0F; });
    const double bad1 = errors.empty() ? 1.0 : static_cast<double>(bad) / errors.size();
    double median = 0;
    if (!errors.empty()) {
      std::nth_element(errors.begin(), errors.begin() + errors.size() / 2, errors.end());
      median = errors[errors.size() / 2];
    }
    std::cout << std::fixed << std::setprecision(3) << "  " << what << " frame " << frame
              << ": valid=" << valid << " epe_mean=" << mean << " epe_median=" << median
              << " bad_1px=" << bad1 << '\n';
    VERIFY_MSG(valid >= kMinValid, what + ": valid share " + std::to_string(valid));
    VERIFY_MSG(mean <= kMaxMeanEpe, what + ": mean EPE " + std::to_string(mean) + " px");
    VERIFY_MSG(median <= kMaxMedianEpe, what + ": median EPE " + std::to_string(median) + " px");
    VERIFY_MSG(bad1 <= kMaxBad1px, what + ": share over 1 px " + std::to_string(bad1));
  }
  const auto timed = config.stats->timed_frames.load();
  std::cout << "  " << what << ": model " << config.stats->model_width << "x"
            << config.stats->model_height << ", gpu_ms(mean/max)="
            << (timed == 0 ? 0.0 : static_cast<double>(config.stats->gpu_ns_total) / 1e6 / timed)
            << '/' << static_cast<double>(config.stats->gpu_ns_max) / 1e6 << ", model_memory_mib="
            << static_cast<double>(config.stats->model_bytes) / (1024.0 * 1024.0) << '\n';
}

}  // namespace
}  // namespace holoscan::foundation_stereo

int main(int argc, char** argv) {
  namespace fs = holoscan::foundation_stereo;
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " MODEL\n";
    return 2;
  }
  fs::g_model = argv[1];
  if (!std::filesystem::is_regular_file(fs::g_model)) {
    std::cout << "SKIPPED: no model at " << fs::g_model.string()
              << "; tools/prepare_model.py writes it\n";
    return 77;
  }
  // The engine is built once, by the first case, and cached beside the model.
  try {
    fs::prepare_engine(fs::FoundationStereoConfig{.model = fs::g_model});
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return fs::test::run({
      {"RGB8 640x480", [] { fs::check_scene(holoscan::schema::ImageEncoding_RGB8, 640, 480); }},
      {"MONO8 1280x720", [] { fs::check_scene(holoscan::schema::ImageEncoding_MONO8, 1280, 720); }},
  });
}
