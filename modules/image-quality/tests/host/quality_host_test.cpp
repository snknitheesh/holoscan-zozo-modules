// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The CPU reference and the evaluator, without the SDK, CUDA or a GPU: every
// accuracy case, and the evaluator's own rules.

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/image_quality/quality.hpp>
#include <holoscan/image_quality/reference.hpp>
#include <holoscan/image_quality/synthetic.hpp>

#include "accuracy_cases.hpp"
#include "test_support.hpp"

namespace holoscan::image_quality::test {
namespace {

class ReferenceSource final : public StatsSource {
 public:
  explicit ReferenceSource(const StatsOptions& options) : stats_(options) {}
  FrameStats gather(const HostImage& image) override { return stats_.gather(image.view()); }

 private:
  ReferenceStats stats_;
};

std::unique_ptr<StatsSource> reference(const StatsOptions& options) {
  return std::make_unique<ReferenceSource>(options);
}

void check_validation() {
  const auto rejects = [](auto change) {
    QualityThresholds t;
    change(t);
    return throws<std::invalid_argument>([&] { validate_thresholds(t); });
  };
  VERIFY(rejects([](QualityThresholds& t) { t.blur_focus = -1; }));
  VERIFY(rejects([](QualityThresholds& t) { t.dark_level = 240; }));
  VERIFY(rejects([](QualityThresholds& t) { t.min_mean_luma = 300; }));
  VERIFY(rejects([](QualityThresholds& t) { t.max_underexposed_fraction = 2; }));
  VERIFY(rejects([](QualityThresholds& t) { t.max_noise_sigma = NAN; }));
  VERIFY(rejects([](QualityThresholds& t) { t.frozen_frames = 0; }));
  VERIFY(rejects([](QualityThresholds& t) { t.obstruction_frames = 0; }));
  VERIFY(rejects([](QualityThresholds& t) { t.obstruction_fraction = 0; }));
  VERIFY(!rejects([](QualityThresholds&) {}));
  VERIFY(throws<std::invalid_argument>([] { validate_stats_options({.tiles_x = 0}); }));
  VERIFY(throws<std::invalid_argument>([] { validate_stats_options({.flat_fraction = 0}); }));
  VERIFY(throws<std::invalid_argument>([] { (void)ReferenceStats({}).gather(ImageView{}); }));
  std::vector<std::uint8_t> pixels(12);
  VERIFY(throws<std::invalid_argument>([&] {
    (void)ReferenceStats({}).gather(ImageView{pixels.data(), 2, 2, 5, PixelFormat::kRgb8});
  }));
}

void check_names() {
  VERIFY(flag_names(0) == "ok");
  VERIFY(flag_names(kBlurred | kNoisy) == "blurred|noisy");
  VERIFY(flag_names(kAllFlags).find("color_cast") != std::string::npos);
  VERIFY(metric_name(Metric::kNoiseSigma) == "noise_sigma");
  VERIFY(metric_name(Metric::kColorCast) == "color_cast");
}

// A row pitch beyond the pixels changes nothing.
void check_pitch() {
  const HostImage packed = render_scene(97, 61, 3, PixelFormat::kRgba8);
  const std::size_t row = 97 * 4;
  const std::size_t pitch = row + 13;
  std::vector<std::uint8_t> padded(pitch * 61, 0xEE);
  for (int y = 0; y < 61; ++y) {
    std::copy_n(packed.pixels.data() + y * row, row, padded.data() + y * pitch);
  }
  ReferenceStats a({});
  ReferenceStats b({});
  VERIFY(a.gather(packed.view()) ==
         b.gather(ImageView{padded.data(), 97, 61, pitch, PixelFormat::kRgba8}));
}

// With obstruction_fraction below 1, a tile may miss a few frames of the window.
void check_obstruction_fraction() {
  QualityThresholds t;
  t.obstruction_frames = 10;
  t.obstruction_fraction = 0.8;
  QualityEvaluator evaluator(t);
  ReferenceStats stats({.tiles_x = 1, .tiles_y = 1});
  const HostImage flat{32, 32, PixelFormat::kMono8, std::vector<std::uint8_t>(32 * 32, 90)};
  // Two textured frames, then flat ones: 8 of the last 10 low-texture after 8 flat.
  for (std::uint32_t frame = 0; frame < 2; ++frame) {
    VERIFY(evaluator.evaluate(stats.gather(render_scene(32, 32, frame, PixelFormat::kMono8).view()))
               .flags == 0U);
  }
  for (int k = 0; k < 8; ++k) {
    const QualityReport r = evaluator.evaluate(stats.gather(flat.view()));
    VERIFY_MSG(((r.flags & kObstructed) != 0U) == (k == 7), "flat frame " + std::to_string(k));
    VERIFY(r.tile(0, 0, TileChannel::kObstruction) == static_cast<float>(k + 1) / 10.0F);
  }
}

}  // namespace
}  // namespace holoscan::image_quality::test

int main() {
  using namespace holoscan::image_quality::test;  // NOLINT(build/namespaces)
  std::vector<TestCase> cases = accuracy_cases("reference_", reference);
  cases.push_back({"validation", check_validation});
  cases.push_back({"names", check_names});
  cases.push_back({"pitch", check_pitch});
  cases.push_back({"obstruction_fraction", check_obstruction_fraction});
  return run(cases);
}
