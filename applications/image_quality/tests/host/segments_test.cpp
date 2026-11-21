// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's ground truth without the SDK, CUDA or a GPU: the camera's schedule
// of segments, the flags Schedule::expected_flags() requires at each segment's
// edges, and, for two whole cycles, that the module's CPU reference gives every
// rendered frame exactly those flags (the GPU gives the reference's, which the
// module's tests check).

#include "segments.hpp"

#include <cstdint>
#include <iostream>
#include <string>

#include <holoscan/image_quality/quality.hpp>
#include <holoscan/image_quality/reference.hpp>

namespace {

namespace iq = holoscan::image_quality;
namespace app = image_quality_app;

int failures = 0;

void check(bool condition, const std::string& what) {
  if (!condition) {
    ++failures;
    std::cerr << "check failed: " << what << '\n';
  }
}

void check_schedule(const app::Schedule& s) {
  check(s.segment(0) == app::Segment::kSharp, "frame 0 is sharp");
  check(s.segment(s.segment_frames - 1) == app::Segment::kSharp, "a segment's last frame");
  check(s.segment(s.segment_frames) == app::Segment::kBlurred, "the next segment's first frame");
  check(s.segment(std::uint64_t{s.segment_frames} * app::kSegmentCount) == app::Segment::kSharp,
        "the cycle repeats");
  const std::uint64_t frozen = std::uint64_t{s.segment_frames} * 4;
  check(s.segment(frozen) == app::Segment::kFrozen, "the fifth segment is frozen");
  check(s.expected_flags(frozen) == 0U, "the frozen segment's first frame is new");
  check(s.expected_flags(frozen + 1) == iq::kFrozen, "its second frame repeats it");
  const std::uint64_t smudged = std::uint64_t{s.segment_frames} * 5;
  check(s.expected_flags(smudged + s.obstruction_frames - 2) == 0U,
        "the smudge is not obstructed before the window is full");
  check(s.expected_flags(smudged + s.obstruction_frames - 1) == iq::kObstructed,
        "the smudge is obstructed once the window is full");
}

// Every frame of two cycles through the CPU reference, as the demo configures
// ImageQualityOp: default thresholds but the schedule's obstruction window.
void check_frames(const app::Schedule& s) {
  iq::QualityThresholds thresholds;
  thresholds.obstruction_frames = static_cast<int>(s.obstruction_frames);
  iq::ReferenceStats stats(iq::StatsOptions{});
  iq::QualityEvaluator evaluator(thresholds);
  const std::uint64_t frames = 2ULL * s.segment_frames * app::kSegmentCount;
  int wrong = 0;
  for (std::uint64_t frame = 0; frame < frames; ++frame) {
    const iq::QualityReport report = evaluator.evaluate(stats.gather(s.render(frame).view()));
    const std::uint32_t expected = s.expected_flags(frame);
    if (report.flags != expected && ++wrong <= 10) {
      check(false, "frame " + std::to_string(frame) + " (" +
                       std::string(app::segment_name(s.segment(frame))) + "): " +
                       iq::flag_names(report.flags) + ", expected " + iq::flag_names(expected));
    }
  }
  check(wrong == 0, std::to_string(wrong) + " frame(s) with other flags than expected");
}

}  // namespace

int main() {
  app::Schedule schedule;
  schedule.segment_frames = 12;
  schedule.obstruction_frames = 6;
  check_schedule(schedule);
  check_frames(schedule);
  std::cout << (failures == 0 ? "all passed" : std::to_string(failures) + " check(s) failed")
            << '\n';
  return failures == 0 ? 0 : 1;
}
