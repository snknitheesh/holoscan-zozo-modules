// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// DetectionReportOp through the SDK's operator test harness: it counts what it
// receives and checks synthetic frames against their ground truth.

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/apriltag/apriltag_detections_schema_traits.hpp>
#include <holoscan/apriltag/synthetic_scene.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "detection_report_op.hpp"
#include "test_support.hpp"

namespace apriltag_app {
namespace {

namespace at = holoscan::apriltag;

constexpr int kWidth = 640;
constexpr int kHeight = 480;

// One tag, id 7, in a 200x100 image; with a pose when `pose` is set.
at::schema::AprilTagDetectionsT one_tag_message(std::uint64_t frame, bool pose) {
  at::schema::AprilTagDetectionsT message;
  message.frame = frame;
  message.image_width = 200;
  message.image_height = 100;
  message.backend = "cuda";
  message.family = "tag36h11";
  message.detect_time_us = 1500.0F;
  const std::array<at::schema::Point2f, 4> corners{
      at::schema::Point2f(20, 80), at::schema::Point2f(80, 80), at::schema::Point2f(80, 20),
      at::schema::Point2f(20, 20)};
  std::array<float, 9> rotation{};
  std::array<float, 3> translation{};
  if (pose) {
    rotation = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    translation = {0.25F, -0.5F, 2.0F};
  }
  message.detections.push_back(
      at::schema::TagDetection(7, 0, 50.0F, at::schema::Point2f(50, 50),
                               ::flatbuffers::span<const at::schema::Point2f, 4>(corners), pose,
                               ::flatbuffers::span<const float, 9>(rotation),
                               ::flatbuffers::span<const float, 3>(translation), 0.0F));
  return message;
}

// The reporter counts frames, tags and latencies; without ground truth it
// passes whatever it saw.
void check_report_counts() {
  auto report = std::make_shared<BackendReport>();
  auto h = holoscan::testing::create_operator_test<DetectionReportOp>(
      DetectionReportConfig{.report = report, .remaining = std::make_shared<std::atomic<int>>()});
  h.push("detections", one_tag_message(0, false));
  h.push("detections", one_tag_message(1, true));
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  VERIFY(report->frames == 2 && report->tags == 2 && report->detect_ms.size() == 2);
  VERIFY(report->detect_ms.front() == 1.5F);
  VERIFY(!report->verified && report->passed());
}

// A perfect detection message for synthetic frame `frame`, numbered as the
// source numbers it.
at::schema::AprilTagDetectionsT truth_message(std::uint32_t frame) {
  const at::SyntheticScene scene = at::make_scene("tag36h11", frame, kWidth, kHeight);
  at::schema::AprilTagDetectionsT message;
  message.device_sequence = std::uint64_t{frame} + 1U;
  message.image_width = kWidth;
  message.image_height = kHeight;
  message.backend = "cpu";
  message.family = "tag36h11";
  for (const at::SceneTag& tag : scene.tags) {
    const auto projected = at::project_corners(tag, scene.intrinsics, scene.tag_size);
    std::array<at::schema::Point2f, 4> corners;
    float cx = 0.0F;
    float cy = 0.0F;
    for (int c = 0; c < 4; ++c) {
      corners[c] = at::schema::Point2f(static_cast<float>(projected[c][0]),
                                       static_cast<float>(projected[c][1]));
      cx += corners[c].x() / 4.0F;
      cy += corners[c].y() / 4.0F;
    }
    const std::array<float, 9> rotation{};
    const std::array<float, 3> translation{};
    message.detections.push_back(
        at::schema::TagDetection(tag.id, 0, 50.0F, at::schema::Point2f(cx, cy),
                                 ::flatbuffers::span<const at::schema::Point2f, 4>(corners), false,
                                 ::flatbuffers::span<const float, 9>(rotation),
                                 ::flatbuffers::span<const float, 3>(translation), 0.0F));
  }
  return message;
}

// Ground truth follows the source's numbering, so frames lost on the way do
// not shift the frames after them; unnumbered frames cannot be checked.
void check_report_truth() {
  const auto run = [](std::vector<at::schema::AprilTagDetectionsT> messages) {
    auto report = std::make_shared<BackendReport>();
    auto h = holoscan::testing::create_operator_test<DetectionReportOp>(
        DetectionReportConfig{.report = report,
                              .remaining = std::make_shared<std::atomic<int>>(),
                              .truth = GroundTruth{"tag36h11", kWidth, kHeight}});
    for (auto& message : messages) h.push("detections", std::move(message));
    return std::make_pair(h.run().ok(), report);
  };
  const auto [ok, report] = run({truth_message(0), truth_message(2), truth_message(5)});
  VERIFY(ok);
  VERIFY_MSG(report->verified && report->passed() && report->expected_tags == 12,
             "missed " + std::to_string(report->missed_tags) + ", extra " +
                 std::to_string(report->extra_tags) + " with frames 1, 3 and 4 lost");
  at::schema::AprilTagDetectionsT unnumbered = truth_message(0);
  unnumbered.device_sequence = 0;
  VERIFY_MSG(!run({std::move(unnumbered)}).first, "an unnumbered frame fails the run");
}

}  // namespace
}  // namespace apriltag_app

int main() {
  return apriltag_app::test::run({
      {"reporter counts", apriltag_app::check_report_counts},
      {"reporter keys ground truth on the source's numbering", apriltag_app::check_report_truth},
  });
}
