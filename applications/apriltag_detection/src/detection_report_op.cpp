// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "detection_report_op.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include <holoscan/apriltag/apriltag_detector_op.hpp>
#include <holoscan/apriltag/synthetic_scene.hpp>

namespace apriltag_app {

namespace at = holoscan::apriltag;

DetectionReportOp::DetectionReportOp(DetectionReportConfig config) : config_(std::move(config)) {
  if (!config_.report || !config_.remaining) {
    throw std::invalid_argument("DetectionReportOp needs a report and a completion counter");
  }
}

void DetectionReportOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(detections, "detections").queue_depth(2U);
}

holoscan::Contract DetectionReportOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{detections});
  return result;
}

void DetectionReportOp::start() { finished_ = false; }

holoscan::expected<void, holoscan::Error> DetectionReportOp::compute(
    holoscan::ExecutionContext& context) {
  auto received = detections.receive_data();
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  const at::schema::AprilTagDetectionsT& message = *received;
  BackendReport& report = *config_.report;
  report.backend = message.backend;
  ++report.frames;
  report.tags += message.detections.size();
  report.dropped += static_cast<std::uint64_t>(std::max(message.dropped, 0));
  report.truncated_frames += message.truncated ? 1U : 0U;
  report.detect_ms.push_back(message.detect_time_us / 1000.0F);
  if (config_.truth) {
    if (auto verified = verify(message); !verified) {
      return verified;
    }
  }
  if (!finished_ && config_.expected_frames != 0 && report.frames >= config_.expected_frames) {
    finished_ = true;
    if (config_.remaining->fetch_sub(1) == 1) {
      context.request_completion();
    }
  }
  return {};
}

holoscan::expected<void, holoscan::Error> DetectionReportOp::verify(
    const at::schema::AprilTagDetectionsT& message) {
  // The source's frame number, not the detector's count, so a frame lost on the
  // way cannot shift every later frame's ground truth.
  if (message.device_sequence == 0) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kInvalidArgument,
        "DetectionReportOp: ground truth needs images numbered by header.device_sequence"});
  }
  BackendReport& report = *config_.report;
  report.verified = true;
  const GroundTruth& truth = *config_.truth;
  const at::SyntheticScene scene =
      at::make_scene(truth.family, static_cast<std::uint32_t>(message.device_sequence - 1U),
                     truth.width, truth.height);
  std::vector<at::Detection> found;
  for (const at::schema::TagDetection& tag : message.detections)
    found.push_back(at::from_message(tag));
  const at::TruthCounts counts =
      at::match_truth(scene, found, [&report](const at::TruthMatch& match) {
        const auto& e = match.corner_errors;
        const double worst = std::max({0.0, e[0], e[1], e[2], e[3]});
        report.max_corner_error_px = std::max(report.max_corner_error_px, worst);
        bool misplaced = worst > kMaxCornerErrorPx;
        if (match.translation_error) {
          report.max_translation_error =
              std::max(report.max_translation_error, *match.translation_error);
          misplaced = misplaced || *match.translation_error > kMaxTranslationError;
        }
        if (misplaced) {
          ++report.misplaced_tags;
        }
      });
  report.expected_tags += scene.tags.size();
  report.extra_tags += counts.extra;
  report.missed_tags += counts.missed;
  return {};
}

}  // namespace apriltag_app
