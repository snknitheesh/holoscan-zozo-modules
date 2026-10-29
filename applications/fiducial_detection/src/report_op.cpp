// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "report_op.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include <holoscan/fiducials/fiducial_messages.hpp>
#include <holoscan/fiducials/synthetic_scene.hpp>

namespace fiducial_app {

namespace fd = holoscan::fiducials;

ReportOp::ReportOp(ReportConfig config) : config_(std::move(config)) {
  if (!config_.report || !config_.remaining) {
    throw std::invalid_argument("ReportOp needs a report and a completion counter");
  }
}

void ReportOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(detections, "detections").queue_depth(2U);
}

holoscan::Contract ReportOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{detections});
  return result;
}

void ReportOp::start() { finished_ = false; }

holoscan::expected<void, holoscan::Error> ReportOp::compute(holoscan::ExecutionContext& context) {
  auto received = detections.receive_data();
  if (!received) return holoscan::make_unexpected(std::move(received).error());
  const fd::schema::FiducialDetectionsT& message = *received;
  Report& report = *config_.report;
  report.backend = message.backend;
  ++report.frames;
  report.markers += message.markers.size();
  report.corners += message.board ? message.board->corners.size() : 0U;
  report.dropped += static_cast<std::uint64_t>(std::max(message.dropped, 0));
  report.truncated_frames += message.truncated ? 1U : 0U;
  report.detect_ms.push_back(message.detect_time_us / 1000.0F);
  if (config_.truth) {
    if (auto verified = verify(message); !verified) return verified;
  }
  if (!finished_ && config_.expected_frames != 0 && report.frames >= config_.expected_frames) {
    finished_ = true;
    if (config_.remaining->fetch_sub(1) == 1) context.request_completion();
  }
  return {};
}

holoscan::expected<void, holoscan::Error> ReportOp::verify(
    const fd::schema::FiducialDetectionsT& message) {
  // The source's frame number, not the detector's count, so a frame lost on the way cannot
  // shift every later frame's ground truth.
  if (message.device_sequence == 0) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kInvalidArgument,
        "ReportOp: ground truth needs images numbered by header.device_sequence"});
  }
  Report& report = *config_.report;
  report.verified = true;
  const fd::SyntheticScene scene =
      make_scene(*config_.truth, static_cast<std::uint32_t>(message.device_sequence - 1U),
                 config_.width, config_.height);
  const auto corner_error = [&report](double error) {
    report.max_corner_px = std::max(report.max_corner_px, error);
    report.sum_corner_px += error;
    ++report.measured_corners;
    return error > kMaxCornerErrorPx;
  };
  const auto pose_error = [&report](const fd::Pose& found, const fd::Pose& truth) {
    const auto& t = truth.translation;
    const auto& p = found.translation;
    const double translation =
        std::hypot(p[0] - t[0], p[1] - t[1], p[2] - t[2]) / std::hypot(t[0], t[1], t[2]);
    const double rotation = fd::rotation_error_deg(found.rotation, truth.rotation);
    report.max_translation = std::max(report.max_translation, translation);
    report.max_rotation_deg = std::max(report.max_rotation_deg, rotation);
    return translation > kMaxTranslationError || rotation > kMaxRotationErrorDeg;
  };

  if (!config_.truth->board) {
    const std::vector<fd::Marker> found = fd::markers_of(message);
    const fd::TruthCounts counts = fd::match_markers(scene, found, [&](const fd::MarkerMatch& m) {
      bool misplaced = false;
      for (const double e : m.corner_errors) misplaced = corner_error(e) || misplaced;
      if (m.found.pose) misplaced = pose_error(*m.found.pose, m.truth.pose) || misplaced;
      report.misplaced += misplaced ? 1U : 0U;
    });
    report.expected += scene.markers.size();
    report.missed += counts.missed;
    report.extra += counts.extra;
    return {};
  }
  const auto view = fd::board_of(message);
  const std::vector<fd::CharucoCorner> truth = fd::board_truth(scene);
  report.expected += truth.size();
  if (!view) {
    report.missed += truth.size();
    ++report.frames_without_pose;
    return {};
  }
  std::size_t matched = 0;
  for (const fd::CharucoCorner& corner : view->corners) {
    const auto it = std::find_if(truth.begin(), truth.end(),
                                 [&](const fd::CharucoCorner& t) { return t.id == corner.id; });
    if (it == truth.end()) {
      ++report.extra;
      continue;
    }
    ++matched;
    const bool off = corner_error(std::hypot(corner.image[0] - it->image[0],
                                             corner.image[1] - it->image[1]));
    const bool elsewhere = std::abs(corner.board[0] - it->board[0]) > 1e-6 ||
                           std::abs(corner.board[1] - it->board[1]) > 1e-6;
    report.misplaced += off || elsewhere ? 1U : 0U;
  }
  report.missed += truth.size() - matched;
  if (view->pose) {
    report.misplaced += pose_error(*view->pose, scene.board->pose) ? 1U : 0U;
  } else {
    ++report.frames_without_pose;
  }
  return {};
}

}  // namespace fiducial_app
