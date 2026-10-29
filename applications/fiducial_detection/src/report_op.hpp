// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ReportOp: the demo's sink. Counts markers, chessboard corners and latencies, checks synthetic
// frames against their ground truth, and requests completion once every reporter has seen its
// expected frames.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/fiducials/fiducial_detections_schema_traits.hpp>

#include "sources.hpp"

namespace fiducial_app {

// One detector's results. Written by its reporter during the run, read after it has stopped.
struct Report {
  std::string backend;
  std::uint32_t frames{};
  std::uint64_t markers{};
  std::uint64_t corners{};  // Chessboard corners.
  std::uint64_t dropped{};
  std::uint32_t truncated_frames{};
  std::vector<float> detect_ms;
  // Ground-truth checks, for synthetic frames.
  bool verified{};
  std::uint64_t expected{};  // Markers, or chessboard corners.
  std::uint64_t missed{};
  std::uint64_t extra{};      // Markers of no scene marker, or corners of no corner in view.
  std::uint64_t misplaced{};  // Found, but off by more than the limits below.
  std::uint32_t frames_without_pose{};  // Boards whose pose was not found.
  double max_corner_px{};
  double sum_corner_px{};
  std::uint64_t measured_corners{};
  double max_translation{};  // Fraction of the distance.
  double max_rotation_deg{};

  [[nodiscard]] bool passed() const noexcept {
    return !verified || (missed == 0 && extra == 0 && misplaced == 0 && frames_without_pose == 0);
  }
};

// Limits a verified marker or corner must keep to.
inline constexpr double kMaxCornerErrorPx = 0.5;
inline constexpr double kMaxTranslationError = 0.01;  // Of the distance.
inline constexpr double kMaxRotationErrorDeg = 2.0;

struct ReportConfig {
  std::shared_ptr<Report> report;
  // Reporters still running, shared by all of a run's; the last requests completion.
  std::shared_ptr<std::atomic<int>> remaining;
  // Frames after which this reporter is done; 0 never finishes.
  std::uint32_t expected_frames{};
  // The synthetic scenes the detections should match: each message's is
  // make_scene(kind, device_sequence - 1, width, height). The camera has none.
  std::optional<SceneKind> truth;
  int width{};
  int height{};
};

class ReportOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `report` or `remaining` is missing.
  explicit ReportOp(ReportConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::fiducials::schema::FiducialDetectionsT> detections;

 private:
  [[nodiscard]] holoscan::expected<void, holoscan::Error> verify(
      const holoscan::fiducials::schema::FiducialDetectionsT& message);

  ReportConfig config_;
  bool finished_{};
};

}  // namespace fiducial_app
