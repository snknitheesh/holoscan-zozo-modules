// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ReportOp: the demo's sink. Counts markers, chessboard corners, latencies, CPU waits and
// copies, checks synthetic frames against their ground truth, keeps each frame's results for
// comparing backends, and requests completion once every reporter has seen its expected frames.

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
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

// What one frame's detections were: by the source's frame number, each marker's id and
// corners and each chessboard corner's id and position, as the message carries them.
struct FrameResult {
  std::uint64_t device_sequence{};
  std::vector<std::pair<int, std::array<float, 8>>> markers;
  std::vector<std::pair<int, std::array<float, 2>>> corners;

  friend bool operator==(const FrameResult&, const FrameResult&) = default;
};

// One detector's results. Written by its reporter during the run, read after it has stopped.
struct Report {
  std::string backend;
  std::uint32_t frames{};
  std::uint64_t markers{};
  std::uint64_t corners{};  // Chessboard corners.
  std::uint64_t dropped{};
  std::uint32_t truncated_frames{};
  std::vector<float> detect_ms;  // Every frame's but the first, which warms the detector up.
  std::uint64_t host_waits{};    // Summed over the frames.
  std::uint64_t copied_bytes{};
  std::vector<FrameResult> results;  // With ReportConfig::keep_results.
  // Ground-truth checks, for synthetic frames.
  bool verified{};
  std::uint64_t expected{};  // Markers, or chessboard corners.
  std::uint64_t missed{};
  std::uint64_t extra{};      // Markers of no scene marker, or corners of no corner in view.
  std::uint64_t misplaced{};  // Found, but off by more than the limits below.
  // Markers found without a pose, or boards whose pose was not found, though the camera is
  // known: misses.
  std::uint32_t without_pose{};
  double max_corner_px{};
  double sum_corner_px{};
  std::uint64_t measured_corners{};
  double max_translation{};  // Fraction of the distance.
  double max_rotation_deg{};

  [[nodiscard]] bool passed() const noexcept {
    return !verified || (missed == 0 && extra == 0 && misplaced == 0 && without_pose == 0);
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
  // Keep every frame's results, to compare backends.
  bool keep_results{};
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
