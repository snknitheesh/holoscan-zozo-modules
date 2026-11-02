// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ReportOp: the demo's sink. For each frame it takes the normals, curvature,
// plane labels and planes and checks them against the synthetic room's
// truth: the normals' angle error on the inside of each surface (median and
// 95th percentile), the curvature on the flat surfaces and the sphere, and
// the floor, both walls and the table top among the planes (angle, offset,
// and how many of their points carry their label, and how few others). It
// completes the run after the expected frames.

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

namespace normals_planes_app {

// What passes, every frame. The k-nearest search's neighborhoods are
// smaller than the window's, so its normals are noisier; main() relaxes the
// normals' limits for it to 3 and 9 degrees.
inline constexpr double kMaxMedianAngleDeg = 1.5;
inline constexpr double kMaxP95AngleDeg = 5.0;
inline constexpr double kMaxPlaneAngleDeg = 1.0;
inline constexpr double kMaxPlaneOffsetM = 0.015;
inline constexpr double kMinRecall = 0.90;
inline constexpr double kMinPrecision = 0.96;

// What one run saw. Written by the reporter during the run, read after it.
struct Report {
  double max_median_deg{kMaxMedianAngleDeg};
  double max_p95_deg{kMaxP95AngleDeg};
  std::uint32_t frames{};
  std::uint32_t frame_gaps{};
  double worst_median_deg{};
  double worst_p95_deg{};
  std::uint32_t flipped{};
  std::uint32_t wrong_planes{};
  double worst_plane_angle_deg{};
  double worst_plane_offset_m{};
  double worst_recall{1.0};
  double worst_precision{1.0};
  // Frames whose median curvature on the sphere was not above the flat surfaces'.
  std::uint32_t flat_sphere{};
  std::uint64_t planes{};

  [[nodiscard]] bool passed() const {
    return frames != 0 && frame_gaps == 0 && worst_median_deg <= max_median_deg &&
           worst_p95_deg <= max_p95_deg && flipped == 0 && wrong_planes == 0 && flat_sphere == 0;
  }
};

struct ReportConfig {
  std::shared_ptr<Report> report;
  int width{};
  int height{};
  // The normals' window radius: the inside of a surface is where the window
  // sees only it.
  int window_radius{3};
  std::uint32_t max_planes{};
  std::uint32_t expected_frames{};
  // Counts the frames done, for the source that waits for them.
  std::shared_ptr<std::atomic<std::uint32_t>> finished;
};

class ReportOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if the report is missing.
  explicit ReportOp(ReportConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> normals;       // [H, W, 3]
  holoscan::Input<holoscan::Tensor> curvature;     // [H, W]
  holoscan::Input<holoscan::Tensor> labels;        // [H, W]
  holoscan::Input<holoscan::Tensor> coefficients;  // [P, 4]
  holoscan::Input<holoscan::Tensor> centroids;     // [P, 3]
  holoscan::Input<holoscan::Tensor> inliers;       // [P]

 private:
  ReportConfig config_;
  std::vector<float> normals_;
  std::vector<float> curvature_;
  std::vector<std::uint8_t> labels_;
  std::uint64_t last_frame_id_{};
  bool finished_{};
};

}  // namespace normals_planes_app
