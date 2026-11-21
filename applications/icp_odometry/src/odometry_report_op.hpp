// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// OdometryReportOp: the demo's sink. Pairs each estimated pose with its
// registration statistics by capture time, measures the trajectory's error
// against the camera's true path (the synthetic path at the frame's id), and
// asks the run to complete after the last frame.

#pragma once

#include <cstdint>
#include <memory>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/icp/rigid_transform.hpp>

namespace icp_odometry_app {

// What one run saw. Written by the reporter during the run and read by main()
// after it has stopped.
struct OdometryReport {
  std::uint32_t frames{};
  // The first frame's id: 1, or 2 when the first frame came before the
  // first camera info and gave no cloud.
  std::uint64_t first_frame_id{};
  std::uint64_t last_frame_id{};
  // Poses whose frame id is not the one after the previous pose's.
  std::uint32_t frame_gaps{};
  // Frames after the first whose registration did not converge.
  std::uint32_t unconverged{};
  double path_m{};  // The true path's length.
  double sum_squared_error_m2{};
  double max_error_m{};
  double final_error_m{};
  double final_rotation_error_rad{};
  double sum_fitness{};
  double sum_rmse_m{};
  std::uint64_t iterations{};

  // Absolute trajectory error: the RMS of the translation errors.
  [[nodiscard]] double ate_rmse_m() const;
  [[nodiscard]] double drift_percent() const;
};

// The largest errors a run may end with and pass.
struct OdometryLimits {
  double max_drift_percent{2.0};
  double max_rotation_error_deg{1.5};
  std::uint32_t max_unconverged{0};
};

struct OdometryReportConfig {
  std::shared_ptr<OdometryReport> report;
  // The camera's frames: the run completes after the one with this id; 0
  // never completes.
  std::uint32_t expected_frames{};
  // The length of the camera's closed path in frames, for its true poses.
  std::uint32_t path_frames{300};
};

// Whether `report` covers the camera's `frames` frames without gaps, from
// the first or the second (when the first came before the camera info),
// within `limits`.
[[nodiscard]] bool passed(const OdometryReport& report, std::uint32_t frames,
                          const OdometryLimits& limits);

class OdometryReportOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `report` is missing.
  explicit OdometryReportOp(OdometryReportConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // float64 [4, 4]: IcpOp's pose.
  holoscan::Input<holoscan::Tensor> pose;
  // IcpOp's float64 statistics.
  holoscan::Input<holoscan::Tensor> stats;

 private:
  OdometryReportConfig config_;
  holoscan::icp::Matrix4 first_truth_{holoscan::icp::identity_transform()};
  holoscan::icp::Matrix4 last_truth_{holoscan::icp::identity_transform()};
  bool finished_{};
};

}  // namespace icp_odometry_app
