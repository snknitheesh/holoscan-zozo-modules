// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// OdometryReportOp: the demo's sink. Pairs each estimated pose and its
// registration statistics with the camera's true pose by capture time,
// measures the trajectory's error against the truth, and asks the run to
// complete after its expected frames.

#pragma once

#include <atomic>
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
// after it has stopped, except `frames`, which the camera reads to pace
// itself.
struct OdometryReport {
  std::atomic<std::uint32_t> frames{};
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
  // Frames after which the run completes; 0 never completes.
  std::uint32_t expected_frames{};
};

// Whether `report` covers `frames` frames, without gaps, within `limits`.
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

  // float64 [4, 4] each: IcpOp's pose, and the camera's true pose.
  holoscan::Input<holoscan::Tensor> pose;
  holoscan::Input<holoscan::Tensor> true_pose;
  // IcpOp's float64 statistics.
  holoscan::Input<holoscan::Tensor> stats;

 private:
  OdometryReportConfig config_;
  holoscan::icp::Matrix4 first_truth_{holoscan::icp::identity_transform()};
  holoscan::icp::Matrix4 last_truth_{holoscan::icp::identity_transform()};
  std::uint64_t last_frame_id_{};
  bool finished_{};
};

}  // namespace icp_odometry_app
