// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "odometry_report_op.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/icp/icp_op.hpp>
#include <holoscan/icp/synthetic_scene.hpp>

namespace icp_odometry_app {
namespace {

using std::chrono_literals::operator""ns;
namespace icp = holoscan::icp;

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "OdometryReportOp: " + message};
}

// The float64 values of a host tensor of `count` elements.
holoscan::expected<std::vector<double>, holoscan::Error> values_of(const holoscan::Tensor& tensor,
                                                                   std::size_t count,
                                                                   const char* what) {
  const DLDataType dtype = tensor.dtype();
  if (dtype.code != kDLFloat || dtype.bits != 64 || dtype.lanes != 1 ||
      tensor.size() != static_cast<std::int64_t>(count)) {
    return holoscan::make_unexpected(
        invalid(std::string(what) + " must hold " + std::to_string(count) + " float64 values"));
  }
  std::vector<double> values(count);
  if (auto copied = tensor.copy_to_host(values.data(), count * sizeof(double)); !copied) {
    return holoscan::make_unexpected(std::move(copied).error());
  }
  return values;
}

}  // namespace

double OdometryReport::ate_rmse_m() const {
  return frames == 0U ? 0.0 : std::sqrt(sum_squared_error_m2 / frames);
}

double OdometryReport::drift_percent() const {
  return path_m > 0.0 ? 100.0 * final_error_m / path_m : 0.0;
}

bool passed(const OdometryReport& report, std::uint32_t frames, const OdometryLimits& limits) {
  const bool complete = (report.first_frame_id == 1U || report.first_frame_id == 2U) &&
                        report.last_frame_id == frames &&
                        report.frames == frames + 1U - report.first_frame_id;
  return complete && report.frame_gaps == 0U && report.unconverged <= limits.max_unconverged &&
         report.drift_percent() <= limits.max_drift_percent &&
         report.final_rotation_error_rad * 180.0 / 3.14159265358979323846 <=
             limits.max_rotation_error_deg;
}

OdometryReportOp::OdometryReportOp(OdometryReportConfig config) : config_(std::move(config)) {
  if (!config_.report) {
    throw std::invalid_argument("OdometryReportOp needs a report");
  }
}

void OdometryReportOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(pose, "pose").queue_depth(2U);
  spec.input(stats, "stats").queue_depth(2U);
}

holoscan::Contract OdometryReportOp::contract() const {
  holoscan::Contract result;
  // The two belong to one frame when their capture times are equal.
  result.trigger(holoscan::OnAligned{.ports = holoscan::ports(pose, stats), .window = 0ns});
  result.input(pose, holoscan::Exact{});
  result.input(stats, holoscan::Exact{});
  return result;
}

holoscan::expected<void, holoscan::Error> OdometryReportOp::compute(
    holoscan::ExecutionContext& context) {
  auto pose_sample = pose.receive();
  if (!pose_sample) {
    return holoscan::make_unexpected(std::move(pose_sample).error());
  }
  auto stats_sample = stats.receive();
  if (!stats_sample) {
    return holoscan::make_unexpected(std::move(stats_sample).error());
  }
  auto estimate = values_of(pose_sample->data, 16U, "pose");
  if (!estimate) return holoscan::make_unexpected(std::move(estimate).error());
  auto values = values_of(stats_sample->data, icp::kStatsSize, "stats");
  if (!values) return holoscan::make_unexpected(std::move(values).error());

  icp::Matrix4 estimated{};
  std::copy(estimate->begin(), estimate->end(), estimated.begin());
  icp::IcpResult result;
  try {
    result = icp::result_from(*values, estimated);
  } catch (const std::invalid_argument& error) {
    return holoscan::make_unexpected(invalid(error.what()));
  }

  OdometryReport& report = *config_.report;
  const std::uint64_t frame_id = pose_sample->metadata.frame_id;
  if (frame_id == 0U) {
    return holoscan::make_unexpected(invalid("a pose without a frame id"));
  }
  // The camera's true pose at this frame: frame i of the path has id i + 1.
  const icp::Matrix4 world_from_camera =
      icp::synthetic_trajectory(static_cast<std::uint32_t>(frame_id - 1U), config_.path_frames);
  if (report.frames > 0U && frame_id != report.last_frame_id + 1U) {
    ++report.frame_gaps;
  }
  if (report.frames == 0U) {
    // Odometry starts at its first frame: the truth is taken relative to it.
    report.first_frame_id = frame_id;
    first_truth_ = world_from_camera;
  } else {
    report.path_m +=
        icp::translation_norm(icp::compose(icp::rigid_inverse(last_truth_), world_from_camera));
    report.unconverged += result.converged ? 0U : 1U;
    report.sum_fitness += result.fitness;
    report.sum_rmse_m += result.rmse;
    report.iterations += static_cast<std::uint64_t>(result.iterations);
  }
  last_truth_ = world_from_camera;
  report.last_frame_id = frame_id;
  const icp::TransformError error = icp::transform_error(
      estimated, icp::compose(icp::rigid_inverse(first_truth_), world_from_camera));
  report.sum_squared_error_m2 += error.translation * error.translation;
  report.max_error_m = std::max(report.max_error_m, error.translation);
  report.final_error_m = error.translation;
  report.final_rotation_error_rad = error.rotation_rad;
  ++report.frames;

  if (!finished_ && config_.expected_frames != 0U && frame_id >= config_.expected_frames) {
    finished_ = true;
    context.request_completion();
  }
  return {};
}

}  // namespace icp_odometry_app
