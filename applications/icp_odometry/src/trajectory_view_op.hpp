// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// TrajectoryViewOp: draws what HoloViz shows. On the left, the trajectory
// from above: the true one in green, the estimate in orange. On the right,
// the current alignment: the previous frame's depth in gray, and the current
// cloud, moved into the previous frame by IcpOp, colored by its distance to
// the previous cloud: green within 2 cm, yellow within 5 cm, red beyond. A
// text layer gives the frame's errors and statistics.
//
// It draws on the CPU: the clouds are copied to the host, which the demo
// can afford for a display.

#pragma once

#include <cstdint>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/icp/rigid_transform.hpp>
#include <holoscan/icp/synthetic_scene.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace icp_odometry_app {

// Text rows the view sends, each of kViewLabelBytes after HoloViz's header.
inline constexpr std::int64_t kViewLabelRows = 3;
inline constexpr std::int64_t kViewLabelBytes = 72;

struct TrajectoryViewConfig {
  holoscan::icp::PinholeCamera camera;
  // The path's length in frames, to fit the plot to it.
  std::uint32_t path_frames{300};
  std::uint32_t count{300};
};

// The size of the view image for `camera`: two panels side by side, each the
// camera's image scaled down to at most 640 pixels wide.
[[nodiscard]] int view_width(const holoscan::icp::PinholeCamera& camera);
[[nodiscard]] int view_height(const holoscan::icp::PinholeCamera& camera);

class TrajectoryViewOp final : public holoscan::Operator<> {
 public:
  explicit TrajectoryViewOp(TrajectoryViewConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> pose;
  holoscan::Input<holoscan::Tensor> true_pose;
  holoscan::Input<holoscan::Tensor> stats;
  // The current cloud as DepthToPointCloudOp made it, and as IcpOp moved it.
  holoscan::Input<holoscan::Tensor> point_cloud;
  holoscan::Input<holoscan::Tensor> aligned;
  // RGBA8 [view_height, view_width, 4], host.
  holoscan::Output<holoscan::schema::ImageT> view;
  // HoloViz text rows: uint8 [kViewLabelRows, 16 + kViewLabelBytes], host.
  holoscan::Output<holoscan::Tensor> labels;

 private:
  void draw_trajectories(std::uint8_t* rgba) const;
  void draw_alignment(std::uint8_t* rgba) const;

  TrajectoryViewConfig config_;
  int panel_width_{};
  int panel_height_{};
  // The plot's world bounds: x across, z up the panel.
  double min_x_{}, max_x_{}, min_z_{}, max_z_{};
  std::vector<holoscan::icp::Matrix4> estimated_;
  std::vector<holoscan::icp::Matrix4> truth_;
  holoscan::icp::Matrix4 first_truth_{holoscan::icp::identity_transform()};
  std::vector<float> previous_;  // The previous frame's cloud.
  std::vector<float> current_;
  std::vector<float> aligned_;
  bool logged_copy_{};
};

}  // namespace icp_odometry_app
