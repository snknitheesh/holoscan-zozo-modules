// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// TrajectoryViewOp: draws what HoloViz shows. On the left, the trajectory
// from above: the true one in green, the estimate in orange. On the right,
// the current alignment: the previous frame's depth in gray, and the current
// cloud, moved into the previous frame by IcpOp, colored by its distance to
// the previous cloud: green within 2 cm, yellow within 5 cm, red beyond. A
// text layer gives the frame's id, errors and statistics.
//
// It draws on the CPU: each frame's two clouds are copied to pinned host
// memory, on the runtime's CUDA stream with one wait, which it logs once.
// The previous cloud is the one with the frame id before the current one's;
// when that frame was not drawn (the display keeps only the newest), the
// panel shows the current cloud alone and says so.

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
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace icp_odometry_app {

// Text rows the view sends, each of kViewLabelBytes after HoloViz's header.
inline constexpr std::int64_t kViewLabelRows = 3;
inline constexpr std::int64_t kViewLabelBytes = 72;

struct TrajectoryViewConfig {
  // The depth camera's image size: the clouds' and the camera info's.
  int width{640};
  int height{480};
  // The path's length in frames, to fit the plot to it and know the truth.
  std::uint32_t path_frames{300};
  std::uint32_t count{300};
};

// The size of the view image for a width x height camera: two panels side
// by side, each the camera's image scaled down to at most 640 pixels wide.
[[nodiscard]] int view_width(int width, int height);
[[nodiscard]] int view_height(int width, int height);

class TrajectoryViewOp final : public holoscan::Operator<> {
 public:
  explicit TrajectoryViewOp(TrajectoryViewConfig config);
  ~TrajectoryViewOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // IcpOp's pose and float64 statistics, host.
  holoscan::Input<holoscan::Tensor> pose;
  holoscan::Input<holoscan::Tensor> stats;
  // The current cloud as DepthToPointCloudOp made it, and as IcpOp moved it:
  // organized [height, width, 3].
  holoscan::Input<holoscan::messages::schema::PointCloudT> point_cloud;
  holoscan::Input<holoscan::messages::schema::PointCloudT> aligned;
  // Mandatory: the depth camera's calibration, to project the clouds.
  holoscan::Input<holoscan::messages::schema::CameraInfoT> camera_info;
  // RGBA8 [view_height, view_width, 4], host.
  holoscan::Output<holoscan::schema::ImageT> view;
  // HoloViz text rows: uint8 [kViewLabelRows, 16 + kViewLabelBytes], host.
  holoscan::Output<holoscan::Tensor> labels;

 private:
  void draw_trajectories(std::uint8_t* rgba) const;
  void draw_alignment(std::uint8_t* rgba, const holoscan::messages::CameraInfo& camera,
                      bool with_previous) const;
  void release() noexcept;

  TrajectoryViewConfig config_;
  int panel_width_{};
  int panel_height_{};
  // The plot's world bounds: x across, z up the panel.
  double min_x_{}, max_x_{}, min_z_{}, max_z_{};
  std::vector<holoscan::icp::Matrix4> estimated_;
  std::vector<holoscan::icp::Matrix4> truth_;
  // Maps an estimated pose into the world: the true pose of the first frame
  // drawn, after that frame's estimate.
  holoscan::icp::Matrix4 world_from_estimate_{holoscan::icp::identity_transform()};
  holoscan::messages::LatestCameraInfo<holoscan::messages::schema::CameraInfoT> info_;
  // Pinned host copies of the clouds: the current frame's, the previous
  // frame's (by frame id), and the current one aligned.
  float* current_{};
  float* previous_{};
  float* aligned_{};
  std::uint64_t previous_frame_id_{};
  bool logged_copy_{};
};

}  // namespace icp_odometry_app
