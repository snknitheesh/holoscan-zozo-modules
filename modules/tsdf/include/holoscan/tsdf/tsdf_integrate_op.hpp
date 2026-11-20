// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// TsdfIntegrateOp: integrates depth images, with their color when
// configured, into a TsdfVolume from known camera poses, on the GPU.
//
//   depth            (input) MONO16 (raw units, scaled by depth_scale) or
//                    MONO32F, [H, W] or [H, W, 1], packed, host or device.
//   color            (input, with `color`) RGB8, BGR8, RGBA8 or BGRA8 of the
//                    depth image's size, aligned with it.
//   pose             (input) float32 or float64 [4, 4] world_from_camera,
//                    row-major; host memory (device memory is copied to the
//                    host, which waits for the GPU).
//   camera_info      (input, mandatory) the depth camera's CameraInfo
//                    (holoscan-messages), for images of the depth image's
//                    size; undistorted or rectified, without skew.
//   integrated_pose  (output, with `pose_output`) float32 [4, 4], host: the
//                    pose just integrated, with the depth image's capture
//                    time and frame id, once the integration is enqueued.
//                    Connect it to TsdfMeshOp to mesh or ray cast the volume
//                    after each image.
//
// The operator runs once per depth image. The depth image, its color and, by
// default, its pose carry one capture time and are aligned exactly; with
// pose_max_age, a pose from a source on its own clock is taken instead (the
// newest at or before the depth image, at most that old). The camera info
// comes at any rate (CameraInfoSourceOp sends it periodically) or with every
// depth image; see TsdfIntegrateConfig::camera_info. Depth images before the
// first camera info, or without a pose, are skipped.
//
// The integration is enqueued on the runtime's stream and ordered against
// every other use of the volume on the GPU. The operator waits on the host
// only for a pose in device memory, and for depth or color in pageable host
// memory, whose copy to the GPU returns once the data is staged.

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/tsdf/tsdf_volume.hpp>
#include <holoscan/tsdf/volume_op.hpp>

namespace holoscan::tsdf {

struct TsdfIntegrateConfig {
  // The volume, shared with TsdfMeshOp. Required.
  std::shared_ptr<TsdfVolume> volume;
  // Multiplies raw depth into meters: 0.001 for uint16 millimeters, 1.0 for
  // float32 meters.
  float depth_scale{0.001F};
  // Depths outside [depth_min, depth_max] meters are no measurement.
  float depth_min{0.1F};
  float depth_max{4.0F};
  // Declare the `color` input; needs a volume with color.
  bool color{false};
  bool pose_output{true};
  // How the camera info arrives: kLatest (the default) for camera info at any
  // rate, kEveryFrame for camera info sent with every depth image, with its
  // capture time.
  holoscan::messages::CameraInfoInputOptions camera_info;
  // 0 (the default): each depth image's pose carries exactly its capture
  // time. Above 0: poses come on their own clock, and each depth image takes
  // the newest pose captured at or before it and at most this long before;
  // a depth image without one is skipped (the camera moves between a pose
  // and the image, so keep this below what the motion allows).
  std::chrono::nanoseconds pose_max_age{0};
  // Largest image accepted; a larger one fails the frame.
  int max_width{1920};
  int max_height{1080};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const TsdfIntegrateConfig& config);

class TsdfIntegrateOp final : public detail::VolumeOp {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit TsdfIntegrateOp(TsdfIntegrateConfig config);
  ~TsdfIntegrateOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::schema::ImageT> depth;
  holoscan::Input<holoscan::schema::ImageT> color;
  holoscan::Input<holoscan::Tensor> pose;
  holoscan::Input<holoscan::messages::schema::CameraInfoT> camera_info;
  holoscan::Output<holoscan::Tensor> integrated_pose;

 private:
  TsdfIntegrateConfig config_;
  holoscan::messages::LatestCameraInfo<holoscan::messages::schema::CameraInfoT> info_;
  // The intrinsics of info_'s camera, once checked that the kernels can use
  // them.
  std::optional<CameraIntrinsics> intrinsics_;
  bool warned_no_pose_{};
};

}  // namespace holoscan::tsdf
