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
//   intrinsics       (input, with `intrinsics_input`) float32
//                    [fx, fy, cx, cy] in host memory.
//   integrated_pose  (output, with `pose_output`) float32 [4, 4], host: the
//                    pose just integrated, with the depth image's capture
//                    time and frame id, once the integration is enqueued.
//                    Connect it to TsdfMeshOp to mesh or ray cast the volume
//                    after each image.
//
// The operator runs once per depth image, when the inputs with exactly the
// same capture time have arrived. The integration is enqueued on the
// runtime's stream and ordered against every other use of the volume on the
// GPU; the operator never waits for it.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/tsdf/tsdf_volume.hpp>
#include <holoscan/tsdf/volume_op.hpp>

namespace holoscan::tsdf {

struct TsdfIntegrateConfig {
  // The volume, shared with TsdfMeshOp. Required.
  std::shared_ptr<TsdfVolume> volume;
  // Used for every frame. Give these or enable `intrinsics_input`, not both.
  std::optional<CameraIntrinsics> intrinsics;
  bool intrinsics_input{false};
  // Multiplies raw depth into meters: 0.001 for uint16 millimeters, 1.0 for
  // float32 meters.
  float depth_scale{0.001F};
  // Depths outside [depth_min, depth_max] meters are no measurement.
  float depth_min{0.1F};
  float depth_max{4.0F};
  // Declare the `color` input; needs a volume with color.
  bool color{false};
  bool pose_output{true};
  // Largest image accepted; a larger one fails the frame.
  int max_width{1920};
  int max_height{1080};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const TsdfIntegrateConfig& config);

class TsdfIntegrateOp final : public VolumeOp {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit TsdfIntegrateOp(TsdfIntegrateConfig config);
  ~TsdfIntegrateOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::schema::ImageT> depth;
  holoscan::Input<holoscan::schema::ImageT> color;
  holoscan::Input<holoscan::Tensor> pose;
  holoscan::Input<holoscan::Tensor> intrinsics;
  holoscan::Output<holoscan::Tensor> integrated_pose;

 private:
  TsdfIntegrateConfig config_;
};

}  // namespace holoscan::tsdf
