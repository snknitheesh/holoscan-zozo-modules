// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// DepthToPointCloudOp: deprojects an organized depth image into an organized
// point cloud on the GPU, one CUDA thread per pixel, through the pinhole
// camera that the depth image's CameraInfo describes. Optionally it takes an
// aligned color image and gives each point its color. The cloud goes out as
// one PointCloud message (holoscan-messages).
//
// The camera info comes on its own input, at any rate (CameraInfoSourceOp
// sends it periodically) or with every depth image; see
// DepthToPointCloudConfig::camera_info and the messages module's
// camera_info_contract(). Depth images that arrive before the first camera
// info give no cloud.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/point_cloud/camera_intrinsics.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::point_cloud {

struct DepthToPointCloudConfig {
  // Multiplies raw depth into meters: 0.001 for uint16 millimeters, 1.0 for
  // float32 meters.
  float depth_scale{0.001F};
  // Valid metric depth range; other pixels give a NaN point.
  float depth_min{0.0F};
  float depth_max{100.0F};
  // Declare the `color` input, and send each point's color in the cloud.
  bool color{false};
  // How the camera info arrives: kLatest (the default) for camera info at any
  // rate, kEveryFrame for camera info sent with every depth image, with its
  // capture time.
  holoscan::messages::CameraInfoInputOptions camera_info;
  // Largest image the output is sized for.
  int max_width{1920};
  int max_height{1080};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const DepthToPointCloudConfig& config);

class DepthToPointCloudOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit DepthToPointCloudOp(DepthToPointCloudConfig config);
  ~DepthToPointCloudOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // MONO16 (raw units, scaled by depth_scale) or MONO32F, [H, W] or [H, W, 1],
  // rows packed, in host or CUDA device memory.
  holoscan::Input<holoscan::schema::ImageT> depth;
  // Mandatory: the depth camera's calibration, for images of the depth
  // image's size. Undistorted or rectified: a raw camera, one with a
  // distortion coefficient that is not zero, fails the frame, as does a
  // skew that is not zero.
  holoscan::Input<holoscan::messages::schema::CameraInfoT> camera_info;
  // With `color`: RGB8, RGBA8, BGR8 or BGRA8 of the depth image's size.
  holoscan::Input<holoscan::schema::ImageT> color;
  // An organized [H, W, 3] cloud in meters in the camera's optical frame, in
  // device memory: NaN for a pixel without valid depth; uint8 [H, W, 3] RGB
  // colors with `color`, else none. Its header has the camera info's frame id
  // and the depth image's capture time; valid_count is -1 (not counted).
  holoscan::Output<holoscan::messages::schema::PointCloudT> point_cloud;

 private:
  // A device copy of a host input, grown as needed.
  struct Staging {
    void* data{};
    std::size_t bytes{};
  };
  [[nodiscard]] holoscan::expected<const void*, holoscan::Error> on_device(
      const holoscan::Tensor& tensor, std::size_t bytes, Staging& staging, cudaStream_t stream);
  void release() noexcept;

  DepthToPointCloudConfig config_;
  holoscan::messages::LatestCameraInfo<holoscan::messages::schema::CameraInfoT> info_;
  // The intrinsics of info_'s camera, once checked that the kernel can use
  // them.
  std::optional<CameraIntrinsics> intrinsics_;
  Staging depth_staging_;
  Staging color_staging_;
};

}  // namespace holoscan::point_cloud
