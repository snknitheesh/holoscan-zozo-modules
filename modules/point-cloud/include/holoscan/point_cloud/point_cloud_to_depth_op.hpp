// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// PointCloudToDepthOp: projects a point cloud (a PointCloud message of
// holoscan-messages) into the pinhole camera that a CameraInfo describes, on
// the GPU, producing a depth image of the camera's size in which the nearest
// point wins each pixel, and optionally the color image of those points.
//
// The camera info comes on its own input, at any rate (CameraInfoSourceOp
// sends it periodically) or with every cloud; see
// PointCloudToDepthConfig::camera_info and the messages module's
// camera_info_contract(). Clouds that arrive before the first camera info
// give no image.
//
// Each frame's depth image is emitted before its color image. If the color
// emit fails, as into a full queue of a blocking connection, the depth image
// has already gone out alone and compute() returns the error. A consumer that
// needs both pairs them by capture time (OnAligned); connect one that only
// displays them with QueuePolicy::kDropOldest, so that it never pushes back.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/point_cloud/camera_intrinsics.hpp>
#include <holoscan/point_cloud/project.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::point_cloud {

// Largest image side and point count the operator accepts.
inline constexpr int kMaxImageSide = 16384;
inline constexpr std::uint32_t kMaxPointsLimit = std::uint32_t{1} << 30;

struct PointCloudToDepthConfig {
  // The largest camera, in pixels, the outputs are sized for: 1..kMaxImageSide.
  // The images have the size the camera info says.
  int max_width{1920};
  int max_height{1080};
  // Maps the input points into the camera optical frame (x right, y down,
  // z forward), as a 3x4 row-major [R | t]; none means they are in it
  // already.
  std::optional<Extrinsics> extrinsics;
  // Points whose camera-frame z is not positive or lies outside
  // [depth_min, depth_max] meters are dropped.
  float depth_min{0.0F};
  float depth_max{100.0F};
  // float32 meters (MONO32F), or uint16 counts of `depth_scale` meters
  // (MONO16) rounded half away from zero, in which a point whose count would
  // be 0 or above 65535 is dropped.
  DepthFormat depth_format{DepthFormat::kFloat32};
  float depth_scale{0.001F};
  // Side of the square of pixels each point covers: 1, 3 or 5.
  int point_size{1};
  // Declare the `color` output, made from the clouds' colors, which every
  // cloud must then carry.
  bool color{false};
  // How the camera info arrives: kLatest (the default) for camera info at any
  // rate, kEveryFrame for camera info sent with every cloud, with its capture
  // time.
  holoscan::messages::CameraInfoInputOptions camera_info;
  // Largest cloud accepted, 1..kMaxPointsLimit points; a larger one fails the
  // frame. It bounds the work per frame and the staging of host input.
  std::uint32_t max_points{std::uint32_t{1} << 24};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const PointCloudToDepthConfig& config);

class PointCloudToDepthOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit PointCloudToDepthOp(PointCloudToDepthConfig config);
  ~PointCloudToDepthOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // float32 points [N, 3], or an organized [H, W, 3] taken as H * W points,
  // with their uint8 RGB colors when `color`; packed, in host, pinned or CUDA
  // device memory on the operator's GPU. Non-finite points are skipped.
  holoscan::Input<holoscan::messages::schema::PointCloudT> point_cloud;
  // Mandatory: the camera to project into. Undistorted or rectified: a raw
  // camera, one with a distortion coefficient that is not zero, fails the
  // frame, as does a skew that is not zero.
  holoscan::Input<holoscan::messages::schema::CameraInfoT> camera_info;
  // [height, width] MONO32F meters or MONO16 counts, 0 where no point
  // projects, on the GPU, of the camera's size. The descriptor's header holds
  // the cloud's capture time (0 without one) and the camera info's frame id;
  // the sample carries the capture time and frame id of the cloud.
  holoscan::Output<holoscan::schema::ImageT> depth;
  // With `color`: [height, width, 3] RGB8 of each pixel's nearest point,
  // black where none projects, on the GPU.
  holoscan::Output<holoscan::schema::ImageT> color;

 private:
  // A device buffer, grown as needed.
  struct DeviceBuffer {
    void* data{};
    std::size_t bytes{};
  };
  [[nodiscard]] holoscan::expected<void*, holoscan::Error> reserve(DeviceBuffer& buffer,
                                                                   std::size_t bytes,
                                                                   cudaStream_t stream);
  [[nodiscard]] holoscan::expected<const void*, holoscan::Error> on_device(
      const holoscan::Tensor& tensor, std::size_t bytes, DeviceBuffer& staging,
      cudaStream_t stream);
  [[nodiscard]] holoscan::expected<cudaStream_t, holoscan::Error> stream_for(
      holoscan::ExecutionContext& context);
  void release() noexcept;

  PointCloudToDepthConfig config_;
  // The settings that do not change from frame to frame.
  ProjectParams params_;
  holoscan::messages::LatestCameraInfo<holoscan::messages::schema::CameraInfoT> info_;
  // The intrinsics of info_'s camera, once checked that the kernels can use
  // them.
  std::optional<CameraIntrinsics> intrinsics_;
  DeviceBuffer zbuffer_;
  DeviceBuffer points_staging_;
  DeviceBuffer colors_staging_;
};

}  // namespace holoscan::point_cloud
