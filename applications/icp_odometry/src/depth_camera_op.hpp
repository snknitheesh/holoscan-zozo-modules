// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// SyntheticDepthCameraOp: a depth camera moving through holoscan-icp's
// synthetic room along its closed path. Each frame is a float32 depth image
// in meters, ray cast on the GPU, with the camera's true pose beside it, both
// with the frame's capture time and id.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/icp/synthetic_scene.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/time_point.hpp>

namespace icp_odometry_app {

struct DepthCameraConfig {
  int width{640};
  int height{480};
  // Frames to send, along a path that closes after `path_frames`.
  std::uint32_t count{300};
  std::uint32_t path_frames{300};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  // Depth noise as a time-of-flight camera has it; none when false.
  bool noise{true};
  float max_depth{8.0F};
  // When set, the count of frames done downstream: a frame goes out only
  // when every frame before it is done, so a lossless pipeline never refuses
  // one.
  std::shared_ptr<const std::atomic<std::uint32_t>> done;
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const DepthCameraConfig& config);

// The noise the camera adds when it has noise: 0.5 mm plus 0.5 mm per square
// meter of depth.
holoscan::icp::DepthNoise camera_noise(std::uint32_t frame);

class SyntheticDepthCameraOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit SyntheticDepthCameraOp(DepthCameraConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // MONO32F [height, width], meters, 0 where there is no depth; on the GPU.
  holoscan::Output<holoscan::schema::ImageT> depth;
  // float64 [4, 4] row-major, host: the camera's true pose in the world.
  holoscan::Output<holoscan::Tensor> true_pose;

 private:
  DepthCameraConfig config_;
  holoscan::icp::SyntheticScene scene_;
  std::uint32_t emitted_{};
  // The frame being sent: its capture time, and whether its depth is out.
  std::optional<holoscan::TimePoint> pending_capture_;
  bool depth_sent_{};
};

}  // namespace icp_odometry_app
