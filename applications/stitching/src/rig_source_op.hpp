// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// RigSourceOp: the cameras of a rig driving through the module's synthetic
// world, rendered on the GPU: one RGBA8 image per camera per frame, all with
// the frame's capture time, in CUDA device memory.

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/stitching/camera_rig.hpp>
#include <holoscan/stitching/synthetic_world.hpp>

namespace stitching_app {

struct RigSourceConfig {
  holoscan::stitching::CameraRig rig;
  holoscan::stitching::SyntheticWorld world;
  // Each camera's exposure; empty means 1 for all.
  std::vector<float> gains;
  // Frames to send, then none; 0 sends until the run is stopped.
  std::uint32_t count{0};
  std::chrono::milliseconds period{33};
  // Rays per pixel along each side (1..8).
  int samples{2};
  // When set, frame n (from 1) is sent only once this returns at least n - 1:
  // one frame in flight, for lossless runs whose consumers report what they
  // finished.
  std::function<std::uint32_t()> done;
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const RigSourceConfig& config);

// The rig's pose in frame n (from 0).
[[nodiscard]] holoscan::stitching::RigPose frame_pose(std::uint32_t frame);

class RigSourceOp final : public holoscan::Operator<> {
 public:
  explicit RigSourceOp(RigSourceConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // Output c, named after camera c: RGBA8 [H, W, 4] on the GPU; the sample's
  // frame id counts from 1.
  std::array<holoscan::Output<holoscan::schema::ImageT>, holoscan::stitching::kMaxCameras> images;

 private:
  RigSourceConfig config_;
  std::uint32_t sent_{};
};

}  // namespace stitching_app
