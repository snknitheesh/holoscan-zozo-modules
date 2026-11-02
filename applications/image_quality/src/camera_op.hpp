// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's synthetic camera: frames of the schedule on a clock, then idle.

#pragma once

#include <chrono>
#include <cstdint>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "segments.hpp"

namespace image_quality_app {

struct CameraConfig {
  Schedule schedule;
  // Frames to send; each is sent with frame id = its index.
  std::uint64_t count{630};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  // Send the frames in CUDA device memory instead of host memory.
  bool device_output{false};
};

class SyntheticCameraOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument for an unusable config.
  explicit SyntheticCameraOp(CameraConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // RGB8 frames.
  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  CameraConfig config_;
  std::uint64_t sent_{};
};

}  // namespace image_quality_app
