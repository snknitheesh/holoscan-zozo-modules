// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// AprilTagDetectorOp: an image in, one AprilTagDetections out, on the configured
// backend (CPU, CUDA or VPI), with poses when the camera's intrinsics are known.

#pragma once

#include <memory>
#include <optional>

#include <holoscan/apriltag/apriltag_detections_schema_traits.hpp>
#include <holoscan/apriltag/detection.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::apriltag {

struct AprilTagDetectorConfig {
  Backend backend{Backend::kCuda};
  DetectorOptions options;
  // With both, each detection carries the tag's pose.
  std::optional<CameraIntrinsics> intrinsics;
  double tag_size{};  // Meters across the tag's black border.
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const AprilTagDetectorConfig& config);

// A message's detection as the backends report it (pose included when present).
[[nodiscard]] Detection from_message(const schema::TagDetection& tag);

class AprilTagDetectorOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit AprilTagDetectorOp(AprilTagDetectorConfig config);
  ~AprilTagDetectorOp();
  AprilTagDetectorOp(const AprilTagDetectorOp&) = delete;
  AprilTagDetectorOp& operator=(const AprilTagDetectorOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // MONO8, RGB8, BGR8, RGBA8 or BGRA8, in host or CUDA device memory.
  holoscan::Input<holoscan::schema::ImageT> image;
  holoscan::Output<schema::AprilTagDetectionsT> detections;

 private:
  // The backend, the gray conversion buffers, a CUDA stream and the counters.
  struct State;

  // Warns about frames over a backend's capacity, at most every few seconds.
  void warn_overflow(const Overflow& overflow);

  AprilTagDetectorConfig config_;
  std::unique_ptr<State> state_;
};

}  // namespace holoscan::apriltag
