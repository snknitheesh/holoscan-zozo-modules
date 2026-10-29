// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ArucoDetectorOp: an image in, the dictionary's markers it holds out, on the configured
// backend, each with its pose when the camera's intrinsics and the markers' size are known.

#pragma once

#include <memory>
#include <optional>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/fiducials/fiducial_detections_schema_traits.hpp>
#include <holoscan/fiducials/types.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::fiducials {

class FrameDetector;

struct ArucoDetectorConfig {
  // kAuto lets the operator choose at start: see resolve_auto().
  Backend backend{Backend::kCuda};
  DetectorOptions options;
  // With both, each marker carries its pose.
  std::optional<CameraIntrinsics> intrinsics;
  double marker_length{};  // Meters across a marker's black border.
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const ArucoDetectorConfig& config);

class ArucoDetectorOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit ArucoDetectorOp(ArucoDetectorConfig config);
  ~ArucoDetectorOp();
  ArucoDetectorOp(const ArucoDetectorOp&) = delete;
  ArucoDetectorOp& operator=(const ArucoDetectorOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // The backend this operator runs: the configured one, or for Backend::kAuto the one start()
  // chose, and kAuto until then. Safe to call from any thread.
  [[nodiscard]] Backend resolved_backend() const noexcept;

  // MONO8, RGB8, BGR8, RGBA8 or BGRA8, in host or CUDA device memory.
  holoscan::Input<holoscan::schema::ImageT> image;
  // One message per image; no board.
  holoscan::Output<schema::FiducialDetectionsT> detections;

 private:
  ArucoDetectorConfig config_;
  std::unique_ptr<FrameDetector> detector_;
};

}  // namespace holoscan::fiducials
