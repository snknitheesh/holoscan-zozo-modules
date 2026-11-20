// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ArucoDetectorOp: an image in, the dictionary's markers it holds out, on the configured
// backend; with estimate_pose, each with its pose, from the camera info that comes on the
// mandatory `camera_info` input (holoscan-messages' contract).

#pragma once

#include <memory>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/fiducials/fiducial_detections_schema_traits.hpp>
#include <holoscan/fiducials/types.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_schema_traits.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::fiducials {

class FrameDetector;

struct ArucoDetectorConfig {
  // kAuto lets the operator choose at start: see resolve_auto().
  Backend backend{Backend::kCuda};
  DetectorOptions options;
  // Give each marker its pose: declares the `camera_info` input, which must then be connected,
  // and needs marker_length.
  bool estimate_pose{false};
  double marker_length{};  // Meters across a marker's black border, > 0 with estimate_pose.
  // With estimate_pose, how the camera info arrives: kLatest (the default) at any rate, as from
  // CameraInfoSourceOp; kEveryFrame with every image, with its capture time.
  holoscan::messages::CameraInfoInputOptions camera_info_input;
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
  // Declared with estimate_pose only, and then mandatory: the camera the images come from, of
  // their size (another size fails the run), any of its lens models. Images before the first
  // camera info are skipped (kLatest).
  holoscan::Input<holoscan::messages::schema::CameraInfoT> camera_info;
  // One message per image; no board.
  holoscan::Output<schema::FiducialDetectionsT> detections;

 private:
  ArucoDetectorConfig config_;
  std::unique_ptr<FrameDetector> detector_;
  holoscan::messages::LatestCameraInfo<holoscan::messages::schema::CameraInfoT> camera_;
};

}  // namespace holoscan::fiducials
