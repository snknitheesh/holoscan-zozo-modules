// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// CharucoDetectorOp: an image in, one view of a ChArUco board out: its markers, its chessboard
// corners with their ids, sub-pixel image positions and board positions, a quality summary and,
// when the camera's intrinsics are known, the board's pose. That is the observation a camera
// calibration takes.

#pragma once

#include <memory>
#include <optional>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/fiducials/charuco.hpp>
#include <holoscan/fiducials/fiducial_detections_schema_traits.hpp>
#include <holoscan/fiducials/types.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::fiducials {

class FrameDetector;

struct CharucoDetectorConfig {
  // kAuto lets the operator choose at start: see resolve_auto().
  Backend backend{Backend::kCuda};
  CharucoBoardConfig board;
  // The detector's settings; the dictionary is the board's, whatever options.dictionary says.
  DetectorOptions options;
  CharucoOptions charuco;
  // With them, the board's pose.
  std::optional<CameraIntrinsics> intrinsics;
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const CharucoDetectorConfig& config);

class CharucoDetectorOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit CharucoDetectorOp(CharucoDetectorConfig config);
  ~CharucoDetectorOp();
  CharucoDetectorOp(const CharucoDetectorOp&) = delete;
  CharucoDetectorOp& operator=(const CharucoDetectorOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // As ArucoDetectorOp::resolved_backend().
  [[nodiscard]] Backend resolved_backend() const noexcept;

  // MONO8, RGB8, BGR8, RGBA8 or BGRA8, in host or CUDA device memory.
  holoscan::Input<holoscan::schema::ImageT> image;
  // One message per image, with the board's view (even when nothing of it was found).
  holoscan::Output<schema::FiducialDetectionsT> detections;

 private:
  CharucoDetectorConfig config_;
  CharucoBoard board_;
  std::unique_ptr<FrameDetector> detector_;
};

}  // namespace holoscan::fiducials
