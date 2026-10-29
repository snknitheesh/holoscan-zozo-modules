// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// FiducialOverlayOp: turns either detector's messages into HoloViz layers: markers' outlines
// and axes, chessboard corners and the board's axes on a transparent RGBA8 image, and labels.

#pragma once

#include <cstdint>
#include <optional>

#include <holoscan/core/domain/tensor.hpp>
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

// Bytes of text per label row after HoloViz's 16-byte run header.
inline constexpr std::uint32_t kOverlayLabelBytes = 24;
// Label rows: one per marker and one for the board.
inline constexpr std::uint32_t kOverlayLabelRows = kMaxMarkersPerFrame + 1;

struct FiducialOverlayConfig {
  int width{};  // Largest image, in pixels.
  int height{};
  // With them, markers and boards with a pose get 3D axes: a marker's half its length
  // (marker_length), a board's two squares long. Without, markers get outlines only.
  std::optional<CameraIntrinsics> intrinsics;
  double marker_length{};
};

class FiducialOverlayOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument for a size that is not positive or unusable intrinsics.
  explicit FiducialOverlayOp(FiducialOverlayConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<schema::FiducialDetectionsT> detections;
  // For a HoloViz RGBA8 image layer: the drawing over a transparent background.
  holoscan::Output<holoscan::schema::ImageT> overlay;
  // For a HoloViz text layer: "id N" per marker and the board's corner count; empty rows unused.
  holoscan::Output<holoscan::Tensor> labels;

 private:
  FiducialOverlayConfig config_;
};

}  // namespace holoscan::fiducials
