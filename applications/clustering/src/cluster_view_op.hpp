// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ClusterViewOp: draws a frame's clusters for HoloViz, on the GPU: each pixel
// in its cluster's color, dark gray for noise and pixels without depth, and
// each cluster's box projected into the camera that its camera_info input
// describes. Frames before the first camera info are not drawn.

#pragma once

#include <cstdint>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace clustering_app {

struct ClusterViewConfig {
  // The largest image drawn, which sizes the output.
  int max_width{};
  int max_height{};
};

class ClusterViewOp final : public holoscan::Operator<> {
 public:
  explicit ClusterViewOp(ClusterViewConfig config) : config_(config) {}

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // [H, W] int32 labels and [K, kSummaryFields] float32 clusters, in device
  // memory.
  holoscan::Input<holoscan::Tensor> labels;
  holoscan::Input<holoscan::Tensor> clusters;
  // Mandatory: the camera the cloud was deprojected with, for images of the
  // labels' size.
  holoscan::Input<holoscan::messages::schema::CameraInfoT> camera_info;
  // RGB8 [H, W, 3], in device memory.
  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  ClusterViewConfig config_;
  holoscan::messages::LatestCameraInfo<holoscan::messages::schema::CameraInfoT> info_;
};

}  // namespace clustering_app
