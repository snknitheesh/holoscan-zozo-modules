// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ViewOp: the demo's picture. Takes a frame's input depth image and the ray
// cast of the volume from the same pose, and emits one RGB8 image twice as
// wide: the depth color-mapped on the left, the ray cast shaded (with the
// volume's colors) on the right, for a HoloViz image layer.

#pragma once

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace tsdf_reconstruction_app {

struct ViewConfig {
  int width{};
  int height{};
  float depth_min{0.1F};
  float depth_max{3.0F};
};

class ViewOp final : public holoscan::Operator<> {
 public:
  explicit ViewOp(ViewConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // MONO16 millimeters or MONO32F meters [H, W], on the GPU.
  holoscan::Input<holoscan::schema::ImageT> depth;
  // The ray cast's organized cloud: its camera-frame normals (NaN where
  // nothing was hit) and RGB8 colors shade the picture.
  holoscan::Input<holoscan::messages::schema::PointCloudT> cloud;
  // RGB8 [H, 2W, 3].
  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  ViewConfig config_;
};

}  // namespace tsdf_reconstruction_app
