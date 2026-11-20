// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ViewOp: the demo's picture. Takes the organized cloud with normals and the
// plane labels of a frame, both in device memory, and emits one RGB8 image
// twice as wide: the normals color-mapped on the left, the planes on the
// right, for a HoloViz image layer.

#pragma once

#include <cstdint>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace normals_planes_app {

struct ViewConfig {
  int width{};
  int height{};
};

class ViewOp final : public holoscan::Operator<> {
 public:
  explicit ViewOp(ViewConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // An organized cloud whose normals are float32 [H, W, 3].
  holoscan::Input<holoscan::messages::schema::PointCloudT> point_cloud;
  // uint8 [H, W].
  holoscan::Input<holoscan::Tensor> labels;
  // RGB8 [H, 2W, 3].
  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  ViewConfig config_;
};

}  // namespace normals_planes_app
