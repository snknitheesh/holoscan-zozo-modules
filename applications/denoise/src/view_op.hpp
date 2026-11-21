// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ViewOp: the demo's picture, for one HoloViz image layer. One RGB8 image of
// 2W x 3H/2: on top the noisy frame and the denoised frame side by side; below
// them, at half size, the three brackets and the fused image.

#pragma once

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace denoise_demo {

struct ViewConfig {
  int width{};  // Of each input, even.
  int height{};
};

class ViewOp final : public holoscan::Operator<> {
 public:
  explicit ViewOp(ViewConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // RGB8 [H, W, 3] in device memory, of one capture time.
  holoscan::Input<holoscan::schema::ImageT> noisy;
  holoscan::Input<holoscan::schema::ImageT> denoised;
  holoscan::Input<holoscan::schema::ImageT> bracket0;
  holoscan::Input<holoscan::schema::ImageT> bracket1;
  holoscan::Input<holoscan::schema::ImageT> bracket2;
  holoscan::Input<holoscan::schema::ImageT> fused;
  // RGB8 [3H/2, 2W, 3].
  holoscan::Output<holoscan::schema::ImageT> view;

 private:
  ViewConfig config_;
};

}  // namespace denoise_demo
