// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ViewOp: the demo's picture. Takes a raw frame and the ISP's image of it and
// emits one RGB8 image twice as wide: the raw mosaic as gray on the left, as a
// sensor delivers it, the processed image on the right, for a HoloViz image layer.

#pragma once

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/isp/isp_params.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace isp_demo {

struct ViewConfig {
  int width{};
  int height{};
  holoscan::isp::RawFormat format{holoscan::isp::RawFormat::kUint16};
  int bit_depth{12};
  float black_level{};
};

class ViewOp final : public holoscan::Operator<> {
 public:
  explicit ViewOp(ViewConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // The raw frame, as RawSourceOp sends it, in device memory.
  holoscan::Input<holoscan::schema::ImageT> raw;
  // RGB8 [H, W, 3] in device memory.
  holoscan::Input<holoscan::schema::ImageT> image;
  // RGB8 [H, 2W, 3].
  holoscan::Output<holoscan::schema::ImageT> view;

 private:
  ViewConfig config_;
};

}  // namespace isp_demo
