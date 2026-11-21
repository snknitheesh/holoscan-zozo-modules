// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// RenderCheckOp: with `--display offscreen`, takes the frames HoloViz renders
// into CUDA memory and counts those that show both views: at least a fifth of
// the pixels of each half (the input depth on the left, the ray cast on the
// right) not black.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace tsdf_reconstruction_app {

struct RenderTally {
  std::uint32_t frames{};
  std::uint32_t showing{};  // Frames with both views on screen.
};

class RenderCheckOp final : public holoscan::Operator<> {
 public:
  explicit RenderCheckOp(std::shared_ptr<RenderTally> tally);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // RGBA8 [H, W, 4], as HoloViz's CUDA sink renders it.
  holoscan::Input<holoscan::schema::ImageT> frame;

 private:
  std::shared_ptr<RenderTally> tally_;
  std::vector<std::uint8_t> pixels_;
};

}  // namespace tsdf_reconstruction_app
