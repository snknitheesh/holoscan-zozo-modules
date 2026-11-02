// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// RenderCheckOp: receives HoloViz's offscreen renders of synthetic frames
// with the stub model's detections drawn over them, and checks each one: it
// must show some frame of the run with the boxes of that frame's red and
// blue rectangles in their class colors, opaque, at their left edges; and at
// each rectangle's center, the frame's color, or, with masks, the frame's
// color blended with the class color at the masks' opacity. The boxes move
// several pixels from frame to frame, so the check finds which frame a render
// shows from where they are.

#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace yolo26_app {

// Renders seen and those that showed a frame of the run correctly. Read by
// main() once the run has stopped.
struct RenderReport {
  std::atomic<std::uint32_t> renders{};
  std::uint32_t matched{};
};

struct RenderCheckConfig {
  std::shared_ptr<RenderReport> report;
  // The synthetic frames' size, which is the render's.
  int width{};
  int height{};
  // The run's frames, numbered from 1, among which a render's is found.
  std::uint32_t frames{};
  // The overlay draws masks, at this opacity.
  bool masks{false};
  std::uint8_t mask_alpha{};
  // Write each render as `image_dir`/render_NNNN.ppm, if set.
  std::filesystem::path image_dir;
};

class RenderCheckOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument for a missing report or an unusable size.
  explicit RenderCheckOp(RenderCheckConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // HoloViz's RGBA8 render, in GPU memory.
  holoscan::Input<holoscan::schema::ImageT> rendered;

 private:
  RenderCheckConfig config_;
  std::vector<std::uint8_t> pixels_;
};

}  // namespace yolo26_app
