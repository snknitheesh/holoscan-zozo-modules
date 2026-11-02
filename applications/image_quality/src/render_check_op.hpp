// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// RenderCheckOp: the offscreen mode's sink. Reads each frame HoloViz rendered into
// device memory and checks that it shows the scene, the text, and the overlay's
// magenta where the smudge is; keeps the frame with the most of it to save.

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
#include <holoscan/image_quality/synthetic.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace image_quality_app {

// What the rendered frames showed. Written during the run, read after it.
struct RenderReport {
  std::uint64_t frames{};
  // Frames with the scene (varied pixels) and with text (white pixels).
  std::uint64_t frames_with_scene{};
  std::uint64_t frames_with_text{};
  // Frames whose smudge area is mostly magenta while the rest is not.
  std::uint64_t frames_with_overlay{};
  // The frame with the most magenta in the smudge area, RGBA8.
  int width{};
  int height{};
  std::vector<std::uint8_t> best;
  double best_fraction{};
};

struct RenderCheckConfig {
  std::shared_ptr<RenderReport> report;
  holoscan::image_quality::Smudge smudge;
};

class RenderCheckOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `report` is missing.
  explicit RenderCheckOp(RenderCheckConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // HoloViz's rendered frames: RGBA8 in CUDA device memory.
  holoscan::Input<holoscan::schema::ImageT> frame;

 private:
  RenderCheckConfig config_;
  std::vector<std::uint8_t> pixels_;
};

// Writes RGBA8 `pixels` as a binary PPM (RGB).
// @throws std::runtime_error if the file cannot be written.
void write_ppm(const char* path, const std::vector<std::uint8_t>& pixels, int width, int height);

}  // namespace image_quality_app
