// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ImageQualityOp: an image in, its quality metrics and pass/fail flags out, per
// frame, computed on the GPU: blur and focus (globally and per tile), exposure,
// noise, frozen frames, lens obstruction over time, and color cast. Optionally
// it also emits the per-tile values and a color-coded tile overlay for HoloViz.

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/image_quality/quality.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::image_quality {

struct ImageQualityConfig {
  StatsOptions stats;
  QualityThresholds thresholds;
  // Declare the `tiles` output.
  bool tiles_output{false};
  // Declare the `overlay` output, with `overlay_cell` pixels per tile side.
  bool overlay_output{false};
  int overlay_cell{16};
  // Least time between two warnings about failed frames; zero turns them off.
  std::chrono::milliseconds log_interval{5000};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const ImageQualityConfig& config);

// The overlay's colors, 0xAABBGGRR as HoloViz reads RGBA8 bytes: obstructed
// tiles magenta, blurred tiles red, low-texture tiles faint yellow, others clear.
inline constexpr std::uint32_t kOverlayObstructed = 0x8CFF00FFU;
inline constexpr std::uint32_t kOverlayBlurred = 0x782828FFU;
inline constexpr std::uint32_t kOverlayLowTexture = 0x3C00E6FFU;

// The overlay of `report`: tiles_y * cell x tiles_x * cell RGBA8 pixels, each
// tile filled with its color but its last row and column, which stay clear so
// the grid shows.
void draw_overlay(const QualityReport& report, int cell, std::uint8_t* rgba);

class ImageQualityOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit ImageQualityOp(ImageQualityConfig config);
  ~ImageQualityOp() override;
  ImageQualityOp(const ImageQualityOp&) = delete;
  ImageQualityOp& operator=(const ImageQualityOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // MONO8, RGB8, BGR8, RGBA8 or BGRA8, [H, W] or [H, W, channels], pixels packed
  // within each row, in host or CUDA device memory. Host images are copied to the
  // GPU, which is logged once.
  holoscan::Input<holoscan::schema::ImageT> image;
  // float32 [kMetricCount] in host memory, in Metric order.
  holoscan::Output<holoscan::Tensor> metrics;
  // uint32 [1] in host memory: the Flag bits; zero passes.
  holoscan::Output<holoscan::Tensor> flags;
  // With `tiles_output`: float32 [tiles_y, tiles_x, kTileChannelCount] in host
  // memory, in TileChannel order.
  holoscan::Output<holoscan::Tensor> tiles;
  // With `overlay_output`: RGBA8 [tiles_y * cell, tiles_x * cell, 4] in host
  // memory, for a HoloViz image layer over the image.
  holoscan::Output<holoscan::schema::ImageT> overlay;

 private:
  // The analyzer, a CUDA stream and the logging state.
  struct State;

  ImageQualityConfig config_;
  std::unique_ptr<State> state_;
};

}  // namespace holoscan::image_quality
