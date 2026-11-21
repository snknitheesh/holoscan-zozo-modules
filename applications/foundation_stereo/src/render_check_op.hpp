// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// RenderCheckOp: takes the frames HoloViz renders offscreen (into CUDA
// memory, with a CudaSinkDesc) and checks what the window would show: the
// left image in the left half, which is not black, and the colorized
// disparity in the right half, most of which is colored. It can also write
// the newest frame as a PPM file, a screenshot of the display without a
// display.

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

namespace foundation_stereo_app {

// What the check saw. Written during the run and read by main() after it has
// stopped.
struct RenderCheck {
  std::uint32_t frames{};
  // Frames whose halves did not look as they should.
  std::uint32_t failed{};
  // The last frame's share of colored pixels in the right half, and the mean
  // brightness of the left half (0..255).
  double colored{};
  double left_brightness{};
  bool written{};
};

// A pixel is colored when its channels differ by more than this.
inline constexpr int kColoredSpread = 40;
inline constexpr double kMinColoredShare = 0.5;
// Mean brightness above black: the infrared images of a recording are dark.
inline constexpr double kMinLeftBrightness = 5.0;

struct RenderCheckConfig {
  std::shared_ptr<RenderCheck> check;
  int width{};
  int height{};
  // Whether the left half shows the left image; otherwise the disparity fills
  // the frame.
  bool left_image{true};
  // Write each frame here as a PPM file, so that the newest remains.
  std::filesystem::path write;
};

class RenderCheckOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if the check is missing or the size is not positive.
  explicit RenderCheckOp(RenderCheckConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // HolovizOp's output: RGBA8 [height, width, 4] in GPU memory.
  holoscan::Input<holoscan::schema::ImageT> frame;

 private:
  RenderCheckConfig config_;
  std::vector<std::uint8_t> pixels_;
};

}  // namespace foundation_stereo_app
