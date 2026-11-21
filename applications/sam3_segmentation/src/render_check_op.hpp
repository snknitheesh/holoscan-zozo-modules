// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// RenderCheckOp: takes the frames HoloViz renders offscreen (into CUDA
// memory, with a CudaSinkDesc), and checks each against the frame expected,
// when there is one: with the stub model and a still synthetic scene, the
// image under its instances' overlay is known, but for the labels' text.
// It can also write the newest frame as a PPM file, a screenshot of the
// display without a display.

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

namespace sam3_app {

// What the check saw. Written during the run and read by main() after it has
// stopped; only `frames` may be read while it runs.
struct RenderCheck {
  std::atomic<std::uint32_t> frames{};
  // Frames checked, and those with more than the allowed fraction of their
  // pixels off by more than kChannelTolerance in a channel.
  std::uint32_t checked{};
  std::uint32_t failed{};
  // The largest fraction of mismatched pixels in a frame.
  double worst_fraction{};
  bool written{};
};

inline constexpr int kChannelTolerance = 8;
inline constexpr double kMaxMismatchedFraction = 0.01;

struct RenderCheckConfig {
  std::shared_ptr<RenderCheck> check;
  int width{};
  int height{};
  // The RGBA8 frame expected, width x height, row by row; empty for none.
  std::vector<std::uint8_t> expected;
  // The largest fraction of a frame's pixels that may differ from it: the
  // labels' text, which it leaves out, covers a few.
  double max_mismatched_fraction{kMaxMismatchedFraction};
  // Write each frame here as a PPM file, so that the newest remains.
  std::filesystem::path write;
};

class RenderCheckOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if the check is missing, the size is not
  // positive, or the expected frame does not have it.
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

}  // namespace sam3_app
