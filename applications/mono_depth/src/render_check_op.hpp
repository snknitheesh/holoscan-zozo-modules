// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// RenderCheckOp: takes the frames HoloViz renders offscreen (into CUDA
// memory, with a CudaSinkDesc), and checks each against the frames expected,
// when they are known: with the stub model and the synthetic scene, the image
// on the left and its colorized depth on the right are known exactly. HoloViz
// gives its frames no frame id, and drops frames when it or the check falls
// behind, so each must match the frame expected for the id last matched or a
// later one (up to `last_id`), both halves at once: a depth shown beside
// another frame's image matches none and fails the check. It can also write the newest frame as a
// PPM file, a screenshot of the display without a display. Each frame is copied to the host, which
// waits for the copy (it says so once).

#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace mono_depth_app {

// What the check saw. Written during the run and read by main() after it has
// stopped; only `frames` may be read while it runs.
struct RenderCheck {
  std::atomic<std::uint32_t> frames{};
  // Frames checked, and those with more than kMaxMismatchedFraction of their
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
  // The RGBA8 frame expected for a frame id, width x height, row by row; unset
  // for none.
  std::function<std::vector<std::uint8_t>(std::uint64_t frame_id)> expected;
  // The last frame id there is.
  std::uint64_t last_id{};
  // Write each frame here as a PPM file, so that the newest remains.
  std::filesystem::path write;
};

class RenderCheckOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if the check is missing or the size is not
  // positive.
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
  bool copy_logged_{};
  // The first frame id a rendered frame may show, and the frame expected for
  // the id last compared, computed once.
  std::uint64_t next_id_{1};
  std::map<std::uint64_t, std::vector<std::uint8_t>> expected_;
};

}  // namespace mono_depth_app
