// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's camera: frames of the module's synthetic test video on a clock.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/time_point.hpp>

namespace codec_app {

struct TestSourceConfig {
  int width{1280};
  int height{720};
  // Frames to send; 0 sends until the run stops.
  std::uint32_t count{0};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  // When set, the frames finished downstream: at most `in_flight` frames are out at once, so no
  // queue on the way fills up. The decoder may hold a frame or two at the start of a stream.
  std::shared_ptr<const std::atomic<std::uint32_t>> done;
  std::uint32_t in_flight{4};
};

// Frame n of the test video as an RGB8 image in device memory, with header.device_sequence and
// the sample's frame id n + 1, and the tick's time as its capture time (also in the header).
class TestSourceOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument for an unusable config.
  explicit TestSourceOp(TestSourceConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  TestSourceConfig config_;
  std::uint32_t emitted_{};
  std::vector<std::uint8_t> pixels_;
  // The capture time of a frame the queue refused, sent again on the next tick.
  std::optional<holoscan::TimePoint> pending_capture_;
};

}  // namespace codec_app
