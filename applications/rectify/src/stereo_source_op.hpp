// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's stereo source: left and right MONO8 images of one capture time,
// rendered from the synthetic distorted pair or read from a recording, on a
// clock.

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/rectify/synthetic_stereo.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "recording.hpp"

namespace rectify_app {

struct StereoSourceConfig {
  // Synthetic frames of this pair, or, without it, the recording's.
  std::optional<holoscan::rectify::SyntheticStereo> scene;
  std::filesystem::path recording;
  int width{};
  int height{};
  std::uint32_t count{};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  // Emit CUDA device images instead of host images.
  bool device_output{};
  // Pairs every consumer has finished with; a pair goes out only once all
  // before it are done, when given. A lossless edge refuses a sample its
  // consumer has no room for, and the rectifier cannot send a refused pair
  // again, so the pipeline must hold no more than its queues: one pair in
  // flight. (An edge back from the reporters would carry this as data, but
  // the SDK refuses a graph with a cycle.)
  std::function<std::uint32_t()> done;
};

// `count` pairs on a clock (a recording's frames in order, repeated if it has
// fewer), each with capture time = activation time and frame id = n + 1;
// then idle.
class StereoSourceOp final : public holoscan::Operator<> {
 public:
  // Renders nothing yet; checks the recording's index.
  // @throws std::invalid_argument for an unusable config or recording.
  explicit StereoSourceOp(StereoSourceConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Output<holoscan::schema::ImageT> left;
  holoscan::Output<holoscan::schema::ImageT> right;

 private:
  // Sends `pixels` on `port`; false (and no error) when the consumer is busy.
  [[nodiscard]] holoscan::expected<bool, holoscan::Error> send(
      holoscan::Output<holoscan::schema::ImageT>& port, const std::vector<std::uint8_t>& pixels,
      holoscan::ExecutionContext& context, const holoscan::EmitOptions& emit);

  StereoSourceConfig config_;
  std::optional<holoscan::rectify::SyntheticView> left_view_;
  std::optional<holoscan::rectify::SyntheticView> right_view_;
  std::vector<RecordedFrame> frames_;
  std::uint32_t emitted_{};
  // The pair being sent: its images and stamp, and whether the left one went
  // out already when the right one met a busy consumer.
  std::vector<std::uint8_t> left_pixels_;
  std::vector<std::uint8_t> right_pixels_;
  std::optional<holoscan::EmitOptions> pending_;
  bool left_sent_{};
};

}  // namespace rectify_app
