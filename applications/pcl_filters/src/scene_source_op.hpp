// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// SceneSourceOp: frames of holoscan-pcl-filters' synthetic scene as a depth
// camera sends them: float32 depth in meters (0 where there is none) and an
// aligned RGB image, in device memory, a fixed number on a clock, then idle.
//
// The filters cannot send a refused output again, so a full queue must never
// meet them: with `finished`, the source keeps one frame in flight, sending
// the next only once the last consumer has counted the previous one there.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace pcl_filters_app {

struct SceneSourceConfig {
  int width{640};
  int height{480};
  std::uint32_t count{100};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  // Frames the last consumer has finished; none sends on every tick.
  std::shared_ptr<const std::atomic<std::uint32_t>> finished;
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const SceneSourceConfig& config);

class SceneSourceOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit SceneSourceOp(SceneSourceConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // MONO32F [H, W] meters, frame f of synthetic_scene(width, height, f).
  holoscan::Output<holoscan::schema::ImageT> depth;
  // RGB8 [H, W, 3].
  holoscan::Output<holoscan::schema::ImageT> color;

 private:
  SceneSourceConfig config_;
  std::uint32_t emitted_{};
  // The frame being sent keeps its capture time until both images are out.
  std::optional<holoscan::TimePoint> pending_capture_;
  bool depth_sent_{};
};

}  // namespace pcl_filters_app
