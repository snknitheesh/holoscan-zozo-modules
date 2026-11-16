// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// SceneSourceOp: frames of holoscan-clustering's synthetic scene as a depth
// camera sends them: float32 depth in meters, 0 where there is none, in
// device memory, a fixed number on a clock, then idle. A full queue or pool
// is backpressure: the same frame goes again on the next tick.

#pragma once

#include <chrono>
#include <cstdint>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace clustering_app {

struct SceneSourceConfig {
  int width{640};
  int height{480};
  std::uint32_t count{100};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
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

  // MONO32F [H, W] meters, frame f of synthetic_scene(width, height, f), with
  // frame id f + 1.
  holoscan::Output<holoscan::schema::ImageT> depth;

 private:
  SceneSourceConfig config_;
  std::uint32_t emitted_{};
};

}  // namespace clustering_app
