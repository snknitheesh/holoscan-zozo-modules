// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// SceneSourceOp: the demo's camera. Each tick it renders, on the GPU, a noisy
// frame of holoscan-denoise's low-light scene (a disc moving over dark
// patches, stripes and a checkerboard) and three exposure brackets of its HDR
// scene, all with one capture time; a fixed number, then idle.
//
// Its consumers cannot send a refused output again, so with `finished` the
// source keeps one frame in flight: it renders the next only once the checker
// has counted the previous one.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/denoise/synthetic_scene.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace denoise_demo {

struct SceneSourceConfig {
  holoscan::denoise::LowLightScene scene;
  holoscan::denoise::NoiseModel noise;
  holoscan::denoise::HdrScene hdr;
  std::uint32_t count{100};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  // Frames the checker has finished; none sends on every tick.
  std::shared_ptr<const std::atomic<std::uint32_t>> finished;
};

class SceneSourceOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument for an unusable configuration.
  explicit SceneSourceOp(SceneSourceConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // RGB8 [H, W, 3] in device memory; frame ids count from 1, frame f + 1 is
  // synthetic frame f.
  holoscan::Output<holoscan::schema::ImageT> noisy;
  // The HDR scene at kBracketExposures[0], [1] and [2], RGB8 [H, W, 3].
  holoscan::Output<holoscan::schema::ImageT> bracket0;
  holoscan::Output<holoscan::schema::ImageT> bracket1;
  holoscan::Output<holoscan::schema::ImageT> bracket2;

 private:
  SceneSourceConfig config_;
  std::uint32_t emitted_{};
};

}  // namespace denoise_demo
