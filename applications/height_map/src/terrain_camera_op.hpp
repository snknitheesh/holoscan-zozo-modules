// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// TerrainCameraOp: a synthetic depth camera moving over holoscan-height-map's
// course terrain, as a robot's camera would send it: float32 depth in meters
// (0 where there is none), in device memory, and the camera's pose, a
// float64 [4, 4] row-major transform from its optical frame into the map
// frame, in host memory, both with the same capture time; a fixed number of
// frames on a clock, then idle.
//
// The operators cannot send a refused output again, so a full queue must
// never meet them: with `finished`, the camera keeps one frame in flight,
// sending the next only once the last consumer has counted the previous one.

#pragma once

#include <atomic>
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
#include <holoscan/height_map/synthetic_terrain.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace height_map_app {

// The camera of frame `frame`: 1.1 m up, tilted 35 degrees down, driving
// along +x at 1 cm a frame from x = 0.5 m and swaying +-8 degrees in yaw, with
// 1 mm + 0.2 % of depth squared of depth noise.
[[nodiscard]] holoscan::height_map::TerrainCamera camera_for_frame(int width, int height,
                                                                   std::uint32_t frame);

struct TerrainCameraConfig {
  int width{640};
  int height{480};
  std::uint32_t count{300};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  // Frames the last consumer has finished; none sends on every tick.
  std::shared_ptr<const std::atomic<std::uint32_t>> finished;
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const TerrainCameraConfig& config);

class TerrainCameraOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit TerrainCameraOp(TerrainCameraConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // MONO32F [H, W] meters.
  holoscan::Output<holoscan::schema::ImageT> depth;
  // float64 [4, 4], optical frame to map frame.
  holoscan::Output<holoscan::Tensor> pose;

 private:
  TerrainCameraConfig config_;
  holoscan::height_map::Terrain terrain_;
  std::uint32_t emitted_{};
};

}  // namespace height_map_app
