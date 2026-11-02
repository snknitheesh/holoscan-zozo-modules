// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// SceneCameraOp: a synthetic RGB-D camera circling holoscan-tsdf's tabletop
// scene, as a tracked camera sends it: depth (uint16 millimeters or float32
// meters, 0 where there is none) with noise, aligned RGB color, both in
// device memory, and the camera's known pose (world_from_camera, float64
// [4, 4], host memory), all with the same capture time and frame id. A fixed
// number of frames on a clock, then idle.
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
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace tsdf_reconstruction_app {

struct SceneCameraConfig {
  int width{640};
  int height{480};
  std::uint32_t count{120};
  // Frames per orbit: frame f is orbit_pose(kTabletop, f, orbit).
  std::uint32_t orbit{120};
  bool millimeters{true};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  // Frames the last consumer has finished; none sends on every tick.
  std::shared_ptr<const std::atomic<std::uint32_t>> finished;
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const SceneCameraConfig& config);

class SceneCameraOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit SceneCameraOp(SceneCameraConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // MONO16 millimeters or MONO32F meters, [H, W].
  holoscan::Output<holoscan::schema::ImageT> depth;
  // RGB8 [H, W, 3].
  holoscan::Output<holoscan::schema::ImageT> color;
  // float64 [4, 4] world_from_camera, host memory.
  holoscan::Output<holoscan::Tensor> pose;

 private:
  SceneCameraConfig config_;
  std::uint32_t emitted_{};
};

}  // namespace tsdf_reconstruction_app
