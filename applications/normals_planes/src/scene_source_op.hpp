// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// SceneSourceOp: frames of holoscan-normals' synthetic room as a depth camera
// sends them: float32 depth in meters (0 where there is none), in device
// memory, with a header naming the camera's frame, a fixed number on a clock,
// then idle. The first tick sends nothing, so that the camera info, which
// CameraInfoSourceOp sends when the run starts, usually comes before the
// first frame; a frame that comes before it gives no cloud.
//
// A frame the consumers cannot take yet (a refused emit, or every slot of the
// output's pool still held) is sent again on the next tick. With `finished`,
// the source keeps at most two frames in flight, sending the next only once
// the last consumer has finished the one before the previous: the operators
// in the middle cannot send a refused output again, so their queues (two
// deep) must never fill.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace normals_planes_app {

struct SceneSourceConfig {
  int width{640};
  int height{480};
  std::uint32_t count{100};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  // The frame id (from 1) of the newest frame the last consumer has
  // finished, or of a later one; none sends on every tick.
  std::shared_ptr<const std::atomic<std::uint32_t>> finished;
  // The camera's frame, in each frame's header.
  std::string frame_id{"synthetic_camera"};
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

 private:
  SceneSourceConfig config_;
  std::uint32_t emitted_{};
  bool delayed_{};
  // The capture time of the frame being sent, kept until it is out.
  std::optional<holoscan::TimePoint> capture_;
};

}  // namespace normals_planes_app
