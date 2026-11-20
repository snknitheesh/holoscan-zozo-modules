// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// TerrainCameraOp: a synthetic depth camera moving over holoscan-height-map's
// course terrain, as a robot's camera would send it: float32 depth in meters
// (0 where there is none), in device memory, with a header (capture time,
// kCameraFrame, frame number), and the camera's pose, a float64 [4, 4]
// row-major transform from its optical frame into the map frame, in host
// memory, both with the same capture time; a fixed number of frames on a
// clock, then idle. Its calibration is camera_info_of(camera_for_frame()),
// which the demo writes into a camera_info file for CameraInfoSourceOp.
//
// The first tick sends nothing, so that the camera info, which
// CameraInfoSourceOp sends when the run starts, comes before the first frame.
// A frame or pose a consumer refuses (backpressure, or no free slot in the
// output's pool) goes again on the next tick, with the frame's own capture
// time; what went is not sent again. The operators after it cannot send a
// refused output again, so in the demo a full queue must never meet them:
// with `finished`, a test sink's pacing, the camera keeps one frame in
// flight, sending the next only once the last consumer has counted the
// previous one.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>

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

// The camera's optical frame: its depth images' and its camera info's frame id.
inline constexpr const char* kCameraFrame = "terrain_camera_optical";

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
  // Renders and sends the depth image of `camera` with `emit`.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> send_depth(
      holoscan::ExecutionContext& context, const holoscan::height_map::TerrainCamera& camera,
      const holoscan::EmitOptions& emit);

  TerrainCameraConfig config_;
  holoscan::height_map::Terrain terrain_;
  std::uint32_t emitted_{};
  bool delayed_{};
  // The frame being sent: its capture time, and whether its depth went.
  std::optional<holoscan::TimePoint> capture_;
  bool depth_sent_{};
};

}  // namespace height_map_app
