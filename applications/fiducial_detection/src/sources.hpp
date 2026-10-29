// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's image sources: synthetic scenes of markers or a ChArUco board, with ground truth,
// and a V4L2 camera.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/fiducials/charuco.hpp>
#include <holoscan/fiducials/synthetic_scene.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace fiducial_app {

// What the synthetic frames show.
struct SceneKind {
  bool board{};                           // A ChArUco board, or loose markers.
  std::string dictionary{"DICT_4X4_50"};  // The markers' (unused for a board).
  holoscan::fiducials::CharucoBoardConfig board_config;
};

// Frame `frame`'s scene: make_marker_scene() or make_board_scene() at width x height.
[[nodiscard]] holoscan::fiducials::SyntheticScene make_scene(const SceneKind& kind,
                                                             std::uint32_t frame, int width,
                                                             int height);

struct SceneSourceConfig {
  SceneKind kind;
  int width{1280};
  int height{720};
  std::uint32_t count{100};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  bool device_output{false};  // Frames in CUDA device memory instead of host memory.
};

// `count` RGB8 renderings of make_scene(kind, n, ...) on a clock, each with
// header.device_sequence n + 1 (zero would mean unnumbered), then idle.
class SceneSourceOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument for an unusable config.
  explicit SceneSourceOp(SceneSourceConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  SceneSourceConfig config_;
  std::uint32_t emitted_{};
};

struct CameraSourceConfig {
  std::string device{"/dev/video0"};
  int width{1280};
  int height{720};
  std::uint32_t count{};  // Frames to emit; 0 runs until the application is stopped.
};

// Checks that `config.device` offers YUYV at the configured size.
// @throws std::invalid_argument naming what is wrong.
void check_camera(const CameraSourceConfig& config);

// RGB8 frames in host memory from a V4L2 camera offering YUYV, stamped with their capture time.
// Only the camera's newest frame goes out; connect it with drop-oldest edges.
class CameraSourceOp final : public holoscan::Operator<> {
 public:
  // The device is opened in start() and closed in stop(): the runtime may construct an
  // operator more than once, and a camera serves one owner at a time.
  explicit CameraSourceOp(CameraSourceConfig config) : config_(std::move(config)) {}
  ~CameraSourceOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  struct Mapping {
    void* data{};
    std::size_t length{};
  };
  void release() noexcept;

  CameraSourceConfig config_;
  int fd_{-1};
  std::size_t pitch_{};
  std::vector<Mapping> buffers_;
  std::uint32_t emitted_{};
};

}  // namespace fiducial_app
