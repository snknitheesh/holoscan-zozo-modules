// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's image sources: frames from synthetic scenes of markers or a ChArUco board, with
// ground truth, or from a recording's color images; and a V4L2 camera.
//
// The frame source takes the camera's calibration from a CameraInfoSourceOp, holds its first
// frame until it has come, and sends it with every frame, first, at the frame's capture time,
// on `image_camera_info`: a detector that takes it with messages::CameraInfoTiming::kEveryFrame
// then has every frame's own, and never skips one. A consumer still busy makes an emit fail
// with kBackpressured, or an allocation with kResourceExhausted; the source then sends the rest
// of the same frame on its next tick, so lossless edges lose nothing.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
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
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_schema_traits.hpp>
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

// Whether an emit or allocation failed only because consumers have no room yet.
[[nodiscard]] bool backpressured(const holoscan::Error& error) noexcept;

struct FrameSourceConfig {
  int width{1280};
  int height{720};
  std::uint32_t count{100};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  bool device_output{false};  // Frames in CUDA device memory instead of host memory.
  // Frame n's packed RGB8 pixels, width x height x 3.
  std::function<std::vector<std::uint8_t>(std::uint32_t n)> render;
  // Frame n's header.device_sequence (never 0, which means unnumbered).
  std::function<std::uint64_t(std::uint32_t n)> sequence;  // NOLINT(readability/casting)
};

// The frames of make_scene(kind, n, width, height), numbered n + 1.
// @throws std::invalid_argument for an unusable kind or size.
[[nodiscard]] FrameSourceConfig scene_source_config(const SceneKind& kind, int width, int height);

// The first `count` color frames of the recording in `clip` (all of them for 0), numbered by
// the camera, and the recording's frame size.
// @throws std::invalid_argument for a recording that cannot be read.
[[nodiscard]] FrameSourceConfig recording_source_config(const std::filesystem::path& clip,
                                                        std::uint32_t count);

// `count` RGB8 frames on a clock, once the camera info has come, each with the newest camera
// info, then idle.
class FrameSourceOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument for an unusable config.
  explicit FrameSourceOp(FrameSourceConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // From a CameraInfoSourceOp, at its own rate.
  holoscan::Input<holoscan::messages::schema::CameraInfoT> camera_info;
  holoscan::Output<holoscan::schema::ImageT> image;
  // The newest camera info, sent with every image at its capture time.
  holoscan::Output<holoscan::messages::schema::CameraInfoT> image_camera_info;

 private:
  FrameSourceConfig config_;
  holoscan::messages::LatestCameraInfo<holoscan::messages::schema::CameraInfoT> info_;
  holoscan::messages::schema::CameraInfoT message_;
  std::uint32_t emitted_{};
  // The frame being sent: its capture time, pixels, and the parts already sent.
  std::optional<holoscan::TimePoint> pending_;
  std::vector<std::uint8_t> pixels_;
  std::uint32_t sent_{};
};

struct CameraSourceConfig {
  std::string device{"/dev/video0"};
  int width{1280};
  int height{720};
  // Frames to emit, after which the run completes once the last ones have had time to pass
  // through; 0 runs until the application is stopped.
  std::uint32_t count{};
};

// Checks that `config.device` offers YUYV at the configured size.
// @throws std::invalid_argument naming what is wrong.
void check_camera(const CameraSourceConfig& config);

// RGB8 frames in host memory from a V4L2 camera offering YUYV, stamped with their capture time
// and numbered by the camera. Only the camera's newest frame goes out; connect it with
// drop-oldest edges.
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
  std::uint32_t idle_ticks_{};  // Ticks since the count was reached.
  bool short_frame_logged_{};
};

}  // namespace fiducial_app
