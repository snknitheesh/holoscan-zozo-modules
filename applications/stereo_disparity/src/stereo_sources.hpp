// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's stereo sources: the module's synthetic scene, and a recording
// played back from its folder. Both emit a fixed number of frames on a clock,
// then idle. Every part of a frame carries the same capture time and frame id
// (counted from 1). A consumer still busy makes an emit fail with
// kBackpressured; the source then sends the rest of the same frame on its
// next tick, so lossless edges lose nothing.
//
// Both also take the pair's camera info from a CameraInfoSourceOp, hold their
// first frame until it has come, and send it with every frame, first, on
// `stereo_camera_info`, as a rectifier does: a matcher that takes it with
// messages::CameraInfoTiming::kEveryFrame then has every frame's own, and
// never skips one.

#pragma once

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <optional>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/stereo_camera_info_schema_traits.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "recording.hpp"

namespace stereo_disparity_app {

// A source's camera info: the newest one received, sent again with each frame.
// Not thread-safe: one per source, used from compute().
class FrameCameraInfo {
 public:
  // Takes this activation's camera info, if there is a new one.
  // @return Whether a camera info has come; an error for one that is not valid.
  [[nodiscard]] holoscan::expected<bool, holoscan::Error> update(
      holoscan::Input<holoscan::messages::schema::StereoCameraInfoT>& port);
  // Sends the newest camera info for the frame `emit` describes, stamped with
  // its capture time. Call only after update() returned true.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> send(
      holoscan::Output<holoscan::messages::schema::StereoCameraInfoT>& port,
      const holoscan::EmitOptions& emit);
  // Forgets the camera info, for a new run.
  void reset() noexcept { latest_.reset(); }

 private:
  holoscan::messages::LatestCameraInfo<holoscan::messages::schema::StereoCameraInfoT> latest_;
  holoscan::messages::schema::StereoCameraInfoT message_;
};

struct SyntheticSourceConfig {
  int width{1280};
  int height{720};
  std::uint32_t count{300};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  // RGB8 images, or MONO8.
  bool color{true};
  // Images in CUDA device memory, or host memory.
  bool device_output{true};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const SyntheticSourceConfig& config);

// Frames 0, 1, ... of holoscan::stereo_disparity::make_scene().
class SyntheticStereoSourceOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit SyntheticStereoSourceOp(SyntheticSourceConfig config);
  ~SyntheticStereoSourceOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::messages::schema::StereoCameraInfoT> camera_info;
  holoscan::Output<holoscan::messages::schema::StereoCameraInfoT> stereo_camera_info;
  holoscan::Output<holoscan::schema::ImageT> left;
  holoscan::Output<holoscan::schema::ImageT> right;

 private:
  void release() noexcept;

  SyntheticSourceConfig config_;
  FrameCameraInfo frame_camera_info_;
  std::uint32_t emitted_{};
  std::optional<holoscan::TimePoint> pending_capture_;
  std::uint32_t sent_{};
  // The rendered pair for host output, and the stream used without a runtime one.
  void* scratch_{};
  cudaStream_t own_stream_{};
};

struct RecordingSourceConfig {
  Recording recording;
  // Frames to send; 0 sends every frame once. With `loop`, the recording
  // starts over until `count` frames are out.
  std::uint32_t count{0};
  bool loop{false};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  // Also send each frame's depth image (MONO16, the recording's units) on
  // `depth`; needs depth aligned with the left image.
  bool depth{false};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const RecordingSourceConfig& config);

// A recording's frames, MONO8 left and right images in host memory, in order.
// A file that cannot be read, or whose size is not the camera info's, fails
// the run.
class RecordingSourceOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit RecordingSourceOp(RecordingSourceConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // Frames this source sends.
  [[nodiscard]] std::uint32_t total() const noexcept { return total_; }

  holoscan::Input<holoscan::messages::schema::StereoCameraInfoT> camera_info;
  holoscan::Output<holoscan::messages::schema::StereoCameraInfoT> stereo_camera_info;
  holoscan::Output<holoscan::schema::ImageT> left;
  holoscan::Output<holoscan::schema::ImageT> right;
  holoscan::Output<holoscan::schema::ImageT> depth;

 private:
  RecordingSourceConfig config_;
  FrameCameraInfo frame_camera_info_;
  std::uint32_t total_{};
  std::uint32_t emitted_{};
  std::optional<holoscan::TimePoint> pending_capture_;
  std::uint32_t sent_{};
  // The frame being sent, read once.
  std::optional<Pgm> left_pixels_;
  std::optional<Pgm> right_pixels_;
  std::optional<Pgm> depth_pixels_;
};

}  // namespace stereo_disparity_app
