// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's stereo sources: the module's synthetic scene, and a recording
// played back from its folder. Both emit a fixed number of frames on a clock,
// then idle. Every part of a frame carries the same capture time and frame id
// (counted from 1). A consumer still busy makes an emit fail with
// kBackpressured; the source then sends the rest of the same frame on its
// next tick, so lossless edges lose nothing.

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
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/stereo_disparity/geometry_message.hpp>

#include "recording.hpp"

namespace stereo_disparity_app {

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

  holoscan::Output<holoscan::schema::ImageT> left;
  holoscan::Output<holoscan::schema::ImageT> right;

 private:
  void release() noexcept;

  SyntheticSourceConfig config_;
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
  // `depth`, and the pair's geometry on `geometry`.
  bool depth{false};
  bool geometry{false};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const RecordingSourceConfig& config);

// The pair's geometry from a recording's calibration.
[[nodiscard]] holoscan::stereo_disparity::StereoGeometry recording_geometry(
    const RecordingCalibration& calibration);

// A recording's frames, MONO8 left and right images in host memory, in order.
// A file that cannot be read, or whose size is not the calibration's, fails
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

  holoscan::Output<holoscan::schema::ImageT> left;
  holoscan::Output<holoscan::schema::ImageT> right;
  holoscan::Output<holoscan::schema::ImageT> depth;
  holoscan::Output<holoscan::stereo_disparity::schema::StereoGeometryT> geometry;

 private:
  RecordingSourceConfig config_;
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
