// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// StereoCameraOp: a stereo camera as a Holoscan source. Each frame goes out as
// a left and a right image, optionally an RGB image and a depth image, and the
// camera's calibration, all with the frame's capture time and frame id, so a
// consumer can align them. The frames come from a backend (source.hpp): a
// RealSense, a side-by-side V4L2 camera, a recording, or any StereoSource.
//
// A live camera's frames are not worth waiting for: connect it with
// QueuePolicy::kDropOldest edges, so that a busy consumer finds the newest frame
// when it is done. A recording plays losslessly: on a blocking edge, a frame a
// consumer has no room for is sent again on the next tick, port by port.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
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
#include <holoscan/stereo_camera/calibration_message.hpp>
#include <holoscan/stereo_camera/source.hpp>

namespace holoscan::stereo_camera {

// Where the images go out.
enum class OutputMemory : std::uint8_t {
  kHost,    // Host memory, as the camera gives them.
  kDevice,  // CUDA device memory, copied on the operator's stream.
};

// What the operator has sent, for the application to read from any thread. The
// runtime may construct an operator more than once, so read these, not the
// operator you constructed.
struct StereoCameraStats {
  // Frames sent in full.
  std::atomic<std::uint64_t> sent{0};
  // Frames of a live camera dropped because a consumer refused them on a
  // blocking edge.
  std::atomic<std::uint64_t> dropped{0};
};

struct StereoCameraConfig {
  // The backend and its settings.
  SourceConfig source;
  // A source of your own instead, for a camera the module has no backend for.
  // Called once in the constructor; the source is opened in start().
  std::function<std::unique_ptr<StereoSource>()> make_source;
  // Frames to send, after which the operator idles; 0 sends until a recording
  // ends, or until the run is stopped.
  std::uint64_t count{0};
  // A recording's pace: one frame per period; zero takes the recorded rate (the
  // median time between its frames). A live camera is polled every 2 ms instead.
  std::chrono::nanoseconds period{0};
  OutputMemory memory{OutputMemory::kHost};
  // Declare the calibration output. Without a consumer for it, turn it off: an
  // emit into an unconnected output fails.
  bool calibration{true};
  // The images' header frame ids are this with "_left", "_right", "_rgb" and
  // "_depth" appended.
  std::string frame_id{"stereo_camera"};
  // Counts what was sent, when given.
  std::shared_ptr<StereoCameraStats> stats;
};

// Checks what the operator needs of `config` beyond the source's own checks.
// @throws std::invalid_argument naming the first unusable field.
void validate_config(const StereoCameraConfig& config);

class StereoCameraOp final : public holoscan::Operator<> {
 public:
  // Makes the source (validating its configuration) but opens nothing: the
  // runtime may construct an operator more than once, and a camera serves one
  // owner at a time. The device is opened in start() and closed in stop().
  // @throws std::invalid_argument if `config` or the source's is not valid.
  explicit StereoCameraOp(StereoCameraConfig config);
  ~StereoCameraOp() override;
  StereoCameraOp(const StereoCameraOp&) = delete;
  StereoCameraOp& operator=(const StereoCameraOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  // @throws holoscan::RuntimeError if the source does not open.
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // The streams the operator sends, known from construction.
  [[nodiscard]] const SourceLayout& layout() const noexcept { return layout_; }

  // MONO8 or RGB8, [H, W] or [H, W, 3], rows packed.
  holoscan::Output<holoscan::schema::ImageT> left;
  holoscan::Output<holoscan::schema::ImageT> right;
  // With an RGB image: RGB8, [H, W, 3].
  holoscan::Output<holoscan::schema::ImageT> rgb;
  // With a depth image: MONO16, [H, W], in units of the calibration's depth_scale.
  holoscan::Output<holoscan::schema::ImageT> depth;
  // With `calibration`: the calibration, with the source's frame number and timestamp.
  holoscan::Output<schema::StereoCalibrationT> calibration;

 private:
  // A frame being sent, kept until every port has taken it.
  struct Pending;

  [[nodiscard]] holoscan::expected<void, holoscan::Error> send(Pending& frame,
                                                               holoscan::ExecutionContext& context);

  StereoCameraConfig config_;
  std::unique_ptr<StereoSource> source_;
  SourceLayout layout_;
  std::chrono::nanoseconds period_{};
  std::unique_ptr<Pending> pending_;
  // The calibration message, made at start().
  std::optional<schema::StereoCalibrationT> message_;
  std::uint64_t frame_ids_{};
};

}  // namespace holoscan::stereo_camera
