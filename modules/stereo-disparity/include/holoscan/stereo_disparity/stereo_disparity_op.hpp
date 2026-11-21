// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// StereoDisparityOp: a rectified left and right image and the pair's stereo
// camera info in, the left image's disparity out, estimated by NVIDIA VPI 4's
// stereo disparity estimator on the GPU (cuda), the Optical Flow Accelerator
// (ofa) or the OFA with the PVA (pva); optionally also the depth, with the
// camera info that goes with it, and VPI's confidence.
//
// The input must already be rectified: rows of the two images must see the
// same scene rows, and the camera info must say so (both cameras
// messages::CameraState::kRectified), as the rectify module's
// StereoRectifyOp sends it, or as CameraInfoSourceOp reads it from a
// rectified pair's camera_info files.
//
// The left and right images pair by capture time: the operator runs once both
// with exactly the same capture time have arrived. The camera info comes as
// messages::camera_info_contract() takes it: by default the newest one
// received (from CameraInfoSourceOp, or one with every pair); frames before
// the first camera info are skipped.
//
// On a blocking edge to a consumer on another worker, an output that consumer
// has no room for fails the run (the emit returns kBackpressured, which
// compute() returns): keep a lossless graph on one worker, where it paces its
// source to what its consumers keep up with. For a live source, connect the
// inputs and outputs with holoscan::ConnectionOptions{.queue_policy =
// holoscan::QueuePolicy::kDropOldest, .queue_depth = 1}, so that the operator
// never waits for a slower consumer and takes the newest pair.
//
// Making the operator with Backend::kOfa or kPva asks VPI and CUDA what the
// platform has, which starts their threads: create
// holoscan::ManagedShutdownSignals first, at the start of main(), or run with
// holoscan::RunSignalPolicy::kCallerManaged (see the README).

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_schema_traits.hpp>
#include <holoscan/messages/stereo_camera_info_schema_traits.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/stereo_disparity/disparity_options.hpp>

namespace holoscan::stereo_disparity {

// What an operator has done, for its owner to read at any time.
struct StereoDisparityStats {
  // The backend that ran the latest frame; kAuto before the first.
  std::atomic<Backend> backend{Backend::kAuto};
  std::atomic<std::uint64_t> frames{};
  // Time in compute() per frame, from receiving the pair to having the outputs
  // written, without building the matcher (once per backend and size).
  std::atomic<std::uint64_t> compute_ns_total{};
  std::atomic<std::uint64_t> compute_ns_max{};
};

struct StereoDisparityConfig {
  // kAuto chooses on the first frame, when the image size is known: see
  // resolve_auto(). The default is cuda, which runs everywhere.
  Backend backend{Backend::kCuda};
  DisparityParams params;
  // How the `camera_info` input is taken: kLatest (the default) for camera
  // info at any rate, kEveryFrame for a producer that sends one with every
  // pair and may send it after the images (see camera_info_input.hpp).
  messages::CameraInfoInputOptions camera_info_input;
  // Declare the `depth` output.
  bool depth{false};
  // Declare the `depth_camera_info` output; needs `depth`.
  bool depth_camera_info{false};
  // Declare the `confidence` output: VPI's confidence map, on cuda and pva.
  bool confidence{false};
  // Largest image the outputs are sized for.
  int max_width{1920};
  int max_height{1080};
  // Counts frames and compute time when given.
  std::shared_ptr<StereoDisparityStats> stats;
};

// With kOfa or kPva, probes the platform (VPI and CUDA start threads).
// @throws std::invalid_argument naming the first unusable field, or a backend
//         this platform does not have.
void validate_config(const StereoDisparityConfig& config);

class StereoDisparityOp final : public holoscan::Operator<> {
 public:
  // With kOfa or kPva, probes the platform (VPI and CUDA start threads).
  // @throws std::invalid_argument if `config` is not valid.
  explicit StereoDisparityOp(StereoDisparityConfig config);
  ~StereoDisparityOp() override;
  StereoDisparityOp(const StereoDisparityOp&) = delete;
  StereoDisparityOp& operator=(const StereoDisparityOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // The backend this operator runs: the configured one, or for kAuto the one
  // chosen on the first frame, and kAuto until then. Safe from any thread.
  [[nodiscard]] Backend resolved_backend() const noexcept;

  // MONO8, RGB8, BGR8, RGBA8 or BGRA8, [H, W] or [H, W, C], rows packed, in
  // host or CUDA device memory; both the same size, already rectified. Color
  // is matched as its BT.601 luma.
  holoscan::Input<holoscan::schema::ImageT> left;
  holoscan::Input<holoscan::schema::ImageT> right;
  // Mandatory: the pair's calibration, both cameras kRectified and of the
  // images' size (an unwired input fails compile()). It gives the depth's fx,
  // baseline and doffs. A camera info that is not valid, or not usable for
  // the frame, fails the run.
  holoscan::Input<messages::schema::StereoCameraInfoT> camera_info;
  // MONO32F [H, W] on the GPU: the left image's disparity in pixels of the
  // input images (left column minus right column), 0 where invalid; H and W
  // are the input's divided by the downscale factor, rounded up. Carries the
  // left image's capture time, frame id and header.
  holoscan::Output<holoscan::schema::ImageT> disparity;
  // With `depth`: MONO32F [H, W] on the GPU, meters along the optical axis
  // (fx * baseline / (disparity + doffs)), 0 where invalid.
  holoscan::Output<holoscan::schema::ImageT> depth;
  // With `depth_camera_info`: the camera of `depth` and `disparity`, the
  // rectified left camera at their size (depth_camera_info() in
  // camera_info.hpp), with the frame's capture time. Sent with every frame,
  // before the depth, so that a consumer with the default
  // messages::camera_info_contract() (kLatest, connected with
  // messages::latest_camera_info_connection()) uses each frame's own; one
  // with kEveryFrame takes it with the depth's edge options.
  holoscan::Output<messages::schema::CameraInfoT> depth_camera_info;
  // With `confidence`: MONO16 [H, W] on the GPU, VPI's confidence, 65535 most
  // confident.
  holoscan::Output<holoscan::schema::ImageT> confidence;

 private:
  // The matcher, the staging buffer, a CUDA stream, the camera info and the
  // one-time logs.
  struct State;

  StereoDisparityConfig config_;
  std::unique_ptr<State> state_;
};

}  // namespace holoscan::stereo_disparity
