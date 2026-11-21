// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// FoundationStereoOp: a rectified left and right image and the pair's stereo
// camera info in, the left image's disparity out, estimated by a learned
// stereo model, Fast-FoundationStereo, with TensorRT through the SDK's
// inference library (HoloInfer); optionally also the depth, with the camera
// info that goes with it. A CUDA kernel resizes each image to the model's
// fixed input size and normalizes it, TensorRT runs the model, and a second
// kernel resizes its disparity back to the image's size and rescales it to
// the image's pixels. Everything after the input stays on the GPU, and the
// outputs are GPU images.
//
// It can take the place of the holoscan-stereo-disparity module's
// StereoDisparityOp in a graph that uses only the ports both have: `left`,
// `right` and `camera_info` in; `disparity`, `depth` and `depth_camera_info`
// out, with the same types and conventions (MONO32F, 0 where invalid). The
// model gives no confidence, so there is no `confidence` output, and the
// outputs are always the input's size (no downscale).
//
// The model is a single-file ONNX export of Fast-FoundationStereo (see
// tools/prepare_model.py): inputs `left_image` and `right_image`, float32
// [1, 3, H, W], ImageNet-normalized RGB, and output `disparity`, float32
// [1, 1, H, W], in the model's pixels. Its fixed H and W are read from it.
// Given an .onnx file, HoloInfer builds a TensorRT engine for this GPU at
// start() and caches it beside the file, which takes minutes for the real
// model; prepare_engine() does that before the graph runs. Any other file is
// taken as such an engine.
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
// compute() returns), as does an output the runtime has no tensor left for
// (kResourceExhausted: consumers still hold every slot of its pool): keep a lossless graph on one
// worker, where it paces its source to what its consumers keep up with (see the messages module's
// README, "Run a graph on one worker"). For a live source, connect the inputs
// and outputs with holoscan::ConnectionOptions{.queue_policy =
// holoscan::QueuePolicy::kDropOldest, .queue_depth = 1}, so that the operator
// never waits for a slower consumer. compute() queues a frame's GPU work and
// returns; it waits for the frame before last to finish first, so at most two
// frames are on the GPU at once, and a frame it takes is at most about two
// model latencies old when its disparity is out.
//
// TensorRT runs on HoloInfer's own CUDA stream, joined to the runtime's
// stream with events: work the operator queues is ordered correctly, but a
// Green Context bound to this operator's partition does not cover the model.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
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

namespace holoscan::foundation_stereo {

// What an operator has done, for its owner to read at any time.
struct FoundationStereoStats {
  std::atomic<std::uint64_t> frames{};
  // GPU time per frame, from the first preprocessing kernel to the last
  // output written, measured with CUDA events: the resize, the model and the
  // postprocessing, without copies from host memory. The first frame of a
  // run, which also captures the CUDA graph, is in gpu_ns_first only; frames
  // whose time was still pending when the operator stopped are not counted.
  std::atomic<std::uint64_t> gpu_ns_first{};
  std::atomic<std::uint64_t> timed_frames{};
  std::atomic<std::uint64_t> gpu_ns_total{};
  std::atomic<std::uint64_t> gpu_ns_max{};
  // The device's free memory before start() loaded the model minus after: an
  // approximation of what the engine, its activations and the buffers take.
  // Any other process allocating or freeing on the device at the same time
  // distorts it, and on Jetson, whose GPU shares the system's memory, it is
  // not reliable.
  std::atomic<std::uint64_t> model_bytes{};
  // The model's input size, once loaded.
  std::atomic<int> model_width{};
  std::atomic<int> model_height{};
};

struct FoundationStereoConfig {
  // An .onnx model, whose engine is built and cached beside it, or a
  // serialized TensorRT engine built for this GPU.
  std::filesystem::path model;
  // Build the engine with FP16 kernels allowed (ignored for an engine file).
  bool fp16{true};
  // Let HoloInfer replay the inference as a CUDA graph. HoloInfer falls back
  // to plain launches by itself, without saying so, if capturing fails.
  bool cuda_graphs{true};
  // TensorRT's workspace limit while it builds the engine. The 480x736 model
  // needs more than 1 GiB on Jetson Orin, where its largest fused layer has
  // no tactic within 1 GiB.
  std::size_t build_workspace_bytes{std::size_t{4} << 30U};
  // How the `camera_info` input is taken: kLatest (the default) for camera
  // info at any rate, kEveryFrame for a producer that sends one with every
  // pair and may send it after the images (see camera_info_input.hpp).
  messages::CameraInfoInputOptions camera_info_input;
  // Declare the `depth` output.
  bool depth{false};
  // Declare the `depth_camera_info` output; needs `depth`.
  bool depth_camera_info{false};
  // Invalidate (set to 0) pixels whose match would lie left of the right
  // image, which the right camera cannot have seen; the model guesses there.
  bool invalidate_out_of_view{true};
  // Largest image the outputs are sized for.
  int max_width{1920};
  int max_height{1080};
  // Counts frames, GPU time and memory when given.
  std::shared_ptr<FoundationStereoStats> stats;
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const FoundationStereoConfig& config);

// Builds and caches the TensorRT engine of an .onnx `config.model` unless one
// is cached already. A cached engine is kept only while the model file's
// content is the one it was built from (recorded in `<model>.digest`
// beside it); otherwise it is deleted and built again. Call it before running the graph: some
// TensorRT releases change the SIGINT and SIGTERM handlers while building, which the SDK's managed
// signal handling accepts only before a run starts. start() then loads the cached engine. TensorRT
// also starts threads, so create the process's holoscan::ManagedShutdownSignals before calling
// this, while main() is the only thread.
//
// It loads the model fully, as start() does, and discards it, so a cached
// engine is loaded twice. HoloInfer names its cached engines internally, so
// whether one exists is known only by loading it.
// @throws std::runtime_error if the model cannot be built.
void prepare_engine(const FoundationStereoConfig& config);

class FoundationStereoOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit FoundationStereoOp(FoundationStereoConfig config);
  ~FoundationStereoOp() override;
  FoundationStereoOp(const FoundationStereoOp&) = delete;
  FoundationStereoOp& operator=(const FoundationStereoOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  // Loads the model, building its engine first if it is not cached.
  // @throws std::runtime_error if the model cannot be loaded or does not have
  // the expected inputs and output.
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // MONO8, RGB8, BGR8, RGBA8 or BGRA8, [H, W] or [H, W, C], pixels packed
  // within rows, in CUDA device memory on device 0, or in pinned or pageable
  // host memory; both the same size, already rectified. MONO8 is given to the
  // model as gray RGB.
  holoscan::Input<holoscan::schema::ImageT> left;
  holoscan::Input<holoscan::schema::ImageT> right;
  // Mandatory: the pair's calibration, both cameras kRectified and of the
  // images' size (an unwired input fails compile()). It gives the depth's fx,
  // baseline and doffs. A camera info that is not valid, or not usable for
  // the frame, fails the run.
  holoscan::Input<messages::schema::StereoCameraInfoT> camera_info;
  // MONO32F [H, W] on the GPU: the left image's disparity in its pixels (left
  // column minus right column), 0 where invalid. Carries the left image's
  // capture time, frame id and header.
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

 private:
  // The loaded model, the staging buffer, the events, the camera info and the
  // one-time logs.
  struct State;

  FoundationStereoConfig config_;
  std::unique_ptr<State> state_;
};

}  // namespace holoscan::foundation_stereo
