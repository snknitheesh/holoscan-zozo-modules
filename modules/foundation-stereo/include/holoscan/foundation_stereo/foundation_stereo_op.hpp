// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// FoundationStereoOp: a rectified left and right image in, the left image's
// disparity out, estimated by a learned stereo model, Fast-FoundationStereo,
// with TensorRT through the SDK's inference library (HoloInfer); optionally
// also the depth, from the pair's geometry. A CUDA kernel resizes each image
// to the model's fixed input size and normalizes it, TensorRT runs the model,
// and a second kernel resizes its disparity back to the image's size and
// rescales it to the image's pixels. Everything after the input stays on the
// GPU, and the outputs are GPU images.
//
// The ports and their conventions are those of the classic matcher in the
// holoscan-stereo-disparity module (`left`, `right`, optional `geometry`;
// `disparity` and optional `depth`, MONO32F with 0 where invalid), so either
// can take the other's place. The model gives no confidence.
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
// same scene rows. A RealSense's infrared pair is rectified by the camera.
// The left and right images, and the geometry when it is an input, pair by
// capture time: the operator runs once all of them with exactly the same
// capture time have arrived.
//
// For a live source, connect the inputs and outputs with
// holoscan::ConnectionOptions{.queue_policy =
// holoscan::QueuePolicy::kDropOldest, .queue_depth = 1}, so that the operator
// takes the newest pair and never waits for a slower consumer.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/foundation_stereo/geometry_message.hpp>
#include <holoscan/foundation_stereo/stereo_geometry.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::foundation_stereo {

// What an operator has done, for its owner to read at any time.
struct FoundationStereoStats {
  std::atomic<std::uint64_t> frames{};
  // GPU time per frame, from the first preprocessing kernel to the last
  // output written, measured with CUDA events: the resize, the model and the
  // postprocessing, without copies from host memory. Frames whose time was
  // still pending when the operator stopped are not counted.
  std::atomic<std::uint64_t> timed_frames{};
  std::atomic<std::uint64_t> gpu_ns_total{};
  std::atomic<std::uint64_t> gpu_ns_max{};
  // GPU memory that loading the model took (the device's free memory before
  // start() minus after): the engine, its activations and the buffers. Only
  // meaningful while nothing else allocates on the device.
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
  // Let HoloInfer replay the inference as a CUDA graph.
  bool cuda_graphs{true};
  // TensorRT's workspace limit while it builds the engine. The 480x736 model
  // needs more than 1 GiB on Jetson Orin, where its largest fused layer has
  // no tactic within 1 GiB.
  std::size_t build_workspace_bytes{std::size_t{4} << 30U};
  // The pair's geometry, for the `depth` output. Give it here, or enable
  // `geometry_input` to take it with every frame, not both.
  std::optional<StereoGeometry> geometry;
  bool geometry_input{false};
  // Declare the `depth` output; needs the geometry.
  bool depth{false};
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
// is cached already. Call it before running the graph: some TensorRT releases
// change the SIGINT and SIGTERM handlers while building, which the SDK's
// managed signal handling accepts only before a run starts. start() then
// loads the cached engine. TensorRT also starts threads, so create the
// process's holoscan::ManagedShutdownSignals before calling this, while
// main() is the only thread.
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
  // within rows, in CUDA device memory on device 0 or in host memory; both
  // the same size, already rectified. MONO8 is given to the model as gray RGB.
  holoscan::Input<holoscan::schema::ImageT> left;
  holoscan::Input<holoscan::schema::ImageT> right;
  // With `geometry_input`: the pair's geometry, with its capture time.
  holoscan::Input<schema::StereoGeometryT> geometry;
  // MONO32F [H, W] on the GPU: the left image's disparity in its pixels (left
  // column minus right column), 0 where invalid. Carries the left image's
  // capture time, frame id and header.
  holoscan::Output<holoscan::schema::ImageT> disparity;
  // With `depth`: MONO32F [H, W] on the GPU, meters along the optical axis
  // (fx * baseline / (disparity + doffs)), 0 where invalid.
  holoscan::Output<holoscan::schema::ImageT> depth;

 private:
  // The loaded model, the staging buffer, the events and the one-time logs.
  struct State;

  FoundationStereoConfig config_;
  std::unique_ptr<State> state_;
};

}  // namespace holoscan::foundation_stereo
