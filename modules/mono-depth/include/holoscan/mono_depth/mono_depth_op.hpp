// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// MonoDepthOp: monocular depth estimation on the GPU. An image goes in; its
// depth comes out, at the image's own size. Inside, a CUDA kernel resizes the
// image into the model's fixed input and normalizes it, the SDK's TensorRT
// inference library (HoloInfer) runs the model, and a second kernel resamples
// its output back onto the image. Optional kernels colorize the depth for
// HoloViz. Everything stays on the GPU, and the outputs are GPU images.
//
// Depth Anything V2's relative models, which tools/prepare_model.py exports,
// give RELATIVE inverse depth: larger is nearer, but in units of an unknown
// scale and shift that change from image to image. It is not metric: it
// ranks and shapes the scene, and needs a reference (a known depth, a stereo
// or depth camera) to become meters. The operator gives metric depth only
// from a metric model, or from a relative one with a scale and shift you
// supply (InverseDepthAffine), which holds only for the camera and scenes it
// was fitted to.
//
// The model is an ONNX export: input float32 [1, 3, H, W], RGB normalized as
// configured; output float32 [1, H', W'] or [1, 1, H', W'], usually H' x W' =
// H x W. Given an .onnx file, HoloInfer builds a TensorRT engine for this GPU
// and TensorRT version and caches it beside the file, so later runs start in
// seconds; any other file is taken as such an engine.
//
// Each image's outputs carry its capture time, frame id and header. They go
// out one after another: should one fail to go out, those before it have been
// delivered alone and compute() returns the error. Pair them by capture time.
//
// Metric depth needs the camera: when the configuration gives metric depth
// (gives_metric_depth()), the operator takes holoscan-messages' CameraInfo on
// a mandatory `camera_info` input, with messages::camera_info_contract() (by
// default the newest one received; frames before the first are skipped), and
// sends the depth's own camera on `depth_camera_info` with every frame, for
// holoscan-point-cloud's DepthToPointCloudOp. The depth is at the image's own
// size, so its camera is the image's: the camera info must be for images of
// that size, and only its capture time changes.
//
// On a blocking edge, an output a busy consumer has no room for is such a
// failure (kBackpressured). Keep a lossless graph on one worker
// (graph.partition("name").include_all(), see holoscan-messages' README): the
// SDK 5.0 runs each automatic partition on a worker of its own, and a source
// on another worker than this operator can fill its queue while it is still
// busy. A source then treats kBackpressured and kResourceExhausted as
// backpressure and sends the same frame again on its next tick. For a live
// source, connect the input and every output with
// holoscan::ConnectionOptions{.queue_policy =
// holoscan::QueuePolicy::kDropOldest, .queue_depth = 1}: a full queue then
// drops its oldest sample, so the operator never waits for a slower consumer
// and takes the newest image.
//
// CPU waits and copies: none per frame for an image in GPU memory. An image
// in host memory is copied to the GPU, and the operator logs that once: plain
// (pageable) memory costs a wait for the image's producer and a copy that the
// driver stages before it returns; pinned memory an asynchronous copy, and a
// wait for it only when the image's allocator cannot wait for the stream
// itself (logged once). HoloInfer runs TensorRT on a CUDA stream of its own,
// joined to the operator's with events: a Green Context on the operator's
// partition does not hold the inference.
//
// The configuration is fixed when the operator is constructed; it has no
// parameters to set later, and no Python bindings.

#pragma once

#include <cuda_runtime_api.h>

#include <array>
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
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_schema_traits.hpp>
#include <holoscan/mono_depth/image_fit.hpp>
#include <holoscan/mono_depth/pixel_math.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::mono_depth {

struct MonoDepthConfig {
  // An .onnx model, whose engine is built and cached beside it, or a
  // serialized TensorRT engine built for this GPU.
  std::filesystem::path model;
  // The model's fixed input size; it must have been exported for it (Depth
  // Anything: multiples of 14).
  int input_width{518};
  int input_height{518};
  // How an image of another aspect ratio fills the input.
  ResizeMode resize_mode{ResizeMode::kLetterbox};
  // The input normalization the model was trained with.
  Normalization normalization{};
  // What the model gives: relative inverse depth, or meters.
  ModelOutput model_output{ModelOutput::kRelativeInverseDepth};
  // For a relative model, a scale and shift that turn its inverse depth into
  // meters, enabling the metric_depth output.
  std::optional<InverseDepthAffine> metric_from_relative;
  // How the `camera_info` input of a metric configuration is taken: kLatest
  // (the default) for camera info at any rate, as CameraInfoSourceOp sends
  // it, or kEveryFrame for a producer that sends one with every image (see
  // holoscan-messages' camera_info_input.hpp).
  messages::CameraInfoInputOptions camera_info_input{};
  // Enables the colorized output.
  bool colorize{false};
  // With `colorize` only: the inverse depths at the colormap's far and near
  // ends; per frame, the image's smallest and largest, when unset.
  std::optional<std::array<float, 2>> color_range;
  // The largest image, which sizes the output buffers.
  int max_width{1920};
  int max_height{1080};
  // Build the engine with FP16 kernels allowed (ignored for an engine file).
  bool fp16{true};
  // Let HoloInfer capture the inference once and replay it as a CUDA graph,
  // which saves TensorRT's per-launch CPU work; false launches it each frame.
  bool cuda_graphs{true};
  // TensorRT's workspace limit while it builds the engine.
  std::size_t build_workspace_bytes{std::size_t{1} << 30U};
};

// Whether `config` enables the metric_depth output: a metric model, or a
// relative one with metric_from_relative.
[[nodiscard]] bool gives_metric_depth(const MonoDepthConfig& config) noexcept;

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const MonoDepthConfig& config);

// Builds and caches the TensorRT engine of an .onnx `config.model` unless one
// is cached already. Call it before running the graph: some TensorRT releases
// (10.13 among them) change the SIGINT and SIGTERM handlers while building,
// which the SDK's managed signal handling accepts only before a run starts.
// start() then loads the cached engine. TensorRT also starts threads, so
// create the process's holoscan::ManagedShutdownSignals before calling this,
// while main() is the only thread.
//
// It loads the model fully, as start() does, and discards it, so a cached
// engine is loaded twice at startup. HoloInfer names its cached engines
// internally, so whether one exists is known only by loading it.
// @throws std::runtime_error if the model cannot be built.
void prepare_engine(const MonoDepthConfig& config);

class MonoDepthOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit MonoDepthOp(MonoDepthConfig config);
  ~MonoDepthOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  // Loads the model, building its engine first if it is not cached.
  // @throws std::runtime_error if the model cannot be loaded or does not have
  // the expected input and output.
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // RGB8, BGR8, RGBA8 or BGRA8: uint8 [height, width, channels], pixels
  // packed within each row and rows any pitch apart, in GPU memory on device
  // 0 or in host memory. At most max_width x max_height.
  holoscan::Input<holoscan::schema::ImageT> image;
  // MONO32F, float32 [height, width]: inverse depth, larger is nearer;
  // relative for a relative model, 1 / meters for a metric one; 0 where the
  // model gave no valid depth.
  holoscan::Output<holoscan::schema::ImageT> depth;
  // With gives_metric_depth() only: the camera of the images, at their size
  // (holoscan-messages' CameraInfo). Mandatory: an unwired one fails
  // compile(). Connect it with messages::latest_camera_info_connection().
  holoscan::Input<messages::schema::CameraInfoT> camera_info;
  // With gives_metric_depth() only: the camera of `depth` and `metric_depth`,
  // which is the image's camera info with the image's capture time. Sent with
  // every frame, before `metric_depth`.
  holoscan::Output<messages::schema::CameraInfoT> depth_camera_info;
  // MONO32F, float32 [height, width]: depth in meters, 0 where none is valid.
  // Declared only when gives_metric_depth(). With `depth_camera_info` it feeds
  // holoscan-point-cloud's DepthToPointCloudOp, configured with a depth_scale
  // of 1 (meters per unit).
  holoscan::Output<holoscan::schema::ImageT> metric_depth;
  // RGBA8 [height, width, 4]: the inverse depth in the Turbo colormap, far
  // blue to near red, opaque. Declared only with `colorize`.
  holoscan::Output<holoscan::schema::ImageT> colorized;

 private:
  struct Engine;

  [[nodiscard]] holoscan::expected<const std::uint8_t*, holoscan::Error> stage(
      const holoscan::Tensor& tensor, int width, int height, int channels, std::size_t pitch,
      cudaStream_t stream);
  void release() noexcept;

  MonoDepthConfig config_;
  std::unique_ptr<Engine> engine_;
  // The newest camera info of a metric configuration, and the message sent
  // on `depth_camera_info`.
  messages::LatestCameraInfo<messages::schema::CameraInfoT> camera_;
  messages::schema::CameraInfoT depth_camera_;
  // One-time notes: host images are copied; a pinned image is waited for.
  bool copy_logged_{};
  bool wait_logged_{};
  // A GPU copy of host images, grown as needed.
  void* staging_{};
  std::size_t staging_bytes_{};
  // The colorization's per-frame range, two ordered floats in GPU memory.
  std::uint32_t* range_{};
  // Recorded after each frame's work. The model's buffers, `staging_` and
  // `range_` are reused by every frame, but the runtime may give each
  // activation another stream, so the next frame's stream waits for it
  // before touching them.
  cudaEvent_t frame_done_{};
};

}  // namespace holoscan::mono_depth
