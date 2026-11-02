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
// Each image's outputs carry its capture time and frame id. They go out one
// after another: should one fail to go out, those before it have been
// delivered alone and compute() returns the error. Pair them by capture time.
//
// On a blocking edge, an output a busy consumer has no room for is such a
// failure (kBackpressured): a lossless graph paces its source to what its
// consumers keep up with. For a live source, connect the input and every
// output with holoscan::ConnectionOptions{.queue_policy =
// holoscan::QueuePolicy::kDropOldest, .queue_depth = 1}: a full queue then
// drops its oldest sample, so the operator never waits for a slower consumer
// and takes the newest image.

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
  // Enables the colorized output.
  bool colorize{false};
  // The inverse depths at the colormap's far and near ends; per frame, the
  // image's smallest and largest, when unset.
  std::optional<std::array<float, 2>> color_range;
  // The largest image, which sizes the output buffers.
  int max_width{1920};
  int max_height{1080};
  // Build the engine with FP16 kernels allowed (ignored for an engine file).
  bool fp16{true};
  // Let HoloInfer replay the inference as a CUDA graph.
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
  // MONO32F, float32 [height, width]: depth in meters, 0 where none is valid.
  // Declared only when gives_metric_depth(); it feeds holoscan-point-cloud's
  // DepthToPointCloudOp as it is.
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
