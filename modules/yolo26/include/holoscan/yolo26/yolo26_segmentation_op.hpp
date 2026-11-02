// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Yolo26SegmentationOp: YOLO26 instance segmentation on the GPU. An image
// goes in; out come the same detections as Yolo26DetectionOp gives (boxes in
// the image's pixels, scores, classes and their count) and each detection's
// mask at the image's size, as one mask per detection and as a label image.
// Inside, the letterbox kernel, TensorRT through HoloInfer and the decode run
// as in Yolo26DetectionOp; the decode also keeps each detection's mask
// coefficients, and a mask kernel combines them with the model's prototype
// masks, upsamples the result bilinearly to the image, crops it to the box
// and thresholds it. Everything stays on the GPU, and the outputs are GPU
// tensors.
//
// The model is an ONNX export of an end-to-end YOLO26 segmentation model
// (yolo26n-seg and its larger siblings): input float32 [1, 3, S, S] RGB in
// [0, 1]; outputs float32 [1, N, 6 + M], rows of [x1, y1, x2, y2, score,
// class] in input pixels and M mask coefficients, best first, and float32
// [1, M, S / 4, S / 4] prototype masks. Its engine is built and cached as for
// Yolo26DetectionOp.
//
// The six outputs carry the image's capture time and frame id, and go out as
// Yolo26DetectionOp's do: pair them by capture time (OnAligned), as
// Yolo26SegmentationOverlayOp does. For a live source, connect them with
// holoscan::ConnectionOptions{.queue_policy =
// holoscan::QueuePolicy::kDropOldest, .queue_depth = 1}.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/yolo26/kernels.hpp>
#include <holoscan/yolo26/letterbox.hpp>

namespace holoscan::yolo26 {

// Most detections, and so masks, per image.
inline constexpr int kMaxSegmentationDetections = 1024;

struct Yolo26SegmentationConfig {
  // An .onnx segmentation model, whose engine is built and cached beside it,
  // or a serialized TensorRT engine built for this GPU.
  std::filesystem::path model;
  // The model's square input side S, a multiple of 32 in
  // kMinInputSize..kMaxInputSize. The model must have been exported for it.
  int input_size{640};
  // Detections need at least this score, in [0, 1].
  float score_threshold{0.25F};
  // Capacity of the outputs, 1..kMaxSegmentationDetections: the number of
  // masks per image. Further detections are dropped, the lowest scores first,
  // and a warning says so once per run. The masks output holds
  // max_detections x height x width bytes.
  int max_detections{32};
  // The model's class count; rows with another class value are dropped.
  int num_classes{80};
  // A pixel belongs to an instance when its mask probability, the sigmoid of
  // the mask value, exceeds this, in (0, 1). Ultralytics uses 0.5.
  float mask_threshold{0.5F};
  // The largest image, which sizes the masks and label image outputs; a
  // larger image fails its frame.
  int max_image_width{1920};
  int max_image_height{1080};
  // The letterbox's border, as an 8-bit gray level.
  std::uint8_t pad_value{114};
  // Build the engine with FP16 kernels allowed (ignored for an engine file).
  bool fp16{false};
  // Let HoloInfer replay the inference as a CUDA graph.
  bool cuda_graphs{true};
  // TensorRT's workspace limit while it builds the engine.
  std::size_t build_workspace_bytes{std::size_t{1} << 30U};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const Yolo26SegmentationConfig& config);

// Builds and caches the TensorRT engine of an .onnx `config.model` unless one
// is cached already. Call it before running the graph, for the reasons
// prepare_engine(const Yolo26DetectionConfig&) gives.
// @throws std::runtime_error if the model cannot be built.
void prepare_engine(const Yolo26SegmentationConfig& config);

class Yolo26SegmentationOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit Yolo26SegmentationOp(Yolo26SegmentationConfig config);
  ~Yolo26SegmentationOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  // Loads the model, building its engine first if it is not cached.
  // @throws std::runtime_error if the model cannot be loaded or does not have
  // the expected input and outputs.
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // As Yolo26DetectionOp's: RGB8, BGR8, RGBA8 or BGRA8, in GPU memory on
  // device 0 or in host memory; at most max_image_width x max_image_height.
  holoscan::Input<holoscan::schema::ImageT> image;
  // As Yolo26DetectionOp's: float32 [max_detections, 4] boxes in image
  // pixels, float32 [max_detections] scores, int32 [max_detections] classes,
  // best first, zero beyond the int32 [1] count.
  holoscan::Output<holoscan::Tensor> boxes;
  holoscan::Output<holoscan::Tensor> scores;
  holoscan::Output<holoscan::Tensor> class_ids;
  holoscan::Output<holoscan::Tensor> count;
  // uint8 [max_detections, height, width], the image's size: per detection,
  // 1 where its mask covers the pixel, 0 elsewhere; all 0 beyond `count`.
  holoscan::Output<holoscan::Tensor> masks;
  // int32 [height, width]: 0 where no mask covers the pixel, otherwise
  // 1 + the index of the best detection whose mask does. Masks overlap; this
  // gives each pixel to one instance.
  holoscan::Output<holoscan::Tensor> label_image;

 private:
  struct Engine;

  void release() noexcept;
  void warn_if_dropped();

  Yolo26SegmentationConfig config_;
  std::unique_ptr<Engine> engine_;
};

}  // namespace holoscan::yolo26
