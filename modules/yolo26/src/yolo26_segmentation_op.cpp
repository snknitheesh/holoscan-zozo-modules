// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/yolo26/yolo26_segmentation_op.hpp>

#include "model_runtime.hpp"

namespace holoscan::yolo26 {
namespace {

constexpr const char* kOp = "Yolo26SegmentationOp";

// What the model must give, for the errors of one that gives something else.
std::string expected_outputs() {
  return "[1, N, 6 + M] end-to-end detections with M mask coefficients and [1, M, S/4, S/4] "
         "prototype masks; export a YOLO26 segmentation model with nms=False";
}

using detail::kFloat32;
using detail::kInt32;
using detail::kUInt8;

holoscan::Error invalid(const std::string& message) { return detail::invalid(kOp, message); }

holoscan::Error failure(const std::string& message) { return detail::failure(kOp, message); }

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return detail::cuda_failure(kOp, what, status);
}

detail::ModelSettings model_settings(const Yolo26SegmentationConfig& config) {
  // HoloInfer binds the outputs by position: the model's two, in its order.
  return detail::ModelSettings{.model = config.model,
                               .input_size = config.input_size,
                               .fp16 = config.fp16,
                               .cuda_graphs = config.cuda_graphs,
                               .build_workspace_bytes = config.build_workspace_bytes,
                               .outputs = {"output0", "output1"},
                               .expected_outputs = expected_outputs()};
}

}  // namespace

// The loaded model, what its outputs hold, and the GPU buffers and host
// state each frame reuses.
struct Yolo26SegmentationOp::Engine {
  std::unique_ptr<detail::LoadedModel> model;
  const float* rows{};
  const float* protos{};
  int row_count{};
  int mask_channels{};
  int proto_width{};
  int proto_height{};
  // capacity x mask_channels coefficients, and capacity x proto plane logits.
  float* coefficients{};
  float* logits{};
  // A GPU copy of host images.
  void* staging{};
  std::size_t staging_bytes{};
  // Recorded after each frame's work; the next frame's stream waits for it
  // before reusing the model's buffers and the ones above.
  cudaEvent_t frame_done{};
  // Mapped host memory where the decode writes how many rows passed.
  std::int32_t* passed{};
  bool warned_dropped{};

  Engine() = default;
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  ~Engine() {
    // Freeing waits for the GPU work that may still use the buffers.
    model.reset();
    if (frame_done != nullptr) {
      cudaEventDestroy(frame_done);
    }
    if (passed != nullptr) {
      cudaFreeHost(passed);
    }
    cudaFree(coefficients);
    cudaFree(logits);
    detail::release_staging(staging, staging_bytes);
  }
};

void validate_config(const Yolo26SegmentationConfig& config) {
  if (config.model.empty()) {
    throw std::invalid_argument("a model path is required");
  }
  if (config.input_size < kMinInputSize || config.input_size > kMaxInputSize ||
      config.input_size % 32 != 0) {
    throw std::invalid_argument("input_size must be a multiple of 32 in " +
                                std::to_string(kMinInputSize) + ".." +
                                std::to_string(kMaxInputSize));
  }
  if (!(config.score_threshold >= 0.0F && config.score_threshold <= 1.0F)) {
    throw std::invalid_argument("score_threshold must be in [0, 1]");
  }
  if (config.max_detections < 1 || config.max_detections > kMaxSegmentationDetections) {
    throw std::invalid_argument("max_detections must be 1.." +
                                std::to_string(kMaxSegmentationDetections));
  }
  if (config.num_classes < 1 || config.num_classes > 65536) {
    throw std::invalid_argument("num_classes must be 1..65536");
  }
  if (!(config.mask_threshold > 0.0F && config.mask_threshold < 1.0F)) {
    throw std::invalid_argument("mask_threshold must be in (0, 1)");
  }
  if (config.max_image_width < 1 || config.max_image_height < 1 ||
      config.max_image_width > kMaxImageSide || config.max_image_height > kMaxImageSide) {
    throw std::invalid_argument("max_image_width and max_image_height must be 1.." +
                                std::to_string(kMaxImageSide));
  }
  if (config.build_workspace_bytes < (std::size_t{1} << 24U)) {
    throw std::invalid_argument("build_workspace_bytes must be at least 16 MiB");
  }
}

void prepare_engine(const Yolo26SegmentationConfig& config) {
  detail::prepare(model_settings(config), kOp);
}

Yolo26SegmentationOp::Yolo26SegmentationOp(Yolo26SegmentationConfig config)
    : config_(std::move(config)) {
  validate_config(config_);
}

Yolo26SegmentationOp::~Yolo26SegmentationOp() { release(); }

void Yolo26SegmentationOp::setup(holoscan::OperatorSpec& spec) {
  const auto capacity = static_cast<std::uint64_t>(config_.max_detections);
  const std::uint64_t pixels = static_cast<std::uint64_t>(config_.max_image_width) *
                               static_cast<std::uint64_t>(config_.max_image_height);
  // Two slots: one image in compute() while the next one arrives.
  spec.input(image, "image").queue_depth(2U);
  spec.output(boxes, "boxes").produces_tensor(detail::device_tensor(kFloat32, 2U, capacity * 16U));
  spec.output(scores, "scores").produces_tensor(detail::device_tensor(kFloat32, 1U, capacity * 4U));
  spec.output(class_ids, "class_ids")
      .produces_tensor(detail::device_tensor(kInt32, 1U, capacity * 4U));
  spec.output(count, "count").produces_tensor(detail::device_tensor(kInt32, 1U, 4U));
  spec.output(masks, "masks").produces_tensor(detail::device_tensor(kUInt8, 3U, capacity * pixels));
  spec.output(label_image, "label_image")
      .produces_tensor(detail::device_tensor(kInt32, 2U, pixels * 4U));
}

holoscan::Contract Yolo26SegmentationOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{image});
  return result;
}

void Yolo26SegmentationOp::start() {
  release();
  const std::string name = config_.model.string();
  auto engine = std::make_unique<Engine>();
  engine->model = detail::load(model_settings(config_), kOp);

  // The model must give [1, N, 6 + M] rows and [1, M, h, w] prototypes, in
  // either order.
  const auto& dims = engine->model->output_dims;
  const auto describe = [&dims] {
    std::string text;
    for (const auto& d : dims) {
      text += (text.empty() ? "" : " and ") + detail::dims_text(d);
    }
    return text.empty() ? std::string("other outputs") : text;
  };
  const auto refuse = [&](const std::string& why) {
    throw std::runtime_error(std::string(kOp) + ": " + name + " gives " + describe() + ", not " +
                             why);
  };
  if (dims.size() != 2 || engine->model->outputs.size() != 2) {
    refuse(expected_outputs());
  }
  const std::size_t rows_index = dims[0].size() == 3 ? 0U : 1U;
  const auto& rows = dims[rows_index];
  const auto& protos = dims[1U - rows_index];
  if (rows.size() != 3 || protos.size() != 4 || rows[0] != 1 || protos[0] != 1 || rows[1] < 1 ||
      rows[1] > kMaxRows || rows[2] <= kRowFields || rows[2] - kRowFields != protos[1] ||
      protos[1] > kMaxMaskChannels || protos[2] < 1 || protos[3] < 1 || protos[2] > kMaxProtoSide ||
      protos[3] > kMaxProtoSide) {
    refuse(expected_outputs());
  }
  engine->rows = engine->model->outputs[rows_index];
  engine->protos = engine->model->outputs[1U - rows_index];
  engine->row_count = static_cast<int>(rows[1]);
  engine->mask_channels = static_cast<int>(protos[1]);
  engine->proto_height = static_cast<int>(protos[2]);
  engine->proto_width = static_cast<int>(protos[3]);

  const auto capacity = static_cast<std::size_t>(config_.max_detections);
  const std::size_t plane = static_cast<std::size_t>(engine->proto_width) *
                            static_cast<std::size_t>(engine->proto_height);
  const auto check = [](cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
      throw std::runtime_error(std::string(kOp) + ": " + what + ": " + cudaGetErrorString(status));
    }
  };
  void* coefficients = nullptr;
  check(cudaMalloc(&coefficients,
                   capacity * static_cast<std::size_t>(engine->mask_channels) * sizeof(float)),
        "cudaMalloc");
  engine->coefficients = static_cast<float*>(coefficients);
  void* logits = nullptr;
  check(cudaMalloc(&logits, capacity * plane * sizeof(float)), "cudaMalloc");
  engine->logits = static_cast<float*>(logits);
  check(cudaEventCreateWithFlags(&engine->frame_done, cudaEventDisableTiming),
        "cudaEventCreateWithFlags");
  void* passed = nullptr;
  check(cudaHostAlloc(&passed, sizeof(std::int32_t), cudaHostAllocMapped), "cudaHostAlloc");
  engine->passed = static_cast<std::int32_t*>(passed);
  *engine->passed = 0;
  HOLOSCAN_LOG_INFO(
      "{}: {} ready, input [1, 3, {}, {}], {} output rows with {} mask coefficients, prototypes "
      "{}x{}",
      kOp, name, config_.input_size, config_.input_size, engine->row_count, engine->mask_channels,
      engine->proto_width, engine->proto_height);
  engine_ = std::move(engine);
}

void Yolo26SegmentationOp::stop() {
  // The last frame's count, before its memory goes.
  if (engine_ && cudaEventSynchronize(engine_->frame_done) == cudaSuccess) {
    warn_if_dropped();
  }
  release();
}

void Yolo26SegmentationOp::warn_if_dropped() {
  if (engine_->warned_dropped || *engine_->passed <= config_.max_detections) {
    return;
  }
  engine_->warned_dropped = true;
  HOLOSCAN_LOG_WARN(
      "{}: {} detections passed the score threshold in a frame, more than max_detections ({}); "
      "the lowest-ranked were dropped. Raise max_detections to keep them",
      kOp, *engine_->passed, config_.max_detections);
}

void Yolo26SegmentationOp::release() noexcept { engine_.reset(); }

holoscan::expected<void, holoscan::Error> Yolo26SegmentationOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = image.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  if (!engine_) {
    return holoscan::make_unexpected(failure("the model is not loaded"));
  }
  Engine& engine = *engine_;
  auto frame_stream = detail::frame_stream(context, kOp);
  if (!frame_stream) {
    return holoscan::make_unexpected(std::move(frame_stream).error());
  }
  cudaStream_t stream = *frame_stream;
  if (detail::finished(engine.frame_done)) {
    warn_if_dropped();
  }
  if (const cudaError_t status = cudaStreamWaitEvent(stream, engine.frame_done, 0);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaStreamWaitEvent", status));
  }
  detail::MarkDone mark_done{engine.frame_done, stream};

  auto pixels =
      detail::image_input(sample->data, stream, engine.staging, engine.staging_bytes, kOp);
  if (!pixels) {
    return holoscan::make_unexpected(std::move(pixels).error());
  }
  const int width = pixels->width;
  const int height = pixels->height;
  if (width > config_.max_image_width || height > config_.max_image_height) {
    return holoscan::make_unexpected(invalid("the image is " + std::to_string(width) + "x" +
                                             std::to_string(height) +
                                             ", larger than max_image_width x max_image_height (" +
                                             std::to_string(config_.max_image_width) + "x" +
                                             std::to_string(config_.max_image_height) + ")"));
  }

  const Letterbox letterbox = make_letterbox(width, height, config_.input_size);
  if (const cudaError_t status =
          launch_letterbox(*pixels, letterbox, config_.pad_value, engine.model->input, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("letterbox kernel", status));
  }
  if (auto inferred = detail::infer_on(*engine.model, stream, kOp); !inferred) {
    return inferred;
  }

  const std::int64_t capacity = config_.max_detections;
  auto boxes_loan = detail::allocate<2>(boxes, {capacity, 4}, kFloat32);
  auto scores_loan = detail::allocate<1>(scores, {capacity}, kFloat32);
  auto class_ids_loan = detail::allocate<1>(class_ids, {capacity}, kInt32);
  auto count_loan = detail::allocate<1>(count, {1}, kInt32);
  auto masks_loan = detail::allocate<3>(masks, {capacity, height, width}, kUInt8);
  auto label_loan = detail::allocate<2>(label_image, {height, width}, kInt32);
  for (const auto* loan :
       {&boxes_loan, &scores_loan, &class_ids_loan, &count_loan, &masks_loan, &label_loan}) {
    if (!*loan) {
      return holoscan::make_unexpected(loan->error());
    }
  }
  auto boxes_writer = boxes_loan->write(stream);
  auto scores_writer = scores_loan->write(stream);
  auto class_ids_writer = class_ids_loan->write(stream);
  auto count_writer = count_loan->write(stream);
  auto masks_writer = masks_loan->write(stream);
  auto label_writer = label_loan->write(stream);
  for (const auto* writer : {&boxes_writer, &scores_writer, &class_ids_writer, &count_writer,
                             &masks_writer, &label_writer}) {
    if (!*writer) {
      return holoscan::make_unexpected(writer->error());
    }
  }
  auto* boxes_out = boxes_writer->data_as<float>();
  auto* count_out = count_writer->data_as<std::int32_t>();
  const DecodeParams params{.rows = engine.row_count,
                            .score_threshold = config_.score_threshold,
                            .num_classes = config_.num_classes,
                            .capacity = config_.max_detections,
                            .letterbox = letterbox};
  if (const cudaError_t status = launch_segmentation_decode(
          engine.rows, params, engine.mask_channels,
          DecodeOutputs{.boxes = boxes_out,
                        .scores = scores_writer->data_as<float>(),
                        .class_ids = class_ids_writer->data_as<std::int32_t>(),
                        .count = count_out,
                        .passed = engine.passed},
          engine.coefficients, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("decode kernel", status));
  }
  // The probability threshold as a logit: log(p / (1 - p)), 0 for one half.
  const float logit_threshold = std::log(config_.mask_threshold / (1.0F - config_.mask_threshold));
  if (const cudaError_t status =
          launch_masks(MaskInputs{.protos = engine.protos,
                                  .coefficients = engine.coefficients,
                                  .boxes = boxes_out,
                                  .count = count_out},
                       MaskParams{.capacity = config_.max_detections,
                                  .mask_channels = engine.mask_channels,
                                  .proto_width = engine.proto_width,
                                  .proto_height = engine.proto_height,
                                  .logit_threshold = logit_threshold,
                                  .letterbox = letterbox},
                       MaskOutputs{.masks = masks_writer->data_as<std::uint8_t>(),
                                   .labels = label_writer->data_as<std::int32_t>(),
                                   .logits = engine.logits},
                       stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("mask kernel", status));
  }
  for (auto* writer : {&boxes_writer, &scores_writer, &class_ids_writer, &count_writer,
                       &masks_writer, &label_writer}) {
    if (auto committed = std::move(**writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
  }
  mark_done.event = nullptr;
  if (const cudaError_t status = cudaEventRecord(engine.frame_done, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaEventRecord", status));
  }

  // Every output carries the image's capture time and frame id.
  const holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(*sample);
  if (auto sent = boxes.emit(std::move(*boxes_loan), emit); !sent) {
    return sent;
  }
  if (auto sent = scores.emit(std::move(*scores_loan), emit); !sent) {
    return sent;
  }
  if (auto sent = class_ids.emit(std::move(*class_ids_loan), emit); !sent) {
    return sent;
  }
  if (auto sent = masks.emit(std::move(*masks_loan), emit); !sent) {
    return sent;
  }
  if (auto sent = label_image.emit(std::move(*label_loan), emit); !sent) {
    return sent;
  }
  return count.emit(std::move(*count_loan), emit);
}

}  // namespace holoscan::yolo26
