// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/yolo26/yolo26_detection_op.hpp>

#include "model_runtime.hpp"

namespace holoscan::yolo26 {
namespace {

constexpr const char* kOp = "Yolo26DetectionOp";
// HoloInfer's name for the output; it binds tensors by position.
constexpr const char* kOutputName = "output0";

// What the model must give, for the errors of one that gives something else.
std::string expected_output() {
  return "one [1, N, 6] output of end-to-end detections; export it with nms=False (a "
         "segmentation model, which gives masks too, needs Yolo26SegmentationOp)";
}

using detail::kFloat32;
using detail::kInt32;

holoscan::Error failure(const std::string& message) { return detail::failure(kOp, message); }

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return detail::cuda_failure(kOp, what, status);
}

detail::ModelSettings model_settings(const Yolo26DetectionConfig& config) {
  return detail::ModelSettings{.model = config.model,
                               .input_size = config.input_size,
                               .fp16 = config.fp16,
                               .cuda_graphs = config.cuda_graphs,
                               .build_workspace_bytes = config.build_workspace_bytes,
                               .outputs = {kOutputName},
                               .expected_outputs = expected_output()};
}

}  // namespace

// The loaded model and its number of output rows.
struct Yolo26DetectionOp::Engine {
  std::unique_ptr<detail::LoadedModel> model;
  int rows{};
};

void validate_config(const Yolo26DetectionConfig& config) {
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
  if (config.max_detections < 1 || config.max_detections > kMaxRows) {
    throw std::invalid_argument("max_detections must be 1.." + std::to_string(kMaxRows));
  }
  if (config.num_classes < 1 || config.num_classes > 65536) {
    throw std::invalid_argument("num_classes must be 1..65536");
  }
  if (config.build_workspace_bytes < (std::size_t{1} << 24U)) {
    throw std::invalid_argument("build_workspace_bytes must be at least 16 MiB");
  }
}

void prepare_engine(const Yolo26DetectionConfig& config) {
  detail::prepare(model_settings(config), kOp);
}

Yolo26DetectionOp::Yolo26DetectionOp(Yolo26DetectionConfig config) : config_(std::move(config)) {
  validate_config(config_);
}

Yolo26DetectionOp::~Yolo26DetectionOp() { release(); }

void Yolo26DetectionOp::setup(holoscan::OperatorSpec& spec) {
  const auto capacity = static_cast<std::uint64_t>(config_.max_detections);
  // Two slots: one image in compute() while the next one arrives.
  spec.input(image, "image").queue_depth(2U);
  spec.output(boxes, "boxes").produces_tensor(detail::device_tensor(kFloat32, 2U, capacity * 16U));
  spec.output(scores, "scores").produces_tensor(detail::device_tensor(kFloat32, 1U, capacity * 4U));
  spec.output(class_ids, "class_ids")
      .produces_tensor(detail::device_tensor(kInt32, 1U, capacity * 4U));
  spec.output(count, "count").produces_tensor(detail::device_tensor(kInt32, 1U, 4U));
}

holoscan::Contract Yolo26DetectionOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{image});
  return result;
}

void Yolo26DetectionOp::start() {
  release();
  auto engine = std::make_unique<Engine>();
  engine->model = detail::load(model_settings(config_), kOp);
  // The model must give [1, N, 6].
  const auto& dims = engine->model->output_dims;
  if (dims.size() != 1 || dims[0].size() != 3 || dims[0][0] != 1 || dims[0][2] != kRowFields ||
      dims[0][1] < 1 || dims[0][1] > kMaxRows) {
    throw std::runtime_error(
        "Yolo26DetectionOp: " + config_.model.string() + " gives " +
        (dims.size() != 1 ? std::string("other outputs") : detail::dims_text(dims[0])) + ", not " +
        expected_output());
  }
  if (engine->model->outputs.size() != 1) {
    throw std::runtime_error("Yolo26DetectionOp: " + config_.model.string() +
                             " must give float32 detections");
  }
  engine->rows = static_cast<int>(dims[0][1]);
  if (const cudaError_t created = cudaEventCreateWithFlags(&frame_done_, cudaEventDisableTiming);
      created != cudaSuccess) {
    frame_done_ = nullptr;
    throw std::runtime_error(std::string("Yolo26DetectionOp: cudaEventCreateWithFlags: ") +
                             cudaGetErrorString(created));
  }
  void* passed = nullptr;
  if (const cudaError_t allocated =
          cudaHostAlloc(&passed, sizeof(std::int32_t), cudaHostAllocMapped);
      allocated != cudaSuccess) {
    throw std::runtime_error(std::string("Yolo26DetectionOp: cudaHostAlloc: ") +
                             cudaGetErrorString(allocated));
  }
  passed_ = static_cast<std::int32_t*>(passed);
  *passed_ = 0;
  warned_dropped_ = false;
  HOLOSCAN_LOG_INFO("Yolo26DetectionOp: {} ready, input [1, 3, {}, {}], {} output rows",
                    config_.model.string(), config_.input_size, config_.input_size, engine->rows);
  engine_ = std::move(engine);
}

void Yolo26DetectionOp::stop() {
  // The last frame's count, before its memory goes.
  if (frame_done_ != nullptr && cudaEventSynchronize(frame_done_) == cudaSuccess) {
    warn_if_dropped();
  }
  release();
}

void Yolo26DetectionOp::warn_if_dropped() {
  if (warned_dropped_ || passed_ == nullptr || *passed_ <= config_.max_detections) {
    return;
  }
  warned_dropped_ = true;
  HOLOSCAN_LOG_WARN(
      "Yolo26DetectionOp: {} detections passed the score threshold in a frame, more than "
      "max_detections ({}); the lowest-ranked were dropped. Raise max_detections to keep them",
      *passed_, config_.max_detections);
}

void Yolo26DetectionOp::release() noexcept {
  // Freeing waits for the GPU work that may still use the buffers.
  engine_.reset();
  if (frame_done_ != nullptr) {
    cudaEventDestroy(frame_done_);
    frame_done_ = nullptr;
  }
  if (passed_ != nullptr) {
    cudaFreeHost(passed_);
    passed_ = nullptr;
  }
  detail::release_staging(staging_, staging_bytes_);
}

holoscan::expected<void, holoscan::Error> Yolo26DetectionOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = image.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  if (!engine_) {
    return holoscan::make_unexpected(failure("the model is not loaded"));
  }
  auto frame_stream = detail::frame_stream(context, kOp);
  if (!frame_stream) {
    return holoscan::make_unexpected(std::move(frame_stream).error());
  }
  cudaStream_t stream = *frame_stream;
  // A finished previous frame's count of passed rows; a frame still running
  // is checked by a later one, or by stop().
  if (detail::finished(frame_done_)) {
    warn_if_dropped();
  }
  // The previous frame may have run on another stream: wait for its work on
  // the buffers this one reuses, and mark this one's when it has been queued,
  // on every way out.
  if (const cudaError_t status = cudaStreamWaitEvent(stream, frame_done_, 0);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaStreamWaitEvent", status));
  }
  detail::MarkDone mark_done{frame_done_, stream};

  // The image, in GPU memory.
  auto pixels = detail::image_input(sample->data, stream, staging_, staging_bytes_, kOp);
  if (!pixels) {
    return holoscan::make_unexpected(std::move(pixels).error());
  }

  // Letterbox into the model input, run the model, decode into the outputs,
  // all in order on the stream; HoloInfer joins it with events. Frames before
  // this one are done with the buffers: see frame_done_.
  const Letterbox letterbox = make_letterbox(pixels->width, pixels->height, config_.input_size);
  if (const cudaError_t status =
          launch_letterbox(*pixels, letterbox, config_.pad_value, engine_->model->input, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("letterbox kernel", status));
  }
  if (auto inferred = detail::infer_on(*engine_->model, stream, kOp); !inferred) {
    return inferred;
  }

  const std::int64_t capacity = config_.max_detections;
  auto boxes_loan = detail::allocate<2>(boxes, {capacity, 4}, kFloat32);
  auto scores_loan = detail::allocate<1>(scores, {capacity}, kFloat32);
  auto class_ids_loan = detail::allocate<1>(class_ids, {capacity}, kInt32);
  auto count_loan = detail::allocate<1>(count, {1}, kInt32);
  for (const auto* loan : {&boxes_loan, &scores_loan, &class_ids_loan, &count_loan}) {
    if (!*loan) {
      return holoscan::make_unexpected(loan->error());
    }
  }
  auto boxes_writer = boxes_loan->write(stream);
  auto scores_writer = scores_loan->write(stream);
  auto class_ids_writer = class_ids_loan->write(stream);
  auto count_writer = count_loan->write(stream);
  for (const auto* writer : {&boxes_writer, &scores_writer, &class_ids_writer, &count_writer}) {
    if (!*writer) {
      return holoscan::make_unexpected(writer->error());
    }
  }
  const DecodeParams params{.rows = engine_->rows,
                            .score_threshold = config_.score_threshold,
                            .num_classes = config_.num_classes,
                            .capacity = config_.max_detections,
                            .letterbox = letterbox};
  if (const cudaError_t status =
          launch_decode(engine_->model->outputs[0], params,
                        DecodeOutputs{.boxes = boxes_writer->data_as<float>(),
                                      .scores = scores_writer->data_as<float>(),
                                      .class_ids = class_ids_writer->data_as<std::int32_t>(),
                                      .count = count_writer->data_as<std::int32_t>(),
                                      .passed = passed_},
                        stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("decode kernel", status));
  }
  for (auto* writer : {&boxes_writer, &scores_writer, &class_ids_writer, &count_writer}) {
    if (auto committed = std::move(**writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
  }
  mark_done.event = nullptr;
  if (const cudaError_t status = cudaEventRecord(frame_done_, stream); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaEventRecord", status));
  }

  // One set of options, so every output carries the image's capture time and
  // frame id.
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
  return count.emit(std::move(*count_loan), emit);
}

}  // namespace holoscan::yolo26
