// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoinfer/holoinfer.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sam3/kernels.hpp>
#include <holoscan/sam3/sam3_segmentation_op.hpp>
#include <holoscan/sensor_io/provenance.hpp>

namespace holoscan::sam3 {
namespace {

namespace infer = holoscan::inference;

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kInt32{kDLInt, 32U, 1U};
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr DLDataType kUInt16{kDLUInt, 16U, 1U};

// HoloInfer's names for the models and their tensors. It binds tensors by
// position, in the order the export writes them.
constexpr const char* kEncoderName = "sam3_encoder";
constexpr const char* kDecoderName = "sam3_decoder";
constexpr const char* kImageName = "image";
using Names = std::vector<std::string>;
const Names kFeatureNames{"fpn0", "fpn1", "fpn2"};
const Names kPromptNames{"text_features", "text_mask", "boxes", "box_labels", "box_valid"};
const Names kDecoderOutputNames{"masks", "boxes_xyxy", "scores"};

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "Sam3SegmentationOp: " + message};
}

holoscan::Error failure(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "Sam3SegmentationOp: " + message};
}

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return failure(std::string(what) + ": " + cudaGetErrorString(status));
}

[[noreturn]] void fail_start(const std::string& message) {
  throw std::runtime_error("Sam3SegmentationOp: " + message);
}

holoscan::TensorOutputSpec device_tensor(DLDataType dtype, std::uint32_t rank,
                                         std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = dtype,
              .rank = rank,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(bytes),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

bool is_onnx(const std::filesystem::path& path) {
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return extension == ".onnx";
}

std::optional<PixelFormat> pixel_format(holoscan::schema::ImageEncoding encoding) {
  switch (encoding) {
    case holoscan::schema::ImageEncoding_RGB8:
      return PixelFormat::kRgb8;
    case holoscan::schema::ImageEncoding_BGR8:
      return PixelFormat::kBgr8;
    case holoscan::schema::ImageEncoding_RGBA8:
      return PixelFormat::kRgba8;
    case holoscan::schema::ImageEncoding_BGRA8:
      return PixelFormat::kBgra8;
    default:
      return std::nullopt;
  }
}

std::string dims_text(const std::vector<std::int64_t>& dims) {
  std::string text = "[";
  for (std::size_t i = 0; i < dims.size(); ++i) {
    text += (i == 0 ? "" : ", ") + std::to_string(dims[i]);
  }
  return text + "]";
}

std::string dims_text(const std::vector<std::vector<std::int64_t>>& all) {
  std::string text;
  for (std::size_t i = 0; i < all.size(); ++i) {
    text += (i == 0 ? "" : ", ") + dims_text(all[i]);
  }
  return text;
}

// Keeps a pinned host tensor's memory from reuse until the work queued on
// `stream` so far is done: its allocator waits for the stream if it can,
// otherwise this waits.
holoscan::expected<void, holoscan::Error> hold_until_done(const holoscan::Tensor& tensor,
                                                          cudaStream_t stream) {
  holoscan::Tensor handle = tensor;
  if (handle.set_deallocation_stream(stream)) {
    return {};
  }
  if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
  }
  return {};
}

// HoloInfer's settings for one model, the same for prepare_engines() and
// start() so that both find the same cached engine.
std::shared_ptr<infer::InferenceSpecs> make_specs(const Sam3SegmentationConfig& config,
                                                  const char* name,
                                                  const std::filesystem::path& model,
                                                  std::vector<std::string> inputs,
                                                  std::vector<std::string> outputs) {
  auto specs = std::make_shared<infer::InferenceSpecs>(
      "trt", infer::Mappings{}, infer::Mappings{{name, model.string()}},
      infer::MultiMappings{{name, std::move(inputs)}},
      infer::MultiMappings{{name, std::move(outputs)}}, infer::Mappings{}, infer::MultiMappings{},
      /*dynamic_input_dims=*/false, /*is_engine_path=*/!is_onnx(model),
      /*parallel_proc=*/false, config.fp16, /*cuda_buffer_in=*/true, /*cuda_buffer_out=*/true,
      config.cuda_graphs, /*dla_core=*/-1, /*dla_gpu_fallback=*/false,
      /*gpu_resident_inference=*/false, std::function<cudaStream_t(std::int32_t)>{});
  specs->max_memory_ = config.build_workspace_bytes;
  return specs;
}

std::shared_ptr<infer::InferenceSpecs> encoder_specs(const Sam3SegmentationConfig& config) {
  return make_specs(config, kEncoderName, config.encoder, {kImageName}, kFeatureNames);
}

std::shared_ptr<infer::InferenceSpecs> decoder_specs(const Sam3SegmentationConfig& config) {
  std::vector<std::string> inputs = kFeatureNames;
  inputs.insert(inputs.end(), kPromptNames.begin(), kPromptNames.end());
  return make_specs(config, kDecoderName, config.decoder, std::move(inputs), kDecoderOutputNames);
}

void load(infer::InferContext& context, std::shared_ptr<infer::InferenceSpecs>& specs,
          const std::filesystem::path& model) {
  const infer::InferStatus status = context.set_inference_params(specs);
  if (status.get_code() != infer::holoinfer_code::H_SUCCESS) {
    fail_start("cannot load " + model.string() + ": " + status.get_message());
  }
}

// GPU memory, freed with the object.
struct DeviceFree {
  void operator()(void* pointer) const noexcept { cudaFree(pointer); }
};
using DeviceMemory = std::unique_ptr<void, DeviceFree>;

DeviceMemory device_alloc(std::size_t bytes) {
  void* pointer = nullptr;
  if (const cudaError_t status = cudaMalloc(&pointer, std::max<std::size_t>(bytes, 1U));
      status != cudaSuccess) {
    throw std::runtime_error(std::string("Sam3SegmentationOp: cudaMalloc: ") +
                             cudaGetErrorString(status));
  }
  return DeviceMemory(pointer);
}

template <std::size_t Rank>
holoscan::expected<holoscan::TensorOutputLoan, holoscan::Error> allocate(
    holoscan::Output<holoscan::Tensor>& port, std::array<std::int64_t, Rank> shape,
    DLDataType dtype) {
  return port.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
}

bool in_unit_interval(float value, bool closed_low) {
  return std::isfinite(value) && (closed_low ? value >= 0.0F : value > 0.0F) && value < 1.0F;
}

}  // namespace

// The loaded models: HoloInfer's contexts and specifications, which own the
// buffers that TensorRT reads and writes, and the buffers around them.
struct Sam3SegmentationOp::Engines {
  std::shared_ptr<infer::InferenceSpecs> encoder_specs;
  std::shared_ptr<infer::InferenceSpecs> decoder_specs;
  std::unique_ptr<infer::InferContext> encoder;
  std::unique_ptr<infer::InferContext> decoder;
  int input_size{};
  // The decoder's prompt inputs, in the order of kPromptNames, and their
  // sizes in floats.
  std::array<float*, 5> prompt_inputs{};
  std::array<std::size_t, 5> prompt_sizes{};
  const float* masks{};
  const float* boxes{};
  const float* scores{};
  int tokens{};
  int dim{};
  int max_boxes{};
  int queries{};
  int mask_width{};
  int mask_height{};
  // Every active prompt's decoder inputs, one after another.
  DeviceMemory prompts;
  int prompt_count{};
  // The candidates, for `candidate_prompts` prompts.
  DeviceMemory candidate_scores;
  DeviceMemory candidate_boxes;
  DeviceMemory candidate_queries;
  DeviceMemory candidate_logits;
  DeviceMemory candidate_counts;
  DeviceMemory order;
  int candidate_prompts{};

  [[nodiscard]] std::size_t prompt_floats() const {
    std::size_t total = 0;
    for (const std::size_t size : prompt_sizes) {
      total += size;
    }
    return total;
  }

  ~Engines() {
    // The contexts' TensorRT objects go before the buffers they were bound to.
    decoder.reset();
    encoder.reset();
    decoder_specs.reset();
    encoder_specs.reset();
  }
};

void validate_config(const Sam3SegmentationConfig& config) {
  if (config.encoder.empty() || config.decoder.empty() || config.prompt_table.empty()) {
    throw std::invalid_argument("an encoder, a decoder and a prompt table are required");
  }
  if (config.prompts.empty() || config.prompts.size() > static_cast<std::size_t>(kMaxPrompts)) {
    throw std::invalid_argument("1.." + std::to_string(kMaxPrompts) + " prompts are needed");
  }
  for (const Sam3Prompt& prompt : config.prompts) {
    validate_prompt(prompt);
  }
  if (!in_unit_interval(config.score_threshold, true)) {
    throw std::invalid_argument("score_threshold must be in [0, 1)");
  }
  if (!in_unit_interval(config.mask_threshold, false)) {
    throw std::invalid_argument("mask_threshold must be in (0, 1)");
  }
  if (config.max_instances < 1 || config.max_instances > kMaxInstances) {
    throw std::invalid_argument("max_instances must be 1.." + std::to_string(kMaxInstances));
  }
  if (config.max_width < 1 || config.max_height < 1 || config.max_width > kMaxImageSide ||
      config.max_height > kMaxImageSide) {
    throw std::invalid_argument("max_width and max_height must be 1.." +
                                std::to_string(kMaxImageSide));
  }
  if (config.build_workspace_bytes < (std::size_t{1} << 24U)) {
    throw std::invalid_argument("build_workspace_bytes must be at least 16 MiB");
  }
}

void prepare_engines(const Sam3SegmentationConfig& config) {
  for (auto [specs, model] : {std::pair{encoder_specs(config), config.encoder},
                              std::pair{decoder_specs(config), config.decoder}}) {
    if (!is_onnx(model)) {
      continue;
    }
    HOLOSCAN_LOG_INFO(
        "Sam3SegmentationOp: preparing {}; building its TensorRT engine takes a while when none "
        "is cached",
        model.string());
    infer::InferContext context;
    load(context, specs, model);
  }
}

Sam3SegmentationOp::Sam3SegmentationOp(Sam3SegmentationConfig config) : config_(std::move(config)) {
  validate_config(config_);
  table_ = read_prompt_table(config_.prompt_table);
  validate_prompts(config_.prompts, table_);
  active_ = config_.prompts;
}

Sam3SegmentationOp::~Sam3SegmentationOp() { release(); }

void Sam3SegmentationOp::poll_control() {
  if (!config_.prompt_control) {
    return;
  }
  auto [generation, prompts] = config_.prompt_control->latest();
  if (generation == control_generation_) {
    return;
  }
  control_generation_ = generation;
  try {
    validate_prompts(prompts, table_);
    if (engines_) {
      for (const Sam3Prompt& prompt : prompts) {
        (void)make_decoder_prompt(prompt, table_, engines_->max_boxes);
      }
    }
  } catch (const std::invalid_argument& error) {
    HOLOSCAN_LOG_WARN("Sam3SegmentationOp: keeping the prompts; the new ones are unusable: {}",
                      error.what());
    return;
  }
  pending_ = std::move(prompts);
}

void Sam3SegmentationOp::setup(holoscan::OperatorSpec& spec) {
  const auto capacity = static_cast<std::uint64_t>(config_.max_instances);
  const std::uint64_t pixels = static_cast<std::uint64_t>(config_.max_width) *
                               static_cast<std::uint64_t>(config_.max_height);
  // Two slots: one image in compute() while the next one arrives.
  spec.input(image, "image").queue_depth(2U);
  if (config_.output_masks) {
    spec.output(masks, "masks").produces_tensor(device_tensor(kUInt8, 3U, capacity * pixels));
  }
  spec.output(labels, "labels").produces_tensor(device_tensor(kUInt16, 2U, pixels * 2U));
  spec.output(scores, "scores").produces_tensor(device_tensor(kFloat32, 1U, capacity * 4U));
  spec.output(boxes, "boxes").produces_tensor(device_tensor(kFloat32, 2U, capacity * 16U));
  spec.output(prompt_ids, "prompt_ids").produces_tensor(device_tensor(kInt32, 1U, capacity * 4U));
  spec.output(count, "count").produces_tensor(device_tensor(kInt32, 1U, 4U));
}

holoscan::Contract Sam3SegmentationOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{image});
  return result;
}

void Sam3SegmentationOp::start() {
  release();
  for (const auto& model : {config_.encoder, config_.decoder}) {
    HOLOSCAN_LOG_INFO(
        "Sam3SegmentationOp: loading {}{}", model.string(),
        is_onnx(model) ? "; building its TensorRT engine takes a while when none is cached" : "");
  }
  auto engines = std::make_unique<Engines>();
  engines->encoder_specs = encoder_specs(config_);
  engines->encoder = std::make_unique<infer::InferContext>();
  load(*engines->encoder, engines->encoder_specs, config_.encoder);
  engines->decoder_specs = decoder_specs(config_);
  engines->decoder = std::make_unique<infer::InferContext>();
  load(*engines->decoder, engines->decoder_specs, config_.decoder);

  // The encoder: one [1, 3, S, S] input, three [1, C, H, W] outputs.
  const auto encoder_inputs = engines->encoder->get_input_dimensions();
  const auto encoder_outputs = engines->encoder->get_output_dimensions();
  const auto in = encoder_inputs.find(kEncoderName);
  if (in == encoder_inputs.end() || in->second.size() != 1 || in->second[0].size() != 4 ||
      in->second[0][0] != 1 || in->second[0][1] != 3 || in->second[0][2] != in->second[0][3] ||
      in->second[0][2] < 1 || in->second[0][2] > kMaxModelSide) {
    fail_start(config_.encoder.string() + " takes " +
               (in == encoder_inputs.end() ? std::string("no input") : dims_text(in->second)) +
               ", not one [1, 3, S, S] image");
  }
  engines->input_size = static_cast<int>(in->second[0][2]);
  const auto features = encoder_outputs.find(kEncoderName);
  if (features == encoder_outputs.end() || features->second.size() != kFeatureNames.size() ||
      std::any_of(features->second.begin(), features->second.end(),
                  [](const auto& dims) { return dims.size() != 4 || dims[0] != 1; })) {
    fail_start(config_.encoder.string() + " gives " +
               (features == encoder_outputs.end() ? std::string("no outputs")
                                                  : dims_text(features->second)) +
               ", not three [1, C, H, W] feature maps");
  }

  // The decoder: the encoder's features, then the prompt.
  const auto decoder_inputs = engines->decoder->get_input_dimensions();
  const auto decoder_outputs = engines->decoder->get_output_dimensions();
  const auto din = decoder_inputs.find(kDecoderName);
  const std::size_t input_count = kFeatureNames.size() + kPromptNames.size();
  if (din == decoder_inputs.end() || din->second.size() != input_count) {
    fail_start(config_.decoder.string() + " takes " +
               (din == decoder_inputs.end() ? std::string("no inputs") : dims_text(din->second)) +
               "; it needs the three feature maps and five prompt inputs of "
               "tools/prepare_model.py's export");
  }
  const auto& dims = din->second;
  for (std::size_t i = 0; i < kFeatureNames.size(); ++i) {
    if (dims[i] != features->second[i]) {
      fail_start("the decoder's " + kFeatureNames[i] + " is " + dims_text(dims[i]) +
                 " but the encoder gives " + dims_text(features->second[i]) +
                 "; use an encoder and a decoder of the same export");
    }
  }
  const auto& text = dims[3];
  const auto& text_mask = dims[4];
  const auto& box_dims = dims[5];
  const auto& label_dims = dims[6];
  const auto& valid_dims = dims[7];
  if (text.size() != 3 || text[0] != 1 || text_mask.size() != 2 || text_mask[0] != 1 ||
      text_mask[1] != text[1] || box_dims.size() != 3 || box_dims[0] != 1 || box_dims[2] != 4 ||
      label_dims != std::vector<std::int64_t>{1, box_dims[1]} || valid_dims != label_dims) {
    fail_start(config_.decoder.string() + " takes prompts as " +
               dims_text(std::vector<std::vector<std::int64_t>>(dims.begin() + 3, dims.end())) +
               ", not text_features [1, T, D], text_mask [1, T], boxes [1, B, 4], box_labels "
               "[1, B] and box_valid [1, B]");
  }
  engines->tokens = static_cast<int>(text[1]);
  engines->dim = static_cast<int>(text[2]);
  engines->max_boxes = static_cast<int>(box_dims[1]);
  if (engines->tokens != table_.tokens || engines->dim != table_.dim) {
    fail_start("the decoder takes text features of " + std::to_string(engines->tokens) + " x " +
               std::to_string(engines->dim) + " but " + config_.prompt_table.string() + " holds " +
               std::to_string(table_.tokens) + " x " + std::to_string(table_.dim) +
               "; use the prompt table of the same export");
  }
  const auto out = decoder_outputs.find(kDecoderName);
  const bool shaped = out != decoder_outputs.end() && out->second.size() == 3 &&
                      out->second[0].size() == 4 && out->second[0][0] == 1 &&
                      out->second[1].size() == 3 && out->second[1][0] == 1 &&
                      out->second[1][1] == out->second[0][1] && out->second[1][2] == 4 &&
                      out->second[2] == std::vector<std::int64_t>{1, out->second[0][1]};
  if (!shaped || out->second[0][1] < 1 || out->second[0][1] > 4096 || out->second[0][2] < 1 ||
      out->second[0][2] > kMaxModelSide || out->second[0][3] < 1 ||
      out->second[0][3] > kMaxModelSide) {
    fail_start(config_.decoder.string() + " gives " +
               (out == decoder_outputs.end() ? std::string("no outputs") : dims_text(out->second)) +
               ", not masks [1, Q, M, M], boxes_xyxy [1, Q, 4] and scores [1, Q]");
  }
  engines->queries = static_cast<int>(out->second[0][1]);
  engines->mask_height = static_cast<int>(out->second[0][2]);
  engines->mask_width = static_cast<int>(out->second[0][3]);
  const auto output_buffer = [&](const std::string& name) {
    const auto found = engines->decoder_specs->output_per_model_.find(name);
    if (found == engines->decoder_specs->output_per_model_.end() ||
        !found->second->device_buffer_ ||
        found->second->get_datatype() != infer::holoinfer_datatype::h_Float32 ||
        found->second->device_buffer_->data() == nullptr) {
      fail_start(config_.decoder.string() + " must give float32 " + name);
    }
    return static_cast<const float*>(found->second->device_buffer_->data());
  };
  engines->masks = output_buffer(kDecoderOutputNames[0]);
  engines->boxes = output_buffer(kDecoderOutputNames[1]);
  engines->scores = output_buffer(kDecoderOutputNames[2]);

  // The encoder's input, which the preprocessing writes and TensorRT reads in
  // place.
  std::vector<std::int64_t> input_dims = in->second[0];
  infer::InferStatus status = infer::allocate_buffers(
      engines->encoder_specs->data_per_tensor_, input_dims, infer::holoinfer_datatype::h_Float32,
      kImageName, /*allocate_cuda=*/true, /*device_id=*/0);
  const auto input = engines->encoder_specs->data_per_tensor_.find(kImageName);
  if (status.get_code() != infer::holoinfer_code::H_SUCCESS ||
      input == engines->encoder_specs->data_per_tensor_.end() || !input->second->device_buffer_ ||
      input->second->device_buffer_->data() == nullptr) {
    fail_start("cannot allocate the encoder input: " + status.get_message());
  }
  // The decoder reads the encoder's outputs where the encoder writes them.
  for (const std::string& name : kFeatureNames) {
    const auto feature = engines->encoder_specs->output_per_model_.find(name);
    if (feature == engines->encoder_specs->output_per_model_.end() ||
        !feature->second->device_buffer_ ||
        feature->second->get_datatype() != infer::holoinfer_datatype::h_Float32) {
      fail_start(config_.encoder.string() + " must give float32 " + name);
    }
    engines->decoder_specs->data_per_tensor_[name] = feature->second;
  }
  // The prompt inputs, which load_prompts() fills for each prompt.
  for (std::size_t i = 0; i < kPromptNames.size(); ++i) {
    std::vector<std::int64_t> prompt_dims = dims[kFeatureNames.size() + i];
    status = infer::allocate_buffers(engines->decoder_specs->data_per_tensor_, prompt_dims,
                                     infer::holoinfer_datatype::h_Float32, kPromptNames[i],
                                     /*allocate_cuda=*/true, /*device_id=*/0);
    const auto buffer = engines->decoder_specs->data_per_tensor_.find(kPromptNames[i]);
    if (status.get_code() != infer::holoinfer_code::H_SUCCESS ||
        buffer == engines->decoder_specs->data_per_tensor_.end() ||
        !buffer->second->device_buffer_ || buffer->second->device_buffer_->data() == nullptr) {
      fail_start("cannot allocate the decoder's " + kPromptNames[i] + ": " + status.get_message());
    }
    engines->prompt_inputs[i] = static_cast<float*>(buffer->second->device_buffer_->data());
    std::size_t size = 1;
    for (const std::int64_t extent : prompt_dims) {
      size *= static_cast<std::size_t>(extent);
    }
    engines->prompt_sizes[i] = size;
  }

  if (const cudaError_t created = cudaEventCreateWithFlags(&frame_done_, cudaEventDisableTiming);
      created != cudaSuccess) {
    frame_done_ = nullptr;
    fail_start(std::string("cudaEventCreateWithFlags: ") + cudaGetErrorString(created));
  }
  void* passed = nullptr;
  if (const cudaError_t allocated =
          cudaHostAlloc(&passed, kMaxPrompts * sizeof(std::int32_t), cudaHostAllocMapped);
      allocated != cudaSuccess) {
    fail_start(std::string("cudaHostAlloc: ") + cudaGetErrorString(allocated));
  }
  passed_ = static_cast<std::int32_t*>(passed);
  std::fill_n(passed_, kMaxPrompts, 0);
  passed_prompts_ = 0;
  warned_dropped_ = false;
  engines->order =
      device_alloc(static_cast<std::size_t>(config_.max_instances) * sizeof(std::int32_t));
  HOLOSCAN_LOG_INFO(
      "Sam3SegmentationOp: ready, input [1, 3, {}, {}], {} queries, {}x{} masks, {} text tokens, "
      "{} box slots{}",
      engines->input_size, engines->input_size, engines->queries, engines->mask_width,
      engines->mask_height, engines->tokens, engines->max_boxes,
      config_.fp16 && (is_onnx(config_.encoder) || is_onnx(config_.decoder)) ? ", FP16 allowed"
                                                                             : "");
  engines_ = std::move(engines);
  // The prompts are uploaded by the first compute(): those of the last run,
  // or the control's newest.
  pending_ = active_;
  try {
    for (const Sam3Prompt& prompt : *pending_) {
      (void)make_decoder_prompt(prompt, table_, engines_->max_boxes);
    }
  } catch (const std::invalid_argument& error) {
    fail_start(error.what());
  }
  control_generation_ = 0;
  poll_control();
}

void Sam3SegmentationOp::stop() {
  // The last image's counts, before their memory goes.
  if (frame_done_ != nullptr && cudaEventSynchronize(frame_done_) == cudaSuccess) {
    warn_if_dropped();
  }
  release();
}

void Sam3SegmentationOp::warn_if_dropped() {
  if (warned_dropped_ || passed_ == nullptr) {
    return;
  }
  int passed = 0;
  for (int p = 0; p < passed_prompts_; ++p) {
    passed += passed_[p];
  }
  if (passed <= config_.max_instances) {
    return;
  }
  warned_dropped_ = true;
  HOLOSCAN_LOG_WARN(
      "Sam3SegmentationOp: {} instances passed the score threshold in an image, more than "
      "max_instances ({}); the lowest-ranked were dropped. Raise max_instances to keep them",
      passed, config_.max_instances);
}

void Sam3SegmentationOp::release() noexcept {
  // Freeing waits for the GPU work that may still use the buffers.
  engines_.reset();
  if (frame_done_ != nullptr) {
    cudaEventDestroy(frame_done_);
    frame_done_ = nullptr;
  }
  if (passed_ != nullptr) {
    cudaFreeHost(passed_);
    passed_ = nullptr;
  }
  if (staging_ != nullptr) {
    cudaFree(staging_);
    staging_ = nullptr;
    staging_bytes_ = 0;
  }
}

holoscan::expected<void, holoscan::Error> Sam3SegmentationOp::load_prompts(cudaStream_t stream) {
  poll_control();
  std::optional<std::vector<Sam3Prompt>> prompts;
  prompts.swap(pending_);
  if (!prompts) {
    return {};
  }
  Engines& e = *engines_;
  std::vector<float> host;
  host.reserve(prompts->size() * e.prompt_floats());
  try {
    for (const Sam3Prompt& prompt : *prompts) {
      const DecoderPrompt inputs = make_decoder_prompt(prompt, table_, e.max_boxes);
      for (const std::vector<float>* part :
           {&inputs.text_features, &inputs.text_mask, &inputs.boxes, &inputs.box_labels,
            &inputs.box_valid}) {
        host.insert(host.end(), part->begin(), part->end());
      }
    }
  } catch (const std::invalid_argument& error) {
    return holoscan::make_unexpected(invalid(error.what()));
  }
  const auto count = static_cast<int>(prompts->size());
  try {
    // A change is rare: wait for the work still using the buffers, then
    // replace them.
    if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
    }
    if (count > e.candidate_prompts) {
      const auto slots =
          static_cast<std::size_t>(count) * static_cast<std::size_t>(config_.max_instances);
      const std::size_t cells =
          static_cast<std::size_t>(e.mask_width) * static_cast<std::size_t>(e.mask_height);
      e.candidate_logits.reset();
      e.candidate_scores = device_alloc(slots * sizeof(float));
      e.candidate_boxes = device_alloc(slots * 4U * sizeof(float));
      e.candidate_queries = device_alloc(slots * sizeof(std::int32_t));
      e.candidate_counts = device_alloc(static_cast<std::size_t>(count) * sizeof(std::int32_t));
      e.candidate_logits = device_alloc(slots * cells * sizeof(float));
      e.candidate_prompts = count;
    }
    if (count > e.prompt_count) {
      e.prompts = device_alloc(host.size() * sizeof(float));
    }
  } catch (const std::runtime_error& error) {
    return holoscan::make_unexpected(failure(error.what()));
  }
  if (const cudaError_t status = cudaMemcpy(e.prompts.get(), host.data(),
                                            host.size() * sizeof(float), cudaMemcpyHostToDevice);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaMemcpy", status));
  }
  e.prompt_count = std::max(e.prompt_count, count);
  active_ = std::move(*prompts);
  return {};
}

holoscan::expected<const std::uint8_t*, holoscan::Error> Sam3SegmentationOp::stage(
    const holoscan::Tensor& tensor, int width, int height, int channels, std::size_t pitch,
    cudaStream_t stream) {
  const std::size_t row = static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
  const std::size_t bytes = row * static_cast<std::size_t>(height);
  if (staging_bytes_ < bytes) {
    // Grows a few times at most; the old buffer is freed once the stream is
    // done with it.
    if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
    }
    const std::size_t grown = std::max(bytes, staging_bytes_ * 2U);
    cudaFree(staging_);
    staging_ = nullptr;
    staging_bytes_ = 0;
    if (const cudaError_t status = cudaMalloc(&staging_, grown); status != cudaSuccess) {
      staging_ = nullptr;
      return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
    }
    staging_bytes_ = grown;
    HOLOSCAN_LOG_INFO(
        "Sam3SegmentationOp: host images are copied to the GPU first; give it GPU images to save "
        "the copy");
  }
  // Plain host memory: the runtime does not order its producer, so wait for
  // it; the copy then stages it before returning. Pinned memory: hold it
  // until the asynchronous copy has read it.
  const bool pinned = tensor.device().device_type == kDLCUDAHost;
  if (!pinned) {
    if (auto ready = tensor.synchronize_for_host(); !ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
  }
  if (const cudaError_t status =
          cudaMemcpy2DAsync(staging_, row, tensor.data(), pitch, row,
                            static_cast<std::size_t>(height), cudaMemcpyHostToDevice, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaMemcpy2DAsync", status));
  }
  if (pinned) {
    if (auto held = hold_until_done(tensor, stream); !held) {
      return holoscan::make_unexpected(std::move(held).error());
    }
  }
  return static_cast<const std::uint8_t*>(staging_);
}

holoscan::expected<void, holoscan::Error> Sam3SegmentationOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = image.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  if (!engines_) {
    return holoscan::make_unexpected(failure("the models are not loaded"));
  }
  // The runtime gives an operator with device outputs a stream, and writes
  // their loans on that stream only.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(failure("the runtime gave no CUDA stream"));
  }
  int device = 0;
  if (const cudaError_t status = cudaStreamGetDevice(stream, &device); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaStreamGetDevice", status));
  }
  if (device != 0) {
    return holoscan::make_unexpected(invalid("runs on GPU 0, where HoloInfer loads the models"));
  }
  // A finished previous image's counts of passed instances; one still running
  // is checked by a later one, or by stop().
  if (const cudaError_t query = cudaEventQuery(frame_done_); query == cudaSuccess) {
    warn_if_dropped();
  } else if (query == cudaErrorNotReady && cudaPeekAtLastError() == cudaErrorNotReady) {
    cudaGetLastError();  // Not an error; keep it from the kernel launch checks.
  }
  // The previous image may have run on another stream: wait for its work on
  // the buffers this one reuses, and mark this one's when it has been queued,
  // on every way out.
  if (const cudaError_t status = cudaStreamWaitEvent(stream, frame_done_, 0);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaStreamWaitEvent", status));
  }
  struct MarkDone {
    cudaEvent_t event;
    cudaStream_t stream;
    ~MarkDone() {
      if (event != nullptr) {
        cudaEventRecord(event, stream);
      }
    }
  } mark_done{frame_done_, stream};

  // The image.
  const holoscan::schema::ImageT& descriptor = sample->data;
  const auto format = pixel_format(descriptor.encoding);
  if (!format) {
    return holoscan::make_unexpected(
        invalid(std::string("unsupported image encoding ") +
                holoscan::schema::EnumNameImageEncoding(descriptor.encoding) +
                "; use RGB8, BGR8, RGBA8 or BGRA8"));
  }
  if (!descriptor.data) {
    return holoscan::make_unexpected(invalid("the image has no tensor"));
  }
  const holoscan::Tensor& tensor = *descriptor.data;
  const int pixel_channels = channels(*format);
  const DLDataType dtype = tensor.dtype();
  const std::span<const std::int64_t> shape = tensor.shape_span();
  if (dtype.code != kUInt8.code || dtype.bits != kUInt8.bits || dtype.lanes != kUInt8.lanes ||
      shape.size() != 3 || shape[2] != pixel_channels) {
    return holoscan::make_unexpected(
        invalid("the image tensor must be uint8 [height, width, channels] for its encoding"));
  }
  const std::vector<std::int64_t> strides = tensor.strides();
  const std::int64_t height = shape[0];
  const std::int64_t width = shape[1];
  if (height < 1 || width < 1 || height > config_.max_height || width > config_.max_width) {
    return holoscan::make_unexpected(
        invalid("the image is " + std::to_string(width) + "x" + std::to_string(height) +
                "; max_width x max_height is " + std::to_string(config_.max_width) + "x" +
                std::to_string(config_.max_height)));
  }
  if (strides.size() != 3 || strides[2] != 1 || strides[1] != pixel_channels ||
      strides[0] < width * pixel_channels) {
    return holoscan::make_unexpected(invalid("image pixels must be packed within each row"));
  }
  if ((descriptor.width != 0 && descriptor.width != width) ||
      (descriptor.height != 0 && descriptor.height != height)) {
    return holoscan::make_unexpected(
        invalid("the image's width and height disagree with its tensor"));
  }
  const DLDevice where = tensor.device();
  const bool on_gpu = where.device_type == kDLCUDA || where.device_type == kDLCUDAManaged;
  if (!on_gpu && where.device_type != kDLCPU && where.device_type != kDLCUDAHost) {
    return holoscan::make_unexpected(invalid("the image must be in GPU or host memory"));
  }
  if (where.device_type == kDLCUDA && where.device_id != device) {
    return holoscan::make_unexpected(invalid("the image is on another GPU than the operator"));
  }
  if (auto loaded = load_prompts(stream); !loaded) {
    return loaded;
  }
  const auto pitch = static_cast<std::size_t>(strides[0]);
  const std::uint8_t* pixels = nullptr;
  std::size_t pixels_pitch = pitch;
  if (on_gpu) {
    // The runtime has ordered the producer's writes before this stream.
    pixels = static_cast<const std::uint8_t*>(tensor.data());
  } else {
    auto staged = stage(tensor, static_cast<int>(width), static_cast<int>(height), pixel_channels,
                        pitch, stream);
    if (!staged) {
      return holoscan::make_unexpected(std::move(staged).error());
    }
    pixels = *staged;
    pixels_pitch = static_cast<std::size_t>(width) * static_cast<std::size_t>(pixel_channels);
  }

  // Preprocess, encode once, decode and select each prompt's instances, all
  // in order on the stream; HoloInfer joins it with events. Images before
  // this one are done with the buffers: see frame_done_.
  Engines& e = *engines_;
  const auto encoder_input = e.encoder_specs->data_per_tensor_.find(kImageName);
  if (const cudaError_t status = launch_preprocess(
          ImageView{.data = pixels,
                    .pitch = pixels_pitch,
                    .width = static_cast<int>(width),
                    .height = static_cast<int>(height),
                    .format = *format},
          e.input_size, e.input_size, Normalization{},
          static_cast<float*>(encoder_input->second->device_buffer_->data()), stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("preprocessing kernel", status));
  }
  infer::InferStatus inferred = e.encoder->execute_inference(e.encoder_specs, stream);
  if (inferred.get_code() != infer::holoinfer_code::H_SUCCESS) {
    return holoscan::make_unexpected(failure("the encoder failed: " + inferred.get_message()));
  }
  const int prompt_count = static_cast<int>(active_.size());
  const Candidates candidates{
      .scores = static_cast<float*>(e.candidate_scores.get()),
      .boxes = static_cast<float*>(e.candidate_boxes.get()),
      .queries = static_cast<std::int32_t*>(e.candidate_queries.get()),
      .logits = static_cast<float*>(e.candidate_logits.get()),
      .counts = static_cast<std::int32_t*>(e.candidate_counts.get()),
      .passed = passed_,
  };
  const float* prompt_inputs = static_cast<const float*>(e.prompts.get());
  for (int p = 0; p < prompt_count; ++p) {
    const float* from = prompt_inputs + static_cast<std::size_t>(p) * e.prompt_floats();
    for (std::size_t i = 0; i < e.prompt_inputs.size(); ++i) {
      if (const cudaError_t status =
              cudaMemcpyAsync(e.prompt_inputs[i], from, e.prompt_sizes[i] * sizeof(float),
                              cudaMemcpyDeviceToDevice, stream);
          status != cudaSuccess) {
        return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", status));
      }
      from += e.prompt_sizes[i];
    }
    inferred = e.decoder->execute_inference(e.decoder_specs, stream);
    if (inferred.get_code() != infer::holoinfer_code::H_SUCCESS) {
      return holoscan::make_unexpected(failure("the decoder failed: " + inferred.get_message()));
    }
    if (const cudaError_t status =
            launch_select(e.scores, e.boxes, e.masks,
                          SelectParams{.prompt = p,
                                       .queries = e.queries,
                                       .capacity = config_.max_instances,
                                       .mask_width = e.mask_width,
                                       .mask_height = e.mask_height,
                                       .score_threshold = config_.score_threshold},
                          candidates, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("selection kernel", status));
    }
  }
  passed_prompts_ = prompt_count;

  const std::int64_t capacity = config_.max_instances;
  std::optional<holoscan::expected<holoscan::TensorOutputLoan, holoscan::Error>> masks_loan;
  if (config_.output_masks) {
    masks_loan.emplace(allocate<3>(masks, {capacity, height, width}, kUInt8));
  }
  auto labels_loan = allocate<2>(labels, {height, width}, kUInt16);
  auto scores_loan = allocate<1>(scores, {capacity}, kFloat32);
  auto boxes_loan = allocate<2>(boxes, {capacity, 4}, kFloat32);
  auto prompt_ids_loan = allocate<1>(prompt_ids, {capacity}, kInt32);
  auto count_loan = allocate<1>(count, {1}, kInt32);
  if (masks_loan && !*masks_loan) {
    return holoscan::make_unexpected(masks_loan->error());
  }
  for (const auto* loan :
       {&labels_loan, &scores_loan, &boxes_loan, &prompt_ids_loan, &count_loan}) {
    if (!*loan) {
      return holoscan::make_unexpected(loan->error());
    }
  }
  std::optional<holoscan::expected<holoscan::TensorOutputWriteGuard, holoscan::Error>> masks_writer;
  if (masks_loan) {
    masks_writer.emplace((*masks_loan)->write(stream));
    if (!*masks_writer) {
      return holoscan::make_unexpected(masks_writer->error());
    }
  }
  auto labels_writer = labels_loan->write(stream);
  auto scores_writer = scores_loan->write(stream);
  auto boxes_writer = boxes_loan->write(stream);
  auto prompt_ids_writer = prompt_ids_loan->write(stream);
  auto count_writer = count_loan->write(stream);
  for (const auto* writer :
       {&labels_writer, &scores_writer, &boxes_writer, &prompt_ids_writer, &count_writer}) {
    if (!*writer) {
      return holoscan::make_unexpected(writer->error());
    }
  }
  const float mask_logit = logit_of(config_.mask_threshold);
  if (const cudaError_t status = launch_finalize(
          candidates,
          FinalizeParams{.prompts = prompt_count,
                         .capacity = config_.max_instances,
                         .mask_width = e.mask_width,
                         .mask_height = e.mask_height,
                         .width = static_cast<int>(width),
                         .height = static_cast<int>(height),
                         .mask_logit_threshold = mask_logit},
          Instances{.masks = masks_writer ? (*masks_writer)->data_as<std::uint8_t>() : nullptr,
                    .labels = labels_writer->data_as<std::uint16_t>(),
                    .scores = scores_writer->data_as<float>(),
                    .boxes = boxes_writer->data_as<float>(),
                    .prompt_ids = prompt_ids_writer->data_as<std::int32_t>(),
                    .count = count_writer->data_as<std::int32_t>(),
                    .order = static_cast<std::int32_t*>(e.order.get())},
          stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("finalization kernel", status));
  }
  if (masks_writer) {
    if (auto committed = std::move(**masks_writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
  }
  for (auto* writer :
       {&labels_writer, &scores_writer, &boxes_writer, &prompt_ids_writer, &count_writer}) {
    if (auto committed = std::move(**writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
  }
  mark_done.event = nullptr;
  if (const cudaError_t status = cudaEventRecord(frame_done_, stream); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaEventRecord", status));
  }

  // One set of options, so every output carries the image's capture time and
  // frame id. The count goes last: a consumer that triggers on it finds the
  // others there.
  const holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(*sample);
  if (masks_loan) {
    if (auto sent = masks.emit(std::move(**masks_loan), emit); !sent) {
      return sent;
    }
  }
  if (auto sent = labels.emit(std::move(*labels_loan), emit); !sent) {
    return sent;
  }
  if (auto sent = scores.emit(std::move(*scores_loan), emit); !sent) {
    return sent;
  }
  if (auto sent = boxes.emit(std::move(*boxes_loan), emit); !sent) {
    return sent;
  }
  if (auto sent = prompt_ids.emit(std::move(*prompt_ids_loan), emit); !sent) {
    return sent;
  }
  return count.emit(std::move(*count_loan), emit);
}

}  // namespace holoscan::sam3
