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
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/mono_depth/kernels.hpp>
#include <holoscan/mono_depth/mono_depth_op.hpp>
#include <holoscan/sensor_io/provenance.hpp>

namespace holoscan::mono_depth {
namespace {

namespace infer = holoscan::inference;

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

// HoloInfer's names for the model and its tensors. It binds tensors by
// position, so they need not match the names inside the model.
constexpr const char* kModelName = "mono_depth";
constexpr const char* kInputName = "image";
constexpr const char* kOutputName = "depth";

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "MonoDepthOp: " + message};
}

holoscan::Error failure(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "MonoDepthOp: " + message};
}

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return failure(std::string(what) + ": " + cudaGetErrorString(status));
}

holoscan::TensorOutputSpec device_image(DLDataType dtype, std::uint32_t rank, std::uint64_t bytes) {
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

holoscan::PayloadOptions image_payload() {
  return holoscan::PayloadOptions{
      .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
  };
}

holoscan::schema::ImageT descriptor(int width, int height,
                                    holoscan::schema::ImageEncoding encoding) {
  holoscan::schema::ImageT image;
  image.width = width;
  image.height = height;
  image.encoding = encoding;
  return image;
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

// HoloInfer's settings for `config`'s model, the same for prepare_engine() and
// start() so that both find the same cached engine.
std::shared_ptr<infer::InferenceSpecs> make_specs(const MonoDepthConfig& config) {
  auto specs = std::make_shared<infer::InferenceSpecs>(
      "trt", infer::Mappings{}, infer::Mappings{{kModelName, config.model.string()}},
      infer::MultiMappings{{kModelName, {kInputName}}},
      infer::MultiMappings{{kModelName, {kOutputName}}}, infer::Mappings{}, infer::MultiMappings{},
      /*dynamic_input_dims=*/false, /*is_engine_path=*/!is_onnx(config.model),
      /*parallel_proc=*/false, config.fp16, /*cuda_buffer_in=*/true, /*cuda_buffer_out=*/true,
      config.cuda_graphs, /*dla_core=*/-1, /*dla_gpu_fallback=*/false,
      /*gpu_resident_inference=*/false, std::function<cudaStream_t(std::int32_t)>{});
  specs->max_memory_ = config.build_workspace_bytes;
  return specs;
}

holoscan::expected<holoscan::TensorOutputLoan, holoscan::Error> allocate(
    holoscan::Output<holoscan::schema::ImageT>& port, std::span<const std::int64_t> shape,
    DLDataType dtype) {
  return port.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
}

bool finite(float value) { return std::isfinite(value); }

}  // namespace

// The loaded model: HoloInfer's context and specifications, which own the
// input and output buffers that TensorRT reads and writes.
struct MonoDepthOp::Engine {
  std::shared_ptr<infer::InferenceSpecs> specs;
  std::unique_ptr<infer::InferContext> context;
  float* input{};
  const float* output{};
  int output_width{};
  int output_height{};

  ~Engine() {
    // The context's TensorRT objects go before the buffers they were bound to.
    context.reset();
    specs.reset();
  }
};

bool gives_metric_depth(const MonoDepthConfig& config) noexcept {
  return config.model_output == ModelOutput::kMetricDepth ||
         config.metric_from_relative.has_value();
}

void validate_config(const MonoDepthConfig& config) {
  if (config.model.empty()) {
    throw std::invalid_argument("a model path is required");
  }
  if (config.input_width < 1 || config.input_height < 1 || config.input_width > kMaxModelSide ||
      config.input_height > kMaxModelSide) {
    throw std::invalid_argument("input_width and input_height must be 1.." +
                                std::to_string(kMaxModelSide));
  }
  for (std::size_t c = 0; c < 3; ++c) {
    if (!finite(config.normalization.mean[c]) || !finite(config.normalization.std[c]) ||
        !(config.normalization.std[c] > 0.0F)) {
      throw std::invalid_argument("the normalization needs a finite mean and a positive std");
    }
  }
  if (config.metric_from_relative) {
    if (config.model_output != ModelOutput::kRelativeInverseDepth) {
      throw std::invalid_argument("metric_from_relative applies to a relative model only");
    }
    if (!finite(config.metric_from_relative->scale) ||
        !finite(config.metric_from_relative->shift) || config.metric_from_relative->scale == 0) {
      throw std::invalid_argument(
          "metric_from_relative needs a finite, nonzero scale and a "
          "finite shift");
    }
  }
  if (config.color_range &&
      !(finite((*config.color_range)[0]) && finite((*config.color_range)[1]) &&
        (*config.color_range)[0] < (*config.color_range)[1])) {
    throw std::invalid_argument("color_range must be finite, its first value below its second");
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

void prepare_engine(const MonoDepthConfig& config) {
  if (!is_onnx(config.model)) {
    return;
  }
  HOLOSCAN_LOG_INFO(
      "MonoDepthOp: preparing {}; building its TensorRT engine takes a while when none is cached",
      config.model.string());
  auto specs = make_specs(config);
  infer::InferContext context;
  const infer::InferStatus status = context.set_inference_params(specs);
  if (status.get_code() != infer::holoinfer_code::H_SUCCESS) {
    throw std::runtime_error("MonoDepthOp: cannot load " + config.model.string() + ": " +
                             status.get_message());
  }
}

MonoDepthOp::MonoDepthOp(MonoDepthConfig config) : config_(std::move(config)) {
  validate_config(config_);
}

MonoDepthOp::~MonoDepthOp() { release(); }

void MonoDepthOp::setup(holoscan::OperatorSpec& spec) {
  const std::uint64_t pixels = static_cast<std::uint64_t>(config_.max_width) *
                               static_cast<std::uint64_t>(config_.max_height);
  // Two slots: one image in compute() while the next one arrives.
  spec.input(image, "image").queue_depth(2U);
  spec.output(depth, "depth")
      .produces_tensor(device_image(kFloat32, 2U, pixels * 4U))
      .payload_options(image_payload());
  if (gives_metric_depth(config_)) {
    spec.output(metric_depth, "metric_depth")
        .produces_tensor(device_image(kFloat32, 2U, pixels * 4U))
        .payload_options(image_payload());
  }
  if (config_.colorize) {
    spec.output(colorized, "colorized")
        .produces_tensor(device_image(kUInt8, 3U, pixels * 4U))
        .payload_options(image_payload());
  }
}

holoscan::Contract MonoDepthOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{image});
  return result;
}

void MonoDepthOp::start() {
  release();
  const bool onnx = is_onnx(config_.model);
  HOLOSCAN_LOG_INFO("MonoDepthOp: loading {}{}", config_.model.string(),
                    onnx ? "; building its TensorRT engine takes a while when none is cached" : "");
  auto engine = std::make_unique<Engine>();
  engine->specs = make_specs(config_);
  engine->context = std::make_unique<infer::InferContext>();
  infer::InferStatus status = engine->context->set_inference_params(engine->specs);
  if (status.get_code() != infer::holoinfer_code::H_SUCCESS) {
    throw std::runtime_error("MonoDepthOp: cannot load " + config_.model.string() + ": " +
                             status.get_message());
  }

  // The model must take [1, 3, H, W] and give [1, H', W'] or [1, 1, H', W'].
  const auto inputs = engine->context->get_input_dimensions();
  const auto outputs = engine->context->get_output_dimensions();
  const std::vector<std::int64_t> expected_input{1, 3, config_.input_height, config_.input_width};
  const auto in = inputs.find(kModelName);
  if (in == inputs.end() || in->second.size() != 1 || in->second[0] != expected_input) {
    throw std::runtime_error(
        "MonoDepthOp: " + config_.model.string() + " takes " +
        (in == inputs.end() || in->second.size() != 1 ? std::string("other inputs")
                                                      : dims_text(in->second[0])) +
        ", not one " + dims_text(expected_input) +
        " input; export it for this input size, and delete any engine cached beside it");
  }
  const auto out = outputs.find(kModelName);
  const std::vector<std::int64_t>* dims =
      out != outputs.end() && out->second.size() == 1 ? &out->second[0] : nullptr;
  const bool shaped = dims != nullptr &&
                      ((dims->size() == 3 && (*dims)[0] == 1) ||
                       (dims->size() == 4 && (*dims)[0] == 1 && (*dims)[1] == 1)) &&
                      dims->back() >= 1 && dims->back() <= kMaxModelSide &&
                      (*dims)[dims->size() - 2] >= 1 && (*dims)[dims->size() - 2] <= kMaxModelSide;
  if (!shaped) {
    throw std::runtime_error("MonoDepthOp: " + config_.model.string() + " gives " +
                             (dims == nullptr ? std::string("other outputs") : dims_text(*dims)) +
                             ", not one [1, H, W] or [1, 1, H, W] depth output");
  }
  engine->output_height = static_cast<int>((*dims)[dims->size() - 2]);
  engine->output_width = static_cast<int>(dims->back());
  const auto output = engine->specs->output_per_model_.find(kOutputName);
  if (output == engine->specs->output_per_model_.end() || !output->second->device_buffer_ ||
      output->second->get_datatype() != infer::holoinfer_datatype::h_Float32) {
    throw std::runtime_error("MonoDepthOp: " + config_.model.string() + " must give float32 depth");
  }
  engine->output = static_cast<const float*>(output->second->device_buffer_->data());

  // The input buffer, which the preprocessing writes and TensorRT reads in
  // place.
  std::vector<std::int64_t> input_dims = expected_input;
  status = infer::allocate_buffers(engine->specs->data_per_tensor_, input_dims,
                                   infer::holoinfer_datatype::h_Float32, kInputName,
                                   /*allocate_cuda=*/true, /*device_id=*/0);
  const auto input = engine->specs->data_per_tensor_.find(kInputName);
  if (status.get_code() != infer::holoinfer_code::H_SUCCESS ||
      input == engine->specs->data_per_tensor_.end() || !input->second->device_buffer_) {
    throw std::runtime_error("MonoDepthOp: cannot allocate the model input: " +
                             status.get_message());
  }
  engine->input = static_cast<float*>(input->second->device_buffer_->data());
  if (engine->input == nullptr || engine->output == nullptr) {
    throw std::runtime_error("MonoDepthOp: the model buffers are not allocated");
  }
  if (const cudaError_t created = cudaEventCreateWithFlags(&frame_done_, cudaEventDisableTiming);
      created != cudaSuccess) {
    frame_done_ = nullptr;
    throw std::runtime_error(std::string("MonoDepthOp: cudaEventCreateWithFlags: ") +
                             cudaGetErrorString(created));
  }
  if (config_.colorize && !config_.color_range) {
    void* range = nullptr;
    if (const cudaError_t allocated = cudaMalloc(&range, 2 * sizeof(std::uint32_t));
        allocated != cudaSuccess) {
      throw std::runtime_error(std::string("MonoDepthOp: cudaMalloc: ") +
                               cudaGetErrorString(allocated));
    }
    range_ = static_cast<std::uint32_t*>(range);
  }
  HOLOSCAN_LOG_INFO("MonoDepthOp: {} ready, input [1, 3, {}, {}], output {}x{}{}",
                    config_.model.string(), config_.input_height, config_.input_width,
                    engine->output_width, engine->output_height,
                    config_.fp16 && onnx ? ", FP16 allowed" : "");
  engine_ = std::move(engine);
}

void MonoDepthOp::stop() { release(); }

void MonoDepthOp::release() noexcept {
  // Freeing waits for the GPU work that may still use the buffers.
  engine_.reset();
  if (frame_done_ != nullptr) {
    cudaEventDestroy(frame_done_);
    frame_done_ = nullptr;
  }
  if (range_ != nullptr) {
    cudaFree(range_);
    range_ = nullptr;
  }
  if (staging_ != nullptr) {
    cudaFree(staging_);
    staging_ = nullptr;
    staging_bytes_ = 0;
  }
}

holoscan::expected<const std::uint8_t*, holoscan::Error> MonoDepthOp::stage(
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
        "MonoDepthOp: host images are copied to the GPU first; give it GPU images to save the "
        "copy");
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

holoscan::expected<void, holoscan::Error> MonoDepthOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = image.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  if (!engine_) {
    return holoscan::make_unexpected(failure("the model is not loaded"));
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
    return holoscan::make_unexpected(invalid("runs on GPU 0, where HoloInfer loads the model"));
  }
  // The previous frame may have run on another stream: wait for its work on
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
  const holoscan::schema::ImageT& descriptor_in = sample->data;
  const auto format = pixel_format(descriptor_in.encoding);
  if (!format) {
    return holoscan::make_unexpected(
        invalid(std::string("unsupported image encoding ") +
                holoscan::schema::EnumNameImageEncoding(descriptor_in.encoding) +
                "; use RGB8, BGR8, RGBA8 or BGRA8"));
  }
  if (!descriptor_in.data) {
    return holoscan::make_unexpected(invalid("the image has no tensor"));
  }
  const holoscan::Tensor& tensor = *descriptor_in.data;
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
  if ((descriptor_in.width != 0 && descriptor_in.width != width) ||
      (descriptor_in.height != 0 && descriptor_in.height != height)) {
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

  // Preprocess into the model input, run the model, resample its output into
  // the outputs, all in order on the stream; HoloInfer joins it with events.
  // Frames before this one are done with the buffers: see frame_done_.
  const ImageFit fit = make_fit(static_cast<int>(width), static_cast<int>(height),
                                config_.input_width, config_.input_height, engine_->output_width,
                                engine_->output_height, config_.resize_mode);
  if (const cudaError_t status =
          launch_preprocess(ImageView{.data = pixels,
                                      .pitch = pixels_pitch,
                                      .width = static_cast<int>(width),
                                      .height = static_cast<int>(height),
                                      .format = *format},
                            fit, config_.normalization, engine_->input, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("preprocessing kernel", status));
  }
  infer::InferStatus inferred = engine_->context->execute_inference(engine_->specs, stream);
  if (inferred.get_code() != infer::holoinfer_code::H_SUCCESS) {
    return holoscan::make_unexpected(failure("inference failed: " + inferred.get_message()));
  }

  const std::array<std::int64_t, 2> plane{height, width};
  const std::array<std::int64_t, 3> color_shape{height, width, 4};
  const bool metric = gives_metric_depth(config_);
  auto depth_loan = allocate(depth, plane, kFloat32);
  if (!depth_loan) {
    return holoscan::make_unexpected(std::move(depth_loan).error());
  }
  std::optional<holoscan::TensorOutputLoan> metric_loan;
  if (metric) {
    auto loan = allocate(metric_depth, plane, kFloat32);
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    metric_loan.emplace(std::move(*loan));
  }
  std::optional<holoscan::TensorOutputLoan> color_loan;
  if (config_.colorize) {
    auto loan = allocate(colorized, color_shape, kUInt8);
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    color_loan.emplace(std::move(*loan));
  }
  auto depth_writer = depth_loan->write(stream);
  if (!depth_writer) {
    return holoscan::make_unexpected(std::move(depth_writer).error());
  }
  std::optional<holoscan::TensorOutputWriteGuard> metric_writer;
  if (metric_loan) {
    auto writer = metric_loan->write(stream);
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    metric_writer.emplace(std::move(*writer));
  }
  std::optional<holoscan::TensorOutputWriteGuard> color_writer;
  if (color_loan) {
    auto writer = color_loan->write(stream);
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    color_writer.emplace(std::move(*writer));
  }

  float* inverse = depth_writer->data_as<float>();
  const InverseDepthAffine* affine =
      config_.metric_from_relative ? &*config_.metric_from_relative : nullptr;
  if (const cudaError_t status = launch_postprocess(
          engine_->output, fit, config_.model_output, affine,
          DepthOutputs{.inverse = inverse,
                       .metric = metric_writer ? metric_writer->data_as<float>() : nullptr},
          stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("postprocessing kernel", status));
  }
  if (color_writer) {
    ColorRange range;
    if (config_.color_range) {
      range.low = (*config_.color_range)[0];
      range.high = (*config_.color_range)[1];
    } else {
      if (const cudaError_t status = launch_depth_range(
              inverse, static_cast<std::size_t>(width) * static_cast<std::size_t>(height), range_,
              stream);
          status != cudaSuccess) {
        return holoscan::make_unexpected(cuda_failure("depth range kernel", status));
      }
      range.range = range_;
    }
    if (const cudaError_t status =
            launch_colorize(inverse, static_cast<int>(width), static_cast<int>(height), range,
                            color_writer->data_as<std::uint8_t>(), stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("colorize kernel", status));
    }
  }
  if (auto committed = std::move(*depth_writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  for (auto* writer : {&metric_writer, &color_writer}) {
    if (*writer) {
      if (auto committed = std::move(**writer).commit(); !committed) {
        return holoscan::make_unexpected(std::move(committed).error());
      }
    }
  }
  mark_done.event = nullptr;
  if (const cudaError_t status = cudaEventRecord(frame_done_, stream); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaEventRecord", status));
  }

  // One set of options, so every output carries the image's capture time and
  // frame id.
  const auto w = static_cast<int>(width);
  const auto h = static_cast<int>(height);
  const holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(*sample);
  if (auto sent = depth.emit_tensor(
          std::move(*depth_loan), descriptor(w, h, holoscan::schema::ImageEncoding_MONO32F), emit);
      !sent) {
    return sent;
  }
  if (metric_loan) {
    if (auto sent = metric_depth.emit_tensor(
            std::move(*metric_loan), descriptor(w, h, holoscan::schema::ImageEncoding_MONO32F),
            emit);
        !sent) {
      return sent;
    }
  }
  if (color_loan) {
    return colorized.emit_tensor(std::move(*color_loan),
                                 descriptor(w, h, holoscan::schema::ImageEncoding_RGBA8), emit);
  }
  return {};
}

}  // namespace holoscan::mono_depth
