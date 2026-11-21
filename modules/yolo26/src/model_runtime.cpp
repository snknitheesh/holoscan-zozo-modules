// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "model_runtime.hpp"

#include <algorithm>
#include <cctype>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>

namespace holoscan::yolo26::detail {
namespace {

// HoloInfer's names for the model and its input. It binds tensors by
// position, so they need not match the names inside the model.
constexpr const char* kModelName = "yolo26";
constexpr const char* kInputName = "images";

std::string prefixed(std::string_view op, const std::string& message) {
  return std::string(op) + ": " + message;
}

// HoloInfer's reason for not loading the model, and what the operator
// expects when the reason is the model's outputs.
std::string load_failure(const ModelSettings& settings, const std::string& reason) {
  std::string message = "cannot load " + settings.model.string() + ": " + reason;
  std::string lower = reason;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lower.find("output") != std::string::npos && !settings.expected_outputs.empty()) {
    message += "; the model must give " + settings.expected_outputs;
  }
  return message;
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

// HoloInfer's settings for the model, the same for prepare() and load() so
// that both find the same cached engine.
std::shared_ptr<infer::InferenceSpecs> make_specs(const ModelSettings& settings) {
  auto specs = std::make_shared<infer::InferenceSpecs>(
      "trt", infer::Mappings{}, infer::Mappings{{kModelName, settings.model.string()}},
      infer::MultiMappings{{kModelName, {kInputName}}},
      infer::MultiMappings{{kModelName, settings.outputs}}, infer::Mappings{},
      infer::MultiMappings{},
      /*dynamic_input_dims=*/false, /*is_engine_path=*/!is_onnx(settings.model),
      /*parallel_proc=*/false, settings.fp16, /*cuda_buffer_in=*/true, /*cuda_buffer_out=*/true,
      settings.cuda_graphs, /*dla_core=*/-1, /*dla_gpu_fallback=*/false,
      /*gpu_resident_inference=*/false, std::function<cudaStream_t(std::int32_t)>{});
  specs->max_memory_ = settings.build_workspace_bytes;
  return specs;
}

// Keeps a pinned host tensor's memory from reuse until the work queued on
// `stream` so far is done: its allocator waits for the stream if it can,
// otherwise this waits.
holoscan::expected<void, holoscan::Error> hold_until_done(const holoscan::Tensor& tensor,
                                                          cudaStream_t stream,
                                                          std::string_view op) {
  holoscan::Tensor handle = tensor;
  if (handle.set_deallocation_stream(stream)) {
    return {};
  }
  if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure(op, "cudaStreamSynchronize", status));
  }
  return {};
}

holoscan::expected<const std::uint8_t*, holoscan::Error> stage(
    const holoscan::Tensor& tensor, int width, int height, int channels, std::size_t pitch,
    cudaStream_t stream, void*& staging, std::size_t& staging_bytes, std::string_view op) {
  const std::size_t row = static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
  const std::size_t bytes = row * static_cast<std::size_t>(height);
  if (staging_bytes < bytes) {
    // Grows a few times at most; the old buffer is freed once the stream is
    // done with it.
    if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure(op, "cudaStreamSynchronize", status));
    }
    const std::size_t grown = std::max(bytes, staging_bytes * 2U);
    cudaFree(staging);
    staging = nullptr;
    staging_bytes = 0;
    if (const cudaError_t status = cudaMalloc(&staging, grown); status != cudaSuccess) {
      staging = nullptr;
      return holoscan::make_unexpected(cuda_failure(op, "cudaMalloc", status));
    }
    staging_bytes = grown;
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
          cudaMemcpy2DAsync(staging, row, tensor.data(), pitch, row,
                            static_cast<std::size_t>(height), cudaMemcpyHostToDevice, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure(op, "cudaMemcpy2DAsync", status));
  }
  if (pinned) {
    if (auto held = hold_until_done(tensor, stream, op); !held) {
      return holoscan::make_unexpected(std::move(held).error());
    }
  }
  return static_cast<const std::uint8_t*>(staging);
}

}  // namespace

holoscan::Error invalid(std::string_view op, const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, prefixed(op, message)};
}

holoscan::Error failure(std::string_view op, const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, prefixed(op, message)};
}

holoscan::Error cuda_failure(std::string_view op, const char* what, cudaError_t status) {
  return failure(op, std::string(what) + ": " + cudaGetErrorString(status));
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

std::string dims_text(const std::vector<std::int64_t>& dims) {
  std::string text = "[";
  for (std::size_t i = 0; i < dims.size(); ++i) {
    text += (i == 0 ? "" : ", ") + std::to_string(dims[i]);
  }
  return text + "]";
}

void prepare(const ModelSettings& settings, std::string_view op) {
  if (!is_onnx(settings.model)) {
    return;
  }
  HOLOSCAN_LOG_INFO(
      "{}: preparing {}; building its TensorRT engine takes a while when none is cached", op,
      settings.model.string());
  auto specs = make_specs(settings);
  infer::InferContext context;
  const infer::InferStatus status = context.set_inference_params(specs);
  if (status.get_code() != infer::holoinfer_code::H_SUCCESS) {
    throw std::runtime_error(prefixed(op, load_failure(settings, status.get_message())));
  }
}

LoadedModel::~LoadedModel() {
  // The context's TensorRT objects go before the buffers they were bound to.
  context.reset();
  specs.reset();
}

std::unique_ptr<LoadedModel> load(const ModelSettings& settings, std::string_view op) {
  const int size = settings.input_size;
  const std::string name = settings.model.string();
  HOLOSCAN_LOG_INFO("{}: loading {}{}", op, name,
                    is_onnx(settings.model)
                        ? "; building its TensorRT engine takes a while when none is cached"
                        : "");
  auto model = std::make_unique<LoadedModel>();
  model->specs = make_specs(settings);
  model->context = std::make_unique<infer::InferContext>();
  infer::InferStatus status = model->context->set_inference_params(model->specs);
  if (status.get_code() != infer::holoinfer_code::H_SUCCESS) {
    throw std::runtime_error(prefixed(op, load_failure(settings, status.get_message())));
  }

  // The model must take [1, 3, S, S].
  const auto inputs = model->context->get_input_dimensions();
  const std::vector<std::int64_t> expected_input{1, 3, size, size};
  const auto in = inputs.find(kModelName);
  if (in == inputs.end() || in->second.size() != 1 || in->second[0] != expected_input) {
    throw std::runtime_error(prefixed(
        op, name + " takes " +
                (in == inputs.end() || in->second.size() != 1 ? std::string("other inputs")
                                                              : dims_text(in->second[0])) +
                ", not one " + dims_text(expected_input) +
                " input; export it for this input size, and delete any engine cached beside it"));
  }
  const auto outputs = model->context->get_output_dimensions();
  const auto out = outputs.find(kModelName);
  if (out != outputs.end() && out->second.size() == settings.outputs.size()) {
    model->output_dims = out->second;
  }
  // The callers check the dimensions; a buffer is needed for each output.
  for (const std::string& output_name : settings.outputs) {
    const auto output = model->specs->output_per_model_.find(output_name);
    if (output == model->specs->output_per_model_.end() || !output->second->device_buffer_ ||
        output->second->get_datatype() != infer::holoinfer_datatype::h_Float32 ||
        output->second->device_buffer_->data() == nullptr) {
      model->outputs.clear();
      break;
    }
    model->outputs.push_back(static_cast<const float*>(output->second->device_buffer_->data()));
  }

  // The input buffer, which the letterbox writes and TensorRT reads in place.
  std::vector<std::int64_t> input_dims = expected_input;
  status = infer::allocate_buffers(model->specs->data_per_tensor_, input_dims,
                                   infer::holoinfer_datatype::h_Float32, kInputName,
                                   /*allocate_cuda=*/true, /*device_id=*/0);
  const auto input = model->specs->data_per_tensor_.find(kInputName);
  if (status.get_code() != infer::holoinfer_code::H_SUCCESS ||
      input == model->specs->data_per_tensor_.end() || !input->second->device_buffer_) {
    throw std::runtime_error(
        prefixed(op, "cannot allocate the model input: " + status.get_message()));
  }
  model->input = static_cast<float*>(input->second->device_buffer_->data());
  if (model->input == nullptr) {
    throw std::runtime_error(prefixed(op, "the model buffers are not allocated"));
  }
  return model;
}

holoscan::expected<void, holoscan::Error> infer_on(LoadedModel& model, cudaStream_t stream,
                                                   std::string_view op) {
  infer::InferStatus inferred = model.context->execute_inference(model.specs, stream);
  if (inferred.get_code() != infer::holoinfer_code::H_SUCCESS) {
    return holoscan::make_unexpected(failure(op, "inference failed: " + inferred.get_message()));
  }
  return {};
}

holoscan::expected<cudaStream_t, holoscan::Error> frame_stream(holoscan::ExecutionContext& context,
                                                               std::string_view op) {
  // The runtime gives an operator with device outputs a stream, and writes
  // their loans on that stream only.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(failure(op, "the runtime gave no CUDA stream"));
  }
  int device = 0;
  if (const cudaError_t status = cudaStreamGetDevice(stream, &device); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure(op, "cudaStreamGetDevice", status));
  }
  if (device != 0) {
    return holoscan::make_unexpected(invalid(op, "runs on GPU 0, where HoloInfer loads the model"));
  }
  return stream;
}

bool finished(cudaEvent_t event) noexcept {
  const cudaError_t query = cudaEventQuery(event);
  if (query == cudaErrorNotReady && cudaPeekAtLastError() == cudaErrorNotReady) {
    cudaGetLastError();  // Not an error; keep it from the kernel launch checks.
  }
  return query == cudaSuccess;
}

holoscan::expected<ImageView, holoscan::Error> image_input(const holoscan::schema::ImageT& image,
                                                           cudaStream_t stream, void*& staging,
                                                           std::size_t& staging_bytes,
                                                           std::string_view op) {
  const auto format = pixel_format(image.encoding);
  if (!format) {
    return holoscan::make_unexpected(
        invalid(op, std::string("unsupported image encoding ") +
                        holoscan::schema::EnumNameImageEncoding(image.encoding) +
                        "; use RGB8, BGR8, RGBA8 or BGRA8"));
  }
  if (!image.data) {
    return holoscan::make_unexpected(invalid(op, "the image has no tensor"));
  }
  const holoscan::Tensor& tensor = *image.data;
  const int pixel_channels = channels(*format);
  const DLDataType dtype = tensor.dtype();
  const std::span<const std::int64_t> shape = tensor.shape_span();
  if (dtype.code != kUInt8.code || dtype.bits != kUInt8.bits || dtype.lanes != kUInt8.lanes ||
      shape.size() != 3 || shape[2] != pixel_channels) {
    return holoscan::make_unexpected(
        invalid(op, "the image tensor must be uint8 [height, width, channels] for its encoding"));
  }
  const std::vector<std::int64_t> strides = tensor.strides();
  const std::int64_t height = shape[0];
  const std::int64_t width = shape[1];
  if (height < 1 || width < 1 || height > kMaxImageSide || width > kMaxImageSide) {
    return holoscan::make_unexpected(
        invalid(op, "image sides must be 1.." + std::to_string(kMaxImageSide)));
  }
  if (strides.size() != 3 || strides[2] != 1 || strides[1] != pixel_channels ||
      strides[0] < width * pixel_channels) {
    return holoscan::make_unexpected(invalid(op, "image pixels must be packed within each row"));
  }
  if ((image.width != 0 && image.width != width) || (image.height != 0 && image.height != height)) {
    return holoscan::make_unexpected(
        invalid(op, "the image's width and height disagree with its tensor"));
  }
  const DLDevice where = tensor.device();
  const bool on_gpu = where.device_type == kDLCUDA || where.device_type == kDLCUDAManaged;
  if (!on_gpu && where.device_type != kDLCPU && where.device_type != kDLCUDAHost) {
    return holoscan::make_unexpected(invalid(op, "the image must be in GPU or host memory"));
  }
  if (where.device_type == kDLCUDA && where.device_id != 0) {
    return holoscan::make_unexpected(invalid(op, "the image is on another GPU than the operator"));
  }
  ImageView view{.data = nullptr,
                 .pitch = static_cast<std::size_t>(strides[0]),
                 .width = static_cast<int>(width),
                 .height = static_cast<int>(height),
                 .format = *format};
  if (on_gpu) {
    // The runtime has ordered the producer's writes before this stream.
    view.data = static_cast<const std::uint8_t*>(tensor.data());
    return view;
  }
  auto staged = stage(tensor, view.width, view.height, pixel_channels, view.pitch, stream, staging,
                      staging_bytes, op);
  if (!staged) {
    return holoscan::make_unexpected(std::move(staged).error());
  }
  view.data = *staged;
  view.pitch = static_cast<std::size_t>(width) * static_cast<std::size_t>(pixel_channels);
  return view;
}

void release_staging(void*& staging, std::size_t& staging_bytes) noexcept {
  if (staging != nullptr) {
    cudaFree(staging);
    staging = nullptr;
  }
  staging_bytes = 0;
}

}  // namespace holoscan::yolo26::detail
