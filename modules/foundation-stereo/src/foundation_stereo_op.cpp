// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/foundation_stereo/foundation_stereo_op.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoinfer/holoinfer.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/foundation_stereo/camera_info.hpp>
#include <holoscan/foundation_stereo/kernels.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/sensor_io/provenance.hpp>

namespace holoscan::foundation_stereo {
namespace {

namespace infer = holoscan::inference;

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr int kMaxSide = 16384;

// HoloInfer's names for the model and its tensors. It binds tensors by
// position, so they need not match the names inside the model, but these are
// those of a single-file Fast-FoundationStereo export.
constexpr const char* kModelName = "foundation_stereo";
constexpr const char* kLeftName = "left_image";
constexpr const char* kRightName = "right_image";
constexpr const char* kOutputName = "disparity";

// Frames whose GPU time may be pending at once; see State::timing.
constexpr std::size_t kTimingSlots = 4;
// Frames on the GPU at once: compute() waits for the frame before last before
// it queues another. This is the operator's only GPU throttle: without it, a
// live source on drop-oldest edges would queue frames faster than the model
// runs them, and each disparity would come out later than the one before.
constexpr std::size_t kMaxFramesInFlight = 2;
static_assert(kMaxFramesInFlight < kTimingSlots);

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "FoundationStereoOp: " + message};
}

holoscan::Error failure(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "FoundationStereoOp: " + message};
}

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return failure(std::string(what) + ": " + cudaGetErrorString(status));
}

// The device's free and total memory, for an error about loading a model.
std::string memory_text() {
  std::size_t free = 0;
  std::size_t total = 0;
  if (cudaMemGetInfo(&free, &total) != cudaSuccess) {
    cudaGetLastError();
    return {};
  }
  return "; the GPU has " + std::to_string(free >> 20U) + " of " + std::to_string(total >> 20U) +
         " MiB free";
}

bool is_onnx(const std::filesystem::path& path) {
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return extension == ".onnx";
}

// A 64-bit FNV-1a digest of a file's bytes, in hex, or nullopt if it cannot be
// read.
std::optional<std::string> file_digest(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::nullopt;
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  std::vector<char> chunk(std::size_t{1} << 20U);
  while (in) {
    in.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
    for (std::streamsize i = 0; i < in.gcount(); ++i) {
      hash = (hash ^ static_cast<unsigned char>(chunk[static_cast<std::size_t>(i)])) *
             0x100000001b3ULL;
    }
  }
  if (in.bad()) return std::nullopt;
  std::ostringstream text;
  text << std::hex << std::setw(16) << std::setfill('0') << hash;
  return text.str();
}

// HoloInfer names the engine it caches for an .onnx model after the model's
// name, the GPU, the TensorRT version and the build options, not after the
// model's content: an engine built for a model that has since been replaced
// under the same name would be loaded as it is. So the content's digest is
// kept beside the model, and when it no longer matches, the cached engines
// (<stem>.*engine*) are deleted, to be built again.
void drop_stale_engines(const std::filesystem::path& model) {
  if (!is_onnx(model)) return;
  const auto digest = file_digest(model);
  if (!digest) return;  // HoloInfer reports a model it cannot read.
  std::filesystem::path record = model;
  record += ".digest";
  {
    std::ifstream in(record);
    std::string recorded;
    if (in && std::getline(in, recorded) && recorded == *digest) return;
  }
  const std::filesystem::path folder = model.has_parent_path() ? model.parent_path() : ".";
  const std::string prefix = model.stem().string() + ".";
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator(folder, error)) {
    const std::string name = entry.path().filename().string();
    if (name.starts_with(prefix) && name.find(".engine", prefix.size()) != std::string::npos) {
      std::error_code removed;
      if (std::filesystem::remove(entry.path(), removed)) {
        HOLOSCAN_LOG_INFO(
            "FoundationStereoOp: deleted {}, cached for another content of {}; it is built again",
            entry.path().string(), model.string());
      }
    }
  }
  // Written whole, then renamed, so that a concurrent reader sees the old or
  // the new digest.
  std::filesystem::path temporary = record;
  temporary += ".tmp";
  {
    std::ofstream out(temporary, std::ios::trunc);
    out << *digest << '\n';
    if (!out.flush()) {
      HOLOSCAN_LOG_WARN("FoundationStereoOp: cannot write {}; a changed {} is not detected",
                        record.string(), model.string());
      return;
    }
  }
  std::filesystem::rename(temporary, record, error);
  if (error) {
    std::filesystem::remove(temporary, error);
    HOLOSCAN_LOG_WARN("FoundationStereoOp: cannot write {}; a changed {} is not detected",
                      record.string(), model.string());
  }
}

std::optional<PixelFormat> pixel_format(holoscan::schema::ImageEncoding encoding) {
  switch (encoding) {
    case holoscan::schema::ImageEncoding_MONO8:
      return PixelFormat::kMono8;
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

// One eye's pixels, after checking that its tensor matches its encoding.
struct EyeView {
  const holoscan::Tensor* tensor{};
  const std::uint8_t* data{};
  int width{};
  int height{};
  std::size_t pitch{};
  PixelFormat format{PixelFormat::kMono8};
  bool on_device{};
  bool pinned{};
};

holoscan::expected<EyeView, holoscan::Error> view_eye(const holoscan::schema::ImageT& image,
                                                      const std::string& side) {
  if (!image.data) {
    return holoscan::make_unexpected(invalid(side + " image has no tensor"));
  }
  const holoscan::Tensor& tensor = *image.data;
  const auto format = pixel_format(image.encoding);
  if (!format) {
    return holoscan::make_unexpected(
        invalid("unsupported " + side + " image encoding " +
                holoscan::schema::EnumNameImageEncoding(image.encoding) +
                "; use MONO8, RGB8, BGR8, RGBA8 or BGRA8"));
  }
  const DLDataType dtype = tensor.dtype();
  if (dtype.code != kDLUInt || dtype.bits != 8 || dtype.lanes != 1) {
    return holoscan::make_unexpected(invalid(side + " image tensor must hold uint8 elements"));
  }
  const std::span<const std::int64_t> shape = tensor.shape_span();
  const int pixel_channels = channels(*format);
  const bool hwc = shape.size() == 3 && shape[2] == pixel_channels;
  const bool hw = shape.size() == 2 && pixel_channels == 1;
  if (!hwc && !hw) {
    return holoscan::make_unexpected(invalid(
        side + " image tensor shape must be [height, width, channels] matching its encoding"));
  }
  if (shape[0] < 1 || shape[1] < 1 || shape[0] > kMaxSide || shape[1] > kMaxSide) {
    return holoscan::make_unexpected(invalid(side + " image sides must be 1..16384"));
  }
  const std::vector<std::int64_t> strides = tensor.strides();
  if (strides[1] != pixel_channels || (hwc && strides[2] != 1) ||
      strides[0] < shape[1] * pixel_channels) {
    return holoscan::make_unexpected(invalid(side + " image pixels must be packed within rows"));
  }
  if ((image.width != 0 && image.width != shape[1]) ||
      (image.height != 0 && image.height != shape[0])) {
    return holoscan::make_unexpected(
        invalid(side + " image width and height disagree with its tensor"));
  }
  const DLDevice device = tensor.device();
  const bool on_device = device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
  if (!on_device && device.device_type != kDLCPU && device.device_type != kDLCUDAHost) {
    return holoscan::make_unexpected(invalid(side + " image must be in host or CUDA memory"));
  }
  if ((device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged) &&
      device.device_id != 0) {
    return holoscan::make_unexpected(
        invalid(side + " image is on GPU " + std::to_string(device.device_id) +
                "; the operator runs on GPU 0, where HoloInfer loads the model"));
  }
  return EyeView{.tensor = &tensor,
                 .data = tensor.data_as<std::uint8_t>(),
                 .width = static_cast<int>(shape[1]),
                 .height = static_cast<int>(shape[0]),
                 .pitch = static_cast<std::size_t>(strides[0]),
                 .format = *format,
                 .on_device = on_device,
                 .pinned = device.device_type == kDLCUDAHost};
}

holoscan::TensorOutputSpec device_image(std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = kFloat32,
              .rank = 2U,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(bytes),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

holoscan::PayloadOptions image_payload() {
  return holoscan::PayloadOptions{.schema = holoscan::schema_identity<holoscan::schema::ImageT>()};
}

// HoloInfer's settings for `config`'s model, the same for prepare_engine() and
// start() so that both find the same cached engine. One image pair a run: the
// batch dimension, dynamic in some exports, is built for 1.
std::shared_ptr<infer::InferenceSpecs> make_specs(const FoundationStereoConfig& config) {
  auto specs = std::make_shared<infer::InferenceSpecs>(
      "trt", infer::Mappings{}, infer::Mappings{{kModelName, config.model.string()}},
      infer::MultiMappings{{kModelName, {kLeftName, kRightName}}},
      infer::MultiMappings{{kModelName, {kOutputName}}}, infer::Mappings{}, infer::MultiMappings{},
      /*dynamic_input_dims=*/false, /*is_engine_path=*/!is_onnx(config.model),
      /*parallel_proc=*/false, config.fp16, /*cuda_buffer_in=*/true, /*cuda_buffer_out=*/true,
      config.cuda_graphs, /*dla_core=*/-1, /*dla_gpu_fallback=*/false,
      /*gpu_resident_inference=*/false, std::function<cudaStream_t(std::int32_t)>{},
      /*max_batch_size=*/1, config.build_workspace_bytes);
  return specs;
}

// Keeps a pinned host tensor's memory from reuse until the work queued on
// `stream` so far is done, by having its allocator wait for the stream;
// false when its memory cannot (an imported tensor), which the caller must
// then hold until its copy is done. It cannot be held past compute(): the
// runtime waits for a frame's inputs to be released before it stops.
bool release_after(const holoscan::Tensor& tensor, cudaStream_t stream) {
  holoscan::Tensor handle = tensor;
  return handle.set_deallocation_stream(stream);
}

}  // namespace

struct FoundationStereoOp::State {
  struct DeviceFree {
    void operator()(void* memory) const noexcept { cudaFree(memory); }
  };

  // The loaded model: HoloInfer's context and specifications, which own the
  // input and output buffers that TensorRT reads and writes.
  std::shared_ptr<infer::InferenceSpecs> specs;
  std::unique_ptr<infer::InferContext> context;
  float* left_input{};
  float* right_input{};
  const float* output{};
  int model_width{};
  int model_height{};

  // Device copies of host images, both eyes, made at the largest size on the
  // first host image.
  std::unique_ptr<void, DeviceFree> staging;
  std::size_t staging_bytes{};
  // A start and an end event per frame, in a ring. The model's buffers and
  // the staging buffer are reused by every frame, but the runtime may give
  // each activation another stream, so each frame's stream waits for the
  // previous frame's end event before touching them. The pair also times the
  // frame, read once it has completed.
  struct Timing {
    cudaEvent_t start{};
    cudaEvent_t end{};
    bool pending{};  // `end` is recorded and not yet seen complete.
    bool timed{};    // The frame ran to the end: its time counts.
    bool first{};    // The run's first frame, which also captures the CUDA graph.
  };
  std::array<Timing, kTimingSlots> timing{};
  std::size_t next_slot{};
  std::uint64_t queued{};  // Frames queued in this run.
  cudaEvent_t last_end{};
  // Recorded after the copies of host images whose memory the runtime cannot
  // hold for the GPU, which compute() then waits for before it returns.
  cudaEvent_t copied{};
  bool copy_logged{};
  // The newest valid stereo camera info, and the depth's camera made from it,
  // with the frame ids it was made from: a camera info whose calibration is
  // the same may still name another frame.
  messages::LatestCameraInfo<messages::schema::StereoCameraInfoT> camera_info;
  messages::schema::CameraInfoT depth_camera_info;
  std::optional<std::pair<std::string, std::string>> depth_frame_ids;

  // Once the frame in `slot` is done (with `wait`, after waiting for it), adds
  // its GPU time to the stats.
  void harvest(Timing& slot, FoundationStereoStats* stats, bool wait) noexcept {
    if (!slot.pending) return;
    const cudaError_t done = wait ? cudaEventSynchronize(slot.end) : cudaEventQuery(slot.end);
    if (done == cudaErrorNotReady) {
      cudaGetLastError();  // Not an error; keep it from the launch checks.
      return;
    }
    slot.pending = false;
    float ms = 0.0F;
    if (!slot.timed || done != cudaSuccess ||
        cudaEventElapsedTime(&ms, slot.start, slot.end) != cudaSuccess) {
      cudaGetLastError();
      return;
    }
    if (stats == nullptr) return;
    const auto ns = static_cast<std::uint64_t>(static_cast<double>(ms) * 1e6);
    if (slot.first) {
      stats->gpu_ns_first = ns;
      return;
    }
    stats->timed_frames.fetch_add(1);
    stats->gpu_ns_total.fetch_add(ns);
    std::uint64_t previous = stats->gpu_ns_max.load();
    while (ns > previous && !stats->gpu_ns_max.compare_exchange_weak(previous, ns)) {
    }
  }

  void release() noexcept {
    // The context's TensorRT objects go before the buffers they were bound
    // to; freeing waits for the GPU work that may still use the buffers.
    context.reset();
    specs.reset();
    left_input = nullptr;
    right_input = nullptr;
    output = nullptr;
    staging.reset();
    staging_bytes = 0;
    for (Timing& slot : timing) {
      if (slot.start != nullptr) cudaEventDestroy(slot.start);
      if (slot.end != nullptr) cudaEventDestroy(slot.end);
      slot = Timing{};
    }
    next_slot = 0;
    queued = 0;
    last_end = nullptr;
    if (copied != nullptr) cudaEventDestroy(copied);
    copied = nullptr;
  }

  ~State() { release(); }
};

void validate_config(const FoundationStereoConfig& config) {
  if (config.model.empty()) {
    throw std::invalid_argument("a model path is required");
  }
  if (config.depth_camera_info && !config.depth) {
    throw std::invalid_argument("the depth_camera_info output goes with the depth output");
  }
  if (config.camera_info_input.max_age <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("camera_info_input.max_age must be > 0");
  }
  if (config.max_width < 1 || config.max_height < 1 || config.max_width > kMaxSide ||
      config.max_height > kMaxSide) {
    throw std::invalid_argument("max_width and max_height must be 1..16384");
  }
  if (config.build_workspace_bytes < (std::size_t{1} << 24U)) {
    throw std::invalid_argument("build_workspace_bytes must be at least 16 MiB");
  }
}

void prepare_engine(const FoundationStereoConfig& config) {
  if (!is_onnx(config.model)) {
    return;
  }
  HOLOSCAN_LOG_INFO(
      "FoundationStereoOp: preparing {}; building its TensorRT engine takes minutes when none "
      "is cached; an interrupt (Ctrl-C) takes effect only once that is done",
      config.model.string());
  drop_stale_engines(config.model);
  auto specs = make_specs(config);
  infer::InferContext context;
  const infer::InferStatus status = context.set_inference_params(specs);
  if (status.get_code() != infer::holoinfer_code::H_SUCCESS) {
    throw std::runtime_error("FoundationStereoOp: cannot load " + config.model.string() + ": " +
                             status.get_message() + memory_text());
  }
}

FoundationStereoOp::FoundationStereoOp(FoundationStereoConfig config)
    : config_(std::move(config)), state_(std::make_unique<State>()) {
  validate_config(config_);
}

FoundationStereoOp::~FoundationStereoOp() = default;

void FoundationStereoOp::setup(holoscan::OperatorSpec& spec) {
  const auto bytes = static_cast<std::uint64_t>(config_.max_width) *
                     static_cast<std::uint64_t>(config_.max_height) * sizeof(float);
  // Two slots per input: one frame in compute() while the next one arrives.
  spec.input(left, "left").queue_depth(2U);
  spec.input(right, "right").queue_depth(2U);
  spec.input(camera_info, "camera_info").queue_depth(2U);
  // Optional outputs exist only when configured: an emit into an unconnected
  // output fails.
  spec.output(disparity, "disparity")
      .produces_tensor(device_image(bytes))
      .payload_options(image_payload());
  if (config_.depth) {
    spec.output(depth, "depth")
        .produces_tensor(device_image(bytes))
        .payload_options(image_payload());
  }
  if (config_.depth_camera_info) {
    spec.output(depth_camera_info, "depth_camera_info");
  }
}

holoscan::Contract FoundationStereoOp::contract() const {
  // The left and right images of a pair carry the same capture time.
  return messages::camera_info_contract(config_.camera_info_input, camera_info, left, right);
}

void FoundationStereoOp::start() {
  State& state = *state_;
  state.release();
  state.copy_logged = false;
  state.camera_info.reset();
  state.depth_frame_ids.reset();
  const bool onnx = is_onnx(config_.model);
  HOLOSCAN_LOG_INFO("FoundationStereoOp: loading {}{}", config_.model.string(),
                    onnx ? "; building its TensorRT engine takes minutes when none is cached" : "");
  std::size_t free_before = 0;
  std::size_t total = 0;
  const bool measured = cudaMemGetInfo(&free_before, &total) == cudaSuccess;

  drop_stale_engines(config_.model);
  state.specs = make_specs(config_);
  state.context = std::make_unique<infer::InferContext>();
  infer::InferStatus status = state.context->set_inference_params(state.specs);
  if (status.get_code() != infer::holoinfer_code::H_SUCCESS) {
    const std::string message = status.get_message();
    state.release();
    throw std::runtime_error("FoundationStereoOp: cannot load " + config_.model.string() + ": " +
                             message + memory_text());
  }

  // The model must take two equal [1, 3, H, W] inputs and give [1, 1, H, W].
  const auto inputs = state.context->get_input_dimensions();
  const auto outputs = state.context->get_output_dimensions();
  const auto in = inputs.find(kModelName);
  const auto out = outputs.find(kModelName);
  const auto model_name = config_.model.string();
  if (in == inputs.end() || in->second.size() != 2 || in->second[0] != in->second[1] ||
      in->second[0].size() != 4 || in->second[0][0] != 1 || in->second[0][1] != 3 ||
      in->second[0][2] < 1 || in->second[0][3] < 1) {
    std::string found = "other inputs";
    if (in != inputs.end()) {
      found.clear();
      for (const auto& dims : in->second) found += (found.empty() ? "" : " and ") + dims_text(dims);
    }
    state.release();
    throw std::runtime_error("FoundationStereoOp: " + model_name + " takes " + found +
                             ", not two [1, 3, H, W] images; export a single-file "
                             "Fast-FoundationStereo model with tools/prepare_model.py");
  }
  const std::vector<std::int64_t> input_dims = in->second[0];
  const std::int64_t height = input_dims[2];
  const std::int64_t width = input_dims[3];
  const std::vector<std::int64_t> expected_output{1, 1, height, width};
  if (out == outputs.end() || out->second.size() != 1 || out->second[0] != expected_output) {
    const std::string found = out == outputs.end() || out->second.size() != 1
                                  ? std::string("other outputs")
                                  : dims_text(out->second[0]);
    state.release();
    throw std::runtime_error("FoundationStereoOp: " + model_name + " gives " + found +
                             ", not one disparity " + dims_text(expected_output));
  }
  const auto output = state.specs->output_per_model_.find(kOutputName);
  if (output == state.specs->output_per_model_.end() || !output->second->device_buffer_ ||
      output->second->get_datatype() != infer::holoinfer_datatype::h_Float32) {
    state.release();
    throw std::runtime_error("FoundationStereoOp: " + model_name + " must give float32 disparity");
  }
  state.output = static_cast<const float*>(output->second->device_buffer_->data());

  // The input buffers, which the preprocessing writes and TensorRT reads in place.
  for (const char* name : {kLeftName, kRightName}) {
    std::vector<std::int64_t> dims = input_dims;
    status = infer::allocate_buffers(state.specs->data_per_tensor_, dims,
                                     infer::holoinfer_datatype::h_Float32, name,
                                     /*allocate_cuda=*/true, /*device_id=*/0);
    const auto buffer = state.specs->data_per_tensor_.find(name);
    if (status.get_code() != infer::holoinfer_code::H_SUCCESS ||
        buffer == state.specs->data_per_tensor_.end() || !buffer->second->device_buffer_ ||
        buffer->second->device_buffer_->data() == nullptr) {
      const std::string message = status.get_message();
      state.release();
      throw std::runtime_error("FoundationStereoOp: cannot allocate the model input " +
                               std::string(name) + ": " + message);
    }
    (std::string_view(name) == kLeftName ? state.left_input : state.right_input) =
        static_cast<float*>(buffer->second->device_buffer_->data());
  }
  if (state.output == nullptr) {
    state.release();
    throw std::runtime_error("FoundationStereoOp: the model output is not allocated");
  }
  if (cudaEventCreateWithFlags(&state.copied, cudaEventDisableTiming) != cudaSuccess) {
    const cudaError_t error = cudaGetLastError();
    state.release();
    throw std::runtime_error(std::string("FoundationStereoOp: cudaEventCreate: ") +
                             cudaGetErrorString(error));
  }
  for (State::Timing& slot : state.timing) {
    if (cudaEventCreate(&slot.start) != cudaSuccess || cudaEventCreate(&slot.end) != cudaSuccess) {
      const cudaError_t error = cudaGetLastError();
      state.release();
      throw std::runtime_error(std::string("FoundationStereoOp: cudaEventCreate: ") +
                               cudaGetErrorString(error));
    }
  }
  state.model_width = static_cast<int>(width);
  state.model_height = static_cast<int>(height);

  std::size_t free_after = 0;
  const std::uint64_t used =
      measured && cudaMemGetInfo(&free_after, &total) == cudaSuccess && free_after < free_before
          ? free_before - free_after
          : 0U;
  if (config_.stats) {
    config_.stats->model_width = state.model_width;
    config_.stats->model_height = state.model_height;
    config_.stats->model_bytes = used;
  }
  HOLOSCAN_LOG_INFO(
      "FoundationStereoOp: {} ready, input [1, 3, {}, {}]{}; the GPU's free memory fell by {:.0f} "
      "MiB while loading it",
      model_name, height, width, config_.fp16 && onnx ? " (FP16 allowed)" : "",
      static_cast<double>(used) / (1024.0 * 1024.0));
}

void FoundationStereoOp::stop() {
  State& state = *state_;
  for (State::Timing& slot : state.timing) {
    state.harvest(slot, config_.stats.get(), /*wait=*/true);
  }
  state.release();
}

holoscan::expected<void, holoscan::Error> FoundationStereoOp::compute(
    holoscan::ExecutionContext& context) {
  State& state = *state_;
  auto info = state.camera_info.update(camera_info);
  if (!info) {
    return holoscan::make_unexpected(std::move(info).error());
  }
  if (*info == nullptr) {
    return {};  // No camera info yet: the frame is skipped.
  }
  const messages::StereoCameraInfo& pair_info = **info;
  auto left_sample = left.receive();
  if (!left_sample) {
    return holoscan::make_unexpected(std::move(left_sample).error());
  }
  auto right_sample = right.receive();
  if (!right_sample) {
    return holoscan::make_unexpected(std::move(right_sample).error());
  }
  if (!state.context) {
    return holoscan::make_unexpected(failure("the model is not loaded"));
  }

  std::array<EyeView, 2> eyes;
  for (int side = 0; side < 2; ++side) {
    auto view =
        view_eye(side == 0 ? left_sample->data : right_sample->data, side == 0 ? "left" : "right");
    if (!view) {
      return holoscan::make_unexpected(std::move(view).error());
    }
    eyes[side] = *view;
  }
  const int width = eyes[0].width;
  const int height = eyes[0].height;
  const auto size = [](int w, int h) { return std::to_string(w) + "x" + std::to_string(h); };
  if (eyes[1].width != width || eyes[1].height != height) {
    return holoscan::make_unexpected(invalid("the left image is " + size(width, height) +
                                             " and the right " +
                                             size(eyes[1].width, eyes[1].height)));
  }
  if (width > config_.max_width || height > config_.max_height) {
    return holoscan::make_unexpected(invalid("image " + size(width, height) +
                                             " exceeds the configured " +
                                             size(config_.max_width, config_.max_height)));
  }
  if (auto problem = camera_info_problem(pair_info, width, height)) {
    return holoscan::make_unexpected(invalid(*problem));
  }
  if (config_.depth_camera_info) {
    auto frame_ids = std::make_pair(pair_info.header.frame_id, pair_info.left.header.frame_id);
    if (state.camera_info.changed() || state.depth_frame_ids != frame_ids) {
      try {
        state.depth_camera_info =
            messages::to_message(foundation_stereo::depth_camera_info(pair_info));
      } catch (const std::invalid_argument& error) {
        return holoscan::make_unexpected(invalid(error.what()));
      }
      state.depth_frame_ids = std::move(frame_ids);
    }
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

  // At most kMaxFramesInFlight frames on the GPU: wait for the frame before
  // last. Earlier frames' GPU time is taken when done; the slot this frame
  // takes, older still, is done too.
  for (State::Timing& slot : state.timing) {
    state.harvest(slot, config_.stats.get(), /*wait=*/false);
  }
  state.harvest(state.timing[(state.next_slot + kTimingSlots - kMaxFramesInFlight) % kTimingSlots],
                config_.stats.get(), /*wait=*/true);
  State::Timing& slot = state.timing[state.next_slot];
  state.harvest(slot, config_.stats.get(), /*wait=*/true);
  state.next_slot = (state.next_slot + 1) % kTimingSlots;
  slot.timed = false;
  slot.first = state.queued++ == 0;

  // The previous frame may have run on another stream: wait for its work on
  // the buffers this one reuses, and mark this one's when it has been queued,
  // on every way out (without timing a frame that did not run to the end).
  if (state.last_end != nullptr) {
    if (const cudaError_t status = cudaStreamWaitEvent(stream, state.last_end, 0);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaStreamWaitEvent", status));
    }
  }
  struct MarkDone {
    State& state;
    State::Timing& slot;
    cudaStream_t stream;
    ~MarkDone() {
      if (cudaEventRecord(slot.end, stream) == cudaSuccess) {
        state.last_end = slot.end;
        slot.pending = true;
      } else {
        cudaGetLastError();
      }
    }
  } mark_done{state, slot, stream};
  // Imported pinned inputs, which the runtime cannot hold for the GPU: kept
  // until their copies are done, which is waited for on every way out, once
  // the frame's other GPU work is queued.
  struct HeldInputs {
    cudaEvent_t copied;
    cudaStream_t stream;
    std::vector<holoscan::Tensor> tensors;
    bool recorded{};
    ~HeldInputs() {
      if (tensors.empty()) return;
      if (!recorded || cudaEventSynchronize(copied) != cudaSuccess) {
        cudaGetLastError();
        cudaStreamSynchronize(stream);
      }
    }
  } held{state.copied, stream, {}};

  // Host images go to the GPU first, through the staging buffer: pageable
  // memory once its producer is done with it, pinned memory held until the
  // asynchronous copy has read it.
  const bool any_host = !eyes[0].on_device || !eyes[1].on_device;
  if (any_host) {
    const std::size_t half = static_cast<std::size_t>(width) * height * 4U;
    if (!state.staging) {
      // Once, at the largest size, so no frame ever waits to grow it.
      const std::size_t bytes = static_cast<std::size_t>(config_.max_width) *
                                static_cast<std::size_t>(config_.max_height) * 4U * 2U;
      void* memory = nullptr;
      if (const cudaError_t status = cudaMalloc(&memory, bytes); status != cudaSuccess) {
        return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
      }
      state.staging.reset(memory);
      state.staging_bytes = bytes;
    }
    if (!state.copy_logged) {
      state.copy_logged = true;
      std::size_t bytes = 0;
      for (const EyeView& eye : eyes) {
        if (!eye.on_device)
          bytes += static_cast<std::size_t>(width) * height * channels(eye.format);
      }
      HOLOSCAN_LOG_INFO(
          "FoundationStereoOp: {} in host memory, so each frame copies {} bytes ({:.1f} MB) "
          "host to device before the model; to avoid the copy, send CUDA device images",
          eyes[0].on_device   ? "the right image is"
          : eyes[1].on_device ? "the left image is"
                              : "both images are",
          bytes, static_cast<double>(bytes) / 1e6);
    }
    for (int side = 0; side < 2; ++side) {
      EyeView& eye = eyes[side];
      if (eye.on_device) continue;
      if (!eye.pinned) {
        if (auto ready = eye.tensor->synchronize_for_host(); !ready) {
          return holoscan::make_unexpected(std::move(ready).error());
        }
      }
      const std::size_t row = static_cast<std::size_t>(width) * channels(eye.format);
      auto* target = static_cast<std::uint8_t*>(state.staging.get()) + side * half;
      if (const cudaError_t status = cudaMemcpy2DAsync(target, row, eye.data, eye.pitch, row,
                                                       height, cudaMemcpyHostToDevice, stream);
          status != cudaSuccess) {
        return holoscan::make_unexpected(cuda_failure("cudaMemcpy2DAsync", status));
      }
      if (eye.pinned && !release_after(*eye.tensor, stream)) {
        held.tensors.push_back(*eye.tensor);
      }
      eye.data = target;
      eye.pitch = row;
    }
    if (!held.tensors.empty()) {
      held.recorded = cudaEventRecord(held.copied, stream) == cudaSuccess;
    }
  }

  // Preprocess both eyes, run the model, postprocess into the outputs, all in
  // order on the stream; HoloInfer joins it with events. Device images: the
  // runtime has ordered their producers' writes before this stream.
  if (const cudaError_t status = cudaEventRecord(slot.start, stream); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaEventRecord", status));
  }
  for (int side = 0; side < 2; ++side) {
    const EyeView& eye = eyes[side];
    if (const cudaError_t status =
            launch_preprocess(ImageView{.data = eye.data,
                                        .pitch = eye.pitch,
                                        .width = width,
                                        .height = height,
                                        .format = eye.format},
                              state.model_width, state.model_height,
                              side == 0 ? state.left_input : state.right_input, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("preprocess kernel", status));
    }
  }
  const infer::InferStatus inferred = state.context->execute_inference(state.specs, stream);
  if (inferred.get_code() != infer::holoinfer_code::H_SUCCESS) {
    return holoscan::make_unexpected(failure("inference failed: " + inferred.get_message()));
  }

  // The loans stay where they are made: a write guard refers to its loan.
  const std::array<std::int64_t, 2> shape{height, width};
  std::optional<holoscan::TensorOutputLoan> disparity_loan;
  std::optional<holoscan::TensorOutputLoan> depth_loan;
  std::optional<holoscan::TensorOutputWriteGuard> disparity_writer;
  std::optional<holoscan::TensorOutputWriteGuard> depth_writer;
  const auto take = [stream, &shape](holoscan::Output<holoscan::schema::ImageT>& output,
                                     std::optional<holoscan::TensorOutputLoan>& loan,
                                     std::optional<holoscan::TensorOutputWriteGuard>& writer)
      -> holoscan::expected<void, holoscan::Error> {
    auto allocated =
        output.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kFloat32});
    if (!allocated) {
      return holoscan::make_unexpected(std::move(allocated).error());
    }
    loan.emplace(std::move(*allocated));
    auto opened = loan->write(stream);
    if (!opened) {
      return holoscan::make_unexpected(std::move(opened).error());
    }
    writer.emplace(std::move(*opened));
    if (writer->data_as<float>() == nullptr) {
      return holoscan::make_unexpected(failure("the runtime gave an output without memory"));
    }
    return {};
  };
  if (auto taken = take(disparity, disparity_loan, disparity_writer); !taken) {
    return taken;
  }
  if (config_.depth) {
    if (auto taken = take(depth, depth_loan, depth_writer); !taken) {
      return taken;
    }
  }
  const PostprocessParams params{
      .model_width = state.model_width,
      .model_height = state.model_height,
      .width = width,
      .height = height,
      .fx_baseline = static_cast<float>(messages::fx_baseline(pair_info)),
      .doffs = static_cast<float>(pair_info.doffs),
      .invalidate_out_of_view = config_.invalidate_out_of_view};
  if (const cudaError_t status =
          launch_postprocess(state.output, params, disparity_writer->data_as<float>(),
                             depth_writer ? depth_writer->data_as<float>() : nullptr, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("postprocess kernel", status));
  }
  slot.timed = true;
  if (config_.stats) {
    config_.stats->frames.fetch_add(1);
  }

  // The outputs carry the left image's capture time, frame id and header.
  const holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(*left_sample);
  const auto send =
      [&](holoscan::Output<holoscan::schema::ImageT>& output, holoscan::TensorOutputLoan& loan,
          holoscan::TensorOutputWriteGuard& writer) -> holoscan::expected<void, holoscan::Error> {
    if (auto committed = std::move(writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    holoscan::schema::ImageT image;
    image.width = width;
    image.height = height;
    image.encoding = holoscan::schema::ImageEncoding_MONO32F;
    image.header = left_sample->data.header;
    return output.emit_tensor(std::move(loan), std::move(image), emit);
  };
  if (auto sent = send(disparity, *disparity_loan, *disparity_writer); !sent) {
    return sent;
  }
  if (config_.depth_camera_info) {
    // Before the depth: a consumer taking the newest camera info gets this one.
    state.depth_camera_info.header->capture_timestamp_ns =
        emit.capture_time ? emit.capture_time->timestamp_ns : 0;
    if (auto sent = depth_camera_info.emit_ref(state.depth_camera_info, emit); !sent) {
      return sent;
    }
  }
  if (depth_loan) {
    return send(depth, *depth_loan, *depth_writer);
  }
  return {};
}

}  // namespace holoscan::foundation_stereo
