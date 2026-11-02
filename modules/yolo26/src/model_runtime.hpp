// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What Yolo26DetectionOp and Yolo26SegmentationOp share, and no user sees:
// loading the model through HoloInfer, checking and staging the input image,
// and ordering one frame's work after the previous one's. Not installed.

#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <holoinfer/holoinfer.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/yolo26/kernels.hpp>
#include <holoscan/yolo26/letterbox.hpp>

namespace holoscan::yolo26::detail {

namespace infer = holoscan::inference;

inline constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
inline constexpr DLDataType kInt32{kDLInt, 32U, 1U};
inline constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

// Errors prefixed with the operator's name.
holoscan::Error invalid(std::string_view op, const std::string& message);
holoscan::Error failure(std::string_view op, const std::string& message);
holoscan::Error cuda_failure(std::string_view op, const char* what, cudaError_t status);

// A runtime-pooled output of `dtype` and `rank`, at most `bytes`, on the GPU.
holoscan::TensorOutputSpec device_tensor(DLDataType dtype, std::uint32_t rank, std::uint64_t bytes);

template <std::size_t Rank>
holoscan::expected<holoscan::TensorOutputLoan, holoscan::Error> allocate(
    holoscan::Output<holoscan::Tensor>& port, std::array<std::int64_t, Rank> shape,
    DLDataType dtype) {
  return port.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
}

std::string dims_text(const std::vector<std::int64_t>& dims);

// The model and how HoloInfer is to run it.
struct ModelSettings {
  std::filesystem::path model;
  int input_size{};
  bool fp16{};
  bool cuda_graphs{};
  std::size_t build_workspace_bytes{};
  // HoloInfer's names for the outputs, in the model's order.
  std::vector<std::string> outputs;
  // What the outputs must be, for the error a model with others gets.
  std::string expected_outputs;
};

// Builds and caches the engine of an .onnx model unless one is cached; see
// prepare_engine().
void prepare(const ModelSettings& settings, std::string_view op);

// The loaded model: HoloInfer's context and specifications, which own the
// input and output buffers that TensorRT reads and writes.
struct LoadedModel {
  std::shared_ptr<infer::InferenceSpecs> specs;
  std::unique_ptr<infer::InferContext> context;
  // The [1, 3, S, S] input, which the letterbox writes.
  float* input{};
  // Each output's float32 buffer and dimensions, in the order of
  // ModelSettings::outputs.
  std::vector<const float*> outputs;
  std::vector<std::vector<std::int64_t>> output_dims;

  LoadedModel() = default;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
  ~LoadedModel();
};

// Loads the model, building its engine first if it is not cached, and
// allocates its input.
// @throws std::runtime_error if it cannot be loaded, does not take one
// [1, 3, S, S] input, or does not give one float32 output per name.
std::unique_ptr<LoadedModel> load(const ModelSettings& settings, std::string_view op);

// Runs the model on `stream`.
holoscan::expected<void, holoscan::Error> infer_on(LoadedModel& model, cudaStream_t stream,
                                                   std::string_view op);

// The stream the runtime gave this activation, on GPU 0.
holoscan::expected<cudaStream_t, holoscan::Error> frame_stream(holoscan::ExecutionContext& context,
                                                               std::string_view op);

// Whether the work recorded in `event` has finished; a "not ready" status is
// cleared, so the kernel launch checks do not see it.
bool finished(cudaEvent_t event) noexcept;

// Records `event` on `stream` when it goes out of scope, unless `event` was
// cleared: so that the next frame waits for whatever this one has queued, on
// every way out of compute().
struct MarkDone {
  cudaEvent_t event;
  cudaStream_t stream;
  MarkDone(cudaEvent_t done, cudaStream_t on) : event(done), stream(on) {}
  MarkDone(const MarkDone&) = delete;
  MarkDone& operator=(const MarkDone&) = delete;
  ~MarkDone() {
    if (event != nullptr) {
      cudaEventRecord(event, stream);
    }
  }
};

// Checks `image` as both operators document their input, and returns its
// pixels in GPU memory: in place when they are there, otherwise copied on
// `stream` to `staging`, a GPU buffer of `staging_bytes` that grows as needed.
holoscan::expected<ImageView, holoscan::Error> image_input(const holoscan::schema::ImageT& image,
                                                           cudaStream_t stream, void*& staging,
                                                           std::size_t& staging_bytes,
                                                           std::string_view op);

// Frees `staging` once the GPU is done with it.
void release_staging(void*& staging, std::size_t& staging_bytes) noexcept;

}  // namespace holoscan::yolo26::detail
