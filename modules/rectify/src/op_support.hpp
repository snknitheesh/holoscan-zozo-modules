// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What UndistortOp and StereoRectifyOp share: reading an input image,
// staging host images on the GPU, building the remapper on the first image,
// and allocating and sending an output image.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/rectify/backend.hpp>
#include <holoscan/rectify/remapper.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::rectify::detail {

[[nodiscard]] holoscan::Error invalid(std::string_view op, const std::string& message);
[[nodiscard]] holoscan::Error failure(std::string_view op, const std::string& message);

// An input image's pixels, after checking its tensor against its encoding.
struct InputImage {
  const void* data{};
  std::size_t pitch{};
  int width{};
  int height{};
  PixelFormat format{PixelFormat::kMono8};
  bool on_device{};
  // [H, W] or [H, W, C], as the tensor has it; the output takes the same rank.
  int rank{};

  [[nodiscard]] std::size_t row_bytes() const noexcept {
    return static_cast<std::size_t>(width) * bytes_per_pixel(format);
  }
};

// Checks `image` (`port` names it in errors) and waits for a host producer.
[[nodiscard]] holoscan::expected<InputImage, holoscan::Error> read_input(
    const holoscan::schema::ImageT& image, std::string_view op, std::string_view port);

// Declares `port` as `name`: images of up to width x height x 4 bytes in CUDA
// device memory, from the runtime's pool.
void declare_output(holoscan::OperatorSpec& spec, holoscan::Output<holoscan::schema::ImageT>& port,
                    const char* name, int width, int height);

// The CUDA stream the runtime gives an operator with device outputs.
[[nodiscard]] holoscan::expected<cudaStream_t, holoscan::Error> stream_of(
    holoscan::ExecutionContext& context, std::string_view op);

// A device copy of host input, grown as needed and kept between frames.
class Staging {
 public:
  Staging() = default;
  ~Staging() { release(); }
  Staging(const Staging&) = delete;
  Staging& operator=(const Staging&) = delete;

  // The image in device memory, ready for work on `stream`: in place when it
  // is there (after its producer's work, on `stream`), else copied on
  // `stream`, with pinned host memory held until the copy is done.
  [[nodiscard]] holoscan::expected<DeviceImage, holoscan::Error> on_device(
      const InputImage& image, const holoscan::Tensor& tensor, cudaStream_t stream,
      std::string_view op);
  void release() noexcept;

 private:
  void* data_{};
  std::size_t bytes_{};
};

// One output image being written: its loan and write guard.
struct OutputImage {
  std::optional<holoscan::TensorOutputLoan> loan;
  std::optional<holoscan::TensorOutputWriteGuard> writer;
  DeviceImage image;
};

// Allocates a `width` x `height` image like `input` on `port` into `output`
// (which must stay where it is until sent), writable on `stream`.
[[nodiscard]] holoscan::expected<void, holoscan::Error> allocate_output(
    holoscan::Output<holoscan::schema::ImageT>& port, const InputImage& input, int width,
    int height, cudaStream_t stream, OutputImage& output);

// Commits `output` and sends it with the input's descriptor (encoding,
// header), resized to its size.
[[nodiscard]] holoscan::expected<void, holoscan::Error> send_output(
    holoscan::Output<holoscan::schema::ImageT>& port, OutputImage& output,
    const holoscan::schema::ImageT& input, const holoscan::EmitOptions& emit);

// The remapper of one camera and what it was built for, rebuilt when the
// pixel format changes.
class RemapperSlot {
 public:
  // The remapper for `format`, built on first use; with `log_auto`, logs
  // auto's choice the first time.
  [[nodiscard]] holoscan::expected<Remapper*, holoscan::Error> get(
      Backend backend, const RemapModel& model, int output_width, int output_height,
      PixelFormat format, Interpolation interpolation, std::string_view op, bool log_auto = true);
  void reset() noexcept;

 private:
  std::unique_ptr<Remapper> remapper_;
  std::optional<PixelFormat> format_;
  bool auto_logged_{};
};

// Says, once per operator, that frames are copied from host to device memory.
void log_host_copy(std::string_view op, std::string_view backend, std::size_t bytes_per_frame);

}  // namespace holoscan::rectify::detail
