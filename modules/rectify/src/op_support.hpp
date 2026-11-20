// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What UndistortOp and StereoRectifyOp share: reading an input image,
// staging host images on the GPU, building the remapper on the first image,
// allocating and sending an output image, and stamping the camera info.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <exception>
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
  // Pinned (page-locked) host memory.
  bool pinned{};
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

// An input image in device memory, whether it was copied there, and whether
// the operator must wait for its stream before compute() returns.
struct StagedImage {
  DeviceImage image;
  bool copied{};
  // The stream reads pinned host memory, in place or for the copy, and the
  // runtime would let its producer reuse that memory as soon as compute()
  // returns: it orders the reuse of device memory after the operator's
  // stream, but not of pinned host memory.
  bool wait_before_return{};
};

// Brings input images to device memory for work on the operator's stream:
// device images in place; pinned host images in place too where the GPU
// shares the host's memory (an integrated GPU, as on Jetson); other host
// images copied into a device buffer, grown as needed and kept between frames.
class Staging {
 public:
  Staging() = default;
  ~Staging() { release(); }
  Staging(const Staging&) = delete;
  Staging& operator=(const Staging&) = delete;

  // The image in device memory, ready for work on `stream` (after its
  // producer's work).
  [[nodiscard]] holoscan::expected<StagedImage, holoscan::Error> on_device(
      const InputImage& image, const holoscan::Tensor& tensor, cudaStream_t stream,
      std::string_view op);
  // Whether the GPU reads pinned host memory in place.
  [[nodiscard]] bool reads_pinned_in_place();
  void release() noexcept;

 private:
  [[nodiscard]] holoscan::expected<void, holoscan::Error> reserve(std::size_t bytes,
                                                                  cudaStream_t stream,
                                                                  std::string_view op);

  void* data_{};
  std::size_t bytes_{};
  std::optional<bool> integrated_;
};

// Waits for `stream` when a staged input's pinned memory must outlive
// compute() and RemapQueue::finish() has not waited already (`queue_waited`).
[[nodiscard]] holoscan::expected<void, holoscan::Error> release_inputs(bool wait_before_return,
                                                                       bool queue_waited,
                                                                       cudaStream_t stream,
                                                                       std::string_view op);

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
  // The remapper for `format`, built on first use; for auto, logs its choice
  // the first time.
  [[nodiscard]] holoscan::expected<Remapper*, holoscan::Error> get(
      Backend backend, const RemapModel& model, int output_width, int output_height,
      PixelFormat format, Interpolation interpolation, std::string_view op);
  void reset() noexcept;

 private:
  std::unique_ptr<Remapper> remapper_;
  std::optional<PixelFormat> format_;
  bool auto_logged_{};
};

// The operator's queue on `stream`, made again when the runtime gives
// another stream.
[[nodiscard]] RemapQueue& queue_on(std::optional<RemapQueue>& queue, cudaStream_t stream);

// After a submit or finish() that threw: waits until nothing queued still
// reads the input or writes the outputs, since compute() returns them, and
// says what failed, invalid input or not.
[[nodiscard]] holoscan::Error remap_failed(std::optional<RemapQueue>& queue, cudaStream_t stream,
                                           std::string_view op, const std::exception& error);

// Sets `header`'s capture time to the one `emit` carries, 0 when it carries
// none, so that a camera info's headers agree with its sample.
void stamp(holoscan::schema::HeaderT& header, const holoscan::EmitOptions& emit) noexcept;

// Says, once per operator, that frames are copied from host to device
// memory, and how to avoid it: device images, or on an integrated GPU
// (`pinned_in_place`) pinned host images too.
void log_host_copy(std::string_view op, std::string_view backend, std::size_t bytes_per_frame,
                   bool pinned_in_place);

}  // namespace holoscan::rectify::detail
