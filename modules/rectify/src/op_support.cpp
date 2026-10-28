// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "op_support.hpp"

#include <array>
#include <exception>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/logger/logger.hpp>

namespace holoscan::rectify::detail {
namespace {

std::optional<PixelFormat> pixel_format(holoscan::schema::ImageEncoding encoding) {
  switch (encoding) {
    case holoscan::schema::ImageEncoding_MONO8:
      return PixelFormat::kMono8;
    case holoscan::schema::ImageEncoding_MONO16:
      return PixelFormat::kMono16;
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

DLDataType element_type(PixelFormat format) {
  return format == PixelFormat::kMono16 ? DLDataType{kDLUInt, 16U, 1U}
                                        : DLDataType{kDLUInt, 8U, 1U};
}

}  // namespace

holoscan::Error invalid(std::string_view op, const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, std::string(op) + ": " + message};
}

holoscan::Error failure(std::string_view op, const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, std::string(op) + ": " + message};
}

holoscan::expected<InputImage, holoscan::Error> read_input(const holoscan::schema::ImageT& image,
                                                           std::string_view op,
                                                           std::string_view port) {
  const std::string what = std::string(port) + " image";
  if (!image.data) return holoscan::make_unexpected(invalid(op, what + " has no tensor"));
  const holoscan::Tensor& tensor = *image.data;
  const auto format = pixel_format(image.encoding);
  if (!format) {
    return holoscan::make_unexpected(
        invalid(op, what + ": unsupported encoding " +
                        holoscan::schema::EnumNameImageEncoding(image.encoding) +
                        "; use MONO8, MONO16, RGB8, BGR8, RGBA8 or BGRA8"));
  }
  const DLDataType dtype = tensor.dtype();
  const DLDataType expected = element_type(*format);
  if (dtype.code != expected.code || dtype.bits != expected.bits || dtype.lanes != 1) {
    return holoscan::make_unexpected(
        invalid(op, what + ": the tensor's element type disagrees with its encoding"));
  }
  const std::span<const std::int64_t> shape = tensor.shape_span();
  const int pixel_channels = channels(*format);
  const bool hwc = shape.size() == 3 && shape[2] == pixel_channels;
  const bool hw = shape.size() == 2 && pixel_channels == 1;
  if (!hwc && !hw) {
    return holoscan::make_unexpected(invalid(
        op, what + ": the tensor must be [height, width] or [height, width, channels] of its "
                   "encoding"));
  }
  const std::vector<std::int64_t> strides = tensor.strides();  // Bytes.
  const std::int64_t pixel_bytes = bytes_per_pixel(*format);
  const std::int64_t element_bytes = expected.bits / 8;
  if (shape[0] < 1 || shape[1] < 1 || strides[1] != pixel_bytes ||
      (hwc && strides[2] != element_bytes) || strides[0] < shape[1] * pixel_bytes) {
    return holoscan::make_unexpected(invalid(op, what + ": pixels must be packed within each row"));
  }
  if ((image.width != 0 && image.width != shape[1]) ||
      (image.height != 0 && image.height != shape[0])) {
    return holoscan::make_unexpected(
        invalid(op, what + ": the width and height disagree with the tensor"));
  }
  const DLDevice device = tensor.device();
  const bool on_device = device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
  if (!on_device && device.device_type != kDLCPU && device.device_type != kDLCUDAHost) {
    return holoscan::make_unexpected(
        invalid(op, what + ": the tensor must be in host or CUDA memory"));
  }
  const auto address = reinterpret_cast<std::uintptr_t>(tensor.data());
  if (address % static_cast<std::uintptr_t>(element_bytes) != 0 ||
      strides[0] % element_bytes != 0) {
    return holoscan::make_unexpected(
        invalid(op, what + ": 16-bit pixels must be aligned to 2 bytes"));
  }
  // A plain host producer is waited for here; Staging::on_device() orders the
  // others on the operator's stream.
  if (device.device_type == kDLCPU) {
    if (auto ready = tensor.synchronize_for_host(); !ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
  }
  return InputImage{.data = tensor.data(),
                    .pitch = static_cast<std::size_t>(strides[0]),
                    .width = static_cast<int>(shape[1]),
                    .height = static_cast<int>(shape[0]),
                    .format = *format,
                    .on_device = on_device,
                    .rank = static_cast<int>(shape.size())};
}

void declare_output(holoscan::OperatorSpec& spec, holoscan::Output<holoscan::schema::ImageT>& port,
                    const char* name, int width, int height) {
  // Four bytes per pixel covers every format: RGBA8 and BGRA8 take four, MONO16 two.
  const std::uint64_t bytes =
      static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) * 4U;
  spec.output(port, name)
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kCudaDevice,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::TensorBounds{.max_rank = 3U, .max_byte_span = bytes},
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      })
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
}

holoscan::expected<cudaStream_t, holoscan::Error> stream_of(holoscan::ExecutionContext& context,
                                                            std::string_view op) {
  // The runtime assigns a stream to an operator with device outputs and
  // writes their loans on it, so there is no other to use.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(
        failure(op, "the runtime gave no CUDA stream for the device outputs"));
  }
  return stream;
}

holoscan::expected<DeviceImage, holoscan::Error> Staging::on_device(const InputImage& image,
                                                                    const holoscan::Tensor& tensor,
                                                                    cudaStream_t stream,
                                                                    std::string_view op) {
  // The producer's writes come first on `stream` (device and pinned memory with
  // a readiness stream; plain host memory read_input() waited for on the host).
  if (tensor.device().device_type != kDLCPU && tensor.has_readiness_stream_contract()) {
    if (auto ready = tensor.synchronize_for_consumer_stream(stream); !ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
  }
  if (image.on_device) {
    return DeviceImage{const_cast<void*>(image.data), image.pitch, image.width, image.height};
  }
  const std::size_t row = image.row_bytes();
  const std::size_t bytes = row * static_cast<std::size_t>(image.height);
  if (bytes_ < bytes) {
    // Free only after the stream is done with the old buffer.
    cudaStreamSynchronize(stream);
    cudaFree(data_);
    data_ = nullptr;
    bytes_ = 0;
    if (const cudaError_t status = cudaMalloc(&data_, bytes); status != cudaSuccess) {
      data_ = nullptr;
      return holoscan::make_unexpected(
          failure(op, std::string("cudaMalloc: ") + cudaGetErrorString(status)));
    }
    bytes_ = bytes;
  }
  if (const cudaError_t status =
          cudaMemcpy2DAsync(data_, row, image.data, image.pitch, row,
                            static_cast<std::size_t>(image.height), cudaMemcpyHostToDevice, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(
        failure(op, std::string("cudaMemcpy2DAsync: ") + cudaGetErrorString(status)));
  }
  // The runtime orders the release of input the GPU reads after this
  // operator's stream, but not of host memory: a producer could reuse a pinned
  // buffer while the copy still reads it. Pageable memory needs nothing:
  // cudaMemcpy2DAsync has staged it by the time it returns.
  if (tensor.device().device_type == kDLCUDAHost) {
    holoscan::Tensor handle = tensor;
    if (!handle.set_deallocation_stream(stream)) {
      if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
        return holoscan::make_unexpected(
            failure(op, std::string("cudaStreamSynchronize: ") + cudaGetErrorString(status)));
      }
    }
  }
  return DeviceImage{data_, row, image.width, image.height};
}

void Staging::release() noexcept {
  // cudaFree waits for the work that may still read the buffer.
  if (data_ != nullptr) cudaFree(data_);
  data_ = nullptr;
  bytes_ = 0;
}

holoscan::expected<void, holoscan::Error> allocate_output(
    holoscan::Output<holoscan::schema::ImageT>& port, const InputImage& input, int width,
    int height, cudaStream_t stream, OutputImage& output) {
  const int pixel_channels = channels(input.format);
  const std::array<std::int64_t, 3> shape{height, width, pixel_channels};
  auto loan = port.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::span<const std::int64_t>(shape.data(), static_cast<std::size_t>(input.rank)),
      .dtype = element_type(input.format)});
  if (!loan) return holoscan::make_unexpected(std::move(loan).error());
  output.loan.emplace(std::move(*loan));
  auto writer = output.loan->write(stream);
  if (!writer) return holoscan::make_unexpected(std::move(writer).error());
  output.writer.emplace(std::move(*writer));
  output.image =
      DeviceImage{output.writer->data(),
                  static_cast<std::size_t>(width) * bytes_per_pixel(input.format), width, height};
  return {};
}

holoscan::expected<void, holoscan::Error> send_output(
    holoscan::Output<holoscan::schema::ImageT>& port, OutputImage& output,
    const holoscan::schema::ImageT& input, const holoscan::EmitOptions& emit) {
  if (auto committed = std::move(*output.writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  output.writer.reset();
  holoscan::schema::ImageT descriptor;
  descriptor.width = output.image.width;
  descriptor.height = output.image.height;
  descriptor.encoding = input.encoding;
  descriptor.color_space = input.color_space;
  descriptor.significant_bits = input.significant_bits;
  descriptor.exposure_time_ns = input.exposure_time_ns;
  descriptor.gain = input.gain;
  if (input.header) {
    descriptor.header = std::make_shared<holoscan::schema::HeaderT>(*input.header);
  }
  auto loan = std::move(*output.loan);
  output.loan.reset();
  return port.emit_tensor(std::move(loan), descriptor, emit);
}

holoscan::expected<Remapper*, holoscan::Error> RemapperSlot::get(
    Backend backend, const RemapModel& model, int output_width, int output_height,
    PixelFormat format, Interpolation interpolation, std::string_view op, bool log_auto) {
  if (remapper_ && format_ == format) return remapper_.get();
  std::string reason;
  try {
    remapper_.reset();
    remapper_ = create_remapper(backend, model, output_width, output_height, format, interpolation,
                                &reason);
  } catch (const std::invalid_argument& error) {
    return holoscan::make_unexpected(invalid(op, error.what()));
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(failure(op, error.what()));
  }
  format_ = format;
  if (backend == Backend::kAuto && log_auto && !auto_logged_) {
    auto_logged_ = true;
    HOLOSCAN_LOG_INFO("{}: backend auto chose {} for {} images: {}", op, remapper_->name(),
                      pixel_format_name(format), reason);
  }
  return remapper_.get();
}

void RemapperSlot::reset() noexcept {
  remapper_.reset();
  format_.reset();
}

void log_host_copy(std::string_view op, std::string_view backend, std::size_t bytes_per_frame) {
  HOLOSCAN_LOG_INFO(
      "{} ({} backend): the images are in host memory and every backend reads CUDA device "
      "memory, so each frame is copied host to device, {} bytes ({:.1f} MB) per frame; to avoid "
      "the copy, send CUDA device images",
      op, backend, bytes_per_frame, static_cast<double>(bytes_per_frame) / 1e6);
}

}  // namespace holoscan::rectify::detail
