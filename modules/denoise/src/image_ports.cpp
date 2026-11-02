// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "image_ports.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>

namespace holoscan::denoise::detail {
namespace {

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

bool same_type(const DLDataType& a, const DLDataType& b) {
  return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

bool on_gpu(const holoscan::Tensor& tensor) {
  const DLDevice device = tensor.device();
  return device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
}

bool readable(const holoscan::Tensor& tensor) {
  const DLDevice device = tensor.device();
  return on_gpu(tensor) || device.device_type == kDLCPU || device.device_type == kDLCUDAHost;
}

std::optional<PixelFormat> pixel_format(holoscan::schema::ImageEncoding encoding) {
  switch (encoding) {
    case holoscan::schema::ImageEncoding_MONO8:
      return PixelFormat::kMono8;
    case holoscan::schema::ImageEncoding_RGB8:
      return PixelFormat::kRgb8;
    case holoscan::schema::ImageEncoding_RGBA8:
      return PixelFormat::kRgba8;
    case holoscan::schema::ImageEncoding_NV12:
      return PixelFormat::kNv12;
    default:
      return std::nullopt;
  }
}

std::optional<HdrFormat> hdr_format(holoscan::schema::ImageEncoding encoding) {
  switch (encoding) {
    case holoscan::schema::ImageEncoding_MONO16:
      return HdrFormat::kMono16;
    case holoscan::schema::ImageEncoding_RGB16:
      return HdrFormat::kRgb16;
    case holoscan::schema::ImageEncoding_RGBA16:
      return HdrFormat::kRgba16;
    case holoscan::schema::ImageEncoding_MONO32F:
      return HdrFormat::kMono32f;
    case holoscan::schema::ImageEncoding_RGB32F:
      return HdrFormat::kRgb32f;
    case holoscan::schema::ImageEncoding_RGBA32F:
      return HdrFormat::kRgba32f;
    default:
      return std::nullopt;
  }
}

std::string size_text(int width, int height) {
  return std::to_string(width) + "x" + std::to_string(height);
}

// Checks a [H, W] or [H, W, channels] tensor of packed `element`-byte samples;
// returns the row pitch in bytes.
holoscan::expected<std::size_t, holoscan::Error> packed_rows(const holoscan::Tensor& tensor,
                                                             int channels, std::size_t element,
                                                             std::string_view op) {
  const std::span<const std::int64_t> shape = tensor.shape_span();
  const bool hwc = shape.size() == 3 && shape[2] == channels;
  const bool hw = shape.size() == 2 && channels == 1;
  if (!hwc && !hw) {
    return holoscan::make_unexpected(
        invalid(op, "image tensor shape must be [height, width, channels] matching its encoding"));
  }
  const std::vector<std::int64_t> strides = tensor.strides();
  const auto pixel = static_cast<std::int64_t>(element) * channels;
  if ((shape[1] > 1 && strides[1] != pixel) ||
      (hwc && channels > 1 && strides[2] != static_cast<std::int64_t>(element))) {
    return holoscan::make_unexpected(invalid(op, "image pixels must be packed within each row"));
  }
  const std::int64_t row = pixel * shape[1];
  const std::int64_t pitch = shape[0] > 1 ? strides[0] : row;
  if (pitch < row) {
    return holoscan::make_unexpected(invalid(op, "image rows overlap"));
  }
  return static_cast<std::size_t>(pitch);
}

holoscan::expected<void, holoscan::Error> check_size(int width, int height, int max_width,
                                                     int max_height, std::string_view op) {
  if (width < 1 || height < 1 || width > max_width || height > max_height) {
    return holoscan::make_unexpected(invalid(op, "image " + size_text(width, height) +
                                                     " is outside 1x1 to the configured " +
                                                     size_text(max_width, max_height)));
  }
  return {};
}

}  // namespace

holoscan::Error invalid(std::string_view op, const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, std::string(op) + ": " + message};
}

holoscan::Error failure(std::string_view op, const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, std::string(op) + ": " + message};
}

holoscan::expected<InputImage, holoscan::Error> describe_image(
    const holoscan::schema::ImageT& image, std::string_view op, bool allow_nv12, int max_width,
    int max_height) {
  if (!image.data) {
    return holoscan::make_unexpected(invalid(op, "image has no tensor"));
  }
  const holoscan::Tensor& tensor = *image.data;
  const auto format = pixel_format(image.encoding);
  if (!format || (*format == PixelFormat::kNv12 && !allow_nv12)) {
    return holoscan::make_unexpected(
        invalid(op, std::string("unsupported image encoding ") +
                        holoscan::schema::EnumNameImageEncoding(image.encoding) +
                        "; use MONO8, RGB8, " + (allow_nv12 ? "RGBA8 or NV12" : "or RGBA8")));
  }
  if (!same_type(tensor.dtype(), kUInt8)) {
    return holoscan::make_unexpected(invalid(op, "image tensors must hold uint8 elements"));
  }
  if (!readable(tensor)) {
    return holoscan::make_unexpected(invalid(op, "image tensors must be in host or CUDA memory"));
  }
  auto* base = static_cast<std::uint8_t*>(const_cast<void*>(tensor.data()));
  InputImage result;
  result.on_device = on_gpu(tensor);
  ImageView& view = result.view;
  view.format = *format;
  if (*format == PixelFormat::kNv12) {
    // Geometry from the descriptor, planes from their layouts, as HoloViz reads them.
    view.width = image.width;
    view.height = image.height;
    if (auto ok = check_size(view.width, view.height, max_width, max_height, op); !ok) {
      return holoscan::make_unexpected(std::move(ok).error());
    }
    if (view.width % 2 != 0 || view.height % 2 != 0) {
      return holoscan::make_unexpected(invalid(op, "NV12 images need an even width and height"));
    }
    if (image.plane_layouts.size() != 2) {
      return holoscan::make_unexpected(invalid(op, "NV12 images need two plane layouts"));
    }
    const auto total = static_cast<std::uint64_t>(tensor.nbytes());
    const int rows[2] = {view.height, view.height / 2};
    for (int p = 0; p < 2; ++p) {
      const holoscan::schema::ImagePlaneLayout& layout = image.plane_layouts[p];
      const std::uint64_t pitch = layout.row_stride_bytes();
      const std::uint64_t span =
          pitch * static_cast<std::uint64_t>(rows[p] - 1) + static_cast<std::uint64_t>(view.width);
      if (pitch < static_cast<std::uint64_t>(view.width) || layout.offset_bytes() > total ||
          span > total - layout.offset_bytes()) {
        return holoscan::make_unexpected(
            invalid(op, "an NV12 plane layout lies outside its tensor or has short rows"));
      }
      view.planes[p] = Plane{.data = base + layout.offset_bytes(), .pitch = pitch};
      result.plane_bytes[p] = span;
    }
    return result;
  }
  const int channels = pixel_bytes(*format);
  auto pitch = packed_rows(tensor, channels, 1U, op);
  if (!pitch) {
    return holoscan::make_unexpected(std::move(pitch).error());
  }
  const std::span<const std::int64_t> shape = tensor.shape_span();
  view.height = static_cast<int>(shape[0]);
  view.width = static_cast<int>(shape[1]);
  if ((image.width != 0 && image.width != view.width) ||
      (image.height != 0 && image.height != view.height)) {
    return holoscan::make_unexpected(
        invalid(op, "image width and height disagree with its tensor"));
  }
  if (auto ok = check_size(view.width, view.height, max_width, max_height, op); !ok) {
    return holoscan::make_unexpected(std::move(ok).error());
  }
  view.planes[0] = Plane{.data = base, .pitch = *pitch};
  result.plane_bytes[0] = *pitch * static_cast<std::size_t>(view.height - 1) +
                          static_cast<std::size_t>(view.width) * channels;
  return result;
}

holoscan::expected<HdrInput, holoscan::Error> describe_hdr(const holoscan::schema::ImageT& image,
                                                           std::string_view op, int max_width,
                                                           int max_height) {
  if (!image.data) {
    return holoscan::make_unexpected(invalid(op, "image has no tensor"));
  }
  const holoscan::Tensor& tensor = *image.data;
  const auto format = hdr_format(image.encoding);
  if (!format) {
    return holoscan::make_unexpected(
        invalid(op, std::string("unsupported image encoding ") +
                        holoscan::schema::EnumNameImageEncoding(image.encoding) +
                        "; use MONO16, RGB16, RGBA16, MONO32F, RGB32F or RGBA32F"));
  }
  const bool is_float = hdr_is_float(*format);
  const DLDataType expected =
      is_float ? DLDataType{kDLFloat, 32U, 1U} : DLDataType{kDLUInt, 16U, 1U};
  if (!same_type(tensor.dtype(), expected)) {
    return holoscan::make_unexpected(
        invalid(op, std::string("a ") + std::string(hdr_format_name(*format)) +
                        " tensor must hold " + (is_float ? "float32" : "uint16") + " elements"));
  }
  if (!readable(tensor)) {
    return holoscan::make_unexpected(invalid(op, "image tensors must be in host or CUDA memory"));
  }
  const std::size_t element = is_float ? sizeof(float) : sizeof(std::uint16_t);
  const int channels = hdr_channels(*format);
  auto pitch = packed_rows(tensor, channels, element, op);
  if (!pitch) {
    return holoscan::make_unexpected(std::move(pitch).error());
  }
  const std::span<const std::int64_t> shape = tensor.shape_span();
  HdrInput result;
  result.on_device = on_gpu(tensor);
  HdrImageView& view = result.view;
  view.format = *format;
  view.height = static_cast<int>(shape[0]);
  view.width = static_cast<int>(shape[1]);
  if ((image.width != 0 && image.width != view.width) ||
      (image.height != 0 && image.height != view.height)) {
    return holoscan::make_unexpected(
        invalid(op, "image width and height disagree with its tensor"));
  }
  if (auto ok = check_size(view.width, view.height, max_width, max_height, op); !ok) {
    return holoscan::make_unexpected(std::move(ok).error());
  }
  view.data = tensor.data();
  view.pitch = *pitch;
  if (reinterpret_cast<std::uintptr_t>(view.data) % element != 0 || view.pitch % element != 0) {
    return holoscan::make_unexpected(invalid(op, "image samples must be aligned to their size"));
  }
  const bool known = image.significant_bits != 0U &&
                     image.significant_bits != holoscan::schema::kUnknownImageSignificantBits;
  view.significant_bits = !is_float && known ? std::min<int>(image.significant_bits, 16) : 16;
  result.bytes = view.pitch * static_cast<std::size_t>(view.height - 1) +
                 static_cast<std::size_t>(view.width) * channels * element;
  return result;
}

holoscan::expected<void, holoscan::Error> wait_for_host_input(const holoscan::Tensor& tensor) {
  if (tensor.device().device_type != kDLCPU) return {};
  return tensor.synchronize_for_host();
}

DeviceStaging::~DeviceStaging() { release(); }

void DeviceStaging::release() noexcept {
  // cudaFree waits for the work that may still read the copy.
  if (data_ != nullptr) cudaFree(data_);
  data_ = nullptr;
  bytes_ = 0;
}

holoscan::expected<std::uint8_t*, holoscan::Error> DeviceStaging::reserve(std::size_t bytes,
                                                                          cudaStream_t stream,
                                                                          std::string_view op) {
  if (bytes_ >= bytes) return data_;
  // The old copy may still be read by queued work.
  if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
    return holoscan::make_unexpected(
        failure(op, std::string("cudaStreamSynchronize: ") + cudaGetErrorString(status)));
  }
  release();
  void* data = nullptr;
  if (const cudaError_t status = cudaMalloc(&data, bytes); status != cudaSuccess) {
    return holoscan::make_unexpected(
        failure(op, std::string("cudaMalloc: ") + cudaGetErrorString(status)));
  }
  data_ = static_cast<std::uint8_t*>(data);
  bytes_ = bytes;
  return data_;
}

holoscan::expected<void, holoscan::Error> DeviceStaging::finish_copy(const holoscan::Tensor& tensor,
                                                                     cudaStream_t stream,
                                                                     std::string_view op,
                                                                     std::string_view port,
                                                                     std::size_t bytes) {
  // A producer must not reuse pinned memory while the copy still reads it;
  // pageable memory is staged by the time cudaMemcpy2DAsync returns.
  if (tensor.device().device_type == kDLCUDAHost) {
    holoscan::Tensor handle = tensor;
    if (!handle.set_deallocation_stream(stream)) {
      if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
        return holoscan::make_unexpected(
            failure(op, std::string("cudaStreamSynchronize: ") + cudaGetErrorString(status)));
      }
    }
  }
  if (!logged_) {
    logged_ = true;
    HOLOSCAN_LOG_INFO(
        "{}: input {} is in host memory and the operator reads CUDA device memory, so each frame "
        "is copied host to device, {} bytes ({:.1f} MB) per frame; to avoid the copy, send CUDA "
        "device images",
        op, port, bytes, static_cast<double>(bytes) / 1e6);
  }
  return {};
}

holoscan::expected<ImageView, holoscan::Error> DeviceStaging::on_device(
    const InputImage& image, const holoscan::Tensor& tensor, cudaStream_t stream,
    std::string_view op, std::string_view port) {
  if (image.on_device) return image.view;
  const ImageView& view = image.view;
  const auto row = static_cast<std::size_t>(view.width) * pixel_bytes(view.format);
  const int planes = view.format == PixelFormat::kNv12 ? 2 : 1;
  const std::size_t y_bytes = row * view.height;
  const std::size_t total = planes == 2 ? y_bytes + y_bytes / 2 : y_bytes;
  auto data = reserve(total, stream, op);
  if (!data) return holoscan::make_unexpected(std::move(data).error());
  ImageView staged = packed_view(view.format, view.width, view.height, *data);
  const int rows[2] = {view.height, view.height / 2};
  for (int p = 0; p < planes; ++p) {
    if (const cudaError_t status =
            cudaMemcpy2DAsync(staged.planes[p].data, staged.planes[p].pitch, view.planes[p].data,
                              view.planes[p].pitch, row, rows[p], cudaMemcpyHostToDevice, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(
          failure(op, std::string("cudaMemcpy2DAsync: ") + cudaGetErrorString(status)));
    }
  }
  if (auto done = finish_copy(tensor, stream, op, port, total); !done) {
    return holoscan::make_unexpected(std::move(done).error());
  }
  return staged;
}

holoscan::expected<const void*, holoscan::Error> DeviceStaging::on_device(
    const HdrInput& image, const holoscan::Tensor& tensor, cudaStream_t stream, std::string_view op,
    std::string_view port) {
  if (image.on_device) return image.view.data;
  const std::size_t total = image.view.pitch * static_cast<std::size_t>(image.view.height);
  auto data = reserve(total, stream, op);
  if (!data) return holoscan::make_unexpected(std::move(data).error());
  if (const cudaError_t status =
          cudaMemcpyAsync(*data, image.view.data, image.bytes, cudaMemcpyHostToDevice, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(
        failure(op, std::string("cudaMemcpyAsync: ") + cudaGetErrorString(status)));
  }
  if (auto done = finish_copy(tensor, stream, op, port, image.bytes); !done) {
    return holoscan::make_unexpected(std::move(done).error());
  }
  return static_cast<const void*>(*data);
}

holoscan::TensorOutputSpec image_output_spec(int max_width, int max_height) {
  const auto bytes =
      static_cast<std::uint64_t>(max_width) * static_cast<std::uint64_t>(max_height) * 4U;
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = kUInt8,
              .rank = 3U,
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

std::array<std::int64_t, 3> image_shape(PixelFormat format, int width, int height) {
  if (format == PixelFormat::kNv12) return {height + height / 2, width, 1};
  return {height, width, pixel_bytes(format)};
}

ImageView packed_view(PixelFormat format, int width, int height, std::uint8_t* data) {
  ImageView view{.format = format, .width = width, .height = height};
  const auto row = static_cast<std::size_t>(width) * pixel_bytes(format);
  view.planes[0] = Plane{.data = data, .pitch = row};
  if (format == PixelFormat::kNv12) {
    view.planes[1] = Plane{.data = data + row * static_cast<std::size_t>(height), .pitch = row};
  }
  return view;
}

holoscan::schema::ImageEncoding encoding_of(PixelFormat format) noexcept {
  switch (format) {
    case PixelFormat::kMono8:
      return holoscan::schema::ImageEncoding_MONO8;
    case PixelFormat::kRgba8:
      return holoscan::schema::ImageEncoding_RGBA8;
    case PixelFormat::kNv12:
      return holoscan::schema::ImageEncoding_NV12;
    case PixelFormat::kRgb8:
      break;
  }
  return holoscan::schema::ImageEncoding_RGB8;
}

holoscan::schema::ImageT image_descriptor(PixelFormat format, int width, int height,
                                          const holoscan::schema::ImageT& source) {
  holoscan::schema::ImageT image;
  image.width = width;
  image.height = height;
  image.encoding = encoding_of(format);
  if (source.header) {
    image.header = std::make_shared<holoscan::schema::HeaderT>(*source.header);
  }
  image.color_space = source.color_space;
  image.exposure_time_ns = source.exposure_time_ns;
  image.gain = source.gain;
  if (format == PixelFormat::kNv12) {
    const auto row = static_cast<std::uint64_t>(width);
    const auto rows = static_cast<std::uint64_t>(height);
    image.plane_layouts = {
        holoscan::schema::ImagePlaneLayout(0, static_cast<std::uint32_t>(row), row * rows),
        holoscan::schema::ImagePlaneLayout(row * rows, static_cast<std::uint32_t>(row),
                                           row * rows / 2)};
  }
  return image;
}

}  // namespace holoscan::denoise::detail
