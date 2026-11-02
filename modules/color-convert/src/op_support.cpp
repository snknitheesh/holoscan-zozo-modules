// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "op_support.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/provenance.hpp>

namespace holoscan::color_convert::detail {
namespace {

namespace schema = holoscan::schema;

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr DLDataType kUInt16{kDLUInt, 16U, 1U};
// The most bytes per pixel of any format: RGB16.
constexpr std::uint32_t kMaxRank = 3U;

holoscan::Error error(holoscan::ErrorCode code, const std::string& op, const std::string& what) {
  return holoscan::Error{code, op + ": " + what};
}

bool same_dtype(const DLDataType& a, const DLDataType& b) {
  return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

// An image message's pixels as a view, after checking that its tensor holds
// what its encoding says, and the bytes the view spans from the tensor's start.
struct InputImage {
  ImageView view;
  std::size_t span{};
};

holoscan::expected<InputImage, holoscan::Error> view_input(const schema::ImageT& image,
                                                           const holoscan::Tensor& tensor,
                                                           std::optional<PixelFormat> custom,
                                                           const std::string& op) {
  const auto invalid = [&op](const std::string& what) {
    return holoscan::make_unexpected(error(holoscan::ErrorCode::kInvalidArgument, op, what));
  };
  const auto format = format_of(image.encoding, custom);
  if (!format) {
    return invalid(std::string("unsupported image encoding ") +
                   schema::EnumNameImageEncoding(image.encoding) +
                   (image.encoding == schema::ImageEncoding_CUSTOM
                        ? "; set custom_format to the format CUSTOM images hold"
                        : "; see the module README for the supported formats"));
  }
  const DLDevice device = tensor.device();
  const bool on_device = device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
  if (!on_device && device.device_type != kDLCPU && device.device_type != kDLCUDAHost) {
    return invalid("image tensors must be in host or CUDA memory");
  }
  const FormatInfo info = format_info(*format);
  InputImage input;
  input.view.format = *format;
  input.view.location = on_device ? MemoryLocation::kDevice : MemoryLocation::kHost;
  auto* base = static_cast<std::uint8_t*>(const_cast<void*>(tensor.data()));
  const std::span<const std::int64_t> shape = tensor.shape_span();
  if (info.planes > 1) {
    if (!same_dtype(tensor.dtype(), kUInt8) || !tensor.is_contiguous()) {
      return invalid(std::string(format_name(*format)) +
                     " images must be a packed uint8 byte buffer");
    }
    if (image.width <= 0 || image.height <= 0) {
      return invalid("planar images need their width and height");
    }
    if (image.plane_layouts.size() != static_cast<std::size_t>(info.planes)) {
      return invalid(std::string(format_name(*format)) + " images need " +
                     std::to_string(info.planes) + " plane layouts, not " +
                     std::to_string(image.plane_layouts.size()));
    }
    input.view.width = image.width;
    input.view.height = image.height;
    input.span = static_cast<std::size_t>(tensor.nbytes());
    for (int plane = 0; plane < info.planes; ++plane) {
      const schema::ImagePlaneLayout& layout = image.plane_layouts[plane];
      const PlaneSize size = plane_size(*format, image.width, image.height, plane);
      const std::size_t pitch = layout.row_stride_bytes();
      if (pitch < size.row_bytes) {
        return invalid("plane " + std::to_string(plane) + " rows are shorter than its pixels");
      }
      const std::size_t end =
          layout.offset_bytes() + pitch * static_cast<std::size_t>(size.rows - 1) + size.row_bytes;
      if (end > input.span) {
        return invalid("plane " + std::to_string(plane) + " reaches past the image tensor");
      }
      input.view.planes[plane] = {base + layout.offset_bytes(), pitch};
    }
    return input;
  }
  const int channels = info.channels;
  const std::size_t sample = static_cast<std::size_t>(info.bytes_per_sample);
  const bool hwc = shape.size() == 3 && shape[2] == channels;
  const bool hw = shape.size() == 2 && channels == 1;
  if (!hwc && !hw) {
    return invalid("a " + std::string(format_name(*format)) +
                   " image tensor must be [height, width" +
                   (channels == 1 ? std::string("] or [height, width, 1]")
                                  : ", " + std::to_string(channels) + "]"));
  }
  if (!same_dtype(tensor.dtype(), sample == 2 ? kUInt16 : kUInt8)) {
    return invalid(std::string(format_name(*format)) + " images must hold " +
                   (sample == 2 ? "uint16" : "uint8") + " elements");
  }
  if (shape[0] <= 0 || shape[1] <= 0) {
    return invalid("the image is empty");
  }
  const std::vector<std::int64_t> strides = tensor.strides();
  const auto pixel = static_cast<std::int64_t>(channels * info.bytes_per_sample);
  if (strides[1] != pixel || (hwc && strides[2] != info.bytes_per_sample) ||
      strides[0] < shape[1] * pixel) {
    return invalid("image pixels must be packed within each row");
  }
  if ((image.width != 0 && image.width != shape[1]) ||
      (image.height != 0 && image.height != shape[0])) {
    return invalid("image width and height disagree with its tensor");
  }
  input.view.width = static_cast<int>(shape[1]);
  input.view.height = static_cast<int>(shape[0]);
  input.view.planes[0] = {base, static_cast<std::size_t>(strides[0])};
  input.span = static_cast<std::size_t>(strides[0]) * static_cast<std::size_t>(shape[0] - 1) +
               static_cast<std::size_t>(shape[1] * pixel);
  return input;
}

schema::ColorSpace color_space_of(ColorMatrix matrix) {
  switch (matrix) {
    case ColorMatrix::kBt709:
      return schema::ColorSpace_REC709;
    case ColorMatrix::kBt2020:
      return schema::ColorSpace_REC2020;
    case ColorMatrix::kBt601:
      break;
  }
  return schema::ColorSpace_REC601;
}

std::vector<schema::ImagePlaneLayout> plane_layouts(PixelFormat format, int width, int height) {
  std::vector<schema::ImagePlaneLayout> layouts;
  if (!is_planar(format)) {
    return layouts;
  }
  std::uint64_t offset = 0;
  for (int plane = 0; plane < format_info(format).planes; ++plane) {
    const PlaneSize size = plane_size(format, width, height, plane);
    const std::uint64_t bytes = size.row_bytes * static_cast<std::uint64_t>(size.rows);
    layouts.emplace_back(offset, static_cast<std::uint32_t>(size.row_bytes), bytes);
    offset += bytes;
  }
  return layouts;
}

// The published image's descriptor: the input's, with the new format, size,
// colour space and planes.
schema::ImageT output_descriptor(const schema::ImageT& input, PixelFormat format, int width,
                                 int height, ColorMatrix matrix, PixelFormat input_format,
                                 bool keep_roi) {
  schema::ImageT image;
  if (input.header) {
    image.header = std::make_shared<schema::HeaderT>(*input.header);
  }
  image.width = width;
  image.height = height;
  image.encoding = encoding_of(format);
  if (is_yuv(format)) {
    image.color_space = color_space_of(matrix);
  } else if (format_info(format).model == ColorModel::kRgb &&
             format_info(input_format).model == ColorModel::kRgb) {
    image.color_space = input.color_space;
  }
  if (keep_roi) {
    image.roi_offset_x = input.roi_offset_x;
    image.roi_offset_y = input.roi_offset_y;
  }
  image.plane_layouts = plane_layouts(format, width, height);
  image.exposure_time_ns = input.exposure_time_ns;
  image.gain = input.gain;
  return image;
}

}  // namespace

DLDataType dtype_of(PixelFormat format) noexcept {
  return format_info(format).bytes_per_sample == 2 ? kUInt16 : kUInt8;
}

std::vector<std::int64_t> shape_of(PixelFormat format, int width, int height) {
  const FormatInfo info = format_info(format);
  if (info.planes > 1) {
    return {static_cast<std::int64_t>(packed_size(format, width, height))};
  }
  if (info.channels == 1) {
    return {height, width};
  }
  return {height, width, info.channels};
}

schema::ImageEncoding encoding_of(PixelFormat format) noexcept {
  switch (format) {
    case PixelFormat::kNv12:
      return schema::ImageEncoding_NV12;
    case PixelFormat::kI420:
      return schema::ImageEncoding_I420;
    case PixelFormat::kYuyv:
      return schema::ImageEncoding_YUYV;
    case PixelFormat::kUyvy:
      return schema::ImageEncoding_UYVY;
    case PixelFormat::kRgb8:
      return schema::ImageEncoding_RGB8;
    case PixelFormat::kBgr8:
      return schema::ImageEncoding_BGR8;
    case PixelFormat::kRgba8:
      return schema::ImageEncoding_RGBA8;
    case PixelFormat::kBgra8:
      return schema::ImageEncoding_BGRA8;
    case PixelFormat::kMono8:
      return schema::ImageEncoding_MONO8;
    case PixelFormat::kMono16:
      return schema::ImageEncoding_MONO16;
    case PixelFormat::kRgb16:
      return schema::ImageEncoding_RGB16;
    case PixelFormat::kNv16:
      break;  // The schema has no NV16.
  }
  return schema::ImageEncoding_CUSTOM;
}

std::optional<PixelFormat> format_of(schema::ImageEncoding encoding,
                                     std::optional<PixelFormat> custom) noexcept {
  switch (encoding) {
    case schema::ImageEncoding_NV12:
      return PixelFormat::kNv12;
    case schema::ImageEncoding_I420:
      return PixelFormat::kI420;
    case schema::ImageEncoding_YUYV:
      return PixelFormat::kYuyv;
    case schema::ImageEncoding_UYVY:
      return PixelFormat::kUyvy;
    case schema::ImageEncoding_RGB8:
      return PixelFormat::kRgb8;
    case schema::ImageEncoding_BGR8:
      return PixelFormat::kBgr8;
    case schema::ImageEncoding_RGBA8:
      return PixelFormat::kRgba8;
    case schema::ImageEncoding_BGRA8:
      return PixelFormat::kBgra8;
    case schema::ImageEncoding_MONO8:
      return PixelFormat::kMono8;
    case schema::ImageEncoding_MONO16:
      return PixelFormat::kMono16;
    case schema::ImageEncoding_RGB16:
      return PixelFormat::kRgb16;
    case schema::ImageEncoding_CUSTOM:
      return custom;
    default:
      return std::nullopt;
  }
}

std::optional<ColorMatrix> matrix_of(schema::ColorSpace space) noexcept {
  switch (space) {
    case schema::ColorSpace_REC601:
      return ColorMatrix::kBt601;
    case schema::ColorSpace_REC709:
      return ColorMatrix::kBt709;
    case schema::ColorSpace_REC2020:
    case schema::ColorSpace_REC2020_PQ:
    case schema::ColorSpace_REC2020_HLG:
      return ColorMatrix::kBt2020;
    default:
      return std::nullopt;
  }
}

holoscan::TensorOutputSpec output_spec(std::optional<PixelFormat> format, std::uint64_t max_bytes) {
  holoscan::TensorRepresentation representation{
      .memory_kind = holoscan::MemoryKind::kCudaDevice,
      .strides = holoscan::TensorStridePolicy::kContiguousOnly,
  };
  holoscan::TensorBounds bounds = holoscan::tensor_bounds(max_bytes);
  if (format) {
    representation.dtype = dtype_of(*format);
    representation.rank = static_cast<std::uint32_t>(shape_of(*format, 2, 2).size());
  } else {
    bounds.max_rank = kMaxRank;
  }
  return holoscan::TensorOutputSpec{
      .representation = representation,
      .bounds = bounds,
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

ConversionRunner::ConversionRunner(std::string op_name, Backend backend)
    : name_(std::move(op_name)), converter_(backend) {}

ConversionRunner::~ConversionRunner() { reset(); }

void ConversionRunner::reset() noexcept {
  if (staging_ != nullptr) {
    cudaFree(staging_);  // Waits for the work that may still read it.
  }
  staging_ = nullptr;
  staging_bytes_ = 0;
  logged_backend_.reset();
  copy_logged_ = false;
}

holoscan::expected<void*, holoscan::Error> ConversionRunner::staging(std::size_t bytes,
                                                                     cudaStream_t stream) {
  if (staging_bytes_ >= bytes) {
    return staging_;
  }
  if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
    return holoscan::make_unexpected(
        error(holoscan::ErrorCode::kFailure, name_,
              std::string("cudaStreamSynchronize: ") + cudaGetErrorString(status)));
  }
  cudaFree(staging_);
  staging_ = nullptr;
  staging_bytes_ = 0;
  if (const cudaError_t status = cudaMalloc(&staging_, bytes); status != cudaSuccess) {
    staging_ = nullptr;
    return holoscan::make_unexpected(
        error(holoscan::ErrorCode::kFailure, name_,
              "cudaMalloc of " + std::to_string(bytes) + " bytes: " + cudaGetErrorString(status)));
  }
  staging_bytes_ = bytes;
  return staging_;
}

holoscan::expected<void, holoscan::Error> ConversionRunner::run(
    holoscan::ExecutionContext& context, holoscan::Input<holoscan::schema::ImageT>& input,
    holoscan::Output<holoscan::schema::ImageT>& output, const Target& target) {
  auto sample = input.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  const schema::ImageT& image = sample->data;
  if (!image.data) {
    return holoscan::make_unexpected(
        error(holoscan::ErrorCode::kInvalidArgument, name_, "image has no tensor"));
  }
  const holoscan::Tensor& tensor = *image.data;
  auto viewed = view_input(image, tensor, target.custom_format, name_);
  if (!viewed) {
    return holoscan::make_unexpected(std::move(viewed).error());
  }
  ImageView source = viewed->view;
  // The runtime gives a stream to an operator with device outputs, and orders
  // device and stream-ready inputs on it before compute().
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(error(holoscan::ErrorCode::kFailure, name_,
                                           "the runtime gave no CUDA stream for the output"));
  }

  const PixelFormat format = target.format.value_or(source.format);
  const int width = target.width.value_or(source.width);
  const int height = target.height.value_or(source.height);
  ConversionParams params = target.params;
  if (target.use_input_color_space && is_yuv(source.format)) {
    params.matrix = matrix_of(image.color_space).value_or(params.matrix);
  }
  const std::size_t bytes = packed_size(format, width, height);
  if (bytes > target.max_bytes) {
    return holoscan::make_unexpected(
        error(holoscan::ErrorCode::kInvalidArgument, name_,
              "a " + std::to_string(width) + "x" + std::to_string(height) + " " +
                  std::string(format_name(format)) + " image needs " + std::to_string(bytes) +
                  " bytes, more than the output's capacity of " + std::to_string(target.max_bytes) +
                  "; raise max_width and max_height"));
  }

  // Every backend reads device memory: host images are staged once a frame.
  if (source.location == MemoryLocation::kHost) {
    if (tensor.device().device_type == kDLCPU) {
      if (auto ready = tensor.synchronize_for_host(); !ready) {
        return holoscan::make_unexpected(std::move(ready).error());
      }
    }
    auto buffer = staging(viewed->span, stream);
    if (!buffer) {
      return holoscan::make_unexpected(std::move(buffer).error());
    }
    if (const cudaError_t status =
            cudaMemcpyAsync(*buffer, tensor.data(), viewed->span, cudaMemcpyHostToDevice, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(
          error(holoscan::ErrorCode::kFailure, name_,
                std::string("cudaMemcpyAsync: ") + cudaGetErrorString(status)));
    }
    // Pinned memory is read by the copy after this returns: keep it from
    // reuse until the stream is past it, or wait here if it cannot wait.
    if (tensor.device().device_type == kDLCUDAHost) {
      holoscan::Tensor handle = tensor;
      if (!handle.set_deallocation_stream(stream)) {
        if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
          return holoscan::make_unexpected(
              error(holoscan::ErrorCode::kFailure, name_,
                    std::string("cudaStreamSynchronize: ") + cudaGetErrorString(status)));
        }
      }
    }
    const auto* host_base = static_cast<const std::uint8_t*>(tensor.data());
    for (int plane = 0; plane < format_info(source.format).planes; ++plane) {
      source.planes[plane].data =
          static_cast<std::uint8_t*>(*buffer) + (source.planes[plane].data - host_base);
    }
    source.location = MemoryLocation::kDevice;
  }

  auto loan = output.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = shape_of(format, width, height), .dtype = dtype_of(format)});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  const ImageView destination =
      packed_view(writer->data(), format, width, height, MemoryLocation::kDevice);

  try {
    const auto [chosen, reason] = converter_.choose(source, destination, params);
    if (!logged_backend_ || *logged_backend_ != chosen) {
      logged_backend_ = chosen;
      if (converter_.backend() == Backend::kAuto) {
        HOLOSCAN_LOG_INFO("{}: backend auto chose {} for {} {}x{} to {} {}x{}: {}", name_,
                          backend_name(chosen), format_name(source.format), source.width,
                          source.height, format_name(format), width, height, reason);
      }
      if (target.on_backend) {
        target.on_backend(chosen);
      }
    }
    if (viewed->view.location == MemoryLocation::kHost && !copy_logged_) {
      copy_logged_ = true;
      HOLOSCAN_LOG_INFO(
          "{} ({} backend): the images are in host memory and the backend reads CUDA device "
          "memory, so each frame is copied host to device, {} bytes ({:.1f} MB) per frame; to "
          "avoid the copy, send CUDA device images",
          name_, backend_name(chosen), viewed->span, static_cast<double>(viewed->span) / 1e6);
    }
    (void)converter_.convert(source, destination, params, stream);
  } catch (const std::exception& failure) {
    // Nothing queued may still use the images after this returns.
    cudaStreamSynchronize(stream);
    const bool bad_input = dynamic_cast<const std::invalid_argument*>(&failure) != nullptr;
    return holoscan::make_unexpected(
        error(bad_input ? holoscan::ErrorCode::kInvalidArgument : holoscan::ErrorCode::kFailure,
              name_, failure.what()));
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  return output.emit_tensor(std::move(*loan),
                            output_descriptor(image, format, width, height, params.matrix,
                                              source.format, target.keep_roi),
                            holoscan::sensor_io::derive_from(*sample));
}

}  // namespace holoscan::color_convert::detail
