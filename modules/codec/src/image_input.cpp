// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "image_input.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>

#include "backend.hpp"

namespace holoscan::codec::detail {
namespace {

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
    case holoscan::schema::ImageEncoding_NV12:
      return PixelFormat::kNv12;
    default:
      return std::nullopt;
  }
}

std::size_t chroma_row_bytes(int width) { return 2 * static_cast<std::size_t>((width + 1) / 2); }

}  // namespace

holoscan::expected<ImageView, std::string> view_image(const holoscan::schema::ImageT& image) {
  using Result = holoscan::expected<ImageView, std::string>;
  if (!image.data) {
    return holoscan::make_unexpected(std::string("the image has no tensor"));
  }
  const auto format = pixel_format(image.encoding);
  if (!format) {
    return holoscan::make_unexpected(std::string("unsupported image encoding ") +
                                     holoscan::schema::EnumNameImageEncoding(image.encoding) +
                                     "; use RGB8, BGR8, RGBA8, BGRA8 or NV12");
  }
  const holoscan::Tensor& tensor = *image.data;
  const DLDataType dtype = tensor.dtype();
  if (dtype.code != kDLUInt || dtype.bits != 8 || dtype.lanes != 1) {
    return holoscan::make_unexpected(std::string("image tensors must hold uint8 elements"));
  }
  const DLDevice device = tensor.device();
  const bool on_device = device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
  if (!on_device && device.device_type != kDLCPU && device.device_type != kDLCUDAHost) {
    return holoscan::make_unexpected(std::string("image tensors must be in host or CUDA memory"));
  }
  const std::span<const std::int64_t> shape = tensor.shape_span();
  const std::vector<std::int64_t> strides = tensor.strides();
  const auto* base = static_cast<const std::uint8_t*>(tensor.data());
  ImageView view{.on_device = on_device};
  DeviceImage& out = view.image;
  out.format = *format;
  if (*format == PixelFormat::kNv12) {
    if (image.width <= 0 || image.height <= 0 || image.plane_layouts.size() != 2) {
      return holoscan::make_unexpected(
          std::string("an NV12 image needs its width, height and two plane layouts"));
    }
    std::size_t extent = 1;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
      extent += static_cast<std::size_t>(shape[axis] - 1) * static_cast<std::size_t>(strides[axis]);
    }
    const auto& luma = image.plane_layouts[0];
    const auto& chroma = image.plane_layouts[1];
    const std::size_t rows = static_cast<std::size_t>(image.height);
    const std::size_t chroma_rows = (rows + 1) / 2;
    if (luma.row_stride_bytes() < static_cast<std::uint32_t>(image.width) ||
        chroma.row_stride_bytes() < chroma_row_bytes(image.width) ||
        luma.offset_bytes() + luma.row_stride_bytes() * rows > extent ||
        chroma.offset_bytes() + chroma.row_stride_bytes() * chroma_rows > extent) {
      return holoscan::make_unexpected(std::string("the NV12 plane layouts do not fit the image"));
    }
    out.data = base + luma.offset_bytes();
    out.pitch = luma.row_stride_bytes();
    out.chroma = base + chroma.offset_bytes();
    out.chroma_pitch = chroma.row_stride_bytes();
    out.width = image.width;
    out.height = image.height;
    return Result{view};
  }
  const int channels = bytes_per_pixel(*format);
  if (shape.size() != 3 || shape[2] != channels) {
    return holoscan::make_unexpected(
        std::string("image tensor shape must be [height, width, channels] matching its encoding"));
  }
  if (strides[2] != 1 || strides[1] != channels || strides[0] < shape[1] * channels) {
    return holoscan::make_unexpected(std::string("image pixels must be packed within each row"));
  }
  if ((image.width != 0 && image.width != shape[1]) ||
      (image.height != 0 && image.height != shape[0])) {
    return holoscan::make_unexpected(
        std::string("image width and height disagree with its tensor"));
  }
  out.data = base;
  out.pitch = static_cast<std::size_t>(strides[0]);
  out.width = static_cast<int>(shape[1]);
  out.height = static_cast<int>(shape[0]);
  return Result{view};
}

std::size_t image_bytes(const DeviceImage& image) {
  const auto rows = static_cast<std::size_t>(image.height);
  if (image.format == PixelFormat::kNv12) {
    return static_cast<std::size_t>(image.width) * rows +
           chroma_row_bytes(image.width) * ((rows + 1) / 2);
  }
  return static_cast<std::size_t>(image.width) * rows *
         static_cast<std::size_t>(bytes_per_pixel(image.format));
}

DeviceImage ImageStaging::to_device(const DeviceImage& host, cudaStream_t stream) {
  const std::size_t bytes = image_bytes(host);
  if (bytes_ < bytes) {
    // Free only after the stream is done with the old copy.
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    release();
    void* memory = nullptr;
    check_cuda(cudaMalloc(&memory, bytes), "cudaMalloc");
    data_ = static_cast<std::uint8_t*>(memory);
    bytes_ = bytes;
  }
  DeviceImage copy = host;
  copy.data = data_;
  if (host.format == PixelFormat::kNv12) {
    const auto width = static_cast<std::size_t>(host.width);
    const auto rows = static_cast<std::size_t>(host.height);
    copy.pitch = width;
    copy.chroma = data_ + width * rows;
    copy.chroma_pitch = chroma_row_bytes(host.width);
    check_cuda(cudaMemcpy2DAsync(data_, copy.pitch, host.data, host.pitch, width, rows,
                                 cudaMemcpyHostToDevice, stream),
               "copying the image to the device");
    check_cuda(cudaMemcpy2DAsync(const_cast<std::uint8_t*>(copy.chroma), copy.chroma_pitch,
                                 host.chroma, host.chroma_pitch, copy.chroma_pitch, (rows + 1) / 2,
                                 cudaMemcpyHostToDevice, stream),
               "copying the image to the device");
  } else {
    const std::size_t row = static_cast<std::size_t>(host.width) *
                            static_cast<std::size_t>(bytes_per_pixel(host.format));
    copy.pitch = row;
    check_cuda(
        cudaMemcpy2DAsync(data_, row, host.data, host.pitch, row,
                          static_cast<std::size_t>(host.height), cudaMemcpyHostToDevice, stream),
        "copying the image to the device");
  }
  // The host image may be pinned memory its producer reuses once this returns.
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  return copy;
}

void ImageStaging::release() noexcept {
  cudaFree(data_);
  data_ = nullptr;
  bytes_ = 0;
}

}  // namespace holoscan::codec::detail
