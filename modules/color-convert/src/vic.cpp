// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Conversion and rescaling on the VIC of Jetson Orin and Thor, through VPI 4.
// VPI wraps the CUDA device images in place, so nothing is copied: a crop or
// a letterbox content rectangle is wrapped as a sub-image at an offset. The
// VIC runs on its own VPI stream, so it waits for the CUDA stream's work
// first, and the conversion returns once the VIC is done. The letterbox
// border is the CUDA kernel's, queued on the CUDA stream meanwhile.

extern "C" {
#include <vpi/ColorSpec.h>
#include <vpi/Context.h>
#include <vpi/Image.h>
#include <vpi/ImageFormat.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>
}

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "kernels.hpp"
#include "vic.hpp"

namespace holoscan::color_convert::detail {
namespace {

void check(VPIStatus status, std::string_view what) {
  if (status != VPI_SUCCESS) {
    char detail[VPI_MAX_STATUS_MESSAGE_LENGTH] = {};
    vpiGetLastStatusMessage(detail, sizeof(detail));
    throw std::runtime_error(std::string(what) + " failed: " + vpiStatusGetName(status) + ": " +
                             detail);
  }
}

void check_cuda(cudaError_t status, std::string_view what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

// The matrix and range, and nothing more. VPI's BT.2020 spec also names
// BT.2020 primaries, which makes the VIC convert the gamut as well, and the
// VIC takes the BT.2020 matrix with no other primaries: vic_limitation()
// keeps BT.2020 conversions between colour models off the VIC.
VPIColorSpec color_spec(ColorMatrix matrix, ColorRange range) {
  const bool full = range == ColorRange::kFull;
  if (matrix == ColorMatrix::kBt709) {
    return full ? VPI_COLOR_SPEC_BT709_ER : VPI_COLOR_SPEC_BT709;
  }
  return full ? VPI_COLOR_SPEC_BT601_ER : VPI_COLOR_SPEC_BT601;
}

// VPI's format for `format`; YCbCr formats carry the conversion's matrix and
// range. MONO8 is full-range luma.
VPIImageFormat vpi_format(PixelFormat format, const ConversionParams& params) {
  switch (format) {
    case PixelFormat::kNv12:
      return vpiImageFormatSetColorSpec(VPI_IMAGE_FORMAT_NV12,
                                        color_spec(params.matrix, params.range));
    case PixelFormat::kYuyv:
      return vpiImageFormatSetColorSpec(VPI_IMAGE_FORMAT_YUYV,
                                        color_spec(params.matrix, params.range));
    case PixelFormat::kUyvy:
      return vpiImageFormatSetColorSpec(VPI_IMAGE_FORMAT_UYVY,
                                        color_spec(params.matrix, params.range));
    case PixelFormat::kRgba8:
      return VPI_IMAGE_FORMAT_RGBA8;
    case PixelFormat::kBgra8:
      return VPI_IMAGE_FORMAT_BGRA8;
    case PixelFormat::kMono8:
      return vpiImageFormatSetColorSpec(VPI_IMAGE_FORMAT_Y8_ER,
                                        color_spec(params.matrix, ColorRange::kFull));
    default:
      break;
  }
  throw std::invalid_argument("the VIC does not handle " + std::string(format_name(format)));
}

// One VPI image wrapping CUDA device memory, re-pointed for each frame.
class Wrapper {
 public:
  Wrapper() = default;
  ~Wrapper() { vpiImageDestroy(image_); }
  Wrapper(const Wrapper&) = delete;
  Wrapper& operator=(const Wrapper&) = delete;

  // `rect` of `view`, as a VPI image of `format`.
  VPIImage wrap(const ImageView& view, const Rect& rect, VPIImageFormat format) {
    VPIImageData data{};
    data.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
    VPIImageBufferPitchLinear& pitch = data.buffer.pitch;
    pitch.format = format;
    pitch.numPlanes = vpiImageFormatGetPlaneCount(format);
    const FormatInfo info = format_info(view.format);
    for (int plane = 0; plane < pitch.numPlanes; ++plane) {
      VPIImagePlanePitchLinear& out = pitch.planes[plane];
      const bool chroma = plane > 0;
      const int x = chroma ? rect.x >> info.chroma_shift_x : rect.x;
      const int y = chroma ? rect.y >> info.chroma_shift_y : rect.y;
      // Bytes per pixel along a row of this plane: packed 4:2:2 is two per
      // pixel, NV12's chroma plane two per chroma sample.
      const std::size_t step = chroma ? 2U
                                      : static_cast<std::size_t>(info.channels) *
                                            static_cast<std::size_t>(info.bytes_per_sample);
      out.pixelType = vpiImageFormatGetPlanePixelType(format, plane);
      out.width = vpiImageFormatGetPlaneWidth(format, rect.width, plane);
      out.height = vpiImageFormatGetPlaneHeight(format, rect.height, plane);
      out.pitchBytes = static_cast<std::int32_t>(view.planes[plane].pitch);
      out.pBase = view.planes[plane].data + static_cast<std::size_t>(y) * view.planes[plane].pitch +
                  static_cast<std::size_t>(x) * step;
    }
    if (image_ != nullptr && format == format_ && rect.width == width_ && rect.height == height_) {
      check(vpiImageSetWrapper(image_, &data), "vpiImageSetWrapper");
      return image_;
    }
    vpiImageDestroy(image_);
    image_ = nullptr;
    check(vpiImageCreateWrapper(&data, nullptr, VPI_BACKEND_VIC | VPI_BACKEND_CUDA, &image_),
          "vpiImageCreateWrapper");
    // VPI leaves out a backend that cannot use the format, rather than failing.
    std::uint64_t flags = 0;
    check(vpiImageGetFlags(image_, &flags), "vpiImageGetFlags");
    if ((flags & VPI_BACKEND_VIC) == 0U) {
      vpiImageDestroy(image_);
      image_ = nullptr;
      throw std::invalid_argument(std::string("the VIC cannot use VPI format ") +
                                  vpiImageFormatGetName(format));
    }
    format_ = format;
    width_ = rect.width;
    height_ = rect.height;
    return image_;
  }

 private:
  VPIImage image_{nullptr};
  VPIImageFormat format_{VPI_IMAGE_FORMAT_INVALID};
  int width_{};
  int height_{};
};

class Vic final : public VicEngine {
 public:
  Vic() { check(vpiStreamCreate(VPI_BACKEND_VIC, &stream_), "vpiStreamCreate"); }
  ~Vic() override {
    vpiStreamSync(stream_);
    vpiStreamDestroy(stream_);
  }
  Vic(const Vic&) = delete;
  Vic& operator=(const Vic&) = delete;

  void convert(const ImageView& source, const ImageView& destination,
               const ConversionParams& params, cudaStream_t stream) override {
    const Mapping mapping = compute_mapping(source.width, source.height, destination.width,
                                            destination.height, destination.format, params);
    const VPIImage in = input_.wrap(source, mapping.source, vpi_format(source.format, params));
    const VPIImage out =
        output_.wrap(destination, mapping.content, vpi_format(destination.format, params));
    // The VIC reads what the CUDA stream has written so far.
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    const bool same_size = mapping.source.width == mapping.content.width &&
                           mapping.source.height == mapping.content.height;
    if (same_size && source.format != destination.format) {
      check(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_VIC, in, out, nullptr),
            "vpiSubmitConvertImageFormat");
    } else {
      check(vpiSubmitRescale(
                stream_, VPI_BACKEND_VIC, in, out,
                params.mode == ResizeMode::kNearest ? VPI_INTERP_NEAREST : VPI_INTERP_LINEAR,
                VPI_BORDER_CLAMP, 0),
            "vpiSubmitRescale");
    }
    // The border, if any, on the GPU while the VIC works: they write apart.
    const bool border =
        mapping.content.width != destination.width || mapping.content.height != destination.height;
    cudaError_t filled = cudaSuccess;
    if (border) {
      filled = launch_convert(make_plan(source, destination, params), true, stream);
    }
    check(vpiStreamSync(stream_), "vpiStreamSync");
    check_cuda(filled, "letterbox border kernel");
  }

 private:
  VPIStream stream_{nullptr};
  Wrapper input_;
  Wrapper output_;
};

}  // namespace

bool vpi_compiled() noexcept { return true; }

bool vpi_has_vic() {
  VPIContext context = nullptr;
  check(vpiContextGetCurrent(&context), "vpiContextGetCurrent");
  std::uint64_t flags = 0;
  check(vpiContextGetFlags(context, &flags), "vpiContextGetFlags");
  return (flags & VPI_BACKEND_VIC) != 0U;
}

std::unique_ptr<VicEngine> make_vic_engine() { return std::make_unique<Vic>(); }

}  // namespace holoscan::color_convert::detail
