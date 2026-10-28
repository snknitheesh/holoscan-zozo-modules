// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The vpi-* backends: VPI 4's Remap on the GPU, the VIC or the PVA.
//
// vpi-cuda reads and writes the operator's CUDA memory in place, through VPI
// image wrappers, on the operator's CUDA stream. The VIC and the PVA cannot
// use CUDA memory directly: VPI keeps a copy of a wrapped buffer for them and
// refreshes it only when the buffer is wrapped again. Measured on Jetson Orin
// at 1280x720 MONO8, rewrapping both images every frame took 4.6 ms on the
// VIC, while copying into and out of VPI's own images with CUDA took 2.7 ms
// (the same 7.2 ms on the PVA), so those two use VPI's own images.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

// VPI's headers declare C linkage themselves: CUDAInterop.h includes the CUDA
// runtime, whose C++ templates cannot sit inside an extern "C" block.
#include <vpi/CUDAInterop.h>  // NOLINT(build/include_order)
#include <vpi/Context.h>      // NOLINT(build/include_order)
#include <vpi/Image.h>        // NOLINT(build/include_order)
#include <vpi/Status.h>       // NOLINT(build/include_order)
#include <vpi/Stream.h>       // NOLINT(build/include_order)
#include <vpi/WarpMap.h>      // NOLINT(build/include_order)
#include <vpi/algo/Remap.h>   // NOLINT(build/include_order)

#include "remapper_factories.hpp"

namespace holoscan::rectify {
namespace {

// Warp-grid spacing on the VIC and PVA: the map is sampled every 4 pixels and
// interpolated between, which more than halves the VIC's time at 1280x720
// against a dense map. vpi-cuda takes the dense map.
constexpr int kHardwareGridInterval = 4;

std::string status_text(VPIStatus status) {
  char detail[VPI_MAX_STATUS_MESSAGE_LENGTH] = {};
  vpiGetLastStatusMessage(detail, sizeof(detail));
  return std::string(vpiStatusGetName(status)) + ": " + detail;
}

void check(VPIStatus status, std::string_view what) {
  if (status != VPI_SUCCESS) {
    throw std::runtime_error(std::string(what) + " failed: " + status_text(status));
  }
}

void check_cuda(cudaError_t status, std::string_view what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(status));
  }
}

std::uint64_t hardware(Backend backend) {
  switch (backend) {
    case Backend::kVpiCuda:
      return VPI_BACKEND_CUDA;
    case Backend::kVpiVic:
      return VPI_BACKEND_VIC;
    case Backend::kVpiPva:
      return VPI_BACKEND_PVA;
    default:
      throw std::invalid_argument("not a VPI backend");
  }
}

VPIImageFormat vpi_format(PixelFormat format) {
  switch (format) {
    case PixelFormat::kMono8:
      return VPI_IMAGE_FORMAT_U8;
    case PixelFormat::kMono16:
      return VPI_IMAGE_FORMAT_U16;
    case PixelFormat::kRgb8:
      return VPI_IMAGE_FORMAT_RGB8;
    case PixelFormat::kBgr8:
      return VPI_IMAGE_FORMAT_BGR8;
    case PixelFormat::kRgba8:
      return VPI_IMAGE_FORMAT_RGBA8;
    case PixelFormat::kBgra8:
      return VPI_IMAGE_FORMAT_BGRA8;
  }
  return VPI_IMAGE_FORMAT_INVALID;
}

VPIImageData wrapper_data(const DeviceImage& image, VPIImageFormat format) {
  VPIImageData data{};
  data.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
  data.buffer.pitch.format = format;
  data.buffer.pitch.numPlanes = 1;
  VPIImagePlanePitchLinear& plane = data.buffer.pitch.planes[0];
  plane.pixelType = vpiImageFormatGetPlanePixelType(format, 0);
  plane.width = image.width;
  plane.height = image.height;
  plane.pitchBytes = static_cast<std::int32_t>(image.pitch);
  plane.pBase = static_cast<VPIByte*>(image.data);
  return data;
}

class VpiRemapper final : public Remapper {
 public:
  VpiRemapper(Backend backend, const RemapModel& model, int output_width, int output_height,
              PixelFormat format, Interpolation interpolation)
      : Remapper(backend, model.input().width, model.input().height, output_width, output_height),
        hardware_(hardware(backend)),
        format_(vpi_format(format)),
        bytes_per_pixel_(bytes_per_pixel(format)),
        interpolation_(interpolation == Interpolation::kNearest ? VPI_INTERP_NEAREST
                                                                : VPI_INTERP_LINEAR) {
    try {
      build_map(model, backend == Backend::kVpiCuda ? 1 : kHardwareGridInterval);
      if (const VPIStatus status = vpiCreateRemap(hardware_, &map_, &payload_);
          status != VPI_SUCCESS) {
        throw std::invalid_argument(std::string(name()) +
                                    ": VPI cannot remap on this backend: " + status_text(status));
      }
      if (backend != Backend::kVpiCuda) {
        check(vpiStreamCreate(hardware_, &stream_), "vpiStreamCreate");
        const std::uint64_t flags = hardware_ | VPI_BACKEND_CUDA;
        check(vpiImageCreate(input_width_, input_height_, format_, flags, &input_image_),
              "vpiImageCreate");
        check(vpiImageCreate(output_width_, output_height_, format_, flags, &output_image_),
              "vpiImageCreate");
      }
      warm_up();
    } catch (...) {
      release();
      throw;
    }
  }

  ~VpiRemapper() override { release(); }

  [[nodiscard]] std::string_view name() const noexcept override {
    switch (backend_) {
      case Backend::kVpiCuda:
        return "vpi-cuda";
      case Backend::kVpiVic:
        return "vpi-vic";
      default:
        return "vpi-pva";
    }
  }

  void finish() override {
    if (!pending_) return;
    pending_ = false;
    check(vpiStreamSync(stream_), "vpiStreamSync");
    if (hardware_ != VPI_BACKEND_CUDA) {
      // The result, out of VPI's image, on the operator's stream. The next
      // submit() waits for that stream before the VIC writes the image again.
      copy(output_image_, VPI_LOCK_READ, pending_output_, pending_stream_, false);
    }
  }

 protected:
  void do_submit(const DeviceImage& input, const DeviceImage& output,
                 cudaStream_t stream) override {
    finish();
    if (hardware_ == VPI_BACKEND_CUDA) {
      // VPI's CUDA work goes on the operator's stream, after the input's producer.
      if (stream != wrapped_stream_ || stream_ == nullptr) {
        if (stream_ != nullptr) vpiStreamDestroy(stream_);
        stream_ = nullptr;
        if (stream == nullptr) {
          throw std::invalid_argument("vpi-cuda: needs a CUDA stream, not the legacy default one");
        }
        check(vpiStreamCreateWrapperCUDA(reinterpret_cast<CUstream>(stream), VPI_BACKEND_CUDA,
                                         &stream_),
              "vpiStreamCreateWrapperCUDA");
        wrapped_stream_ = stream;
      }
      wrap(input_image_, input);
      wrap(output_image_, output);
    } else {
      // The input, into VPI's image, after its producer; the VIC and the PVA
      // know nothing of CUDA streams, so wait for the copy here.
      copy(input_image_, VPI_LOCK_WRITE, input, stream, true);
      pending_output_ = output;
      pending_stream_ = stream;
    }
    check(vpiSubmitRemap(stream_, hardware_, payload_, input_image_, output_image_, interpolation_,
                         VPI_BORDER_ZERO, 0),
          "vpiSubmitRemap");
    pending_ = true;
  }

 private:
  void build_map(const RemapModel& model, int interval) {
    map_.grid.numHorizRegions = 1;
    map_.grid.numVertRegions = 1;
    map_.grid.regionWidth[0] = static_cast<std::int16_t>(output_width_);
    map_.grid.regionHeight[0] = static_cast<std::int16_t>(output_height_);
    map_.grid.horizInterval[0] = static_cast<std::int16_t>(interval);
    map_.grid.vertInterval[0] = static_cast<std::int16_t>(interval);
    if (const VPIStatus status = vpiWarpMapAllocData(&map_); status != VPI_SUCCESS) {
      throw std::invalid_argument(std::string(name()) +
                                  ": VPI refuses the warp grid: " + status_text(status));
    }
    map_allocated_ = true;
    check(vpiWarpMapGenerateIdentity(&map_), "vpiWarpMapGenerateIdentity");
    // Each control point, at its output position, takes the input position it samples.
    for (int row = 0; row < map_.numVertPoints; ++row) {
      auto* points =
          reinterpret_cast<VPIKeypointF32*>(reinterpret_cast<std::uint8_t*>(map_.keypoints) +
                                            static_cast<std::ptrdiff_t>(row) * map_.pitchBytes);
      for (int col = 0; col < map_.numHorizPoints; ++col) {
        const Point2 source = model.clamped_source(points[col].x, points[col].y);
        points[col].x = static_cast<float>(source.x);
        points[col].y = static_cast<float>(source.y);
      }
    }
  }

  // Copies between a VPI-owned image and `image` on `stream`: into it for
  // VPI_LOCK_WRITE, out of it for VPI_LOCK_READ. With `wait`, returns once
  // the copy is done.
  void copy(VPIImage vpi_image, VPILockMode mode, const DeviceImage& image, cudaStream_t stream,
            bool wait) {
    VPIImageData data{};
    check(vpiImageLockData(vpi_image, mode, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &data),
          "vpiImageLockData");
    const VPIImagePlanePitchLinear& plane = data.buffer.pitch.planes[0];
    const std::size_t row = static_cast<std::size_t>(image.width) * bytes_per_pixel_;
    cudaError_t status =
        mode == VPI_LOCK_WRITE
            ? cudaMemcpy2DAsync(plane.pBase, static_cast<std::size_t>(plane.pitchBytes), image.data,
                                image.pitch, row, static_cast<std::size_t>(image.height),
                                cudaMemcpyDeviceToDevice, stream)
            : cudaMemcpy2DAsync(
                  image.data, image.pitch, plane.pBase, static_cast<std::size_t>(plane.pitchBytes),
                  row, static_cast<std::size_t>(image.height), cudaMemcpyDeviceToDevice, stream);
    if (status == cudaSuccess && wait) status = cudaStreamSynchronize(stream);
    const VPIStatus unlocked = vpiImageUnlock(vpi_image);
    check_cuda(status, "cudaMemcpy2DAsync");
    check(unlocked, "vpiImageUnlock");
  }

  // Wraps `image` as `vpi_image`, reusing the wrapper object when there is one.
  void wrap(VPIImage& vpi_image, const DeviceImage& image) {
    const VPIImageData data = wrapper_data(image, format_);
    if (vpi_image != nullptr) {
      check(vpiImageSetWrapper(vpi_image, &data), "vpiImageSetWrapper");
      return;
    }
    check(vpiImageCreateWrapper(&data, nullptr, hardware_ | VPI_BACKEND_CUDA, &vpi_image),
          "vpiImageCreateWrapper");
  }

  // One remap of blank images, so that a backend that cannot take these
  // images or this interpolation fails here, at construction, not mid-run.
  void warm_up() {
    struct Buffer {
      void* data{};
      ~Buffer() { cudaFree(data); }
    } input, output;
    const std::size_t input_pitch = static_cast<std::size_t>(input_width_) * bytes_per_pixel_;
    const std::size_t output_pitch = static_cast<std::size_t>(output_width_) * bytes_per_pixel_;
    check_cuda(cudaMalloc(&input.data, input_pitch * input_height_), "cudaMalloc");
    check_cuda(cudaMalloc(&output.data, output_pitch * output_height_), "cudaMalloc");
    check_cuda(cudaMemset(input.data, 0, input_pitch * input_height_), "cudaMemset");
    // The memset runs on the legacy default stream, which `cuda_stream` does not wait for.
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    cudaStream_t cuda_stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking), "cudaStreamCreate");
    struct StreamDestroy {
      cudaStream_t stream;
      ~StreamDestroy() {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
      }
    } destroy{cuda_stream};
    try {
      do_submit(DeviceImage{input.data, input_pitch, input_width_, input_height_},
                DeviceImage{output.data, output_pitch, output_width_, output_height_}, cuda_stream);
      finish();
    } catch (const std::runtime_error& error) {
      pending_ = false;
      throw std::invalid_argument(std::string(name()) +
                                  ": VPI cannot remap these images: " + error.what());
    }
    // The wrappers and a wrapped stream point at the buffers and stream that go now.
    check_cuda(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize");
    if (hardware_ == VPI_BACKEND_CUDA) {
      vpiImageDestroy(input_image_);
      vpiImageDestroy(output_image_);
      input_image_ = nullptr;
      output_image_ = nullptr;
      vpiStreamDestroy(stream_);
      stream_ = nullptr;
      wrapped_stream_ = nullptr;
    }
  }

  void release() noexcept {
    if (stream_ != nullptr) {
      vpiStreamSync(stream_);
      vpiStreamDestroy(stream_);
      stream_ = nullptr;
    }
    vpiImageDestroy(input_image_);
    vpiImageDestroy(output_image_);
    input_image_ = nullptr;
    output_image_ = nullptr;
    vpiPayloadDestroy(payload_);
    payload_ = nullptr;
    if (map_allocated_) {
      vpiWarpMapFreeData(&map_);
      map_allocated_ = false;
    }
  }

  std::uint64_t hardware_;
  VPIImageFormat format_;
  int bytes_per_pixel_;
  VPIInterpolationType interpolation_;
  VPIWarpMap map_{};
  bool map_allocated_{};
  VPIPayload payload_{};
  VPIStream stream_{};
  cudaStream_t wrapped_stream_{};
  // vpi-cuda: wrappers of the operator's images. VIC and PVA: VPI's own images.
  VPIImage input_image_{};
  VPIImage output_image_{};
  bool pending_{};
  // VIC and PVA: where finish() copies the result, and on which stream.
  DeviceImage pending_output_{};
  cudaStream_t pending_stream_{};
};

}  // namespace

void vpi_platform(bool& has_vic, bool& has_pva) {
  VPIContext context = nullptr;
  check(vpiContextGetCurrent(&context), "vpiContextGetCurrent");
  std::uint64_t flags = 0;
  check(vpiContextGetFlags(context, &flags), "vpiContextGetFlags");
  has_vic = (flags & VPI_BACKEND_VIC) != 0U;
  has_pva = (flags & VPI_BACKEND_PVA) != 0U;
}

std::unique_ptr<Remapper> create_vpi_remapper(Backend backend, const RemapModel& model,
                                              int output_width, int output_height,
                                              PixelFormat format, Interpolation interpolation) {
  return std::make_unique<VpiRemapper>(backend, model, output_width, output_height, format,
                                       interpolation);
}

}  // namespace holoscan::rectify
