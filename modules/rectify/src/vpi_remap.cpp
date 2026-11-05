// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The vpi-vic and vpi-pva backends: VPI 4's Remap on the VIC or the PVA.
//
// The VIC and the PVA cannot use CUDA memory directly: VPI keeps a copy of a
// wrapped buffer for them and refreshes it only when the buffer is wrapped
// again. Measured on Jetson Orin at 1280x720 MONO8, rewrapping both images
// every frame took 4.6 ms on the VIC, while copying into and out of VPI's own
// images took 2.7 ms, so the remap runs between VPI-owned images and the
// copies run on the GPU. Each frame is three tasks on the queue's VPI stream,
// which wraps the operator's CUDA stream: a CUDA copy from the input into
// VPI's input image, the remap on the VIC or PVA, and a CUDA copy from VPI's
// output image into the output. VPI orders the three across the engines; the
// input and output are reached through wrappers that only the CUDA backend
// uses, so VPI keeps no copy of them.

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

// VPI's headers declare C linkage themselves: CUDAInterop.h includes the CUDA
// runtime, whose C++ templates cannot sit inside an extern "C" block.
#include <vpi/CUDAInterop.h>  // NOLINT(build/include_order)
#include <vpi/Context.h>      // NOLINT(build/include_order)
#include <vpi/Image.h>        // NOLINT(build/include_order)
#include <vpi/Status.h>       // NOLINT(build/include_order)
#include <vpi/Stream.h>       // NOLINT(build/include_order)
#include <vpi/WarpMap.h>      // NOLINT(build/include_order)
#include <vpi/algo/Copy.h>    // NOLINT(build/include_order)
#include <vpi/algo/Remap.h>   // NOLINT(build/include_order)

#include "remapper_factories.hpp"
#include "vpi_status.hpp"

namespace holoscan::rectify {

struct RemapQueue::Vpi {
  VPIStream stream{};
  // This stream among all made in the process, which its address, reused
  // after a stream's destruction, would not tell.
  std::uint64_t id{};
  // Work was submitted since the last wait.
  bool pending{};
};

namespace {

using detail::refuse_or_fail;
using detail::status_text;

std::atomic<std::uint64_t> g_next_stream_id{0};

// Warp-grid spacing: the map is sampled every 4 pixels and interpolated
// between, which more than halves the VIC's time at 1280x720 against a dense
// map.
constexpr int kGridInterval = 4;

void check(VPIStatus status, std::string_view what) {
  if (status != VPI_SUCCESS) {
    throw std::runtime_error(std::string(what) + " failed: " + status_text(status));
  }
}

std::uint64_t hardware(Backend backend) {
  switch (backend) {
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

// VPI objects that go with the scope.
struct ScopedVpi {
  VPIStream stream{};
  VPIPayload payload{};
  VPIImage input{};
  VPIImage output{};
  ScopedVpi() = default;
  ~ScopedVpi() {
    vpiStreamDestroy(stream);  // Waits for the stream's work.
    vpiPayloadDestroy(payload);
    vpiImageDestroy(input);
    vpiImageDestroy(output);
  }
  ScopedVpi(const ScopedVpi&) = delete;
  ScopedVpi& operator=(const ScopedVpi&) = delete;
};

class VpiRemapper final : public Remapper {
 public:
  VpiRemapper(Backend backend, const RemapModel& model, int output_width, int output_height,
              PixelFormat format, Interpolation interpolation)
      : Remapper(backend, format, static_cast<int>(model.input().width),
                 static_cast<int>(model.input().height), output_width, output_height),
        hardware_(hardware(backend)),
        vpi_format_(vpi_format(format)),
        interpolation_(interpolation == Interpolation::kNearest ? VPI_INTERP_NEAREST
                                                                : VPI_INTERP_LINEAR) {
    try {
      build_map(model);
      create_objects();
      warm_up();
    } catch (...) {
      release();
      throw;
    }
  }

  ~VpiRemapper() override { release(); }

  [[nodiscard]] std::string_view name() const noexcept override {
    return backend_ == Backend::kVpiVic ? "vpi-vic" : "vpi-pva";
  }

 protected:
  // Queues copy in, remap and copy out on the queue's VPI stream, and returns.
  void do_submit(const DeviceImage& input, const DeviceImage& output, RemapQueue& queue) override {
    RemapQueue::Vpi& vpi = vpi_of(queue);
    if (stream_id_ && *stream_id_ != vpi.id) {
      // Another VPI stream than the last submit's (another queue, or the
      // queue's new stream after a failure): VPI objects used on one stream
      // and then on another make VPI report NvError_InvalidState, so this
      // stream gets new ones. The last submit is finished, or its stream
      // destroyed, which waited for it, so nothing uses the old ones.
      destroy_objects();
      create_objects();
    }
    stream_id_ = vpi.id;
    wrap(input_wrapper_, input);
    wrap(output_wrapper_, output);
    vpi.pending = true;
    check(vpiSubmitCopyImage(vpi.stream, VPI_BACKEND_CUDA, input_wrapper_.image, input_image_),
          "vpiSubmitCopyImage");
    check(vpiSubmitRemap(vpi.stream, hardware_, payload_, input_image_, output_image_,
                         interpolation_, VPI_BORDER_ZERO, 0),
          "vpiSubmitRemap");
    check(vpiSubmitCopyImage(vpi.stream, VPI_BACKEND_CUDA, output_image_, output_wrapper_.image),
          "vpiSubmitCopyImage");
  }

 private:
  void build_map(const RemapModel& model) {
    map_.grid.numHorizRegions = 1;
    map_.grid.numVertRegions = 1;
    map_.grid.regionWidth[0] = static_cast<std::int16_t>(output_width_);
    map_.grid.regionHeight[0] = static_cast<std::int16_t>(output_height_);
    map_.grid.horizInterval[0] = static_cast<std::int16_t>(kGridInterval);
    map_.grid.vertInterval[0] = static_cast<std::int16_t>(kGridInterval);
    if (const VPIStatus status = vpiWarpMapAllocData(&map_); status != VPI_SUCCESS) {
      refuse_or_fail(status, std::string(name()) + ": VPI refuses the warp grid");
    }
    map_allocated_ = true;
    check(vpiWarpMapGenerateIdentity(&map_), "vpiWarpMapGenerateIdentity");
    // Each control point, at its output position, takes the input position it
    // samples, unclamped (grid_source()), since VPI interpolates between them.
    for (int row = 0; row < map_.numVertPoints; ++row) {
      auto* points =
          reinterpret_cast<VPIKeypointF32*>(reinterpret_cast<std::uint8_t*>(map_.keypoints) +
                                            static_cast<std::ptrdiff_t>(row) * map_.pitchBytes);
      for (int col = 0; col < map_.numHorizPoints; ++col) {
        const Point2 source = model.grid_source(points[col].x, points[col].y);
        points[col].x = static_cast<float>(source.x);
        points[col].y = static_cast<float>(source.y);
      }
    }
  }

  // A CUDA-only wrapper and the row pitch it wraps.
  struct Wrapper {
    VPIImage image{};
    std::size_t pitch{};
  };

  // The payload and VPI's own images, for one VPI stream.
  void create_objects() {
    if (const VPIStatus status = vpiCreateRemap(hardware_, &map_, &payload_);
        status != VPI_SUCCESS) {
      refuse_or_fail(status, std::string(name()) + ": VPI cannot remap on this backend");
    }
    const std::uint64_t flags = hardware_ | VPI_BACKEND_CUDA;
    check(vpiImageCreate(input_width_, input_height_, vpi_format_, flags, &input_image_),
          "vpiImageCreate");
    check(vpiImageCreate(output_width_, output_height_, vpi_format_, flags, &output_image_),
          "vpiImageCreate");
  }

  // Points `wrapper` at `image`, reusing the wrapper object when there is one
  // of the same pitch: VPI re-points a wrapper only at memory of the same
  // capacity, so another pitch takes a new one (the last submit, which used
  // the old one, is finished). Only the CUDA backend reads the wrapper, so VPI
  // keeps no copy of it.
  void wrap(Wrapper& wrapper, const DeviceImage& image) {
    const VPIImageData data = wrapper_data(image, vpi_format_);
    if (wrapper.image != nullptr && wrapper.pitch == image.pitch) {
      check(vpiImageSetWrapper(wrapper.image, &data), "vpiImageSetWrapper");
      return;
    }
    vpiImageDestroy(wrapper.image);
    wrapper.image = nullptr;
    check(vpiImageCreateWrapper(&data, nullptr, VPI_BACKEND_CUDA, &wrapper.image),
          "vpiImageCreateWrapper");
    wrapper.pitch = image.pitch;
  }

  // One remap with a throwaway payload, images and stream, so that a backend
  // that cannot take these images or this interpolation fails here, at
  // construction, not mid-run. The remapper's own payload and images are left
  // for the queue's stream: VPI objects used on one stream and then on
  // another make VPI report NvError_InvalidState when the second goes.
  void warm_up() {
    ScopedVpi trial;
    check(vpiCreateRemap(hardware_, &map_, &trial.payload), "vpiCreateRemap");
    check(vpiStreamCreate(hardware_, &trial.stream), "vpiStreamCreate");
    check(vpiImageCreate(input_width_, input_height_, vpi_format_, hardware_, &trial.input),
          "vpiImageCreate");
    check(vpiImageCreate(output_width_, output_height_, vpi_format_, hardware_, &trial.output),
          "vpiImageCreate");
    VPIStatus status = vpiSubmitRemap(trial.stream, hardware_, trial.payload, trial.input,
                                      trial.output, interpolation_, VPI_BORDER_ZERO, 0);
    if (status == VPI_SUCCESS) {
      count_host_wait();
      status = vpiStreamSync(trial.stream);
    }
    if (status != VPI_SUCCESS) {
      refuse_or_fail(status, std::string(name()) + ": VPI cannot remap these images");
    }
  }

  // Everything create_objects() and wrap() make.
  void destroy_objects() noexcept {
    for (Wrapper* wrapper : {&input_wrapper_, &output_wrapper_}) {
      vpiImageDestroy(wrapper->image);
      *wrapper = Wrapper{};
    }
    vpiImageDestroy(input_image_);
    vpiImageDestroy(output_image_);
    input_image_ = nullptr;
    output_image_ = nullptr;
    vpiPayloadDestroy(payload_);
    payload_ = nullptr;
  }

  void release() noexcept {
    destroy_objects();
    if (map_allocated_) {
      vpiWarpMapFreeData(&map_);
      map_allocated_ = false;
    }
  }

  std::uint64_t hardware_;
  VPIImageFormat vpi_format_;
  VPIInterpolationType interpolation_;
  VPIWarpMap map_{};
  bool map_allocated_{};
  VPIPayload payload_{};
  // VPI's own images, which the VIC or PVA reads and writes.
  VPIImage input_image_{};
  VPIImage output_image_{};
  // CUDA-only wrappers of the last submit's input and output.
  Wrapper input_wrapper_;
  Wrapper output_wrapper_;
  // The VPI stream (RemapQueue::Vpi::id) of the last submit.
  std::optional<std::uint64_t> stream_id_;
};

}  // namespace

void RemapQueue::VpiDelete::operator()(Vpi* vpi) const noexcept {
  if (vpi == nullptr) return;
  vpiStreamDestroy(vpi->stream);  // Waits for the stream's work.
  delete vpi;
}

RemapQueue::Vpi& RemapQueue::vpi() {
  if (vpi_) return *vpi_;
  // CUDA for the copies; every VPI hardware engine of this platform for the remaps.
  VPIContext context = nullptr;
  check(vpiContextGetCurrent(&context), "vpiContextGetCurrent");
  std::uint64_t context_flags = 0;
  check(vpiContextGetFlags(context, &context_flags), "vpiContextGetFlags");
  const std::uint64_t flags =
      VPI_BACKEND_CUDA | (context_flags & (VPI_BACKEND_VIC | VPI_BACKEND_PVA));
  std::unique_ptr<Vpi, VpiDelete> created(new Vpi);
  check(vpiStreamCreateWrapperCUDA(reinterpret_cast<CUstream>(stream_), flags, &created->stream),
        "vpiStreamCreateWrapperCUDA");
  created->id = g_next_stream_id.fetch_add(1, std::memory_order_relaxed);
  vpi_ = std::move(created);
  return *vpi_;
}

bool RemapQueue::finish_vpi() {
  if (!vpi_->pending) return false;
  vpi_->pending = false;
  count_host_wait();
  if (const VPIStatus status = vpiStreamSync(vpi_->stream); status != VPI_SUCCESS) {
    const std::string message = "vpiStreamSync failed: " + status_text(status);
    // A stream whose task failed runs nothing more; the next submit makes a
    // new one, and each remapper new VPI objects for it.
    vpi_.reset();
    throw std::runtime_error(message);
  }
  return true;
}

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
