// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

// NVIDIA VPI's AprilTag detector on the PVA (Jetson Orin and Thor) or the CPU.
// VPI has no CUDA AprilTag detector.

// VPI's C API.
extern "C" {
#include <vpi/Array.h>
#include <vpi/Context.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/AprilTags.h>
}

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/apriltag/detector_backend.hpp>

namespace holoscan::apriltag {
namespace {

// Image sides the PVA detector accepts, measured with VPI 4 on Jetson Thor: others
// fail to create or launch, or (one past a multiple of 128) fail every detect.
constexpr bool pva_side_usable(int side) noexcept {
  return side >= 256 && side <= 4096 && side % 128 != 1;
}
// Results VPI returns per frame. It discards any past its array's capacity
// without counting them, so one more than the frame reports shows that some
// were dropped.
constexpr std::int32_t kMaxDetections = static_cast<std::int32_t>(kMaxDetectionsPerFrame) + 1;

void check(VPIStatus status, std::string_view what) {
  if (status != VPI_SUCCESS) {
    char detail[VPI_MAX_STATUS_MESSAGE_LENGTH] = {};
    vpiGetLastStatusMessage(detail, sizeof(detail));
    throw std::runtime_error(std::string(what) + " failed: " + vpiStatusGetName(status) + ": " +
                             detail);
  }
}

VPIAprilTagFamily vpi_family(std::string_view name) {
  constexpr std::pair<std::string_view, VPIAprilTagFamily> kMap[] = {
      {"tag16h5", VPI_APRILTAG_16H5},
      {"tag25h9", VPI_APRILTAG_25H9},
      {"tag36h10", VPI_APRILTAG_36H10},
      {"tag36h11", VPI_APRILTAG_36H11},
      {"tagCircle21h7", VPI_APRILTAG_CIRCLE21H7},
      {"tagCircle49h12", VPI_APRILTAG_CIRCLE49H12},
      {"tagCustom48h12", VPI_APRILTAG_CUSTOM48H12},
      {"tagStandard41h12", VPI_APRILTAG_STANDARD41H12},
      {"tagStandard52h13", VPI_APRILTAG_STANDARD52H13},
  };
  for (const auto& [family_name, family] : kMap) {
    if (family_name == name) {
      return family;
    }
  }
  throw std::invalid_argument("VPI does not support AprilTag family '" + std::string(name) + "'");
}

// Whether VPI reports a PVA on this platform.
bool platform_has_pva() {
  VPIContext context = nullptr;
  check(vpiContextGetCurrent(&context), "vpiContextGetCurrent");
  std::uint64_t flags = 0;
  check(vpiContextGetFlags(context, &flags), "vpiContextGetFlags");
  return (flags & VPI_BACKEND_PVA) != 0U;
}

class VpiBackend final : public DetectorBackend {
 public:
  VpiBackend(const DetectorOptions& options, int width, int height)
      : DetectorBackend(width, height) {
    // The PVA where there is one and the image suits it, else VPI's CPU detector.
    if (platform_has_pva() && pva_side_usable(width) && pva_side_usable(height)) {
      hardware_ = VPI_BACKEND_PVA;
    }
    VPIAprilTagDecodeParams params{};
    check(vpiInitAprilTagDecodeParams(&params), "vpiInitAprilTagDecodeParams");
    params.family = vpi_family(options.family);
    params.maxBitsCorrected = options.max_hamming;
    try {
      check(vpiCreateAprilTagDetector(hardware_, width, height, &params, &payload_),
            "vpiCreateAprilTagDetector");
      // With the PVA, frames arrive in device memory and are copied in through CUDA.
      const std::uint64_t image_backends =
          hardware_ | VPI_BACKEND_CPU |
          (hardware_ == VPI_BACKEND_PVA ? std::uint64_t{VPI_BACKEND_CUDA} : std::uint64_t{0});
      check(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, image_backends, &gray_),
            "vpiImageCreate");
      check(vpiArrayCreate(kMaxDetections, VPI_ARRAY_TYPE_APRILTAG_DETECTION,
                           hardware_ | VPI_BACKEND_CPU, &detections_),
            "vpiArrayCreate");
      check(vpiStreamCreate(hardware_ | VPI_BACKEND_CPU, &stream_), "vpiStreamCreate");
    } catch (...) {
      release();
      throw;
    }
  }

  ~VpiBackend() override { release(); }

  [[nodiscard]] std::string_view name() const noexcept override {
    return hardware_ == VPI_BACKEND_PVA ? "vpi-pva" : "vpi-cpu";
  }
  [[nodiscard]] MemoryLocation input_location() const noexcept override {
    // The PVA reads device-visible memory; the CPU detector reads host memory.
    return hardware_ == VPI_BACKEND_PVA ? MemoryLocation::kDevice : MemoryLocation::kHost;
  }

  [[nodiscard]] FrameDetections find(const GrayImageView& image, cudaStream_t stream) override {
    upload(image, stream);
    check(
        vpiSubmitAprilTagDetector(stream_, hardware_, payload_, kMaxDetections, gray_, detections_),
        "vpiSubmitAprilTagDetector");
    check(vpiStreamSync(stream_), "vpiStreamSync");

    VPIArrayData data{};
    check(vpiArrayLockData(detections_, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &data),
          "vpiArrayLockData");
    // Unlock however the copy below ends.
    struct Unlock {
      VPIArray array;
      ~Unlock() { vpiArrayUnlock(array); }
    } unlock{detections_};
    FrameDetections result;
    const auto count = static_cast<std::size_t>(*data.buffer.aos.sizePointer);
    result.tags.reserve(count);
    const auto* base = static_cast<const std::byte*>(data.buffer.aos.data);
    for (std::size_t i = 0; i < count; ++i) {
      const auto* element = reinterpret_cast<const VPIAprilTagDetection*>(
          base + i * static_cast<std::size_t>(data.buffer.aos.strideBytes));
      const double center[2] = {element->center.x, element->center.y};
      double corners[4][2];
      for (int c = 0; c < 4; ++c) {
        corners[c][0] = element->corners[c].x;
        corners[c][1] = element->corners[c].y;
      }
      result.tags.push_back(make_detection(element->id, element->correctedBits,
                                           element->decisionMargin, center, corners));
    }
    return result;
  }

 private:
  // Copies `image` into the VPI-owned gray image, through the buffer type that
  // matches where the image lives.
  void upload(const GrayImageView& image, cudaStream_t stream) {
    const bool device = image.location == MemoryLocation::kDevice;
    VPIImageData data{};
    check(vpiImageLockData(
              gray_, VPI_LOCK_WRITE,
              device ? VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR : VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,
              &data),
          "vpiImageLockData");
    const VPIImagePlanePitchLinear& plane = data.buffer.pitch.planes[0];
    const auto pitch = static_cast<std::size_t>(plane.pitchBytes);
    const auto width = static_cast<std::size_t>(width_);
    cudaError_t synced = cudaSuccess;
    if (device) {
      synced =
          cudaMemcpy2DAsync(plane.pBase, pitch, image.data, image.pitch, width,
                            static_cast<std::size_t>(height_), cudaMemcpyDeviceToDevice, stream);
      if (synced == cudaSuccess) {
        synced = cudaStreamSynchronize(stream);
      }
    } else {
      auto* rows = static_cast<std::uint8_t*>(plane.pBase);
      for (int y = 0; y < height_; ++y) {
        std::memcpy(rows + static_cast<std::size_t>(y) * pitch,
                    image.data + static_cast<std::size_t>(y) * image.pitch, width);
      }
    }
    check(vpiImageUnlock(gray_), "vpiImageUnlock");
    if (synced != cudaSuccess) {
      throw std::runtime_error(std::string("copy into the VPI image failed: ") +
                               cudaGetErrorString(synced));
    }
  }

  void release() noexcept {
    if (stream_ != nullptr) {
      vpiStreamSync(stream_);
      vpiStreamDestroy(stream_);
      stream_ = nullptr;
    }
    vpiArrayDestroy(detections_);
    detections_ = nullptr;
    vpiImageDestroy(gray_);
    gray_ = nullptr;
    vpiPayloadDestroy(payload_);
    payload_ = nullptr;
  }

  std::uint64_t hardware_{VPI_BACKEND_CPU};
  VPIPayload payload_{nullptr};
  VPIImage gray_{nullptr};
  VPIArray detections_{nullptr};
  VPIStream stream_{nullptr};
};

}  // namespace

std::unique_ptr<DetectorBackend> create_vpi_backend(const DetectorOptions& options, int width,
                                                    int height) {
  return std::make_unique<VpiBackend>(options, width, height);
}

}  // namespace holoscan::apriltag
