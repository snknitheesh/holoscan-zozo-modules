// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "vpi_matcher.hpp"

// VPI's CUDA interop includes the CUDA runtime's C++ headers, so it stays out
// of extern "C"; it declares its functions' C linkage itself.
#include <vpi/CUDAInterop.h>

// VPI's C API.
extern "C" {
#include <vpi/Context.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Copy.h>
#include <vpi/algo/StereoDisparity.h>
}

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "holoscan/stereo_disparity/platform.hpp"

namespace holoscan::stereo_disparity {
namespace detail {
namespace {

std::uint64_t vpi_backend(Backend backend) {
  switch (backend) {
    case Backend::kCuda:
      return VPI_BACKEND_CUDA;
    case Backend::kOfa:
      return VPI_BACKEND_OFA;
    case Backend::kPva:
      return VPI_BACKEND_PVA | VPI_BACKEND_OFA | VPI_BACKEND_VIC;
    case Backend::kAuto:
      break;
  }
  throw std::invalid_argument("the matcher needs a resolved backend, not auto");
}

VPIStereoDisparityConfidenceType vpi_confidence_type(ConfidenceType type) {
  switch (type) {
    case ConfidenceType::kRelative:
      return VPI_STEREO_CONFIDENCE_RELATIVE;
    case ConfidenceType::kInference:
      return VPI_STEREO_CONFIDENCE_INFERENCE;
    case ConfidenceType::kAbsolute:
      break;
  }
  return VPI_STEREO_CONFIDENCE_ABSOLUTE;
}

VPIImageFormat vpi_format(PixelFormat format) {
  switch (format) {
    case PixelFormat::kRgb8:
      return VPI_IMAGE_FORMAT_RGB8;
    case PixelFormat::kBgr8:
      return VPI_IMAGE_FORMAT_BGR8;
    case PixelFormat::kRgba8:
      return VPI_IMAGE_FORMAT_RGBA8;
    case PixelFormat::kBgra8:
      return VPI_IMAGE_FORMAT_BGRA8;
    case PixelFormat::kMono8:
      break;
  }
  return VPI_IMAGE_FORMAT_Y8_ER;
}

// CUDA device memory described for VPI. VPI only reads an input, so the
// const_cast does not let anything write it.
VPIImageData cuda_plane(const void* data, std::size_t pitch, int width, int height,
                        VPIImageFormat format) {
  VPIImageData image{};
  image.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
  image.buffer.pitch.format = format;
  image.buffer.pitch.numPlanes = 1;
  VPIImagePlanePitchLinear& plane = image.buffer.pitch.planes[0];
  plane.width = width;
  plane.height = height;
  plane.pitchBytes = static_cast<std::int32_t>(pitch);
  plane.pBase = static_cast<VPIByte*>(const_cast<void*>(data));
  return image;
}

void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

// Jetson Thor's integrated GPU: compute capability 11.0 with CUDA 13 (10.1
// with CUDA 12). Orin's is 8.7.
bool thor_gpu() noexcept {
  int device = 0;
  int integrated = 0;
  int major = 0;
  int minor = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device) != cudaSuccess ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
    return false;
  }
  return integrated != 0 && (major == 11 || (major == 10 && minor == 1));
}

// An image locked for CUDA to read, unlocked when it goes.
class CudaReadLock {
 public:
  CudaReadLock(VPIImage image, const char* what) : image_(image) {
    VPIImageData data{};
    vpi_check(vpiImageLockData(image, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &data),
              what);
    plane_ = data.buffer.pitch.planes[0];
  }
  ~CudaReadLock() { vpiImageUnlock(image_); }
  CudaReadLock(const CudaReadLock&) = delete;
  CudaReadLock& operator=(const CudaReadLock&) = delete;

  [[nodiscard]] const VPIImagePlanePitchLinear& plane() const noexcept { return plane_; }

 private:
  VPIImage image_;
  VPIImagePlanePitchLinear plane_{};
};

}  // namespace

void vpi_check(VPIStatus status, const char* what) {
  if (status != VPI_SUCCESS) {
    char detail[VPI_MAX_STATUS_MESSAGE_LENGTH] = {};
    vpiGetLastStatusMessage(detail, sizeof(detail));
    throw std::runtime_error(std::string(what) + ": " + vpiStatusGetName(status) + ": " + detail);
  }
}

Matcher::Matcher(Backend backend, const DisparityParams& params, bool confidence, int width,
                 int height, cudaStream_t stream)
    : backend_(backend),
      params_(params),
      width_(width),
      height_(height),
      output_width_(output_side(width, params.downscale)),
      output_height_(output_side(height, params.downscale)),
      searched_(searched_disparity(backend, params)),
      num_passes_(num_passes_for(params, probe_platform())),
      cuda_stream_(stream) {
  const std::uint64_t algorithm = vpi_backend(backend);
  if (auto problem = params_problem(backend, params, confidence)) {
    throw std::invalid_argument(*problem);
  }
  if (auto problem = size_problem(backend, params, confidence, width, height)) {
    throw std::invalid_argument(*problem);
  }
  try {
    const bool cuda = backend == Backend::kCuda;
    const bool ofa = backend == Backend::kOfa;
    if (backend == Backend::kPva) {
      vpi_check(vpiStreamCreate(algorithm | VPI_BACKEND_CUDA, &stream_), "vpiStreamCreate");
    } else {
      vpi_check(vpiStreamCreateWrapperCUDA(stream, algorithm, &stream_),
                "vpiStreamCreateWrapperCUDA");
    }
    const std::size_t output_row = static_cast<std::size_t>(output_width_) * 2U;
    allocate(disparity_, output_width_, output_height_, output_row, VPI_IMAGE_FORMAT_S16,
             VPI_BACKEND_CUDA);
    if (confidence) {
      allocate(confidence_, output_width_, output_height_, output_row, VPI_IMAGE_FORMAT_U16,
               VPI_BACKEND_CUDA);
    }
    if (!cuda) {
      // Written by the OFA (block-linear, locked for CUDA) or the PVA (copied
      // by VPI's CUDA backend).
      const std::uint64_t backends = VPI_BACKEND_CUDA | VPI_BACKEND_VIC | algorithm;
      vpi_check(vpiImageCreate(output_width_, output_height_,
                               ofa ? VPI_IMAGE_FORMAT_S16_BL : VPI_IMAGE_FORMAT_S16, backends,
                               &accelerator_disparity_),
                "vpiImageCreate(disparity)");
      if (confidence) {
        vpi_check(vpiImageCreate(output_width_, output_height_, VPI_IMAGE_FORMAT_U16, backends,
                                 &accelerator_confidence_),
                  "vpiImageCreate(confidence)");
      }
    }
    const VPIImageFormat input = cuda ? VPI_IMAGE_FORMAT_Y8_ER : VPI_IMAGE_FORMAT_Y8_ER_BL;
    for (VPIImage& image : gray_) {
      vpi_check(
          vpiImageCreate(width, height, input, VPI_BACKEND_CUDA | (cuda ? 0U : algorithm), &image),
          "vpiImageCreate(gray)");
    }
    VPIStereoDisparityEstimatorCreationParams creation{};
    vpi_check(vpiInitStereoDisparityEstimatorCreationParams(&creation),
              "vpiInitStereoDisparityEstimatorCreationParams");
    creation.maxDisparity = params.max_disparity;
    creation.includeDiagonals = include_diagonals_for(backend, params) ? 1 : 0;
    creation.downscaleFactor = params.downscale;
    vpi_check(
        vpiCreateStereoDisparityEstimator(algorithm, width, height, input, &creation, &payload_),
        "vpiCreateStereoDisparityEstimator");
  } catch (...) {
    release();
    throw;
  }
}

Matcher::~Matcher() { release(); }

template <typename T>
void Matcher::allocate(OwnedPlane<T>& plane, int width, int height, std::size_t row_bytes,
                       VPIImageFormat format, std::uint64_t backends) {
  void* memory = nullptr;
  std::size_t pitch = 0;
  cuda_check(cudaMallocPitch(&memory, &pitch, row_bytes, static_cast<std::size_t>(height)),
             "cudaMallocPitch");
  plane.memory.reset(memory);
  plane.data = static_cast<T*>(memory);
  plane.pitch = pitch;
  const VPIImageData data = cuda_plane(memory, pitch, width, height, format);
  vpi_check(vpiImageCreateWrapper(&data, nullptr, backends, &plane.image), "vpiImageCreateWrapper");
}

void Matcher::match(const std::array<DeviceImage, 2>& pair) {
  try {
    std::array<VPIImage, 2> sources{};
    std::array<bool, 2> convert{};
    for (int side = 0; side < 2; ++side) {
      sources[side] = wrap_input(side, pair[side]);
      // cuda matches MONO8 as it is; everything else goes to (block-linear) gray.
      convert[side] = backend_ != Backend::kCuda || pair[side].format != PixelFormat::kMono8;
    }
    if (backend_ == Backend::kPva) {
      // pva's VPI stream is its own: what is queued on the work stream (the
      // inputs' producers, copies and expansion) must be done before VPI reads.
      cuda_check(cudaStreamSynchronize(cuda_stream_), "cudaStreamSynchronize");
    }
    submit(sources, convert);
  } catch (...) {
    // Nothing queued may still read the inputs or write the outputs.
    vpiStreamSync(stream_);
    cudaStreamSynchronize(cuda_stream_);
    throw;
  }
  vpi_check(vpiStreamSync(stream_), "vpiStreamSync");
  if (backend_ == Backend::kOfa) {
    copy_out();
  }
}

void Matcher::copy_out() {
  const CudaReadLock disparity(accelerator_disparity_, "vpiImageLockData(disparity)");
  const VPIImagePlanePitchLinear& view = disparity.plane();
  const cudaError_t copied = cudaMemcpy2DAsync(
      disparity_.memory.get(), disparity_.pitch, view.pBase,
      static_cast<std::size_t>(view.pitchBytes), static_cast<std::size_t>(output_width_) * 2U,
      static_cast<std::size_t>(output_height_), cudaMemcpyDeviceToDevice, cuda_stream_);
  // VPI may move the memory once it is unlocked, when the lock goes: whatever
  // was queued must be done first.
  const cudaError_t waited = cudaStreamSynchronize(cuda_stream_);
  cuda_check(copied, "cudaMemcpy2DAsync");
  cuda_check(waited, "cudaStreamSynchronize");
}

VPIImage Matcher::wrap_input(int side, const DeviceImage& image) {
  if (image.format == PixelFormat::kMono8 && backend_ != Backend::kCuda) {
    OwnedPlane<std::uint8_t>& expanded = expanded_[side];
    if (expanded.image == nullptr) {
      allocate(expanded, width_, height_, static_cast<std::size_t>(width_) * 4U,
               VPI_IMAGE_FORMAT_RGBA8, VPI_BACKEND_CUDA);
    }
    // Queued straight on the CUDA stream: VPI has nothing queued yet.
    cuda_check(expand_gray(image.data, image.pitch, width_, height_, expanded.data, expanded.pitch,
                           cuda_stream_),
               "expand_gray");
    return expanded.image;
  }
  const VPIImageFormat format = vpi_format(image.format);
  const VPIImageData data = cuda_plane(image.data, image.pitch, width_, height_, format);
  VPIImage& wrapper = input_[side];
  if (wrapper != nullptr && input_format_[side] == image.format) {
    vpi_check(vpiImageSetWrapper(wrapper, &data), "vpiImageSetWrapper(input)");
    return wrapper;
  }
  vpiImageDestroy(wrapper);
  wrapper = nullptr;
  vpi_check(vpiImageCreateWrapper(&data, nullptr, VPI_BACKEND_CUDA, &wrapper),
            "vpiImageCreateWrapper(input)");
  input_format_[side] = image.format;
  return wrapper;
}

void Matcher::submit(const std::array<VPIImage, 2>& sources, const std::array<bool, 2>& convert) {
  VPIStereoDisparityEstimatorParams params{};
  vpi_check(vpiInitStereoDisparityEstimatorParams(&params),
            "vpiInitStereoDisparityEstimatorParams");
  params.windowSize = params_.window;
  params.maxDisparity = 0;  // The payload's.
  params.confidenceThreshold = params_.confidence_threshold;
  params.confidenceType = vpi_confidence_type(params_.confidence_type);
  params.p1 = params_.p1;
  params.p2 = params_.p2;
  params.p2Alpha = params_.p2_alpha;
  params.uniqueness = params_.uniqueness;
  params.numPasses = static_cast<std::int8_t>(num_passes_);

  const bool cuda = backend_ == Backend::kCuda;
  std::array<VPIImage, 2> pair = sources;
  for (int side = 0; side < 2; ++side) {
    if (convert[side]) {
      vpi_check(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA, sources[side], gray_[side],
                                            nullptr),
                cuda ? "vpiSubmitConvertImageFormat(to gray)"
                     : "vpiSubmitConvertImageFormat(to block-linear gray)");
      pair[side] = gray_[side];
    }
  }
  if (cuda) {
    vpi_check(
        vpiSubmitStereoDisparityEstimator(stream_, VPI_BACKEND_CUDA, payload_, pair[0], pair[1],
                                          disparity_.image, confidence_.image, &params),
        "vpiSubmitStereoDisparityEstimator");
    return;
  }
  vpi_check(
      vpiSubmitStereoDisparityEstimator(stream_, vpi_backend(backend_), payload_, pair[0], pair[1],
                                        accelerator_disparity_, accelerator_confidence_, &params),
      "vpiSubmitStereoDisparityEstimator");
  if (backend_ == Backend::kPva) {
    // On its own stream, VPI moves the PVA's result to CUDA itself.
    vpi_check(
        vpiSubmitCopyImage(stream_, VPI_BACKEND_CUDA, accelerator_disparity_, disparity_.image),
        "vpiSubmitCopyImage(disparity)");
    if (accelerator_confidence_ != nullptr) {
      vpi_check(
          vpiSubmitCopyImage(stream_, VPI_BACKEND_CUDA, accelerator_confidence_, confidence_.image),
          "vpiSubmitCopyImage(confidence)");
    }
  }
}

void Matcher::release() noexcept {
  if (stream_ != nullptr) {
    vpiStreamSync(stream_);
    vpiStreamDestroy(stream_);
    stream_ = nullptr;
  }
  vpiPayloadDestroy(payload_);
  payload_ = nullptr;
  for (VPIImage* image : {&input_[0], &input_[1], &gray_[0], &gray_[1], &accelerator_disparity_,
                          &accelerator_confidence_, &expanded_[0].image, &expanded_[1].image,
                          &disparity_.image, &confidence_.image}) {
    vpiImageDestroy(*image);
    *image = nullptr;
  }
}

}  // namespace detail

PlatformFacts probe_platform() noexcept {
  VPIContext context = nullptr;
  std::uint64_t flags = 0;
  if (vpiContextGetCurrent(&context) != VPI_SUCCESS ||
      vpiContextGetFlags(context, &flags) != VPI_SUCCESS) {
    return {};
  }
  const bool has_ofa = (flags & VPI_BACKEND_OFA) != 0U;
  return PlatformFacts{.has_ofa = has_ofa,
                       .has_vic = (flags & VPI_BACKEND_VIC) != 0U,
                       .has_pva = (flags & VPI_BACKEND_PVA) != 0U,
                       .thor = has_ofa && detail::thor_gpu()};
}

std::vector<Backend> available_backends() { return available_backends(probe_platform()); }

}  // namespace holoscan::stereo_disparity
