// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "vpi_matcher.hpp"

// VPI's C API.
extern "C" {
#include <vpi/Context.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
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

}  // namespace

void vpi_check(VPIStatus status, const char* what) {
  if (status != VPI_SUCCESS) {
    char detail[VPI_MAX_STATUS_MESSAGE_LENGTH] = {};
    vpiGetLastStatusMessage(detail, sizeof(detail));
    throw std::runtime_error(std::string(what) + ": " + vpiStatusGetName(status) + ": " + detail);
  }
}

LockedImage::LockedImage(VPIImage image, VPILockMode mode) : image_(image) {
  VPIImageData data{};
  vpi_check(vpiImageLockData(image, mode, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &data),
            "vpiImageLockData");
  plane_ = data.buffer.pitch.planes[0];
}

LockedImage::~LockedImage() { vpiImageUnlock(image_); }

Matcher::Matcher(Backend backend, const DisparityParams& params, bool confidence, int width,
                 int height)
    : backend_(backend),
      params_(params),
      width_(width),
      height_(height),
      searched_(searched_disparity(backend, params)) {
  const std::uint64_t algorithm = vpi_backend(backend);
  if (auto problem = params_problem(backend, params, confidence)) {
    throw std::invalid_argument(*problem);
  }
  if (auto problem = size_problem(backend, params, width, height)) {
    throw std::invalid_argument(*problem);
  }
  try {
    const bool cuda = backend == Backend::kCuda;
    const bool pva = backend == Backend::kPva;
    // VIC converts to and from block-linear for ofa and pva.
    const std::uint64_t shared = VPI_BACKEND_CUDA | (cuda ? 0U : VPI_BACKEND_VIC);
    vpi_check(vpiStreamCreate(algorithm | shared, &stream_), "vpiStreamCreate");
    for (VPIImage& image : gray_) {
      vpi_check(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_Y8_ER, shared, &image),
                "vpiImageCreate(gray)");
    }
    // The PVA writes the pitch-linear disparity and confidence itself.
    const std::uint64_t output = shared | (pva ? algorithm : 0U);
    vpi_check(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_S16, output, &disparity_),
              "vpiImageCreate(disparity)");
    if (confidence) {
      vpi_check(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U16, output, &confidence_),
                "vpiImageCreate(confidence)");
    }
    VPIStereoDisparityEstimatorCreationParams creation{};
    vpi_check(vpiInitStereoDisparityEstimatorCreationParams(&creation),
              "vpiInitStereoDisparityEstimatorCreationParams");
    creation.maxDisparity = params.max_disparity;
    creation.includeDiagonals = params.include_diagonals ? 1 : 0;
    VPIImageFormat input = VPI_IMAGE_FORMAT_Y8_ER;
    if (!cuda) {
      input = VPI_IMAGE_FORMAT_Y8_ER_BL;
      for (VPIImage& image : block_linear_) {
        vpi_check(vpiImageCreate(width, height, input, algorithm | VPI_BACKEND_VIC, &image),
                  "vpiImageCreate(block-linear gray)");
      }
      if (!pva) {
        vpi_check(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_S16_BL,
                                 algorithm | VPI_BACKEND_VIC, &block_linear_disparity_),
                  "vpiImageCreate(block-linear disparity)");
      }
    }
    vpi_check(
        vpiCreateStereoDisparityEstimator(algorithm, width, height, input, &creation, &payload_),
        "vpiCreateStereoDisparityEstimator");
  } catch (...) {
    release();
    throw;
  }
}

Matcher::~Matcher() { release(); }

void Matcher::match() {
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
  params.numPasses = static_cast<std::int8_t>(params_.num_passes);
  if (backend_ == Backend::kCuda) {
    vpi_check(vpiSubmitStereoDisparityEstimator(stream_, VPI_BACKEND_CUDA, payload_, gray_[0],
                                                gray_[1], disparity_, confidence_, &params),
              "vpiSubmitStereoDisparityEstimator(cuda)");
  } else {
    for (int side = 0; side < 2; ++side) {
      vpi_check(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_VIC, gray_[side],
                                            block_linear_[side], nullptr),
                "vpiSubmitConvertImageFormat(to block-linear)");
    }
    const bool pva = backend_ == Backend::kPva;
    vpi_check(
        vpiSubmitStereoDisparityEstimator(
            stream_, vpi_backend(backend_), payload_, block_linear_[0], block_linear_[1],
            pva ? disparity_ : block_linear_disparity_, confidence_, &params),
        pva ? "vpiSubmitStereoDisparityEstimator(pva)" : "vpiSubmitStereoDisparityEstimator(ofa)");
    if (!pva) {
      vpi_check(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_VIC, block_linear_disparity_,
                                            disparity_, nullptr),
                "vpiSubmitConvertImageFormat(to pitch-linear)");
    }
  }
  vpi_check(vpiStreamSync(stream_), "vpiStreamSync");
}

void Matcher::release() noexcept {
  if (stream_ != nullptr) {
    vpiStreamSync(stream_);
    vpiStreamDestroy(stream_);
    stream_ = nullptr;
  }
  vpiPayloadDestroy(payload_);
  payload_ = nullptr;
  for (VPIImage* image : {&gray_[0], &gray_[1], &block_linear_[0], &block_linear_[1],
                          &block_linear_disparity_, &disparity_, &confidence_}) {
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
  return PlatformFacts{.has_ofa = (flags & VPI_BACKEND_OFA) != 0U,
                       .has_vic = (flags & VPI_BACKEND_VIC) != 0U,
                       .has_pva = (flags & VPI_BACKEND_PVA) != 0U};
}

std::vector<Backend> available_backends() { return available_backends(probe_platform()); }

}  // namespace holoscan::stereo_disparity
