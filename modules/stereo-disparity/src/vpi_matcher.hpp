// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// VPI 4's stereo disparity estimator for one backend and image size, and small
// helpers for VPI's C API.

#pragma once

#include <cuda_runtime_api.h>

// VPI's C API.
extern "C" {
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Types.h>
}

#include <array>
#include <cstddef>

#include "holoscan/stereo_disparity/disparity_options.hpp"

namespace holoscan::stereo_disparity::detail {

// @throws std::runtime_error naming `what` and VPI's message unless `status`
//         is VPI_SUCCESS.
void vpi_check(VPIStatus status, const char* what);

// A VPI image's first plane, locked as CUDA pitch-linear memory until
// destroyed.
class LockedImage {
 public:
  // @throws std::runtime_error if VPI cannot lock the image.
  LockedImage(VPIImage image, VPILockMode mode);
  ~LockedImage();
  LockedImage(const LockedImage&) = delete;
  LockedImage& operator=(const LockedImage&) = delete;

  template <typename T>
  [[nodiscard]] T* data() const noexcept {
    return reinterpret_cast<T*>(plane_.pBase + plane_.offsetBytes);
  }
  [[nodiscard]] std::size_t pitch() const noexcept {
    return static_cast<std::size_t>(plane_.pitchBytes);
  }

 private:
  VPIImage image_;
  VPIImagePlanePitchLinear plane_{};
};

// The estimator and its images for one backend (cuda, ofa or pva) and size.
// The caller writes the gray pair into gray(0) and gray(1) through CUDA,
// calls match(), and reads disparity() (S16, Q10.5) and, with a confidence
// map, confidence() (U16) through CUDA.
//
// cuda matches the gray pair as it is. ofa and pva need block-linear input:
// VIC copies the pair into block-linear images first. The OFA writes a
// block-linear disparity, which VIC copies back to pitch-linear; the PVA
// writes pitch-linear itself.
class Matcher {
 public:
  // @throws std::invalid_argument for kAuto, or parameters or a size the
  //         backend does not take.
  // @throws std::runtime_error if VPI fails to create what it needs (for
  //         instance, the backend's hardware is missing).
  Matcher(Backend backend, const DisparityParams& params, bool confidence, int width, int height);
  ~Matcher();
  Matcher(const Matcher&) = delete;
  Matcher& operator=(const Matcher&) = delete;

  [[nodiscard]] Backend backend() const noexcept { return backend_; }
  [[nodiscard]] int width() const noexcept { return width_; }
  [[nodiscard]] int height() const noexcept { return height_; }
  // The largest disparity the estimator searches.
  [[nodiscard]] int searched() const noexcept { return searched_; }

  // 0 for the left image, 1 for the right: Y8_ER, lockable for CUDA.
  [[nodiscard]] VPIImage gray(int side) const noexcept { return gray_[side]; }
  // S16 (Q10.5), and U16 (null without a confidence map), lockable for CUDA.
  [[nodiscard]] VPIImage disparity() const noexcept { return disparity_; }
  [[nodiscard]] VPIImage confidence() const noexcept { return confidence_; }

  // Matches the gray pair, and waits for the result. The gray images must be
  // unlocked.
  // @throws std::runtime_error on a VPI failure.
  void match();

 private:
  void release() noexcept;

  Backend backend_;
  DisparityParams params_;
  int width_;
  int height_;
  int searched_;
  VPIStream stream_{};
  VPIPayload payload_{};
  std::array<VPIImage, 2> gray_{};
  std::array<VPIImage, 2> block_linear_{};
  VPIImage block_linear_disparity_{};
  VPIImage disparity_{};
  VPIImage confidence_{};
};

}  // namespace holoscan::stereo_disparity::detail
