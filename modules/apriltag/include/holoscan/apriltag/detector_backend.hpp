// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The interface every detector backend implements, and its factory. Backends
// find tags and corners; pose.hpp estimates poses the same way for all of them.

#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <string_view>
#include <vector>

#include <holoscan/apriltag/detection.hpp>

namespace holoscan::apriltag {

class DetectorBackend {
 public:
  virtual ~DetectorBackend() = default;
  DetectorBackend(const DetectorBackend&) = delete;
  DetectorBackend& operator=(const DetectorBackend&) = delete;

  // "cpu", "cuda", or the hardware VPI uses: "vpi-pva" or "vpi-cpu".
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
  // Where `detect` wants its image; the caller converts and copies to match.
  [[nodiscard]] virtual MemoryLocation input_location() const noexcept = 0;
  // Finds tags in `image`, which must be the size the backend was built for and
  // in input_location(). Device work is ordered on `stream`; results are on the
  // host when this returns. Poses are left empty. Reports at most
  // kMaxDetectionsPerFrame tags, and counts what did not fit.
  // @throws std::invalid_argument for an image of another size or location.
  // @throws std::runtime_error on a backend failure.
  [[nodiscard]] FrameDetections detect(const GrayImageView& image, cudaStream_t stream);

 protected:
  DetectorBackend(int width, int height) : width_(width), height_(height) {}
  // detect() once the image is checked. May return more than
  // kMaxDetectionsPerFrame tags; detect() keeps the first ones and counts the rest.
  [[nodiscard]] virtual FrameDetections find(const GrayImageView& image, cudaStream_t stream) = 0;

  int width_;
  int height_;
};

// A detection from coordinates with pixel centers at +0.5 (AprilTag 3, VPI and
// the CUDA decoder), moved to integer pixel centers.
[[nodiscard]] inline Detection make_detection(int id, int hamming, float margin,
                                              const double (&center)[2],
                                              const double (&corners)[4][2]) {
  Detection detection{.id = id, .hamming = hamming, .decision_margin = margin};
  detection.center = {center[0] - 0.5, center[1] - 0.5};
  for (int i = 0; i < 4; ++i) {
    detection.corners[i] = {corners[i][0] - 0.5, corners[i][1] - 0.5};
  }
  return detection;
}

// Checks that `backend` is compiled into this build, without building it.
// kAuto always is.
// @throws std::invalid_argument naming what is missing.
void check_available(Backend backend);

// The facts about this process's platform that resolve_auto() needs. A fact that
// cannot be established (VPI or CUDA fails to answer) is false.
[[nodiscard]] PlatformFacts probe_platform() noexcept;
// `backend`, or for kAuto, resolve_auto() of this platform's facts.
[[nodiscard]] Backend resolve_backend(Backend backend) noexcept;

// Builds `backend` for `width` x `height` gray images; kAuto builds resolve_backend()'s.
// @throws std::invalid_argument if the options or size are not usable with this
//         backend, or the backend is not compiled into this build.
// @throws std::runtime_error if the backend fails to initialize.
[[nodiscard]] std::unique_ptr<DetectorBackend> create_backend(Backend backend,
                                                              const DetectorOptions& options,
                                                              int width, int height);

// Whether VPI reports a PVA on this platform; false in a build without VPI.
// @throws std::runtime_error if VPI fails to answer.
[[nodiscard]] bool vpi_platform_has_pva();

// Each backend's factory, in its own source file; create_backend validates
// their arguments first. VPI's throws in a build without VPI.
[[nodiscard]] std::unique_ptr<DetectorBackend> create_cpu_backend(const DetectorOptions& options,
                                                                  int width, int height);
[[nodiscard]] std::unique_ptr<DetectorBackend> create_cuda_backend(const DetectorOptions& options,
                                                                   int width, int height);
[[nodiscard]] std::unique_ptr<DetectorBackend> create_vpi_backend(const DetectorOptions& options,
                                                                  int width, int height);

}  // namespace holoscan::apriltag
