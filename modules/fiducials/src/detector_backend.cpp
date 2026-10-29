// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/fiducials/detector_backend.hpp"

#include <memory>
#include <stdexcept>

#include "backends.hpp"
#include "cpu_detector.hpp"

namespace holoscan::fiducials {
namespace {

// The cpu backend: CpuDetector, which needs no stream.
class CpuBackend final : public DetectorBackend {
 public:
  CpuBackend(const DetectorOptions& options, int width, int height)
      : detector_(options, width, height) {}

  [[nodiscard]] std::string_view name() const noexcept override { return "cpu"; }
  [[nodiscard]] MemoryLocation input_location() const noexcept override {
    return MemoryLocation::kHost;
  }
  FrameMarkers detect(const GrayImageView& image, cudaStream_t) override {
    return detector_.detect(image);
  }
  std::vector<std::optional<Point2>> refine_saddles(const GrayImageView& image,
                                                    std::span<const SaddleSeed> seeds,
                                                    cudaStream_t) override {
    return detector_.refine_saddles(image, seeds);
  }

 private:
  CpuDetector detector_;
};

}  // namespace

std::unique_ptr<DetectorBackend> create_cpu_backend(const DetectorOptions& options, int width,
                                                    int height) {
  return std::make_unique<CpuBackend>(options, width, height);
}

PlatformFacts probe_platform() noexcept {
  int devices = 0;
  return PlatformFacts{.cuda_device = cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0};
}

Backend resolve_backend(Backend backend) noexcept {
  return backend == Backend::kAuto ? resolve_auto(probe_platform()) : backend;
}

std::unique_ptr<DetectorBackend> create_backend(Backend backend, const DetectorOptions& options,
                                                int width, int height) {
  switch (resolve_backend(backend)) {
    case Backend::kCpu:
      return create_cpu_backend(options, width, height);
    case Backend::kCuda:
      return create_cuda_backend(options, width, height);
    case Backend::kAuto:
      break;  // resolve_backend() never returns it.
  }
  throw std::invalid_argument("unknown backend");
}

}  // namespace holoscan::fiducials
