// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/apriltag/detector_backend.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tag_family.hpp"

namespace holoscan::apriltag {
namespace {

constexpr std::array<std::pair<Backend, std::string_view>, 3> kBackendNames{{
    {Backend::kCpu, "cpu"},
    {Backend::kCuda, "cuda"},
    {Backend::kVpi, "vpi"},
}};

// Decode-table entries allowed: AprilTag advises two-bit correction only for small families.
constexpr std::uint64_t kMaxDecodeEntries = 8'000'000;

constexpr std::string_view kNoVpi = "backend vpi: this build has no VPI; install VPI 4 and rebuild";

}  // namespace

std::string_view backend_name(Backend backend) noexcept {
  for (const auto& [kind, name] : kBackendNames) {
    if (kind == backend) {
      return name;
    }
  }
  return "unknown";
}

std::optional<Backend> parse_backend(std::string_view name) noexcept {
  for (const auto& [kind, backend_name] : kBackendNames) {
    if (backend_name == name) {
      return kind;
    }
  }
  return std::nullopt;
}

std::vector<Backend> available_backends() {
  std::vector<Backend> backends{Backend::kCpu, Backend::kCuda};
#if HOLOSCAN_APRILTAG_WITH_VPI
  backends.push_back(Backend::kVpi);
#endif
  return backends;
}

void validate_options(const DetectorOptions& options) {
  if (!is_known_family(options.family)) {
    throw std::invalid_argument("family: unknown AprilTag family '" + options.family + "'");
  }
  if (options.max_hamming < 0 || options.max_hamming > 2) {
    throw std::invalid_argument("max_hamming must be in 0..2");
  }
  if (options.max_hamming == 2 && decode_table_entries(options.family, 2) > kMaxDecodeEntries) {
    throw std::invalid_argument("max_hamming 2 with " + options.family +
                                " needs decode tables of several gigabytes; use 0 or 1");
  }
  if (options.decimate < 1 || options.decimate > 8) {
    throw std::invalid_argument("decimate must be in 1..8");
  }
  if (!(options.decode_sharpening >= 0.0 && options.decode_sharpening <= 1.0)) {
    throw std::invalid_argument("decode_sharpening must be in 0..1");
  }
  if (options.min_white_black_diff < 1 || options.min_white_black_diff > 255) {
    throw std::invalid_argument("min_white_black_diff must be in 1..255");
  }
}

void check_available(Backend backend) {
  if (backend == Backend::kVpi && !HOLOSCAN_APRILTAG_WITH_VPI) {
    throw std::invalid_argument(std::string(kNoVpi));
  }
}

#if !HOLOSCAN_APRILTAG_WITH_VPI
std::unique_ptr<DetectorBackend> create_vpi_backend(const DetectorOptions&, int, int) {
  throw std::invalid_argument(std::string(kNoVpi));
}
#endif

FrameDetections DetectorBackend::detect(const GrayImageView& image, cudaStream_t stream) {
  if (image.location != input_location() || image.width != width_ || image.height != height_) {
    throw std::invalid_argument(std::string(name()) +
                                " backend: the image is not of the configured size and location");
  }
  FrameDetections result = find(image, stream);
  if (result.tags.size() > kMaxDetectionsPerFrame) {
    result.overflow.detections +=
        static_cast<std::uint32_t>(result.tags.size() - kMaxDetectionsPerFrame);
    result.tags.resize(kMaxDetectionsPerFrame);
  }
  return result;
}

std::unique_ptr<DetectorBackend> create_backend(Backend backend, const DetectorOptions& options,
                                                int width, int height) {
  validate_options(options);
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("image size must be positive");
  }
  switch (backend) {
    case Backend::kCpu:
      return create_cpu_backend(options, width, height);
    case Backend::kCuda:
      return create_cuda_backend(options, width, height);
    case Backend::kVpi:
      return create_vpi_backend(options, width, height);
  }
  throw std::invalid_argument("unknown backend");
}

}  // namespace holoscan::apriltag
