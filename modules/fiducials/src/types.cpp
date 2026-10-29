// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/fiducials/types.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <holoscan/fiducials/dictionary.hpp>

namespace holoscan::fiducials {
namespace {

constexpr std::array<std::pair<Backend, std::string_view>, 3> kBackendNames{{
    {Backend::kCpu, "cpu"},
    {Backend::kCuda, "cuda"},
    {Backend::kAuto, "auto"},
}};

}  // namespace

std::string_view backend_name(Backend backend) noexcept {
  for (const auto& [kind, name] : kBackendNames) {
    if (kind == backend) return name;
  }
  return "unknown";
}

std::optional<Backend> parse_backend(std::string_view name) noexcept {
  for (const auto& [kind, known] : kBackendNames) {
    if (known == name) return kind;
  }
  return std::nullopt;
}

std::vector<Backend> available_backends() { return {Backend::kCpu, Backend::kCuda}; }

void validate_options(const DetectorOptions& options) {
  (void)Dictionary(options.dictionary);  // Throws for an unknown name.
  if (!(options.error_correction_rate >= 0.0 && options.error_correction_rate <= 1.0)) {
    throw std::invalid_argument("error_correction_rate must be in 0..1");
  }
  if (options.tile_size < 2 || options.tile_size > 64) {
    throw std::invalid_argument("tile_size must be in 2..64");
  }
  if (options.min_contrast < 1 || options.min_contrast > 255) {
    throw std::invalid_argument("min_contrast must be in 1..255");
  }
  if (options.min_marker_px < 4 || options.min_marker_px > 1000) {
    throw std::invalid_argument("min_marker_px must be in 4..1000");
  }
}

}  // namespace holoscan::fiducials
