// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/denoise/denoise_params.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef HOLOSCAN_DENOISE_WITH_VPI
#define HOLOSCAN_DENOISE_WITH_VPI 0
#endif

namespace holoscan::denoise {
namespace {

template <typename Enum, std::size_t N>
std::string_view name_of(const std::array<std::pair<Enum, std::string_view>, N>& names,
                         Enum value) noexcept {
  for (const auto& [kind, name] : names) {
    if (kind == value) return name;
  }
  return "unknown";
}

template <typename Enum, std::size_t N>
std::optional<Enum> parse(const std::array<std::pair<Enum, std::string_view>, N>& names,
                          std::string_view text) noexcept {
  for (const auto& [kind, name] : names) {
    if (name == text) return kind;
  }
  return std::nullopt;
}

constexpr std::array<std::pair<Backend, std::string_view>, 3> kBackends{{
    {Backend::kCuda, "cuda"},
    {Backend::kVpi, "vpi"},
    {Backend::kAuto, "auto"},
}};

constexpr std::array<std::pair<SpatialFilter, std::string_view>, 3> kSpatial{{
    {SpatialFilter::kNone, "none"},
    {SpatialFilter::kBilateral, "bilateral"},
    {SpatialFilter::kNlm, "nlm"},
}};

constexpr std::array<std::pair<VpiEngine, std::string_view>, 3> kEngines{{
    {VpiEngine::kAuto, "auto"},
    {VpiEngine::kVic, "vic"},
    {VpiEngine::kCuda, "cuda"},
}};

constexpr std::array<std::pair<PixelFormat, std::string_view>, 4> kFormats{{
    {PixelFormat::kMono8, "MONO8"},
    {PixelFormat::kRgb8, "RGB8"},
    {PixelFormat::kRgba8, "RGBA8"},
    {PixelFormat::kNv12, "NV12"},
}};

}  // namespace

std::string_view backend_name(Backend backend) noexcept { return name_of(kBackends, backend); }
std::optional<Backend> parse_backend(std::string_view name) noexcept {
  return parse(kBackends, name);
}

std::vector<Backend> available_backends() {
  std::vector<Backend> backends{Backend::kCuda};
  if (HOLOSCAN_DENOISE_WITH_VPI) backends.push_back(Backend::kVpi);
  return backends;
}

std::string_view spatial_name(SpatialFilter filter) noexcept { return name_of(kSpatial, filter); }
std::optional<SpatialFilter> parse_spatial(std::string_view name) noexcept {
  return parse(kSpatial, name);
}

std::string_view vpi_engine_name(VpiEngine engine) noexcept { return name_of(kEngines, engine); }
std::optional<VpiEngine> parse_vpi_engine(std::string_view name) noexcept {
  return parse(kEngines, name);
}

std::string_view format_name(PixelFormat format) noexcept { return name_of(kFormats, format); }

void validate_params(const DenoiseParams& params) {
  if (params.spatial != SpatialFilter::kNone && params.spatial != SpatialFilter::kBilateral &&
      params.spatial != SpatialFilter::kNlm) {
    throw std::invalid_argument("spatial must be none, bilateral or nlm");
  }
  if (!params.temporal && params.spatial == SpatialFilter::kNone) {
    throw std::invalid_argument("enable the temporal filter or a spatial filter");
  }
  if (!(params.strength >= 0.0F && params.strength <= 1.0F)) {
    throw std::invalid_argument("strength must be in 0..1");
  }
  if (!(params.noise_sigma >= 0.0F && params.noise_sigma <= 100.0F)) {
    throw std::invalid_argument("noise_sigma must be 0 (estimate) or up to 100 code values");
  }
  if (!(params.motion_threshold >= 1.0F && params.motion_threshold <= 20.0F)) {
    throw std::invalid_argument("motion_threshold must be in 1..20");
  }
  if (params.max_history < 2 || params.max_history > 255) {
    throw std::invalid_argument("max_history must be in 2..255");
  }
  if (!(params.scene_cut_fraction > 0.0F && params.scene_cut_fraction <= 1.0F)) {
    throw std::invalid_argument("scene_cut_fraction must be in (0, 1]");
  }
}

std::string_view unsupported_reason(Backend backend, const DenoiseParams& params) noexcept {
  if (backend != Backend::kVpi) return {};
  if (!params.temporal) {
    return "the vpi backend is VPI's temporal noise reduction; enable temporal";
  }
  if (params.spatial != SpatialFilter::kNone) {
    return "the vpi backend filters spatially itself; use spatial none, or the cuda backend";
  }
  return {};
}

}  // namespace holoscan::denoise
