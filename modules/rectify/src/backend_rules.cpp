// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The backends' names and rules, without CUDA or VPI, so that the SDK-free
// tests check them too.

#include <array>
#include <string>
#include <utility>

#include "holoscan/rectify/backend.hpp"

namespace holoscan::rectify {
namespace {

constexpr std::array<std::pair<Backend, std::string_view>, 5> kBackendNames{{
    {Backend::kCuda, "cuda"},
    {Backend::kVpiCuda, "vpi-cuda"},
    {Backend::kVpiVic, "vpi-vic"},
    {Backend::kVpiPva, "vpi-pva"},
    {Backend::kAuto, "auto"},
}};

// VPI's warp grid needs regions of at least this size.
constexpr int kVpiMinWidth = 64;
constexpr int kVpiMinHeight = 16;
constexpr int kVicMaxSide = 16384;
constexpr int kPvaWidthMultiple = 256;
constexpr int kPvaMinHeight = 64;

}  // namespace

std::string_view backend_name(Backend backend) noexcept {
  for (const auto& [kind, name] : kBackendNames) {
    if (kind == backend) return name;
  }
  return "unknown";
}

std::optional<Backend> parse_backend(std::string_view name) noexcept {
  for (const auto& [kind, backend_name] : kBackendNames) {
    if (backend_name == name) return kind;
  }
  return std::nullopt;
}

std::string_view interpolation_name(Interpolation interpolation) noexcept {
  return interpolation == Interpolation::kNearest ? "nearest" : "linear";
}

std::optional<Interpolation> parse_interpolation(std::string_view name) noexcept {
  if (name == "linear") return Interpolation::kLinear;
  if (name == "nearest") return Interpolation::kNearest;
  return std::nullopt;
}

std::string_view pixel_format_name(PixelFormat format) noexcept {
  switch (format) {
    case PixelFormat::kMono8:
      return "MONO8";
    case PixelFormat::kMono16:
      return "MONO16";
    case PixelFormat::kRgb8:
      return "RGB8";
    case PixelFormat::kBgr8:
      return "BGR8";
    case PixelFormat::kRgba8:
      return "RGBA8";
    case PixelFormat::kBgra8:
      return "BGRA8";
  }
  return "unknown";
}

int channels(PixelFormat format) noexcept {
  switch (format) {
    case PixelFormat::kMono8:
    case PixelFormat::kMono16:
      return 1;
    case PixelFormat::kRgb8:
    case PixelFormat::kBgr8:
      return 3;
    case PixelFormat::kRgba8:
    case PixelFormat::kBgra8:
      return 4;
  }
  return 0;
}

int bytes_per_pixel(PixelFormat format) noexcept {
  return format == PixelFormat::kMono16 ? 2 : channels(format);
}

std::optional<std::string> backend_limitation(Backend backend, const PlatformFacts& facts,
                                              PixelFormat format, Interpolation interpolation,
                                              int width, int height) {
  const std::string size = std::to_string(width) + "x" + std::to_string(height);
  if (format == PixelFormat::kMono16 && interpolation != Interpolation::kNearest) {
    return std::string(
        "MONO16 images (depth) take nearest interpolation only: blending would mix depths across "
        "edges and with holes");
  }
  switch (backend) {
    case Backend::kAuto:
      return std::nullopt;
    case Backend::kCuda:
      if (!facts.cuda_device) return std::string("there is no usable CUDA device");
      return std::nullopt;
    case Backend::kVpiCuda:
    case Backend::kVpiVic:
    case Backend::kVpiPva:
      break;
  }
  const std::string name(backend_name(backend));
  if (!facts.vpi_compiled) {
    return name + ": this build has no VPI; install VPI 4 and rebuild";
  }
  if (width < kVpiMinWidth || height < kVpiMinHeight) {
    return name + ": VPI remaps images of at least 64x16, not " + size;
  }
  if (backend == Backend::kVpiCuda) {
    if (!facts.cuda_device) return name + ": there is no usable CUDA device";
    return std::nullopt;
  }
  if (backend == Backend::kVpiVic) {
    if (!facts.has_vic) return name + ": VPI reports no VIC on this platform";
    if (format != PixelFormat::kMono8 && format != PixelFormat::kMono16) {
      return name + ": the VIC remaps MONO8 and MONO16 only, not " +
             std::string(pixel_format_name(format));
    }
    if (width % 2 != 0 || height % 2 != 0 || width > kVicMaxSide || height > kVicMaxSide) {
      return name + ": the VIC needs an even width and height up to 16384, not " + size;
    }
    return std::nullopt;
  }
  if (!facts.has_pva) return name + ": VPI reports no PVA on this platform";
  if (format != PixelFormat::kMono8) {
    return name + ": the PVA remaps MONO8 only, not " + std::string(pixel_format_name(format));
  }
  if (width % kPvaWidthMultiple != 0 || height % 2 != 0 || height < kPvaMinHeight) {
    return name +
           ": the PVA needs a width that is a multiple of 256 and an even height of at "
           "least 64, not " +
           size;
  }
  return std::nullopt;
}

Backend resolve_auto(const PlatformFacts& facts, PixelFormat format, Interpolation interpolation,
                     int width, int height) {
  if (!backend_limitation(Backend::kVpiVic, facts, format, interpolation, width, height)) {
    return Backend::kVpiVic;
  }
  return Backend::kCuda;
}

}  // namespace holoscan::rectify
