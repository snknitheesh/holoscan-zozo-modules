// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/stereo_disparity/disparity_options.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/stereo_disparity/stereo_geometry.hpp"

namespace holoscan::stereo_disparity {
namespace {

constexpr std::array<std::pair<Backend, std::string_view>, 4> kBackendNames{{
    {Backend::kCuda, "cuda"},
    {Backend::kOfa, "ofa"},
    {Backend::kPva, "pva"},
    {Backend::kAuto, "auto"},
}};

// Image sides VPI's stereo estimator takes: 16..16384 on the OFA, by its
// documentation; the same upper bound for cuda.
constexpr int kMinOfaSide = 16;
constexpr int kMaxSide = 16384;
// The PVA's median-filter windows.
constexpr std::array<int, 4> kPvaWindows{1, 3, 5, 7};
constexpr std::array<int, 5> kP2Alphas{0, 1, 2, 4, 8};

template <typename T, std::size_t N>
bool one_of(T value, const std::array<T, N>& values) {
  for (const T& allowed : values) {
    if (value == allowed) return true;
  }
  return false;
}

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

int searched_disparity(Backend backend, const DisparityParams& params) noexcept {
  if (backend == Backend::kOfa || backend == Backend::kPva) {
    return params.max_disparity;
  }
  return params.max_disparity <= 64 ? 64 : params.max_disparity <= 128 ? 128 : 256;
}

std::optional<std::string> params_problem(Backend backend, const DisparityParams& params,
                                          bool confidence) {
  const bool ofa = backend == Backend::kOfa;
  const bool pva = backend == Backend::kPva;
  const std::string name(backend_name(backend == Backend::kAuto ? Backend::kCuda : backend));
  if (ofa || pva) {
    if (params.max_disparity != 128 && params.max_disparity != 256) {
      return "max_disparity must be 128 or 256 on " + name;
    }
  } else if (params.max_disparity < 1 || params.max_disparity > 256) {
    return "max_disparity must be 1..256 on cuda";
  }
  if (params.confidence_threshold < 0 || params.confidence_threshold > 65535) {
    return "confidence_threshold must be 0..65535";
  }
  if (params.p1 < 1 || params.p2 < params.p1) {
    return "the penalties need 1 <= p1 <= p2";
  }
  const bool inference = params.confidence_type == ConfidenceType::kInference;
  if (ofa || pva) {
    const int limit = pva && inference ? 89 : 217;
    if (params.p1 + params.p2 > limit) {
      return "p1 + p2 must be at most " + std::to_string(limit) + " on " + name;
    }
  } else if (params.p2 > 255) {
    return "p2 must be at most 255 on cuda";
  }
  if (!one_of(params.p2_alpha, kP2Alphas)) {
    return "p2_alpha must be 0, 1, 2, 4 or 8";
  }
  if (!(params.uniqueness == -1.0F || (params.uniqueness >= 0.0F && params.uniqueness <= 1.0F))) {
    return "uniqueness must be 0..1, or -1 to skip the check";
  }
  if (params.num_passes < 1 || params.num_passes > 3) {
    return "num_passes must be 1..3";
  }
  if (pva && !one_of(params.window, kPvaWindows)) {
    return "window must be 1, 3, 5 or 7 on pva";
  }
  if (params.window < 1) {
    return "window must be positive";
  }
  if (inference && !pva) {
    return "the inference confidence runs on pva only";
  }
  if (confidence && ofa) {
    return "the ofa backend has no confidence output; use cuda or pva";
  }
  return std::nullopt;
}

std::optional<std::string> size_problem(Backend backend, const DisparityParams& params, int width,
                                        int height) {
  const std::string size = std::to_string(width) + "x" + std::to_string(height);
  if (width < 1 || height < 1 || width > kMaxSide || height > kMaxSide) {
    return "image " + size + " is outside 1..16384 pixels a side";
  }
  if (backend == Backend::kOfa || backend == Backend::kPva) {
    if (width < kMinOfaSide || height < kMinOfaSide) {
      return "image " + size + " is too small for " + std::string(backend_name(backend)) +
             ", which takes sides of 16..16384";
    }
  }
  if (backend == Backend::kPva) {
    const int min_width = params.max_disparity > 64 ? params.max_disparity : 64;
    if (width < min_width) {
      return "image " + size + " is too narrow for pva, which needs a width of at least " +
             std::to_string(min_width);
    }
  }
  return std::nullopt;
}

std::vector<Backend> available_backends(const PlatformFacts& facts) {
  std::vector<Backend> backends{Backend::kCuda};
  if (facts.has_ofa && facts.has_vic) {
    backends.push_back(Backend::kOfa);
    if (facts.has_pva) {
      backends.push_back(Backend::kPva);
    }
  }
  return backends;
}

AutoChoice resolve_auto(const PlatformFacts& facts, const DisparityParams& params, bool confidence,
                        int width, int height) {
  if (!facts.has_ofa || !facts.has_vic) {
    return {Backend::kCuda, "VPI reports no Optical Flow Accelerator"};
  }
  if (auto problem = params_problem(Backend::kOfa, params, confidence)) {
    return {Backend::kCuda, "VPI reports an OFA, but " + *problem};
  }
  if (auto problem = size_problem(Backend::kOfa, params, width, height)) {
    return {Backend::kCuda, "VPI reports an OFA, but the " + *problem};
  }
  return {Backend::kOfa, "VPI reports an OFA, and the parameters and image size suit it"};
}

void validate_geometry(const StereoGeometry& geometry) {
  if (!(std::isfinite(geometry.fx) && geometry.fx > 0)) {
    throw std::invalid_argument("geometry: fx must be finite and positive");
  }
  if (!(std::isfinite(geometry.baseline) && geometry.baseline > 0)) {
    throw std::invalid_argument("geometry: baseline must be finite and positive, in meters");
  }
  if (geometry.width < 1 || geometry.height < 1 || geometry.width > kMaxSide ||
      geometry.height > kMaxSide) {
    throw std::invalid_argument("geometry: width and height must be 1..16384");
  }
  if ((geometry.cx && !std::isfinite(*geometry.cx)) ||
      (geometry.cy && !std::isfinite(*geometry.cy)) || !std::isfinite(geometry.doffs)) {
    throw std::invalid_argument("geometry: cx, cy and doffs must be finite");
  }
}

}  // namespace holoscan::stereo_disparity
