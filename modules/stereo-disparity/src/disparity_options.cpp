// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/stereo_disparity/disparity_options.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <utility>
#include <vector>

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
constexpr std::array<int, 4> kDownscales{1, 2, 4, 8};
// VPI 4.1.4 on the PVA pads the right edge of a full-size frame with
// max_disparity 256 and a confidence map by 384 - width % 256 pixels, and
// takes at most 255 (measured on Jetson Orin over widths 256..2048).
constexpr int kPvaPaddingPeriod = 256;
constexpr int kPvaPaddingMinRemainder = 128;

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

bool include_diagonals_for(Backend backend, const DisparityParams& params) noexcept {
  return params.include_diagonals.value_or(backend != Backend::kCuda);
}

int num_passes_for(const DisparityParams& params, const PlatformFacts& facts) noexcept {
  return params.num_passes.value_or(facts.thor ? 3 : 1);
}

int output_side(int input_side, int downscale) noexcept {
  return downscale < 1 ? input_side : (input_side + downscale - 1) / downscale;
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
  if (params.num_passes && (*params.num_passes < 1 || *params.num_passes > 3)) {
    return "num_passes must be 1..3";
  }
  if (!one_of(params.downscale, kDownscales)) {
    return "downscale must be 1, 2, 4 or 8";
  }
  if (params.downscale != 1 && !ofa && !pva) {
    return "downscale needs the ofa or pva backend" +
           std::string(backend == Backend::kAuto ? ", not auto, which may choose cuda" : "");
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

std::optional<std::string> size_problem(Backend backend, const DisparityParams& params,
                                        bool confidence, int width, int height) {
  const std::string size = std::to_string(width) + "x" + std::to_string(height);
  if (width < 1 || height < 1 || width > kMaxSide || height > kMaxSide) {
    return "image " + size + " is outside 1..16384 pixels a side";
  }
  if (backend != Backend::kOfa && backend != Backend::kPva) {
    return std::nullopt;
  }
  const int downscale = params.downscale;
  if (output_side(width, downscale) < kMinOfaSide || output_side(height, downscale) < kMinOfaSide) {
    return "image " + size + " is too small for " + std::string(backend_name(backend)) +
           ", which takes sides of at least 16 once downscaled by " + std::to_string(downscale);
  }
  if (backend != Backend::kPva) {
    return std::nullopt;
  }
  const int min_width = std::max(64, params.max_disparity / downscale) * downscale;
  if (width < min_width) {
    return "image " + size + " is too narrow for pva, which needs a width of at least " +
           std::to_string(min_width);
  }
  const int remainder = width % kPvaPaddingPeriod;
  if (confidence && params.max_disparity == 256 && downscale == 1 && remainder != 0 &&
      remainder < kPvaPaddingMinRemainder) {
    return "image " + size +
           " does not suit pva with max_disparity 256, downscale 1 and a confidence map, "
           "where VPI 4.1.4 needs a width whose remainder modulo 256 is 0 or at least 128 (it "
           "pads the right edge by 384 - width % 256 pixels, and the PVA takes at most 255); "
           "use max_disparity 128, downscale 2 or more, no confidence map, or another width";
  }
  return std::nullopt;
}

std::optional<std::string> platform_problem(Backend backend, const DisparityParams& params,
                                            const PlatformFacts& facts) {
  if (!facts.thor || (backend != Backend::kOfa && backend != Backend::kPva)) {
    return std::nullopt;
  }
  const std::string name(backend_name(backend));
  if (params.num_passes && *params.num_passes < 3) {
    return "num_passes " + std::to_string(*params.num_passes) + " on " + name +
           " takes over a second per frame on Jetson Thor with VPI 4.1.4; use 3";
  }
  if (params.downscale == 8) {
    return "VPI 4.1.4 on Jetson Thor refuses downscale 8 on " + name + "; use 1, 2 or 4";
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
  if (auto problem = platform_problem(Backend::kOfa, params, facts)) {
    return {Backend::kCuda, "VPI reports an OFA, but " + *problem};
  }
  if (auto problem = size_problem(Backend::kOfa, params, confidence, width, height)) {
    return {Backend::kCuda, "VPI reports an OFA, but the " + *problem};
  }
  return {Backend::kOfa, "VPI reports an OFA, and the parameters and image size suit it"};
}

}  // namespace holoscan::stereo_disparity
