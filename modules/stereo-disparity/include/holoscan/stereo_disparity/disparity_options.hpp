// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The matcher's backends and parameters, and the `auto` backend's rule. No
// SDK, CUDA or VPI: the rules here are what the operator and the tests share.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace holoscan::stereo_disparity {

// Where VPI 4's stereo disparity estimator runs (vpiSubmitStereoDisparityEstimator).
// VPI 4 has no CPU implementation.
enum class Backend : std::uint8_t {
  kCuda,  // Semi-global matching on the GPU; every platform.
  kOfa,   // VPI's advanced SGM on the Optical Flow Accelerator (Jetson Orin, Thor).
  kPva,   // The same on the OFA, with the PVA and VIC: confidence, median filter.
  kAuto,  // ofa where it suits, else cuda; see resolve_auto().
};

[[nodiscard]] std::string_view backend_name(Backend backend) noexcept;
// Parses "cuda", "ofa", "pva" or "auto"; nullopt for anything else.
[[nodiscard]] std::optional<Backend> parse_backend(std::string_view name) noexcept;

// How VPI computes the confidence map (VPIStereoDisparityConfidenceType).
enum class ConfidenceType : std::uint8_t {
  kAbsolute,   // 1 - |D_lr - D_rl| / max_disparity; cuda and pva.
  kRelative,   // 1 - |D_lr - D_rl| / D_lr; cuda and pva.
  kInference,  // A small network's; pva only.
};

// The matcher's parameters. The defaults are VPI's, but max_disparity, which
// is VPI's 64: 128 covers a RealSense at 1280x720 (fx 337 px, 95 mm) down to
// 0.25 m, and ofa and pva take only 128 or 256. Each field says which backends
// read it; the others ignore it.
struct DisparityParams {
  // Largest disparity searched, in pixels. cuda: 1..256, rounded up to 64, 128
  // or 256 (see searched_disparity()); ofa, pva: 128 or 256.
  int max_disparity{128};
  // Census window on pva only, as a median filter of 1, 3, 5 or 7. cuda uses a
  // fixed 9x7 window, and ofa none.
  int window{5};
  // cuda, pva: VPI's confidence (0..65535) below which a disparity is invalid.
  // 0 keeps every disparity.
  int confidence_threshold{32767};
  // cuda, pva: how the confidence is computed.
  ConfidenceType confidence_type{ConfidenceType::kAbsolute};
  // cuda, ofa: SGM penalties for a disparity change of one pixel (p1) and of
  // more (p2) between neighbors; 1 <= p1 <= p2. cuda: p2 <= 255; ofa, pva:
  // p1 + p2 <= 217 (89 with the kInference confidence).
  int p1{3};
  int p2{48};
  // ofa: adaptive p2, which lowers p2 across intensity edges; 0 (off), 1, 2,
  // 4 or 8.
  int p2_alpha{0};
  // cuda: how far the best match must beat the second best, 0..1, or -1 to
  // skip the check. Lower values invalidate more pixels.
  float uniqueness{-1.0F};
  // ofa, pva: SGM passes, 1..3; fewer is faster.
  int num_passes{3};
  // cuda, ofa: include the diagonal SGM paths; slower, better.
  bool include_diagonals{true};
};

// The largest disparity `backend` searches with `params`: max_disparity,
// rounded up to 64, 128 or 256 on cuda.
[[nodiscard]] int searched_disparity(Backend backend, const DisparityParams& params) noexcept;

// Why `params` cannot run on `backend` (with a confidence output when
// `confidence`), or nullopt when they can. kAuto checks what cuda needs, which
// it always may fall back to.
[[nodiscard]] std::optional<std::string> params_problem(Backend backend,
                                                        const DisparityParams& params,
                                                        bool confidence);

// Why `width` x `height` images cannot run on `backend` with `params`, or
// nullopt. ofa takes sides of 16..16384; pva also needs a width of at least
// max(64, max_disparity); cuda any size up to 16384.
[[nodiscard]] std::optional<std::string> size_problem(Backend backend,
                                                      const DisparityParams& params, int width,
                                                      int height);

// The accelerators VPI reports on a platform (probe_platform() in
// platform.hpp asks it); all false on x86_64.
struct PlatformFacts {
  bool has_ofa{};  // An Optical Flow Accelerator (Jetson Orin, Thor).
  bool has_vic{};  // A Video Image Compositor, which ofa and pva convert with.
  bool has_pva{};  // A Programmable Vision Accelerator.
};

// The backends a platform with `facts` runs, in a stable order: cuda always;
// ofa with an OFA and a VIC; pva with all three. Never kAuto.
[[nodiscard]] std::vector<Backend> available_backends(const PlatformFacts& facts);

// What Backend::kAuto chose, and why, in a sentence for the log.
struct AutoChoice {
  Backend backend{Backend::kCuda};
  std::string reason;
};

// Backend::kAuto's choice for `width` x `height` images: ofa where VPI reports
// an OFA and a VIC and the parameters and size suit it (max_disparity 128 or 256, no
// confidence output, p1 + p2 <= 217, sides of 16..16384), else cuda.
[[nodiscard]] AutoChoice resolve_auto(const PlatformFacts& facts, const DisparityParams& params,
                                      bool confidence, int width, int height);

}  // namespace holoscan::stereo_disparity
