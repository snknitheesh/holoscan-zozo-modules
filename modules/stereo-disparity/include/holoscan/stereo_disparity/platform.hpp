// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What this platform's VPI offers the stereo matcher.

#pragma once

#include <vector>

#include <holoscan/stereo_disparity/disparity_options.hpp>

namespace holoscan::stereo_disparity {

// The accelerators VPI reports for the current context, and whether this is
// Jetson Thor: VPI reports an OFA, and the current CUDA device is an
// integrated GPU of compute capability 11.0 (10.1 before CUDA 13; Orin's is
// 8.7). A fact that cannot be established is false.
[[nodiscard]] PlatformFacts probe_platform() noexcept;

// available_backends() of this platform.
[[nodiscard]] std::vector<Backend> available_backends();

}  // namespace holoscan::stereo_disparity
