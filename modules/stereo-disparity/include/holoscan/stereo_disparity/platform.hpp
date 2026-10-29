// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What this platform's VPI offers the stereo matcher.

#pragma once

#include <vector>

#include <holoscan/stereo_disparity/disparity_options.hpp>

namespace holoscan::stereo_disparity {

// The accelerators VPI reports for the current context. A fact VPI cannot
// establish is false.
[[nodiscard]] PlatformFacts probe_platform() noexcept;

// available_backends() of this platform.
[[nodiscard]] std::vector<Backend> available_backends();

}  // namespace holoscan::stereo_disparity
