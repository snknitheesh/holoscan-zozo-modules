// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's --benchmark mode: VideoEncoder and VideoDecoder throughput, without a graph.

#pragma once

#include <cstdint>

namespace codec_app {

// Encodes and decodes `frames` frames at 1920x1080 and 3840x2160 for each codec on the backend
// `auto` picks, and prints frames per second. Returns 0, 1 when a backend failed, or 77 when no
// backend encodes and decodes here.
int run_benchmark(std::uint32_t frames);

}  // namespace codec_app
