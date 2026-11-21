// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// What a CI lane must exercise. A test that finds no backend skips (exit 77) on a developer's
// machine, but in CI (HOLOSCAN_CI_PLATFORM set) a runner that lost its encoder or decoder must
// fail, not report a skip.

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/codec/codec_types.hpp>
#include <holoscan/codec/video_codec.hpp>

namespace holoscan::codec::test {

// The backend this CI lane must run: nvcodec on x86_64 and on Jetson Thor (a GPU of compute
// capability 10 or later), v4l2 on Jetson Orin. Empty outside CI.
inline std::optional<Backend> ci_backend() {
  const char* platform = std::getenv("HOLOSCAN_CI_PLATFORM");
  if (platform == nullptr || *platform == '\0') {
    return std::nullopt;
  }
  if (std::string_view(platform) != "jetson") {
    return Backend::kNvcodec;
  }
  int device = 0;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    return Backend::kNvcodec;  // No GPU: the check below fails, naming nvcodec.
  }
  return properties.major >= 10 ? Backend::kNvcodec : Backend::kV4l2;
}

// Empty when `backend` encodes and decodes both codecs here; otherwise why not.
inline std::string backend_problem(Backend backend) {
  std::string problem;
  for (const Codec codec : {Codec::kH264, Codec::kH265}) {
    for (const bool encoder : {true, false}) {
      const std::vector<BackendStatus> statuses =
          encoder ? probe_encoder_backends(codec) : probe_decoder_backends(codec);
      const auto found = std::find_if(statuses.begin(), statuses.end(),
                                      [&](const BackendStatus& s) { return s.backend == backend; });
      if (found == statuses.end() || !found->available) {
        problem += std::string(problem.empty() ? "" : "; ") + std::string(backend_name(backend)) +
                   (encoder ? " cannot encode " : " cannot decode ") +
                   std::string(codec_name(codec)) +
                   (found != statuses.end() ? ": " + found->detail : "");
      }
    }
  }
  return problem;
}

}  // namespace holoscan::codec::test
