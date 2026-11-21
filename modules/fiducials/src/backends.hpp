// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Each backend's factory, in its own source file; create_backend() checks their arguments.

#pragma once

#include <memory>

#include <holoscan/fiducials/detector_backend.hpp>

namespace holoscan::fiducials {

[[nodiscard]] std::unique_ptr<DetectorBackend> create_cpu_backend(const DetectorOptions& options,
                                                                  int width, int height);
[[nodiscard]] std::unique_ptr<DetectorBackend> create_cuda_backend(const DetectorOptions& options,
                                                                   int width, int height);

}  // namespace holoscan::fiducials
