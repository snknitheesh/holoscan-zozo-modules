// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A CPU reference of the registration, for small clouds: brute-force nearest
// neighbors, normals by Jacobi eigen decomposition, and the normal equations
// summed in double. It runs a fixed number of iterations and does not
// subsample, so the GPU's result after the same iterations must match it.

#pragma once

#include <cstdint>
#include <vector>

#include <holoscan/icp/registration.hpp>

namespace holoscan::icp::test {

struct ReferenceResult {
  Matrix4 transform{identity_transform()};
  double fitness{};
  double rmse{};
  std::uint32_t correspondences{};
};

// Aligns packed x, y, z `source` to `target` from `initial` with exactly
// `iterations` updates, using options.method, max_correspondence_distance,
// normal_radius and the robust kernel; voxel_size must be 0.
[[nodiscard]] ReferenceResult reference_icp(const std::vector<float>& source,
                                            const std::vector<float>& target,
                                            const IcpOptions& options, const Matrix4& initial,
                                            int iterations);

}  // namespace holoscan::icp::test
