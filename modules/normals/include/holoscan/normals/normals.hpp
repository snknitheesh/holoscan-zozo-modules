// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Surface normals and curvature on the GPU, without a Holoscan graph. A cloud
// is `count` packed float32 x, y, z triples in device memory, the layout of
// holoscan-point-cloud's clouds; an organized cloud is `height` rows of
// `width` points. A point is invalid when any coordinate is not finite
// (DepthToPointCloudOp writes NaN).
//
// Each valid point's neighborhood is gathered, its covariance computed about
// the neighborhood's mean, and the eigenvector of the smallest eigenvalue is
// the normal (principal component analysis, as PCL's NormalEstimation). The
// normal is oriented toward a viewpoint, and the curvature is the surface
// variation lambda0 / (lambda0 + lambda1 + lambda2), 0 on a plane and up to
// 1/3 for points spread evenly in every direction.
//
// The neighborhood is either a window of the image grid around the pixel,
// for organized clouds, or the points within a radius (all of them, or the k
// nearest) found in a uniform grid sorted on the GPU, for any cloud.
//
// The results are deterministic: the same input gives the same bits on every
// run. NormalEstimator keeps its device scratch between calls, grown as
// needed, and is for one stream at a time. All work is ordered on the given
// stream; nothing waits for it, except that growing a scratch buffer first
// waits for the stream to finish with the old one.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>

namespace holoscan::normals {

// Largest cloud accepted.
inline constexpr std::uint32_t kMaxPointsLimit = std::uint32_t{1} << 30;
// Largest window radius: a window of (2 * 15 + 1)^2 pixels.
inline constexpr int kMaxWindowRadius = 15;
// Largest k of the k-nearest search.
inline constexpr std::uint32_t kMaxNeighbors = 64;

enum class NeighborSearch : std::uint8_t {
  // kWindow for an organized cloud, kKnn for an unorganized one.
  kAuto,
  // The pixels within window_radius of the pixel in the image grid, whose
  // range from the viewpoint differs from the point's by at most
  // depth_change_factor * range per pixel of distance. Organized clouds only.
  kWindow,
  // Every point within `radius`.
  kRadius,
  // The k nearest points within `radius`.
  kKnn,
};

struct NormalParams {
  NeighborSearch search{NeighborSearch::kAuto};
  // kWindow: the window is (2 * window_radius + 1)^2 pixels, 1..kMaxWindowRadius.
  int window_radius{3};
  // kWindow: a neighbor counts when its range from the viewpoint differs from
  // the point's by at most this fraction of the point's range per pixel of
  // (Chebyshev) distance. It keeps the window from reaching across depth
  // edges; finite and positive.
  float depth_change_factor{0.02F};
  // kRadius, kKnn: the neighborhood's radius in meters.
  float radius{0.05F};
  // kKnn: neighbors, the point itself included, 3..kMaxNeighbors.
  std::uint32_t k{24};
  // Fewer neighbors than this, the point itself included, give no normal;
  // 3..(2 * kMaxWindowRadius + 1)^2.
  std::uint32_t min_neighbors{5};
  // Normals point toward this point, in the cloud's frame: the camera at its
  // origin for a camera's cloud.
  float viewpoint[3]{0.0F, 0.0F, 0.0F};
};

// The search an estimate uses: `search`, or for kAuto the one for the
// cloud's kind.
[[nodiscard]] inline NeighborSearch resolve_search(NeighborSearch search, bool organized) {
  if (search != NeighborSearch::kAuto) {
    return search;
  }
  return organized ? NeighborSearch::kWindow : NeighborSearch::kKnn;
}

// Whether `params` are usable; an unusable field makes estimate() fail with
// cudaErrorInvalidValue. `what`, when given, names the first unusable field.
[[nodiscard]] bool valid_params(const NormalParams& params, const char** what = nullptr);

class NormalEstimator {
 public:
  NormalEstimator();
  ~NormalEstimator();
  NormalEstimator(const NormalEstimator&) = delete;
  NormalEstimator& operator=(const NormalEstimator&) = delete;

  // Writes each point's unit normal (3 floats) and curvature (1 float), NaN
  // for an invalid point or one without enough neighbors (or whose neighbors
  // lie on a line). `width` x `height` must be `count` for kWindow; for the
  // other searches the grid is ignored, and may be count x 1. Fails with
  // cudaErrorInvalidValue for unusable params or a window over an
  // unorganized cloud (height 1 with kWindow is a single row, and allowed).
  cudaError_t estimate(const float* xyz, std::uint32_t count, int width, int height,
                       const NormalParams& params, float* normals, float* curvature,
                       cudaStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::normals
