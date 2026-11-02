// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Multi-plane segmentation on the GPU, without a Holoscan graph: the planes
// of a cloud with normals, found one after another by RANSAC with normal
// agreement, each removed before the next is searched.
//
// For each plane, every hypothesis is drawn at once, one thread each: a
// random unassigned point, then two more whose normals agree with its own,
// from a SplitMix64 sequence of `seed`, so the same cloud gives the same
// planes on every run. A hypothesis whose plane disagrees with any of the
// three normals is dropped. Each is scored on a fixed, evenly spread sample of
// the unassigned points, which counts a point when it is within
// distance_threshold of the plane and its normal within max_normal_angle_deg
// of the plane's (either way round). The best hypothesis (the lowest index on
// a tie) is refit to all its inliers by least squares, refine_iterations
// times, and its inliers are the plane's points. A plane with fewer than
// min_inliers ends the search. With merge_coplanar, planes that agree within
// merge_angle_deg and merge_distance are then merged, and refit.
//
// Everything is decided on the GPU, so the host never waits; floating-point
// sums are in double, reduced in a fixed order.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>

namespace holoscan::normals {

// Largest number of planes.
inline constexpr std::uint32_t kMaxPlanes = 64;
// Largest number of hypotheses per plane.
inline constexpr std::uint32_t kMaxIterations = 16384;
// Largest number of points each hypothesis is scored on.
inline constexpr std::uint32_t kMaxSamples = 65536;

// Labels: plane k is k + 1, in the order found (merged planes renumbered in
// order). These two are not planes.
inline constexpr std::uint8_t kLabelNoPlane = 0;    // A point with a normal on no plane.
inline constexpr std::uint8_t kLabelInvalid = 255;  // An invalid point, or one without a normal.

struct PlaneParams {
  // Planes to find, 1..kMaxPlanes.
  std::uint32_t max_planes{8};
  // Meters from a plane that are on it.
  float distance_threshold{0.02F};
  // Degrees between a point's normal and the plane's, 0..90, within which the
  // point can be on the plane.
  float max_normal_angle_deg{20.0F};
  // Hypotheses per plane, 1..kMaxIterations.
  std::uint32_t iterations{256};
  // Unassigned points each hypothesis is scored on, 1..kMaxSamples; all of
  // them when there are fewer.
  std::uint32_t samples{8192};
  // A plane needs at least this many inliers, at least 3.
  std::uint32_t min_inliers{500};
  std::uint64_t seed{0x5EEDU};
  // Least-squares refits of each plane to its inliers, 0..8.
  std::uint32_t refine_iterations{2};
  // Plane normals point toward this point, in the cloud's frame.
  float viewpoint[3]{0.0F, 0.0F, 0.0F};
  // Merge planes whose normals are within merge_angle_deg (0..90) and whose
  // centroids are each within merge_distance of the other's plane.
  bool merge_coplanar{false};
  float merge_angle_deg{3.0F};
  float merge_distance{0.03F};
};

// Whether `params` are usable; an unusable field makes segment() fail with
// cudaErrorInvalidValue. `what`, when given, names the first unusable field.
[[nodiscard]] bool valid_params(const PlaneParams& params, const char** what = nullptr);

// Where segment() writes, in device memory.
struct PlaneOutputs {
  // One label per point, see kLabelNoPlane and kLabelInvalid.
  std::uint8_t* labels{};
  // [max_planes, 4] (a, b, c, d): a x + b y + c z + d = 0 with (a, b, c) a
  // unit normal toward the viewpoint; NaN past the planes found.
  float* coefficients{};
  // [max_planes, 3] the mean of each plane's inliers; NaN past the planes found.
  float* centroids{};
  // [max_planes] each plane's inliers; 0 past the planes found.
  std::uint32_t* inliers{};
  // The number of planes found, when given.
  std::uint32_t* plane_count{};
};

class PlaneSegmenter {
 public:
  PlaneSegmenter();
  ~PlaneSegmenter();
  PlaneSegmenter(const PlaneSegmenter&) = delete;
  PlaneSegmenter& operator=(const PlaneSegmenter&) = delete;

  // Segments `count` points with their normals (3 floats each; a point with a
  // non-finite normal is kLabelInvalid). Every output but plane_count is
  // required.
  cudaError_t segment(const float* xyz, const float* normals, std::uint32_t count,
                      const PlaneParams& params, const PlaneOutputs& outputs, cudaStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::normals
