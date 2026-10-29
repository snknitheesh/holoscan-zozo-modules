// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The point-cloud filters on the GPU, without a Holoscan graph: a cloud is
// `count` packed float32 x, y, z triples in device memory, optionally with
// packed uint8 RGB triples, the layout of holoscan-point-cloud's clouds. A
// point is invalid when any coordinate is not finite (DepthToPointCloudOp
// writes NaN); no filter keeps an invalid point.
//
// The filters that only remove points (crop box, ground removal, outlier
// removal) write a keep mask, one byte per point; apply_mask() turns it into
// an organized cloud of the same size, removed points NaN, and Compactor into
// a dense cloud of the kept points in their input order. The voxel grid makes
// a dense cloud of one point per voxel. Every result is deterministic: the
// same input gives the same bits on every run.
//
// Each class keeps its device scratch between calls, grown as needed, and is
// for one stream at a time. All work is ordered on the given stream; nothing
// waits for it, except that growing a scratch buffer first waits for the
// stream to finish with the old one.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <memory>

namespace holoscan::pcl_filters {

// Largest cloud the filters accept.
inline constexpr std::uint32_t kMaxPointsLimit = std::uint32_t{1} << 30;

// A rigid (or any affine) transform, 3x4 row-major [R | t]: p' = R * p + t.
struct Transform {
  float matrix[12]{1.0F, 0.0F, 0.0F, 0.0F,  //
                   0.0F, 1.0F, 0.0F, 0.0F,  //
                   0.0F, 0.0F, 1.0F, 0.0F};
};

// ---- Keep masks ---------------------------------------------------------------------------------

// Writes the kept points of a mask into an organized cloud of the same size:
// kept points and their colors as they are, removed and invalid points NaN
// with black. `rgb` and `out_rgb` may both be null.
cudaError_t apply_mask(const float* xyz, const std::uint8_t* rgb, const std::uint8_t* keep,
                       std::uint32_t count, float* out_xyz, std::uint8_t* out_rgb,
                       cudaStream_t stream);

// Packs the kept points of a mask, in their input order.
class Compactor {
 public:
  Compactor();
  ~Compactor();
  Compactor(const Compactor&) = delete;
  Compactor& operator=(const Compactor&) = delete;

  // Counts the kept points into device_count(). Keeps `keep` for write().
  cudaError_t count(const std::uint8_t* keep, std::uint32_t count, cudaStream_t stream);
  // The number of kept points, in device memory, once count() has run.
  [[nodiscard]] const std::uint32_t* device_count() const noexcept;
  // Writes the kept points (and colors, when both are given) of the cloud
  // count() saw, into buffers of at least that many points.
  cudaError_t write(const float* xyz, const std::uint8_t* rgb, float* out_xyz,
                    std::uint8_t* out_rgb, cudaStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// ---- Crop box and pass-through ------------------------------------------------------------------

struct CropBoxParams {
  // The box, in the box frame: min <= p <= max on every axis. An axis left at
  // -inf..inf is not limited, so one finite axis is a pass-through filter.
  float min[3]{-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
               -std::numeric_limits<float>::infinity()};
  float max[3]{std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
               std::numeric_limits<float>::infinity()};
  // And the distance from the box frame's origin within [min_distance,
  // max_distance].
  float min_distance{0.0F};
  float max_distance{std::numeric_limits<float>::infinity()};
  // Maps input points into the box frame; the output points stay in the
  // input frame. Ignored unless `use_transform`.
  bool use_transform{false};
  Transform transform;
  // Keep the points outside the region instead of inside.
  bool keep_outside{false};
};

// Marks the points to keep: valid, and inside the region (outside it with
// keep_outside).
cudaError_t launch_crop_box(const float* xyz, std::uint32_t count, const CropBoxParams& params,
                            std::uint8_t* keep, cudaStream_t stream);

// ---- Voxel grid ---------------------------------------------------------------------------------

enum class VoxelMode : std::uint8_t {
  kCentroid,    // The mean of the voxel's points.
  kFirstPoint,  // The voxel's point with the lowest input index.
};

// A voxel index is limited to [-kVoxelIndexRange, kVoxelIndexRange) on each
// axis; points beyond it (2^20 voxels from the origin) are dropped.
inline constexpr std::int32_t kVoxelIndexRange = std::int32_t{1} << 20;

// One point per occupied voxel of size `voxel_size`, the voxel of p being
// floor(p / voxel_size) on each axis. Voxels come out in ascending order of
// their (x, y, z) index, x slowest; colors are the voxel's mean color,
// rounded, in both modes.
class VoxelGrid {
 public:
  VoxelGrid();
  ~VoxelGrid();
  VoxelGrid(const VoxelGrid&) = delete;
  VoxelGrid& operator=(const VoxelGrid&) = delete;

  // Groups the points into voxels and counts them into device_count().
  cudaError_t group(const float* xyz, std::uint32_t count, float voxel_size, cudaStream_t stream);
  // The number of voxels, in device memory, once group() has run.
  [[nodiscard]] const std::uint32_t* device_count() const noexcept;
  // Writes one point (and color, when both are given) per voxel of the cloud
  // group() saw, into buffers of at least that many points.
  cudaError_t write(const float* xyz, const std::uint8_t* rgb, VoxelMode mode, float* out_xyz,
                    std::uint8_t* out_rgb, cudaStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// ---- Ground removal -----------------------------------------------------------------------------

enum class GroundMode : std::uint8_t {
  kRansac,  // Fit the plane with RANSAC.
  kHeight,  // The plane up . p = height.
};

struct GroundParams {
  GroundMode mode{GroundMode::kRansac};
  // Which way is up, in the cloud's frame; normalized by the filter. The
  // plane's normal points this way.
  float up[3]{0.0F, 0.0F, 1.0F};
  // RANSAC: points within this distance of the plane are ground.
  float distance_threshold{0.05F};
  // RANSAC: plane hypotheses, each from three valid points drawn with `seed`.
  std::uint32_t max_iterations{256};
  std::uint64_t seed{0x5EEDU};
  // RANSAC: a plane whose normal is more than this many degrees from `up` is
  // not ground.
  float max_tilt_deg{30.0F};
  // RANSAC: refit the best hypothesis to its inliers by least squares.
  bool refine{true};
  // Height mode: points with up . p <= height are ground.
  float height{0.0F};
};

// Hypotheses scored per pass of the scoring kernel; max_iterations above this
// are scored in several passes.
inline constexpr std::uint32_t kMaxIterationsLimit = 65536;

// Finds the ground plane and marks the points off it. The plane (a, b, c, d),
// with a x + b y + c z + d = 0, (a, b, c) a unit normal toward `up`, goes to
// `plane` in device memory; all four are NaN, and every valid point is kept,
// when RANSAC finds no plane (fewer than three valid points, or no hypothesis
// within max_tilt_deg). The number of ground points goes to `ground_count`
// in device memory, when given.
class GroundSegmenter {
 public:
  GroundSegmenter();
  ~GroundSegmenter();
  GroundSegmenter(const GroundSegmenter&) = delete;
  GroundSegmenter& operator=(const GroundSegmenter&) = delete;

  cudaError_t segment(const float* xyz, std::uint32_t count, const GroundParams& params,
                      float* plane, std::uint8_t* keep, std::uint32_t* ground_count,
                      cudaStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// ---- Outlier removal ----------------------------------------------------------------------------

enum class OutlierMode : std::uint8_t {
  // Remove points whose mean distance to their k nearest neighbors is above
  // mean + std_multiplier * standard deviation of that distance over the cloud.
  kStatistical,
  // Remove points with fewer than min_neighbors other points within radius.
  kRadius,
};

// Largest k of the statistical mode.
inline constexpr std::uint32_t kMaxNeighbors = 64;

struct OutlierParams {
  OutlierMode mode{OutlierMode::kStatistical};
  // Statistical: neighbors per point, 1..kMaxNeighbors, searched within
  // search_radius. A point with fewer than k other points within it is an
  // outlier, and is left out of the mean and deviation.
  std::uint32_t k{8};
  float std_multiplier{1.0F};
  float search_radius{0.1F};
  // Radius: the neighborhood and the neighbors a point needs to stay.
  float radius{0.05F};
  std::uint32_t min_neighbors{2};
};

// Neighbors are found in a uniform grid (search_radius / 4 or radius a side)
// whose occupied cells are sorted by key and looked up by binary search; the
// k nearest within the search radius are exact.
class OutlierFilter {
 public:
  OutlierFilter();
  ~OutlierFilter();
  OutlierFilter(const OutlierFilter&) = delete;
  OutlierFilter& operator=(const OutlierFilter&) = delete;

  cudaError_t filter(const float* xyz, std::uint32_t count, const OutlierParams& params,
                     std::uint8_t* keep, cudaStream_t stream);

  // Statistical mode, after filter(), in device memory: each point's mean
  // distance to its k nearest neighbors (NaN without k of them), and the
  // threshold applied. For tests and diagnostics.
  [[nodiscard]] const float* device_mean_distances() const noexcept;
  [[nodiscard]] const double* device_threshold() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::pcl_filters
