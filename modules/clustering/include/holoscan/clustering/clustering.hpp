// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Euclidean clustering on the GPU, without a Holoscan graph. A cloud is
// `count` packed float32 x, y, z triples in device memory, the layout of
// holoscan-point-cloud's clouds: unorganized [N, 3], or organized
// [height, width, 3] read as `count` = height * width points. A point is
// invalid when any coordinate is not finite (DepthToPointCloudOp writes NaN).
//
// Two valid points are linked when they are at most `tolerance` apart, and a
// cluster is a connected component of these links: every point of a cluster
// can be reached from every other in steps of at most `tolerance`. This is
// the Euclidean cluster extraction of PCL. Components whose size is outside
// [min_cluster_size, max_cluster_size] are dropped; of the rest, the
// max_clusters largest are kept. Their points are labeled 0, 1, 2, ... by
// decreasing size, ties going to the cluster whose lowest point index is
// lower; every other point, invalid ones included, is labeled -1. The result
// is deterministic: the same cloud gives the same bits on every run.
//
// EuclideanClusterer keeps its device scratch between calls, grown as
// needed, and is for one stream at a time. All work is ordered on the given
// stream. cluster() waits for the stream to learn how many components there
// are (and before that, with kEuclidean, how many grid cells), and
// clustered_count() waits once more; nothing else waits, except that growing
// a scratch buffer first waits for the stream to finish with the old one.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>

namespace holoscan::clustering {

// Largest cloud accepted: point counts are exact in the float32 summaries.
inline constexpr std::uint32_t kMaxPointsLimit = std::uint32_t{1} << 24;
// Largest number of clusters kept per cloud.
inline constexpr std::uint32_t kMaxClustersLimit = 65536;
// Largest image window radius of Connectivity::kImage.
inline constexpr int kMaxImageWindow = 8;

enum class Connectivity : std::uint8_t {
  // Any two points within the tolerance are linked: exact, for any cloud.
  // Neighbors are found in a uniform grid on the GPU.
  kEuclidean,
  // Organized clouds only: a point is linked to the points within the
  // tolerance among the pixels at most `image_window` rows and columns away.
  // Faster, but points that are close in space and far apart in the image,
  // such as across an occlusion edge, are not linked.
  kImage,
};

struct ClusterParams {
  // Largest distance between linked points, in the cloud's units (meters).
  float tolerance{0.02F};
  // Clusters with fewer or more points are dropped (labeled -1).
  std::uint32_t min_cluster_size{1};
  std::uint32_t max_cluster_size{kMaxPointsLimit};
  // At most this many of the largest clusters are kept, 1..kMaxClustersLimit.
  std::uint32_t max_clusters{1024};
  Connectivity connectivity{Connectivity::kEuclidean};
  // kImage: the window's radius in pixels, 1..kMaxImageWindow.
  int image_window{2};
};

// Whether `params` are usable: a positive, finite tolerance, sizes with
// 1 <= min <= max, max_clusters and image_window in range.
[[nodiscard]] bool valid_params(const ClusterParams& params);

// One row of the cluster summary: kSummaryFields float32, in this order. The
// id equals the row; points and their counts are exact.
struct ClusterSummary {
  float id;
  float point_count;
  float centroid[3];
  // The axis-aligned bounding box of the cluster's points.
  float min[3];
  float max[3];
  // The oriented bounding box from PCA of the points: its center, its axes as
  // three unit rows, by decreasing variance and right-handed, and its half
  // extents along them. All NaN unless asked for.
  float obb_center[3];
  float obb_axes[3][3];
  float obb_half_extents[3];
};

inline constexpr int kSummaryFields = 26;
static_assert(sizeof(ClusterSummary) == kSummaryFields * sizeof(float));

class EuclideanClusterer {
 public:
  EuclideanClusterer();
  ~EuclideanClusterer();
  EuclideanClusterer(const EuclideanClusterer&) = delete;
  EuclideanClusterer& operator=(const EuclideanClusterer&) = delete;

  // Clusters the cloud and returns the number of clusters kept in
  // `clusters`. `width` is the organized cloud's width (count a multiple of
  // it), or 0 for an unorganized cloud; kImage needs it. Waits once with
  // kImage, twice with kEuclidean.
  cudaError_t cluster(const float* xyz, std::uint32_t count, std::uint32_t width,
                      const ClusterParams& params, std::uint32_t& clusters, cudaStream_t stream);

  // The following write what the last cluster() found, for the cloud it saw.

  // The label of every point, in input order: count int32.
  cudaError_t write_labels(std::int32_t* labels, cudaStream_t stream) const;
  // The number of clustered points, the sum of the clusters' sizes. Waits.
  cudaError_t clustered_count(std::uint32_t& points, cudaStream_t stream);
  // The clustered points packed and grouped by cluster, in label order and in
  // input order within a cluster, into buffers of clustered_count() points:
  // their coordinates, and their colors and labels when given. Cluster k is
  // the rows from the sum of the sizes of the clusters before it.
  cudaError_t write_grouped(const float* xyz, const std::uint8_t* rgb, float* out_xyz,
                            std::uint8_t* out_rgb, std::int32_t* out_labels,
                            cudaStream_t stream) const;
  // The cloud with every point that is not clustered set to NaN (and black),
  // in input order. `rgb` and `out_rgb` may both be null.
  cudaError_t write_kept(const float* xyz, const std::uint8_t* rgb, float* out_xyz,
                         std::uint8_t* out_rgb, cudaStream_t stream) const;
  // One ClusterSummary per cluster, by id: `clusters` rows of kSummaryFields
  // float32. With `oriented`, the oriented boxes too, which takes another
  // pass over the points. Sums are in double, in a fixed order.
  cudaError_t write_summaries(const float* xyz, bool oriented, float* summaries,
                              cudaStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::clustering
