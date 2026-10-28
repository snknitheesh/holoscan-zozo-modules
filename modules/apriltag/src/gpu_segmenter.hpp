// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The first half of the CUDA backend: threshold, connected components, boundary
// points per black/white component pair (a cluster), angular order, line-fit
// moments and corner candidates. Fixed capacities and device-side counts, so a
// frame needs no host synchronization.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include <holoscan/apriltag/detection.hpp>

#include "homography.hpp"  // TAG_DETECTION_HOST_DEVICE

namespace holoscan::apriltag {

inline void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

// `count` elements of device memory, released on destruction.
template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), (count > 0 ? count : 1) * sizeof(T)),
               "cudaMalloc");
  }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  [[nodiscard]] T* get() const noexcept { return data_; }

 private:
  T* data_{};
};

// Weighted sums for fitting a line to points: x and y in decimated pixels with
// centers at +0.5, the units the fit thresholds are given in.
struct LineMoments {
  double mx{}, my{}, mxx{}, mxy{}, myy{}, w{};

  TAG_DETECTION_HOST_DEVICE LineMoments operator+(const LineMoments& o) const {
    return {mx + o.mx, my + o.my, mxx + o.mxx, mxy + o.mxy, myy + o.myy, w + o.w};
  }
  TAG_DETECTION_HOST_DEVICE LineMoments operator-(const LineMoments& o) const {
    return {mx - o.mx, my - o.my, mxx - o.mxx, mxy - o.mxy, myy - o.myy, w - o.w};
  }
};

#if defined(__CUDACC__)
// The moments of a cluster's points first .. last inclusive, wrapping past its
// end when first > last: `before` sums the points before first, `through` those
// up to last, `total` all of them. Each .cu keeps its own FMA setting.
__device__ inline LineMoments range_moments(const LineMoments& before, const LineMoments& through,
                                            const LineMoments& total, bool wraps) {
  return wraps ? (total - before) + through : through - before;
}

// The points' weighted covariance; its eigenvalues are half_trace -/+ spread.
struct Covariance {
  double ex, ey, cxx, cxy, cyy, half_trace, spread;

  __device__ explicit Covariance(const LineMoments& m)
      : ex(m.mx / m.w),
        ey(m.my / m.w),
        cxx(m.mxx / m.w - ex * ex),
        cxy(m.mxy / m.w - ex * ey),
        cyy(m.myy / m.w - ey * ey),
        half_trace(0.5 * (cxx + cyy)),
        spread(sqrt(0.25 * (cxx - cyy) * (cxx - cyy) + cxy * cxy)) {}
  // The mean squared distance of the points from their best line.
  [[nodiscard]] __device__ double smallest() const { return fmax(0.0, half_trace - spread); }
};
#endif

// Counts the segmenter leaves on the device for each frame.
struct SegmentCounts {
  std::uint32_t points;  // Boundary points found; those past the capacity are dropped.
  std::uint32_t runs;    // Point runs after sorting, including the padding run.
  std::uint32_t unique;  // Points left after repeated points are removed.
  std::uint32_t maxima;  // Corner candidates over all clusters.
};

// The segmenter's device results. Cluster c < clusters() has points offsets[c] ..
// + point_counts[c] - 1 in angular order and corner candidates (point indices)
// maxima[maxima_offsets[c]] .. + maxima_counts[c] - 1.
struct ClusterData {
  const SegmentCounts* counts;
  std::uint32_t capacity;
  const std::uint32_t* raw_points;  // Boundary points before repeats were removed.
  const std::uint32_t* offsets;
  const std::uint32_t* point_counts;
  const std::int32_t* box;  // Per cluster in doubled decimated pixels: x min, x max, y min, y max.
  const double* polarity;   // Sum of (point - center) . gradient: > 0 for a dark inside.
  const LineMoments* prefix;  // Inclusive per-cluster prefix sums of the points' moments.
  const double* smoothed;     // The smoothed line-fit error at each point.
  const std::uint32_t* maxima;
  const std::uint32_t* maxima_offsets;
  const std::uint32_t* maxima_counts;

#if defined(__CUDACC__)
  // Clusters in this frame: the runs, less the run of padding when there is one.
  __device__ std::uint32_t clusters() const {
    const std::uint32_t found = counts->points < capacity ? counts->points : capacity;
    return counts->runs - (found < capacity ? 1U : 0U);
  }
#endif
};

class GpuSegmenter {
 public:
  // @throws std::invalid_argument for a size or option out of range.
  // @throws std::runtime_error if a device allocation fails.
  GpuSegmenter(int width, int height, int decimate, int min_white_black_diff);
  ~GpuSegmenter();
  GpuSegmenter(const GpuSegmenter&) = delete;
  GpuSegmenter& operator=(const GpuSegmenter&) = delete;

  // Enqueues the segmentation of `image` (device memory, the configured size,
  // checked by DetectorBackend::detect) on `stream`; results are in data().
  // @throws std::runtime_error on a CUDA failure.
  void segment(const GrayImageView& image, cudaStream_t stream);

  [[nodiscard]] ClusterData data() const noexcept;
  // Most boundary points, and so clusters, a frame can hold.
  [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }
  // Longest cluster that can still be a tag border inside the image.
  [[nodiscard]] std::uint32_t max_cluster_points() const noexcept {
    return static_cast<std::uint32_t>(3 * (2 * dec_width_ + 2 * dec_height_));
  }

 private:
  struct Buffers;
  // Runs CUB call `stage` over the whole capacity; with `temp` null it only
  // sizes its scratch.
  void run_cub(int stage, void* temp, std::size_t& bytes, cudaStream_t stream);

  int decimate_;
  int min_white_black_diff_;
  int dec_width_;
  int dec_height_;
  int tiles_x_;
  int tiles_y_;
  std::uint32_t capacity_;
  int component_bits_;  // Bits of a component root (a decimated pixel index).
  int cluster_bits_;    // Bits of a cluster index.
  std::unique_ptr<Buffers> buffers_;
};

}  // namespace holoscan::apriltag
