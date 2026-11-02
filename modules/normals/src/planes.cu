// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Multi-plane RANSAC. The host queues max_planes rounds; each reads the state
// the previous one left on the device, and once a round finds no plane the
// rest do nothing, so the host never waits. A round lists the unassigned
// points (a CUB scan), draws every hypothesis at once (one thread each),
// scores them on an evenly spread sample of those points staged in shared
// memory (one thread per hypothesis, integer counts), keeps the best (the
// lowest index on a tie), refits it by least squares (sums in double over a
// fixed grid, reduced in a fixed order, and the smallest eigenvector of their
// covariance), and labels its inliers when there are enough. Coplanar planes
// are merged by one thread at the end. Compiled without fused multiply-adds,
// so a host reference that computes the same expressions draws and scores the
// same hypotheses.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <utility>

#include <holoscan/normals/normals.hpp>
#include <holoscan/normals/planes.hpp>

#include "device_buffer.hpp"
#include "kernel_common.cuh"

namespace holoscan::normals {
namespace {

using detail::blocks_for;
using detail::kBlock;
using detail::nan_float;

// Blocks of the least-squares sums: a fixed number, so that the order of the
// additions, and their result, never changes.
constexpr unsigned int kSumBlocks = 64;
// Sums: n, x, y, z, xx, xy, xz, yy, yz, zz.
constexpr int kSums = 10;
// Draws a hypothesis makes for each of its second and third points.
constexpr std::uint32_t kTries = 16;

// What the rounds share, on the device.
struct State {
  std::uint32_t active;      // Whether the search goes on.
  std::uint32_t found;       // Planes found so far.
  std::uint32_t candidates;  // Unassigned points this round.
  std::uint32_t accepted;    // Whether this round's plane was kept.
  float4 plane;              // This round's plane.
};

struct Agreement {
  float threshold;  // Distance.
  float cosine;     // Of the largest normal angle.
};

__device__ inline float4 nan_plane() {
  const float nan = nan_float();
  return make_float4(nan, nan, nan, nan);
}

__device__ inline float3 load3(const float* data, std::uint32_t i) {
  return make_float3(data[3 * i], data[3 * i + 1], data[3 * i + 2]);
}

__device__ inline float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__device__ inline bool agrees(float3 a, float3 b, float cosine) {
  return fabsf(dot3(a, b)) >= cosine;
}

// Whether point p with normal n is on `plane`.
__device__ inline bool on_plane(float4 plane, float3 p, float3 n, Agreement agreement) {
  const float3 normal = make_float3(plane.x, plane.y, plane.z);
  return fabsf(dot3(normal, p) + plane.w) <= agreement.threshold &&
         agrees(normal, n, agreement.cosine);
}

__global__ void start_kernel(const float* __restrict__ xyz, const float* __restrict__ normals,
                             std::uint32_t count, std::uint8_t* __restrict__ labels,
                             State* __restrict__ state) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0U) {
    *state = State{1U, 0U, 0U, 0U, nan_plane()};
  }
  if (i < count) {
    const bool usable = detail::valid_point(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]) &&
                        detail::valid_point(normals[3 * i], normals[3 * i + 1], normals[3 * i + 2]);
    labels[i] = usable ? kLabelNoPlane : kLabelInvalid;
  }
}

__global__ void flags_kernel(const std::uint8_t* __restrict__ labels, std::uint32_t count,
                             const State* __restrict__ state, std::uint32_t* __restrict__ flags) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    flags[i] = state->active != 0U && labels[i] == kLabelNoPlane ? 1U : 0U;
  }
}

__global__ void candidates_kernel(const std::uint32_t* __restrict__ flags,
                                  const std::uint32_t* __restrict__ offsets, std::uint32_t count,
                                  std::uint32_t* __restrict__ candidates,
                                  State* __restrict__ state) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  if (flags[i] != 0U) {
    candidates[offsets[i]] = i;
  }
  if (i == count - 1U) {
    state->candidates = offsets[i] + flags[i];
  }
}

// Hypothesis h of round `round`: the plane through a random candidate and two
// more whose normals agree with its own, toward the viewpoint; NaN when they
// cannot be found, are (nearly) collinear, or the plane disagrees with any of
// their normals.
__global__ void hypotheses_kernel(const float* __restrict__ xyz, const float* __restrict__ normals,
                                  const std::uint32_t* __restrict__ candidates,
                                  const State* __restrict__ state, std::uint32_t iterations,
                                  std::uint64_t seed, std::uint32_t round, float cosine,
                                  float3 view, float4* __restrict__ planes) {
  const std::uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= iterations) {
    return;
  }
  planes[h] = nan_plane();
  const std::uint32_t n = state->candidates;
  if (state->active == 0U || n < 3U) {
    return;
  }
  const std::uint64_t key =
      detail::mix64(seed ^ detail::mix64((static_cast<std::uint64_t>(round) << 32U) | h));
  const auto draw = [&](std::uint64_t j) {
    return candidates[detail::mix64(key ^ detail::mix64(j)) % n];
  };
  std::uint32_t picks[3];
  picks[0] = draw(0U);
  const float3 n0 = load3(normals, picks[0]);
  for (std::uint32_t j = 1; j < 3U; ++j) {
    bool found = false;
    for (std::uint32_t attempt = 0; attempt < kTries && !found; ++attempt) {
      const std::uint32_t pick = draw(1U + (j - 1U) * kTries + attempt);
      if (pick != picks[0] && (j == 1U || pick != picks[1]) &&
          agrees(load3(normals, pick), n0, cosine)) {
        picks[j] = pick;
        found = true;
      }
    }
    if (!found) {
      return;
    }
  }
  const float3 p0 = load3(xyz, picks[0]);
  const float3 p1 = load3(xyz, picks[1]);
  const float3 p2 = load3(xyz, picks[2]);
  const float3 a = make_float3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
  const float3 b = make_float3(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);
  float3 normal = make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
  const float length = sqrtf(dot3(normal, normal));
  const float spread = sqrtf(dot3(a, a)) * sqrtf(dot3(b, b));
  if (!(length > 1e-6F * spread) || !(length > 0.0F)) {
    return;  // Collinear or repeated points.
  }
  normal = make_float3(normal.x / length, normal.y / length, normal.z / length);
  float d = -dot3(normal, p0);
  if (dot3(normal, view) + d < 0.0F) {
    normal = make_float3(-normal.x, -normal.y, -normal.z);
    d = -d;
  }
  for (const std::uint32_t pick : picks) {
    if (!agrees(normal, load3(normals, pick), cosine)) {
      return;
    }
  }
  planes[h] = make_float4(normal.x, normal.y, normal.z, d);
}

// Thread h of row blockIdx.y counts hypothesis h's inliers among this
// block's tiles of the sample; integer sums, so the order does not matter.
__global__ void score_kernel(const float* __restrict__ xyz, const float* __restrict__ normals,
                             const std::uint32_t* __restrict__ candidates,
                             const State* __restrict__ state, std::uint32_t samples,
                             const float4* __restrict__ planes, std::uint32_t iterations,
                             Agreement agreement, std::uint32_t* __restrict__ scores) {
  __shared__ float tile[6 * kBlock];
  const std::uint32_t n = state->candidates;
  if (state->active == 0U || n == 0U) {
    return;
  }
  const std::uint32_t taken = samples < n ? samples : n;
  const std::uint32_t h = blockIdx.y * blockDim.x + threadIdx.x;
  const float4 plane = h < iterations ? planes[h] : nan_plane();
  std::uint32_t hits = 0;
  for (std::uint32_t base = blockIdx.x * kBlock; base < taken; base += gridDim.x * kBlock) {
    const std::uint32_t j = base + threadIdx.x;
    if (j < taken) {
      const std::uint32_t i = candidates[static_cast<std::uint64_t>(j) * n / taken];
      for (int c = 0; c < 3; ++c) {
        tile[6 * threadIdx.x + c] = xyz[3 * i + c];
        tile[6 * threadIdx.x + 3 + c] = normals[3 * i + c];
      }
    }
    __syncthreads();
    const std::uint32_t in_tile = taken - base < kBlock ? taken - base : kBlock;
    if (isfinite(plane.x)) {
      for (std::uint32_t t = 0; t < in_tile; ++t) {
        const float* s = tile + 6 * t;
        hits +=
            on_plane(plane, make_float3(s[0], s[1], s[2]), make_float3(s[3], s[4], s[5]), agreement)
                ? 1U
                : 0U;
      }
    }
    __syncthreads();
  }
  if (h < iterations && hits != 0U) {
    atomicAdd(scores + h, hits);
  }
}

// The hypothesis with the most inliers, the lowest index on a tie, becomes
// the round's plane; none ends the search.
__global__ void select_kernel(const float4* __restrict__ planes,
                              const std::uint32_t* __restrict__ scores, std::uint32_t iterations,
                              State* __restrict__ state) {
  __shared__ std::uint32_t best_score[kBlock];
  __shared__ std::uint32_t best_index[kBlock];
  if (state->active == 0U) {
    return;
  }
  std::uint32_t score = 0;
  std::uint32_t index = 0xFFFFFFFFU;
  for (std::uint32_t h = threadIdx.x; h < iterations; h += blockDim.x) {
    if (scores[h] > score) {
      score = scores[h];
      index = h;
    }
  }
  best_score[threadIdx.x] = score;
  best_index[threadIdx.x] = index;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride /= 2U) {
    if (threadIdx.x < stride) {
      const std::uint32_t other = threadIdx.x + stride;
      if (best_score[other] > best_score[threadIdx.x] ||
          (best_score[other] == best_score[threadIdx.x] &&
           best_index[other] < best_index[threadIdx.x])) {
        best_score[threadIdx.x] = best_score[other];
        best_index[threadIdx.x] = best_index[other];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (best_score[0] == 0U) {
      state->active = 0U;
      state->plane = nan_plane();
    } else {
      state->plane = planes[best_index[0]];
    }
  }
}

// Per-block sums over the unassigned inliers of the round's plane, each block
// over a fixed set of points, reduced in a fixed tree.
__global__ void sums_kernel(const float* __restrict__ xyz, const float* __restrict__ normals,
                            const std::uint8_t* __restrict__ labels, std::uint32_t count,
                            const State* __restrict__ state, Agreement agreement,
                            double* __restrict__ partials) {
  __shared__ double shared[kSums][kBlock];
  double sums[kSums] = {};
  if (state->active != 0U) {
    const float4 plane = state->plane;
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x) {
      if (labels[i] != kLabelNoPlane) {
        continue;
      }
      const float3 p = load3(xyz, i);
      if (on_plane(plane, p, load3(normals, i), agreement)) {
        const double x = p.x;
        const double y = p.y;
        const double z = p.z;
        sums[0] += 1.0;
        sums[1] += x;
        sums[2] += y;
        sums[3] += z;
        sums[4] += x * x;
        sums[5] += x * y;
        sums[6] += x * z;
        sums[7] += y * y;
        sums[8] += y * z;
        sums[9] += z * z;
      }
    }
  }
  for (int k = 0; k < kSums; ++k) {
    shared[k][threadIdx.x] = sums[k];
  }
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride /= 2U) {
    if (threadIdx.x < stride) {
      for (int k = 0; k < kSums; ++k) {
        shared[k][threadIdx.x] += shared[k][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (int k = 0; k < kSums; ++k) {
      partials[blockIdx.x * kSums + k] = shared[k][0];
    }
  }
}

__device__ inline void reduce_partials(const double* __restrict__ partials, double s[kSums]) {
  for (int k = 0; k < kSums; ++k) {
    s[k] = 0.0;
  }
  for (unsigned int b = 0; b < kSumBlocks; ++b) {
    for (int k = 0; k < kSums; ++k) {
      s[k] += partials[b * kSums + k];
    }
  }
}

// The least-squares plane of points with sums `s`, toward `view`; false for
// fewer than three points or points on a line.
__device__ bool fit_plane(const double s[kSums], float3 view, float4& plane) {
  const double n = s[0];
  if (n < 3.0) {
    return false;
  }
  const double mean[3] = {s[1] / n, s[2] / n, s[3] / n};
  double cov[3][3];
  cov[0][0] = s[4] / n - mean[0] * mean[0];
  cov[0][1] = s[5] / n - mean[0] * mean[1];
  cov[0][2] = s[6] / n - mean[0] * mean[2];
  cov[1][1] = s[7] / n - mean[1] * mean[1];
  cov[1][2] = s[8] / n - mean[1] * mean[2];
  cov[2][2] = s[9] / n - mean[2] * mean[2];
  cov[1][0] = cov[0][1];
  cov[2][0] = cov[0][2];
  cov[2][1] = cov[1][2];
  double normal[3];
  double variation = 0.0;
  if (!detail::pca_normal(cov, normal, variation)) {
    return false;
  }
  double d = -(normal[0] * mean[0] + normal[1] * mean[1] + normal[2] * mean[2]);
  if (normal[0] * view.x + normal[1] * view.y + normal[2] * view.z + d < 0.0) {
    for (double& value : normal) {
      value = -value;
    }
    d = -d;
  }
  plane = make_float4(static_cast<float>(normal[0]), static_cast<float>(normal[1]),
                      static_cast<float>(normal[2]), static_cast<float>(d));
  return true;
}

__global__ void refit_kernel(const double* __restrict__ partials, float3 view,
                             State* __restrict__ state) {
  if (state->active == 0U) {
    return;
  }
  double s[kSums];
  reduce_partials(partials, s);
  float4 plane;
  if (fit_plane(s, view, plane)) {
    state->plane = plane;
  }
}

// Keeps the round's plane when it has min_inliers, and ends the search when
// it has not. With `refine`, the plane kept is the least-squares plane of
// exactly its inliers; they are the points the round's plane labels.
__global__ void accept_kernel(const double* __restrict__ partials, std::uint32_t min_inliers,
                              bool refine, float3 view, State* __restrict__ state,
                              float4* __restrict__ found_planes, double* __restrict__ found_sums) {
  state->accepted = 0U;
  if (state->active == 0U) {
    return;
  }
  double s[kSums];
  reduce_partials(partials, s);
  if (s[0] < static_cast<double>(min_inliers) || s[0] < 3.0) {
    state->active = 0U;
    return;
  }
  const std::uint32_t k = state->found;
  float4 plane = state->plane;
  if (refine) {
    fit_plane(s, view, plane);
  }
  found_planes[k] = plane;
  for (int j = 0; j < kSums; ++j) {
    found_sums[k * kSums + j] = s[j];
  }
  state->found = k + 1U;
  state->accepted = 1U;
}

__global__ void mark_kernel(const float* __restrict__ xyz, const float* __restrict__ normals,
                            std::uint32_t count, const State* __restrict__ state,
                            Agreement agreement, std::uint8_t* __restrict__ labels) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count || state->accepted == 0U || labels[i] != kLabelNoPlane) {
    return;
  }
  if (on_plane(state->plane, load3(xyz, i), load3(normals, i), agreement)) {
    labels[i] = static_cast<std::uint8_t>(state->found);
  }
}

__device__ inline float3 centroid_of(const double* s) {
  return make_float3(static_cast<float>(s[1] / s[0]), static_cast<float>(s[2] / s[0]),
                     static_cast<float>(s[3] / s[0]));
}

// Merges each plane into the first earlier one it is coplanar with, refits
// the merged planes, and renumbers them in order: map[k] is plane k's new
// index.
__global__ void merge_kernel(float cosine, float distance, float3 view, State* __restrict__ state,
                             float4* __restrict__ planes, double* __restrict__ sums,
                             std::uint32_t* __restrict__ map) {
  const std::uint32_t found = state->found;
  std::uint32_t root[kMaxPlanes];
  for (std::uint32_t j = 0; j < found; ++j) {
    root[j] = j;
    for (std::uint32_t i = 0; i < j; ++i) {
      if (root[i] != i) {
        continue;
      }
      const float4 a = planes[i];
      const float4 b = planes[j];
      const float3 ca = centroid_of(sums + i * kSums);
      const float3 cb = centroid_of(sums + j * kSums);
      const float3 na = make_float3(a.x, a.y, a.z);
      const float3 nb = make_float3(b.x, b.y, b.z);
      if (dot3(na, nb) >= cosine && fabsf(dot3(na, cb) + a.w) <= distance &&
          fabsf(dot3(nb, ca) + b.w) <= distance) {
        for (int k = 0; k < kSums; ++k) {
          sums[i * kSums + k] += sums[j * kSums + k];
        }
        float4 merged;
        if (fit_plane(sums + i * kSums, view, merged)) {
          planes[i] = merged;
        }
        root[j] = i;
        break;
      }
    }
  }
  std::uint32_t kept = 0;
  for (std::uint32_t j = 0; j < found; ++j) {
    if (root[j] == j) {
      map[j] = kept;
      planes[kept] = planes[j];
      for (int k = 0; k < kSums; ++k) {
        sums[kept * kSums + k] = sums[j * kSums + k];
      }
      ++kept;
    } else {
      map[j] = map[root[j]];
    }
  }
  state->found = kept;
}

__global__ void relabel_kernel(std::uint32_t count, const std::uint32_t* __restrict__ map,
                               std::uint8_t* __restrict__ labels) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count && labels[i] != kLabelNoPlane && labels[i] != kLabelInvalid) {
    labels[i] = static_cast<std::uint8_t>(map[labels[i] - 1U] + 1U);
  }
}

__global__ void outputs_kernel(const State* __restrict__ state, const float4* __restrict__ planes,
                               const double* __restrict__ sums, std::uint32_t max_planes,
                               float* __restrict__ coefficients, float* __restrict__ centroids,
                               std::uint32_t* __restrict__ inliers,
                               std::uint32_t* __restrict__ plane_count) {
  const std::uint32_t k = threadIdx.x;
  const std::uint32_t found = state->found;
  if (k == 0U && plane_count != nullptr) {
    *plane_count = found;
  }
  if (k >= max_planes) {
    return;
  }
  const float nan = nan_float();
  if (k < found) {
    const float4 plane = planes[k];
    const float3 centroid = centroid_of(sums + k * kSums);
    coefficients[4 * k] = plane.x;
    coefficients[4 * k + 1] = plane.y;
    coefficients[4 * k + 2] = plane.z;
    coefficients[4 * k + 3] = plane.w;
    centroids[3 * k] = centroid.x;
    centroids[3 * k + 1] = centroid.y;
    centroids[3 * k + 2] = centroid.z;
    inliers[k] = static_cast<std::uint32_t>(sums[k * kSums]);
  } else {
    for (int c = 0; c < 4; ++c) {
      coefficients[4 * k + c] = nan;
    }
    for (int c = 0; c < 3; ++c) {
      centroids[3 * k + c] = nan;
    }
    inliers[k] = 0U;
  }
}

float cosine_of(float degrees) {
  return static_cast<float>(std::cos(static_cast<double>(degrees) * std::numbers::pi / 180.0));
}

}  // namespace

bool valid_params(const PlaneParams& params, const char** what) {
  const auto fail = [what](const char* field) {
    if (what != nullptr) {
      *what = field;
    }
    return false;
  };
  if (params.max_planes < 1U || params.max_planes > kMaxPlanes) {
    return fail("max_planes must be 1..64");
  }
  if (!(params.distance_threshold >= 0.0F) || !std::isfinite(params.distance_threshold)) {
    return fail("distance_threshold must be finite and not negative");
  }
  if (!(params.max_normal_angle_deg >= 0.0F && params.max_normal_angle_deg <= 90.0F)) {
    return fail("max_normal_angle_deg must be 0..90");
  }
  if (params.iterations < 1U || params.iterations > kMaxIterations) {
    return fail("iterations must be 1..16384");
  }
  if (params.samples < 1U || params.samples > kMaxSamples) {
    return fail("samples must be 1..65536");
  }
  if (params.min_inliers < 3U) {
    return fail("min_inliers must be at least 3");
  }
  if (params.refine_iterations > 8U) {
    return fail("refine_iterations must be 0..8");
  }
  if (!std::isfinite(params.viewpoint[0]) || !std::isfinite(params.viewpoint[1]) ||
      !std::isfinite(params.viewpoint[2])) {
    return fail("viewpoint must be finite");
  }
  if (params.merge_coplanar &&
      (!(params.merge_angle_deg >= 0.0F && params.merge_angle_deg <= 90.0F) ||
       !(params.merge_distance >= 0.0F) || !std::isfinite(params.merge_distance))) {
    return fail("merge_angle_deg must be 0..90 and merge_distance finite and not negative");
  }
  return true;
}

struct PlaneSegmenter::Impl {
  detail::DeviceBuffer flags;
  detail::DeviceBuffer offsets;
  detail::DeviceBuffer candidates;
  detail::DeviceBuffer hypotheses;
  detail::DeviceBuffer scores;
  detail::DeviceBuffer partials;
  detail::DeviceBuffer state;
  detail::DeviceBuffer planes;
  detail::DeviceBuffer sums;
  detail::DeviceBuffer map;
  detail::DeviceBuffer temp;
};

PlaneSegmenter::PlaneSegmenter() : impl_(std::make_unique<Impl>()) {}
PlaneSegmenter::~PlaneSegmenter() = default;

cudaError_t PlaneSegmenter::segment(const float* xyz, const float* normals, std::uint32_t count,
                                    const PlaneParams& params, const PlaneOutputs& outputs,
                                    cudaStream_t stream) {
  if (!valid_params(params) || count > kMaxPointsLimit || outputs.coefficients == nullptr ||
      outputs.centroids == nullptr || outputs.inliers == nullptr ||
      (count != 0U && (xyz == nullptr || normals == nullptr || outputs.labels == nullptr))) {
    return cudaErrorInvalidValue;
  }
  Impl& s = *impl_;
  const std::size_t n = std::max<std::size_t>(count, 1U);
  const std::uint32_t iterations = params.iterations;
  for (auto [buffer, bytes] :
       {std::pair{&s.flags, n * 4U}, std::pair{&s.offsets, n * 4U},
        std::pair{&s.candidates, n * 4U},
        std::pair{&s.hypotheses, std::size_t{iterations} * sizeof(float4)},
        std::pair{&s.scores, std::size_t{iterations} * 4U},
        std::pair{&s.partials, std::size_t{kSumBlocks} * kSums * sizeof(double)},
        std::pair{&s.state, sizeof(State)},
        std::pair{&s.planes, std::size_t{kMaxPlanes} * sizeof(float4)},
        std::pair{&s.sums, std::size_t{kMaxPlanes} * kSums * sizeof(double)},
        std::pair{&s.map, std::size_t{kMaxPlanes} * 4U}}) {
    if (const cudaError_t status = buffer->reserve(bytes, stream); status != cudaSuccess) {
      return status;
    }
  }
  auto* state = s.state.as<State>();
  const float3 view = make_float3(params.viewpoint[0], params.viewpoint[1], params.viewpoint[2]);
  const Agreement agreement{params.distance_threshold, cosine_of(params.max_normal_angle_deg)};

  start_kernel<<<std::max(blocks_for(count), 1U), kBlock, 0, stream>>>(xyz, normals, count,
                                                                       outputs.labels, state);
  if (count != 0U) {
    const dim3 score_grid(std::min(blocks_for(params.samples), blocks_for(count)),
                          blocks_for(iterations));
    for (std::uint32_t round = 0; round < params.max_planes; ++round) {
      flags_kernel<<<blocks_for(count), kBlock, 0, stream>>>(outputs.labels, count, state,
                                                             s.flags.as<std::uint32_t>());
      if (const cudaError_t status = detail::exclusive_sum(
              s.flags.as<std::uint32_t>(), s.offsets.as<std::uint32_t>(), count, s.temp, stream);
          status != cudaSuccess) {
        return status;
      }
      candidates_kernel<<<blocks_for(count), kBlock, 0, stream>>>(
          s.flags.as<std::uint32_t>(), s.offsets.as<std::uint32_t>(), count,
          s.candidates.as<std::uint32_t>(), state);
      hypotheses_kernel<<<blocks_for(iterations), kBlock, 0, stream>>>(
          xyz, normals, s.candidates.as<std::uint32_t>(), state, iterations, params.seed, round,
          agreement.cosine, view, s.hypotheses.as<float4>());
      if (const cudaError_t status =
              cudaMemsetAsync(s.scores.as<void>(), 0, std::size_t{iterations} * 4U, stream);
          status != cudaSuccess) {
        return status;
      }
      score_kernel<<<score_grid, kBlock, 0, stream>>>(
          xyz, normals, s.candidates.as<std::uint32_t>(), state, params.samples,
          s.hypotheses.as<float4>(), iterations, agreement, s.scores.as<std::uint32_t>());
      select_kernel<<<1, kBlock, 0, stream>>>(s.hypotheses.as<float4>(),
                                              s.scores.as<std::uint32_t>(), iterations, state);
      for (std::uint32_t refit = 0; refit < params.refine_iterations; ++refit) {
        sums_kernel<<<kSumBlocks, kBlock, 0, stream>>>(xyz, normals, outputs.labels, count, state,
                                                       agreement, s.partials.as<double>());
        refit_kernel<<<1, 1, 0, stream>>>(s.partials.as<double>(), view, state);
      }
      sums_kernel<<<kSumBlocks, kBlock, 0, stream>>>(xyz, normals, outputs.labels, count, state,
                                                     agreement, s.partials.as<double>());
      accept_kernel<<<1, 1, 0, stream>>>(s.partials.as<double>(), params.min_inliers,
                                         params.refine_iterations != 0U, view, state,
                                         s.planes.as<float4>(), s.sums.as<double>());
      mark_kernel<<<blocks_for(count), kBlock, 0, stream>>>(xyz, normals, count, state, agreement,
                                                            outputs.labels);
      if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
        return status;
      }
    }
    if (params.merge_coplanar) {
      merge_kernel<<<1, 1, 0, stream>>>(cosine_of(params.merge_angle_deg), params.merge_distance,
                                        view, state, s.planes.as<float4>(), s.sums.as<double>(),
                                        s.map.as<std::uint32_t>());
      relabel_kernel<<<blocks_for(count), kBlock, 0, stream>>>(count, s.map.as<std::uint32_t>(),
                                                               outputs.labels);
    }
  }
  outputs_kernel<<<1, kMaxPlanes, 0, stream>>>(
      state, s.planes.as<float4>(), s.sums.as<double>(), params.max_planes, outputs.coefficients,
      outputs.centroids, outputs.inliers, outputs.plane_count);
  return cudaGetLastError();
}

}  // namespace holoscan::normals
