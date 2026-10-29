// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Ground removal. RANSAC runs every hypothesis at once: one thread draws each
// plane from three valid points, one thread per hypothesis then counts its
// inliers over tiles of points staged in shared memory, and the hypothesis
// with the most inliers wins, the lowest index on a tie. The winner is refit
// to its inliers by least squares (sums in double, reduced in a fixed order,
// and the smallest eigenvector of their covariance), and the points within
// the threshold of the final plane are ground. Every step reads the counts it
// needs on the device, so the host never waits, and every result, counts and
// sums alike, comes out the same on every run.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <utility>

#include <holoscan/pcl_filters/filters.hpp>

#include "device_buffer.hpp"
#include "kernel_common.cuh"

namespace holoscan::pcl_filters {
namespace {

using detail::blocks_for;
using detail::kBlock;
using detail::valid_point;

// Blocks of the least-squares sums: a fixed number, so that the order of the
// additions, and their result, never changes.
constexpr unsigned int kSumBlocks = 64;
// Sums per block: n, x, y, z, xx, xy, xz, yy, yz, zz.
constexpr int kSums = 10;

__device__ inline float nan_float() { return __int_as_float(0x7FC00000); }

__global__ void valid_flags_kernel(const float* __restrict__ xyz, std::uint32_t count,
                                   std::uint32_t* __restrict__ flags) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    flags[i] = valid_point(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]) ? 1U : 0U;
  }
}

__global__ void valid_indices_kernel(const std::uint32_t* __restrict__ flags,
                                     const std::uint32_t* __restrict__ offsets, std::uint32_t count,
                                     std::uint32_t* __restrict__ indices,
                                     std::uint32_t* __restrict__ valid) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  if (flags[i] != 0U) {
    indices[offsets[i]] = i;
  }
  if (i == count - 1U) {
    *valid = offsets[i] + flags[i];
  }
}

__device__ inline float3 point_at(const float* xyz, std::uint32_t i) {
  return make_float3(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
}

// Hypothesis h: the plane through three valid points drawn from the seed, or
// NaN when they are (nearly) collinear or the plane is too steep.
__global__ void hypotheses_kernel(const float* __restrict__ xyz,
                                  const std::uint32_t* __restrict__ indices,
                                  const std::uint32_t* __restrict__ valid, std::uint32_t iterations,
                                  std::uint64_t seed, float3 up, float min_cos,
                                  float4* __restrict__ planes) {
  const std::uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= iterations) {
    return;
  }
  const float nan = nan_float();
  planes[h] = make_float4(nan, nan, nan, nan);
  const std::uint32_t n = *valid;
  if (n < 3U) {
    return;
  }
  float3 p[3];
  for (std::uint32_t j = 0; j < 3U; ++j) {
    const std::uint64_t draw = detail::mix64(seed ^ detail::mix64(3ULL * h + j));
    p[j] = point_at(xyz, indices[draw % n]);
  }
  const float3 a = make_float3(p[1].x - p[0].x, p[1].y - p[0].y, p[1].z - p[0].z);
  const float3 b = make_float3(p[2].x - p[0].x, p[2].y - p[0].y, p[2].z - p[0].z);
  float3 normal = make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
  const float length = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
  const float spread =
      sqrtf(a.x * a.x + a.y * a.y + a.z * a.z) * sqrtf(b.x * b.x + b.y * b.y + b.z * b.z);
  if (!(length > 1e-6F * spread) || !(length > 0.0F)) {
    return;  // Collinear or repeated points.
  }
  normal = make_float3(normal.x / length, normal.y / length, normal.z / length);
  float along = normal.x * up.x + normal.y * up.y + normal.z * up.z;
  if (along < 0.0F) {
    normal = make_float3(-normal.x, -normal.y, -normal.z);
    along = -along;
  }
  if (along < min_cos) {
    return;
  }
  const float d = -(normal.x * p[0].x + normal.y * p[0].y + normal.z * p[0].z);
  planes[h] = make_float4(normal.x, normal.y, normal.z, d);
}

__device__ inline bool near_plane(float4 plane, float x, float y, float z, float threshold) {
  return fabsf(plane.x * x + plane.y * y + plane.z * z + plane.w) <= threshold;
}

// Thread h of row blockIdx.y counts hypothesis h's inliers among the points
// of this block's tiles; integer sums, so the order does not matter.
__global__ void score_kernel(const float* __restrict__ xyz, std::uint32_t count,
                             const float4* __restrict__ planes, std::uint32_t iterations,
                             float threshold, std::uint32_t* __restrict__ scores) {
  __shared__ float tile[3 * kBlock];
  const std::uint32_t h = blockIdx.y * blockDim.x + threadIdx.x;
  const float nan = nan_float();
  const float4 plane = h < iterations ? planes[h] : make_float4(nan, nan, nan, nan);
  std::uint32_t hits = 0;
  const std::size_t floats = 3U * static_cast<std::size_t>(count);
  for (std::size_t base = static_cast<std::size_t>(blockIdx.x) * kBlock; base < count;
       base += static_cast<std::size_t>(gridDim.x) * kBlock) {
    for (unsigned int k = threadIdx.x; k < 3U * kBlock; k += blockDim.x) {
      const std::size_t at = 3U * base + k;
      tile[k] = at < floats ? xyz[at] : nan;
    }
    __syncthreads();
    for (int j = 0; j < kBlock; ++j) {
      hits += near_plane(plane, tile[3 * j], tile[3 * j + 1], tile[3 * j + 2], threshold) ? 1U : 0U;
    }
    __syncthreads();
  }
  if (h < iterations && hits != 0U) {
    atomicAdd(scores + h, hits);
  }
}

// The hypothesis with the most inliers, the lowest index on a tie, into
// `plane`; NaN when none has any.
__global__ void select_kernel(const float4* __restrict__ planes,
                              const std::uint32_t* __restrict__ scores, std::uint32_t iterations,
                              float4* __restrict__ plane) {
  __shared__ std::uint32_t best_score[kBlock];
  __shared__ std::uint32_t best_index[kBlock];
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
    const float nan = nan_float();
    *plane = best_score[0] == 0U ? make_float4(nan, nan, nan, nan) : planes[best_index[0]];
  }
}

// Per-block sums over the inliers of `plane`, each block over a fixed set of
// points, reduced in a fixed tree.
__global__ void sums_kernel(const float* __restrict__ xyz, std::uint32_t count,
                            const float4* __restrict__ plane_in, float threshold,
                            double* __restrict__ partials) {
  __shared__ double shared[kSums][kBlock];
  const float4 plane = *plane_in;
  double sums[kSums] = {};
  if (isfinite(plane.x)) {
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x) {
      const float x = xyz[3 * i];
      const float y = xyz[3 * i + 1];
      const float z = xyz[3 * i + 2];
      if (near_plane(plane, x, y, z, threshold)) {
        const double dx = x;
        const double dy = y;
        const double dz = z;
        sums[0] += 1.0;
        sums[1] += dx;
        sums[2] += dy;
        sums[3] += dz;
        sums[4] += dx * dx;
        sums[5] += dx * dy;
        sums[6] += dx * dz;
        sums[7] += dy * dy;
        sums[8] += dy * dz;
        sums[9] += dz * dz;
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

// The unit eigenvector of the smallest eigenvalue of symmetric `a`, by cyclic
// Jacobi rotations.
__device__ void smallest_eigenvector(double a[3][3], double out[3]) {
  double v[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  for (int sweep = 0; sweep < 32; ++sweep) {
    const double off = fabs(a[0][1]) + fabs(a[0][2]) + fabs(a[1][2]);
    if (off <= 1e-300) {
      break;
    }
    for (int p = 0; p < 2; ++p) {
      for (int q = p + 1; q < 3; ++q) {
        if (a[p][q] == 0.0) {
          continue;
        }
        const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        const double t = (theta >= 0.0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
        const double c = 1.0 / sqrt(t * t + 1.0);
        const double s = t * c;
        for (int k = 0; k < 3; ++k) {
          const double akp = a[k][p];
          const double akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < 3; ++k) {
          const double apk = a[p][k];
          const double aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 3; ++k) {
          const double vkp = v[k][p];
          const double vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  int smallest = 0;
  for (int k = 1; k < 3; ++k) {
    if (a[k][k] < a[smallest][smallest]) {
      smallest = k;
    }
  }
  for (int k = 0; k < 3; ++k) {
    out[k] = v[k][smallest];
  }
}

// Sums the blocks' partials in order and replaces the plane by the
// least-squares plane of its inliers, unless that one is too steep.
__global__ void refit_kernel(const double* __restrict__ partials, unsigned int blocks, float3 up,
                             float min_cos, float4* __restrict__ plane) {
  double s[kSums] = {};
  for (unsigned int b = 0; b < blocks; ++b) {
    for (int k = 0; k < kSums; ++k) {
      s[k] += partials[b * kSums + k];
    }
  }
  const double n = s[0];
  if (n < 3.0) {
    return;
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
  smallest_eigenvector(cov, normal);
  const double length = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
  if (!(length > 0.0)) {
    return;
  }
  double along = (normal[0] * up.x + normal[1] * up.y + normal[2] * up.z) / length;
  const double sign = along < 0.0 ? -1.0 : 1.0;
  along *= sign;
  if (along < min_cos) {
    return;
  }
  for (double& value : normal) {
    value *= sign / length;
  }
  const double d = -(normal[0] * mean[0] + normal[1] * mean[1] + normal[2] * mean[2]);
  *plane = make_float4(static_cast<float>(normal[0]), static_cast<float>(normal[1]),
                       static_cast<float>(normal[2]), static_cast<float>(d));
}

__global__ void height_plane_kernel(float3 up, float height, float4* __restrict__ plane) {
  *plane = make_float4(up.x, up.y, up.z, -height);
}

// Ground: within the threshold of the plane (RANSAC), or at or below it
// (height). Without a plane, every valid point stays.
__global__ void mark_kernel(const float* __restrict__ xyz, std::uint32_t count,
                            const float4* __restrict__ plane_in, bool height_mode, float threshold,
                            std::uint8_t* __restrict__ keep, std::uint32_t* __restrict__ ground) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const float4 plane = *plane_in;
  const float x = xyz[3 * i];
  const float y = xyz[3 * i + 1];
  const float z = xyz[3 * i + 2];
  bool on_ground = false;
  if (valid_point(x, y, z) && isfinite(plane.x)) {
    on_ground = height_mode ? plane.x * x + plane.y * y + plane.z * z + plane.w <= 0.0F
                            : near_plane(plane, x, y, z, threshold);
  }
  keep[i] = valid_point(x, y, z) && !on_ground ? 1U : 0U;
  if (on_ground && ground != nullptr) {
    atomicAdd(ground, 1U);
  }
}

}  // namespace

struct GroundSegmenter::Impl {
  detail::DeviceBuffer flags;
  detail::DeviceBuffer offsets;
  detail::DeviceBuffer indices;
  detail::DeviceBuffer valid;
  detail::DeviceBuffer planes;
  detail::DeviceBuffer scores;
  detail::DeviceBuffer partials;
  detail::DeviceBuffer temp;
  int score_blocks{};
};

GroundSegmenter::GroundSegmenter() : impl_(std::make_unique<Impl>()) {}
GroundSegmenter::~GroundSegmenter() = default;

cudaError_t GroundSegmenter::segment(const float* xyz, std::uint32_t count,
                                     const GroundParams& params, float* plane_out,
                                     std::uint8_t* keep, std::uint32_t* ground_count,
                                     cudaStream_t stream) {
  const float up_length = std::sqrt(params.up[0] * params.up[0] + params.up[1] * params.up[1] +
                                    params.up[2] * params.up[2]);
  if (!(up_length > 0.0F) || !std::isfinite(up_length) || count > kMaxPointsLimit ||
      plane_out == nullptr) {
    return cudaErrorInvalidValue;
  }
  const bool ransac = params.mode == GroundMode::kRansac;
  if (ransac && (params.max_iterations < 1U || params.max_iterations > kMaxIterationsLimit ||
                 !(params.distance_threshold >= 0.0F) || !(params.max_tilt_deg >= 0.0F) ||
                 !(params.max_tilt_deg <= 90.0F))) {
    return cudaErrorInvalidValue;
  }
  const float3 up =
      make_float3(params.up[0] / up_length, params.up[1] / up_length, params.up[2] / up_length);
  auto* plane = reinterpret_cast<float4*>(plane_out);
  if (reinterpret_cast<std::uintptr_t>(plane_out) % alignof(float4) != 0U) {
    return cudaErrorInvalidValue;
  }
  if (ground_count != nullptr) {
    if (const cudaError_t status = cudaMemsetAsync(ground_count, 0, sizeof(std::uint32_t), stream);
        status != cudaSuccess) {
      return status;
    }
  }

  if (!ransac) {
    height_plane_kernel<<<1, 1, 0, stream>>>(up, params.height, plane);
  } else {
    Impl& s = *impl_;
    const std::size_t n = count;
    const std::uint32_t iterations = params.max_iterations;
    for (auto [buffer, bytes] :
         {std::pair{&s.flags, n * 4U}, std::pair{&s.offsets, n * 4U}, std::pair{&s.indices, n * 4U},
          std::pair{&s.valid, std::size_t{4}},
          std::pair{&s.planes, std::size_t{iterations} * sizeof(float4)},
          std::pair{&s.scores, std::size_t{iterations} * 4U},
          std::pair{&s.partials, std::size_t{kSumBlocks} * kSums * sizeof(double)}}) {
      if (const cudaError_t status = buffer->reserve(bytes, stream); status != cudaSuccess) {
        return status;
      }
    }
    if (s.score_blocks == 0) {
      int device = 0;
      int multiprocessors = 0;
      if (const cudaError_t status = cudaGetDevice(&device); status != cudaSuccess) {
        return status;
      }
      if (const cudaError_t status =
              cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device);
          status != cudaSuccess) {
        return status;
      }
      s.score_blocks = 4 * multiprocessors;
    }
    if (count == 0U) {
      if (const cudaError_t status = cudaMemsetAsync(s.valid.as<void>(), 0, 4U, stream);
          status != cudaSuccess) {
        return status;
      }
    } else {
      valid_flags_kernel<<<blocks_for(count), kBlock, 0, stream>>>(xyz, count,
                                                                   s.flags.as<std::uint32_t>());
      if (const cudaError_t status = detail::exclusive_sum(
              s.flags.as<std::uint32_t>(), s.offsets.as<std::uint32_t>(), count, s.temp, stream);
          status != cudaSuccess) {
        return status;
      }
      valid_indices_kernel<<<blocks_for(count), kBlock, 0, stream>>>(
          s.flags.as<std::uint32_t>(), s.offsets.as<std::uint32_t>(), count,
          s.indices.as<std::uint32_t>(), s.valid.as<std::uint32_t>());
    }
    const auto min_cos = static_cast<float>(
        std::cos(static_cast<double>(params.max_tilt_deg) * std::numbers::pi / 180.0));
    hypotheses_kernel<<<blocks_for(iterations), kBlock, 0, stream>>>(
        xyz, s.indices.as<std::uint32_t>(), s.valid.as<std::uint32_t>(), iterations, params.seed,
        up, min_cos, s.planes.as<float4>());
    if (const cudaError_t status =
            cudaMemsetAsync(s.scores.as<void>(), 0, std::size_t{iterations} * 4U, stream);
        status != cudaSuccess) {
      return status;
    }
    if (count != 0U) {
      const dim3 grid(std::min(blocks_for(count), static_cast<unsigned int>(s.score_blocks)),
                      blocks_for(iterations));
      score_kernel<<<grid, kBlock, 0, stream>>>(xyz, count, s.planes.as<float4>(), iterations,
                                                params.distance_threshold,
                                                s.scores.as<std::uint32_t>());
    }
    select_kernel<<<1, kBlock, 0, stream>>>(s.planes.as<float4>(), s.scores.as<std::uint32_t>(),
                                            iterations, plane);
    if (params.refine && count != 0U) {
      sums_kernel<<<kSumBlocks, kBlock, 0, stream>>>(xyz, count, plane, params.distance_threshold,
                                                     s.partials.as<double>());
      refit_kernel<<<1, 1, 0, stream>>>(s.partials.as<double>(), kSumBlocks, up, min_cos, plane);
    }
  }
  if (count != 0U) {
    mark_kernel<<<blocks_for(count), kBlock, 0, stream>>>(
        xyz, count, plane, !ransac, params.distance_threshold, keep, ground_count);
  }
  return cudaGetLastError();
}

}  // namespace holoscan::pcl_filters
