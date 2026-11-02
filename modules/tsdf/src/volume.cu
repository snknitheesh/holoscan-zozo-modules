// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// TsdfVolume: allocation, reset, integration and ray casting. Compiled
// without fused multiply-adds, so that the integration computes bit for bit
// what tests/reference.hpp computes on the host.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/tsdf/marching_cubes.hpp>
#include <holoscan/tsdf/tsdf_volume.hpp>

#include "device_buffer.hpp"
#include "ply.hpp"

namespace holoscan::tsdf {
namespace {

constexpr int kBlock = 256;

bool finite_positive(float value) { return std::isfinite(value) && value > 0.0F; }

bool valid_intrinsics(const CameraIntrinsics& k) {
  return finite_positive(k.fx) && finite_positive(k.fy) && std::isfinite(k.cx) &&
         std::isfinite(k.cy);
}

// The kernels' view of a volume.
struct Grid {
  int nx{};
  int ny{};
  int nz{};
  float origin[3]{};
  float voxel{};
  float truncation{};
  float weight_cap{};
  float* tsdf{};
  float* weight{};
  uchar4* rgba{};
};

__global__ void reset_kernel(float* __restrict__ tsdf, float* __restrict__ weight,
                             uchar4* __restrict__ rgba, std::uint64_t count) {
  const std::uint64_t i = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  tsdf[i] = 1.0F;
  weight[i] = 0.0F;
  if (rgba != nullptr) {
    rgba[i] = make_uchar4(0U, 0U, 0U, 0U);
  }
}

struct IntegrateArgs {
  Grid grid;
  float camera_from_world[12]{};
  CameraIntrinsics k;
  int width{};
  int height{};
  const void* depth{};
  DepthType type{};
  float scale{};
  float depth_min{};
  float depth_max{};
  const std::uint8_t* color{};
  int channels{};
  bool bgr{};
  // The box of voxels visited.
  int box_min[3]{};
  int box_size[3]{};
};

// One thread per voxel of the box. Each voxel is written by one thread only,
// so the result does not depend on scheduling. tests/reference.hpp repeats
// this arithmetic in the same order.
__global__ void integrate_kernel(IntegrateArgs a) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto bx = static_cast<std::uint32_t>(a.box_size[0]);
  const auto by = static_cast<std::uint32_t>(a.box_size[1]);
  const auto bz = static_cast<std::uint32_t>(a.box_size[2]);
  if (index >= bx * by * bz) {
    return;
  }
  const int x = a.box_min[0] + static_cast<int>(index % bx);
  const int y = a.box_min[1] + static_cast<int>((index / bx) % by);
  const int z = a.box_min[2] + static_cast<int>(index / (bx * by));
  const Grid& g = a.grid;
  const float px = g.origin[0] + g.voxel * static_cast<float>(x);
  const float py = g.origin[1] + g.voxel * static_cast<float>(y);
  const float pz = g.origin[2] + g.voxel * static_cast<float>(z);
  const float* m = a.camera_from_world;
  const float xc = m[0] * px + m[1] * py + m[2] * pz + m[3];
  const float yc = m[4] * px + m[5] * py + m[6] * pz + m[7];
  const float zc = m[8] * px + m[9] * py + m[10] * pz + m[11];
  if (!(zc > 0.0F)) {
    return;
  }
  const float u = a.k.fx * xc / zc + a.k.cx;
  const float v = a.k.fy * yc / zc + a.k.cy;
  const float uf = floorf(u + 0.5F);
  const float vf = floorf(v + 0.5F);
  if (!(uf >= 0.0F && uf < static_cast<float>(a.width) && vf >= 0.0F &&
        vf < static_cast<float>(a.height))) {
    return;
  }
  const std::size_t pixel = static_cast<std::size_t>(vf) * static_cast<std::size_t>(a.width) +
                            static_cast<std::size_t>(uf);
  const float raw = a.type == DepthType::kUint16
                        ? static_cast<float>(static_cast<const std::uint16_t*>(a.depth)[pixel])
                        : static_cast<const float*>(a.depth)[pixel];
  const float d = raw * a.scale;
  if (!(d > 0.0F && d >= a.depth_min && d <= a.depth_max)) {
    return;
  }
  const float sdf = d - zc;
  if (sdf < -g.truncation) {
    return;
  }
  const float t = fminf(1.0F, sdf / g.truncation);
  const std::size_t i = static_cast<std::size_t>(x) +
                        static_cast<std::size_t>(g.nx) *
                            (static_cast<std::size_t>(y) +
                             static_cast<std::size_t>(g.ny) * static_cast<std::size_t>(z));
  const float w = g.weight[i];
  const float w1 = w + 1.0F;
  g.tsdf[i] = (g.tsdf[i] * w + t) / w1;
  g.weight[i] = fminf(w1, g.weight_cap);
  if (g.rgba != nullptr && a.color != nullptr) {
    const std::uint8_t* c = a.color + pixel * static_cast<std::size_t>(a.channels);
    const float rgb[3] = {static_cast<float>(c[a.bgr ? 2 : 0]), static_cast<float>(c[1]),
                          static_cast<float>(c[a.bgr ? 0 : 2])};
    const uchar4 old = g.rgba[i];
    const float before[3] = {static_cast<float>(old.x), static_cast<float>(old.y),
                             static_cast<float>(old.z)};
    std::uint8_t out[3];
    for (int ch = 0; ch < 3; ++ch) {
      const float mixed = (before[ch] * w + rgb[ch]) / w1;
      out[ch] = static_cast<std::uint8_t>(fminf(255.0F, floorf(mixed + 0.5F)));
    }
    g.rgba[i] = make_uchar4(out[0], out[1], out[2], 255U);
  }
}

// The trilinear distance at grid point (gx, gy, gz) in voxel units, and the
// eight corners it came from; false where a corner is unobserved or the
// point is outside the volume.
struct Cell {
  int x{};
  int y{};
  int z{};
  float fx{};
  float fy{};
  float fz{};
  float c[8]{};  // Corner (dx, dy, dz) at c[dx + 2 * dy + 4 * dz].
};

__device__ bool sample(const Grid& g, float gx, float gy, float gz, Cell& cell, float& value) {
  const float bx = floorf(gx);
  const float by = floorf(gy);
  const float bz = floorf(gz);
  if (!(bx >= 0.0F && by >= 0.0F && bz >= 0.0F && bx <= static_cast<float>(g.nx - 2) &&
        by <= static_cast<float>(g.ny - 2) && bz <= static_cast<float>(g.nz - 2))) {
    return false;
  }
  cell.x = static_cast<int>(bx);
  cell.y = static_cast<int>(by);
  cell.z = static_cast<int>(bz);
  cell.fx = gx - bx;
  cell.fy = gy - by;
  cell.fz = gz - bz;
  const std::size_t sx = 1;
  const std::size_t sy = static_cast<std::size_t>(g.nx);
  const std::size_t sz = sy * static_cast<std::size_t>(g.ny);
  const std::size_t base = static_cast<std::size_t>(cell.x) +
                           sy * static_cast<std::size_t>(cell.y) +
                           sz * static_cast<std::size_t>(cell.z);
  for (int corner = 0; corner < 8; ++corner) {
    const std::size_t i = base + ((corner & 1) != 0 ? sx : 0) + ((corner & 2) != 0 ? sy : 0) +
                          ((corner & 4) != 0 ? sz : 0);
    if (!(g.weight[i] > 0.0F)) {
      return false;
    }
    cell.c[corner] = g.tsdf[i];
  }
  const float* c = cell.c;
  const float x00 = c[0] + (c[1] - c[0]) * cell.fx;
  const float x10 = c[2] + (c[3] - c[2]) * cell.fx;
  const float x01 = c[4] + (c[5] - c[4]) * cell.fx;
  const float x11 = c[6] + (c[7] - c[6]) * cell.fx;
  const float y0 = x00 + (x10 - x00) * cell.fy;
  const float y1 = x01 + (x11 - x01) * cell.fy;
  value = y0 + (y1 - y0) * cell.fz;
  return true;
}

struct RaycastArgs {
  Grid grid;
  float world_from_camera[12]{};
  RaycastParams params;
  RaycastImages images;
};

__global__ void raycast_kernel(RaycastArgs a) {
  const int u = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int v = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const RaycastParams& p = a.params;
  if (u >= p.width || v >= p.height) {
    return;
  }
  const Grid& g = a.grid;
  const float* m = a.world_from_camera;
  // The ray in the camera frame with z 1, so the ray parameter is the depth.
  const float ray[3] = {(static_cast<float>(u) - p.intrinsics.cx) / p.intrinsics.fx,
                        (static_cast<float>(v) - p.intrinsics.cy) / p.intrinsics.fy, 1.0F};
  float go[3];  // Ray origin and direction in voxel units.
  float gd[3];
  for (int r = 0; r < 3; ++r) {
    const float d = m[4 * r] * ray[0] + m[4 * r + 1] * ray[1] + m[4 * r + 2] * ray[2];
    go[r] = (m[4 * r + 3] - g.origin[r]) / g.voxel;
    gd[r] = d / g.voxel;
  }
  // Clip the ray to the cells of the volume and the depth range.
  float t0 = p.depth_min;
  float t1 = p.depth_max;
  const float extent[3] = {static_cast<float>(g.nx - 1), static_cast<float>(g.ny - 1),
                           static_cast<float>(g.nz - 1)};
  for (int r = 0; r < 3; ++r) {
    if (gd[r] == 0.0F) {
      if (go[r] < 0.0F || go[r] > extent[r]) {
        t1 = -1.0F;
      }
      continue;
    }
    float ta = (0.0F - go[r]) / gd[r];
    float tb = (extent[r] - go[r]) / gd[r];
    if (ta > tb) {
      const float swap = ta;
      ta = tb;
      tb = swap;
    }
    t0 = fmaxf(t0, ta);
    t1 = fminf(t1, tb);
  }
  const std::size_t pixel =
      static_cast<std::size_t>(v) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(u);
  float hit_t = 0.0F;
  Cell hit_cell;
  bool hit = false;
  if (t0 < t1) {
    // Steps in t: a voxel of world distance is 1 / |gd| of t.
    const float norm = sqrtf(gd[0] * gd[0] + gd[1] * gd[1] + gd[2] * gd[2]);
    const float voxel_t = 1.0F / norm;
    const float band = g.truncation / g.voxel;  // The truncation in voxels.
    float t = t0;
    float prev_t = 0.0F;
    float prev_f = 0.0F;
    bool prev_valid = false;
    Cell cell;
    while (t <= t1) {
      float f = 0.0F;
      const bool valid =
          sample(g, go[0] + t * gd[0], go[1] + t * gd[1], go[2] + t * gd[2], cell, f);
      if (valid && prev_valid && prev_f > 0.0F && f <= 0.0F) {
        hit_t = prev_t + (t - prev_t) * prev_f / (prev_f - f);
        float refined = 0.0F;
        if (!sample(g, go[0] + hit_t * gd[0], go[1] + hit_t * gd[1], go[2] + hit_t * gd[2],
                    hit_cell, refined)) {
          hit_cell = cell;
        }
        hit = true;
        break;
      }
      prev_valid = valid;
      prev_f = f;
      prev_t = t;
      // Half a voxel near and behind surfaces, up to half the truncation in
      // the open; a full voxel through unobserved space.
      const float step = !valid ? 1.0F : f > 0.0F ? fmaxf(0.5F, 0.5F * f * band) : 0.5F;
      t += step * voxel_t;
    }
  }
  if (!hit) {
    if (a.images.depth != nullptr) {
      a.images.depth[pixel] = 0.0F;
    }
    if (a.images.normals != nullptr) {
      const float nan = __int_as_float(0x7FC00000);
      for (int r = 0; r < 3; ++r) {
        a.images.normals[3 * pixel + r] = nan;
      }
    }
    if (a.images.colors != nullptr) {
      for (int r = 0; r < 3; ++r) {
        a.images.colors[3 * pixel + r] = 0U;
      }
    }
    return;
  }
  if (a.images.depth != nullptr) {
    a.images.depth[pixel] = hit_t;
  }
  if (a.images.normals != nullptr) {
    // The gradient of the trilinear interpolant in the hit's cell, which
    // points out of the surface, toward the camera.
    const float* c = hit_cell.c;
    const float fx = hit_cell.fx;
    const float fy = hit_cell.fy;
    const float fz = hit_cell.fz;
    const float n_world[3] = {
        (1.0F - fy) * (1.0F - fz) * (c[1] - c[0]) + fy * (1.0F - fz) * (c[3] - c[2]) +
            (1.0F - fy) * fz * (c[5] - c[4]) + fy * fz * (c[7] - c[6]),
        (1.0F - fx) * (1.0F - fz) * (c[2] - c[0]) + fx * (1.0F - fz) * (c[3] - c[1]) +
            (1.0F - fx) * fz * (c[6] - c[4]) + fx * fz * (c[7] - c[5]),
        (1.0F - fx) * (1.0F - fy) * (c[4] - c[0]) + fx * (1.0F - fy) * (c[5] - c[1]) +
            (1.0F - fx) * fy * (c[6] - c[2]) + fx * fy * (c[7] - c[3])};
    // Into the camera frame: the transpose of the pose's rotation.
    float n[3];
    for (int r = 0; r < 3; ++r) {
      n[r] = m[r] * n_world[0] + m[4 + r] * n_world[1] + m[8 + r] * n_world[2];
    }
    const float length = sqrtf(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    const float nan = __int_as_float(0x7FC00000);
    const bool usable = length > 0.0F;
    // Facing the camera: against the ray.
    const float sign = n[0] * ray[0] + n[1] * ray[1] + n[2] * ray[2] > 0.0F ? -1.0F : 1.0F;
    for (int r = 0; r < 3; ++r) {
      a.images.normals[3 * pixel + r] = usable ? sign * n[r] / length : nan;
    }
  }
  if (a.images.colors != nullptr) {
    const float gx = go[0] + hit_t * gd[0];
    const float gy = go[1] + hit_t * gd[1];
    const float gz = go[2] + hit_t * gd[2];
    const int x = min(max(static_cast<int>(floorf(gx + 0.5F)), 0), g.nx - 1);
    const int y = min(max(static_cast<int>(floorf(gy + 0.5F)), 0), g.ny - 1);
    const int z = min(max(static_cast<int>(floorf(gz + 0.5F)), 0), g.nz - 1);
    const uchar4 rgba = g.rgba[static_cast<std::size_t>(x) +
                               static_cast<std::size_t>(g.nx) *
                                   (static_cast<std::size_t>(y) +
                                    static_cast<std::size_t>(g.ny) * static_cast<std::size_t>(z))];
    a.images.colors[3 * pixel] = rgba.x;
    a.images.colors[3 * pixel + 1] = rgba.y;
    a.images.colors[3 * pixel + 2] = rgba.z;
  }
}

std::string mib(std::uint64_t bytes) {
  return std::to_string((bytes + (std::uint64_t{1} << 20) - 1) >> 20) + " MiB";
}

}  // namespace

bool valid_pose(const Matrix4& pose, float tolerance) {
  for (const float value : pose) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  if (pose[12] != 0.0F || pose[13] != 0.0F || pose[14] != 0.0F || pose[15] != 1.0F) {
    return false;
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      double dot = 0.0;
      for (int r = 0; r < 3; ++r) {
        dot += static_cast<double>(pose[4 * r + a]) * static_cast<double>(pose[4 * r + b]);
      }
      if (std::abs(dot - (a == b ? 1.0 : 0.0)) > tolerance) {
        return false;
      }
    }
  }
  const auto at = [&pose](int r, int c) { return static_cast<double>(pose[4 * r + c]); };
  const double det = at(0, 0) * (at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1)) -
                     at(0, 1) * (at(1, 0) * at(2, 2) - at(1, 2) * at(2, 0)) +
                     at(0, 2) * (at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0));
  return det > 0.0;
}

Matrix4 inverse_pose(const Matrix4& pose) {
  Matrix4 inverse = identity_pose();
  for (int r = 0; r < 3; ++r) {
    double translation = 0.0;
    for (int c = 0; c < 3; ++c) {
      inverse[4 * r + c] = pose[4 * c + r];
      translation -= static_cast<double>(pose[4 * c + r]) * static_cast<double>(pose[4 * c + 3]);
    }
    inverse[4 * r + 3] = static_cast<float>(translation);
  }
  return inverse;
}

void validate_config(const TsdfVolumeConfig& config) {
  std::uint64_t voxels = 1;
  for (const int dim : config.dims) {
    if (dim < 2 || dim > 1024) {
      throw std::invalid_argument("dims must each be 2..1024");
    }
    voxels *= static_cast<std::uint64_t>(dim);
  }
  if (voxels > kMaxVoxels) {
    throw std::invalid_argument("a volume holds at most 2^30 voxels");
  }
  if (!finite_positive(config.voxel_size)) {
    throw std::invalid_argument("voxel_size must be finite and positive");
  }
  for (const float o : config.origin) {
    if (!std::isfinite(o)) {
      throw std::invalid_argument("origin must be finite");
    }
  }
  if (!(std::isfinite(config.truncation) && config.truncation >= config.voxel_size)) {
    throw std::invalid_argument("truncation must be finite and at least one voxel");
  }
  if (!(config.weight_cap >= 1.0F && config.weight_cap <= 65535.0F)) {
    throw std::invalid_argument("weight_cap must be 1..65535");
  }
}

std::uint64_t volume_bytes(const TsdfVolumeConfig& config) {
  const std::uint64_t voxels = static_cast<std::uint64_t>(config.dims[0]) *
                               static_cast<std::uint64_t>(config.dims[1]) *
                               static_cast<std::uint64_t>(config.dims[2]);
  return voxels * (config.color ? 12U : 8U);
}

void validate_params(const IntegrateParams& params) {
  if (!valid_intrinsics(params.intrinsics)) {
    throw std::invalid_argument("intrinsics need finite, positive fx and fy and finite cx, cy");
  }
  if (!(std::isfinite(params.depth_min) && std::isfinite(params.depth_max) &&
        params.depth_min >= 0.0F && params.depth_max > params.depth_min)) {
    throw std::invalid_argument("the depth range needs 0 <= depth_min < depth_max, finite");
  }
}

void validate_params(const RaycastParams& params) {
  if (!valid_intrinsics(params.intrinsics)) {
    throw std::invalid_argument("intrinsics need finite, positive fx and fy and finite cx, cy");
  }
  if (params.width < 1 || params.height < 1 || params.width > 8192 || params.height > 8192) {
    throw std::invalid_argument("the ray-cast image must be 1x1 to 8192x8192");
  }
  if (!(std::isfinite(params.depth_min) && std::isfinite(params.depth_max) &&
        params.depth_min > 0.0F && params.depth_max > params.depth_min)) {
    throw std::invalid_argument("the ray-cast depth range needs 0 < depth_min < depth_max");
  }
}

struct TsdfVolume::Impl {
  float* tsdf{};
  float* weight{};
  uchar4* rgba{};
  cudaEvent_t last{};
  bool recorded{};
  std::mutex mutex;

  ~Impl() {
    // cudaFree waits for the work that still uses the memory.
    cudaFree(tsdf);
    cudaFree(weight);
    cudaFree(rgba);
    if (last != nullptr) {
      cudaEventDestroy(last);
    }
  }

  Grid grid(const TsdfVolumeConfig& config) const {
    Grid g;
    g.nx = config.dims[0];
    g.ny = config.dims[1];
    g.nz = config.dims[2];
    for (int r = 0; r < 3; ++r) {
      g.origin[r] = config.origin[r];
    }
    g.voxel = config.voxel_size;
    g.truncation = config.truncation;
    g.weight_cap = config.weight_cap;
    g.tsdf = tsdf;
    g.weight = weight;
    g.rgba = rgba;
    return g;
  }
};

namespace {

// Holds the volume for one operation enqueued on a stream.
class Access {
 public:
  Access(const TsdfVolume& volume, cudaStream_t stream,
         cudaError_t (TsdfVolume::*begin)(cudaStream_t) const,
         cudaError_t (TsdfVolume::*end)(cudaStream_t) const)
      : volume_(volume), stream_(stream), end_(end), status_((volume.*begin)(stream)) {}
  ~Access() {
    if (status_ == cudaSuccess) {
      static_cast<void>((volume_.*end_)(stream_));
    }
  }
  Access(const Access&) = delete;
  Access& operator=(const Access&) = delete;

  [[nodiscard]] cudaError_t status() const noexcept { return status_; }
  // Ends the access early, returning the first failure of the operation or
  // of recording its end.
  cudaError_t finish(cudaError_t launched) {
    const cudaError_t ended = (volume_.*end_)(stream_);
    status_ = cudaErrorNotReady;  // Ended.
    return launched != cudaSuccess ? launched : ended;
  }

 private:
  const TsdfVolume& volume_;
  cudaStream_t stream_;
  cudaError_t (TsdfVolume::*end_)(cudaStream_t) const;
  cudaError_t status_;
};

}  // namespace

TsdfVolume::TsdfVolume(TsdfVolumeConfig config) : config_(config), impl_(std::make_unique<Impl>()) {
  validate_config(config_);
  if (const cudaError_t status = cudaGetDevice(&device_); status != cudaSuccess) {
    throw std::runtime_error(std::string("TsdfVolume: cudaGetDevice: ") +
                             cudaGetErrorString(status));
  }
  const std::uint64_t voxels = voxel_count();
  const auto allocate = [&](void** where, std::uint64_t bytes) {
    if (const cudaError_t status = cudaMalloc(where, bytes); status != cudaSuccess) {
      cudaGetLastError();
      std::size_t free = 0;
      std::size_t total = 0;
      cudaMemGetInfo(&free, &total);
      throw std::runtime_error(
          "TsdfVolume: allocating " + mib(volume_bytes(config_)) + " for a " +
          std::to_string(config_.dims[0]) + "x" + std::to_string(config_.dims[1]) + "x" +
          std::to_string(config_.dims[2]) + " volume failed (" + cudaGetErrorString(status) +
          "); the GPU had " + mib(free) + " free of " + mib(total));
    }
  };
  allocate(reinterpret_cast<void**>(&impl_->tsdf), voxels * sizeof(float));
  allocate(reinterpret_cast<void**>(&impl_->weight), voxels * sizeof(float));
  if (config_.color) {
    allocate(reinterpret_cast<void**>(&impl_->rgba), voxels * sizeof(uchar4));
  }
  if (const cudaError_t status = cudaEventCreateWithFlags(&impl_->last, cudaEventDisableTiming);
      status != cudaSuccess) {
    impl_->last = nullptr;
    throw std::runtime_error(std::string("TsdfVolume: cudaEventCreate: ") +
                             cudaGetErrorString(status));
  }
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
    throw std::runtime_error("TsdfVolume: cudaStreamCreate failed");
  }
  const cudaError_t reset_status = reset(stream);
  const cudaError_t sync_status = cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  if (reset_status != cudaSuccess || sync_status != cudaSuccess) {
    throw std::runtime_error(
        std::string("TsdfVolume: reset failed: ") +
        cudaGetErrorString(reset_status != cudaSuccess ? reset_status : sync_status));
  }
}

TsdfVolume::~TsdfVolume() = default;

std::uint64_t TsdfVolume::voxel_count() const noexcept {
  return static_cast<std::uint64_t>(config_.dims[0]) * static_cast<std::uint64_t>(config_.dims[1]) *
         static_cast<std::uint64_t>(config_.dims[2]);
}

TsdfVolume::DeviceView TsdfVolume::view() const noexcept {
  return DeviceView{impl_->tsdf, impl_->weight, reinterpret_cast<const std::uint8_t*>(impl_->rgba)};
}

cudaError_t TsdfVolume::begin_access(cudaStream_t stream) const {
  int current = 0;
  if (const cudaError_t status = cudaGetDevice(&current); status != cudaSuccess) {
    return status;
  }
  if (current != device_) {
    return cudaErrorInvalidDevice;
  }
  if (stream != nullptr) {
    int device = 0;
    if (const cudaError_t status = cudaStreamGetDevice(stream, &device); status != cudaSuccess) {
      return status;
    }
    if (device != device_) {
      return cudaErrorInvalidDevice;
    }
  }
  impl_->mutex.lock();
  if (impl_->recorded) {
    if (const cudaError_t status = cudaStreamWaitEvent(stream, impl_->last, 0);
        status != cudaSuccess) {
      impl_->mutex.unlock();
      return status;
    }
  }
  return cudaSuccess;
}

cudaError_t TsdfVolume::end_access(cudaStream_t stream) const {
  const cudaError_t status = cudaEventRecord(impl_->last, stream);
  impl_->recorded = impl_->recorded || status == cudaSuccess;
  impl_->mutex.unlock();
  return status;
}

cudaError_t TsdfVolume::reset(cudaStream_t stream) {
  Access access(*this, stream, &TsdfVolume::begin_access, &TsdfVolume::end_access);
  if (access.status() != cudaSuccess) {
    return access.status();
  }
  const std::uint64_t voxels = voxel_count();
  reset_kernel<<<static_cast<unsigned>((voxels + kBlock - 1) / kBlock), kBlock, 0, stream>>>(
      impl_->tsdf, impl_->weight, impl_->rgba, voxels);
  return access.finish(cudaGetLastError());
}

cudaError_t TsdfVolume::integrate(const DepthImage& depth, const ColorImage* color,
                                  const Matrix4& world_from_camera, const IntegrateParams& params,
                                  cudaStream_t stream) {
  if (depth.width < 0 || depth.height < 0 ||
      (depth.data == nullptr && depth.width * depth.height > 0) || !finite_positive(depth.scale) ||
      !valid_intrinsics(params.intrinsics) ||
      !(params.depth_min >= 0.0F && params.depth_max > params.depth_min) ||
      !valid_pose(world_from_camera)) {
    return cudaErrorInvalidValue;
  }
  if (color != nullptr && color->data != nullptr && color->channels != 3 && color->channels != 4) {
    return cudaErrorInvalidValue;
  }
  IntegrateArgs a;
  a.grid = impl_->grid(config_);
  const Matrix4 camera_from_world = inverse_pose(world_from_camera);
  for (int i = 0; i < 12; ++i) {
    a.camera_from_world[i] = camera_from_world[i];
  }
  a.k = params.intrinsics;
  a.width = depth.width;
  a.height = depth.height;
  a.depth = depth.data;
  a.type = depth.type;
  a.scale = depth.scale;
  a.depth_min = params.depth_min;
  a.depth_max = params.depth_max;
  if (color != nullptr && color->data != nullptr && config_.color) {
    a.color = color->data;
    a.channels = color->channels;
    a.bgr = color->bgr;
  }

  // The box of voxels the camera can update: within its view, from its
  // center out to depth_max plus the truncation, with a voxel to spare.
  double lo[3];
  double hi[3];
  for (int r = 0; r < 3; ++r) {
    lo[r] = hi[r] = world_from_camera[4 * r + 3];
  }
  const double far = static_cast<double>(params.depth_max) + config_.truncation;
  const CameraIntrinsics& k = params.intrinsics;
  for (const double u : {-0.5, static_cast<double>(depth.width) - 0.5}) {
    for (const double v : {-0.5, static_cast<double>(depth.height) - 0.5}) {
      const double camera[3] = {(u - k.cx) / k.fx * far, (v - k.cy) / k.fy * far, far};
      for (int r = 0; r < 3; ++r) {
        const double world =
            world_from_camera[4 * r] * camera[0] + world_from_camera[4 * r + 1] * camera[1] +
            world_from_camera[4 * r + 2] * camera[2] + world_from_camera[4 * r + 3];
        lo[r] = std::min(lo[r], world);
        hi[r] = std::max(hi[r], world);
      }
    }
  }
  std::uint64_t count = 1;
  for (int r = 0; r < 3; ++r) {
    const double first = std::floor((lo[r] - config_.origin[r]) / config_.voxel_size) - 1.0;
    const double last = std::ceil((hi[r] - config_.origin[r]) / config_.voxel_size) + 1.0;
    const int from = static_cast<int>(std::clamp(first, 0.0, static_cast<double>(config_.dims[r])));
    const int to =
        static_cast<int>(std::clamp(last, -1.0, static_cast<double>(config_.dims[r] - 1)));
    a.box_min[r] = from;
    a.box_size[r] = std::max(0, to - from + 1);
    count *= static_cast<std::uint64_t>(a.box_size[r]);
  }

  Access access(*this, stream, &TsdfVolume::begin_access, &TsdfVolume::end_access);
  if (access.status() != cudaSuccess) {
    return access.status();
  }
  if (count == 0 || depth.width == 0 || depth.height == 0) {
    return access.finish(cudaSuccess);
  }
  integrate_kernel<<<static_cast<unsigned>((count + kBlock - 1) / kBlock), kBlock, 0, stream>>>(a);
  return access.finish(cudaGetLastError());
}

cudaError_t TsdfVolume::raycast(const Matrix4& world_from_camera, const RaycastParams& params,
                                const RaycastImages& images, cudaStream_t stream) {
  if (!valid_pose(world_from_camera) || !valid_intrinsics(params.intrinsics) || params.width < 1 ||
      params.height < 1 || !(params.depth_min > 0.0F && params.depth_max > params.depth_min) ||
      (images.colors != nullptr && !config_.color)) {
    return cudaErrorInvalidValue;
  }
  RaycastArgs a;
  a.grid = impl_->grid(config_);
  for (int i = 0; i < 12; ++i) {
    a.world_from_camera[i] = world_from_camera[i];
  }
  a.params = params;
  a.images = images;
  Access access(*this, stream, &TsdfVolume::begin_access, &TsdfVolume::end_access);
  if (access.status() != cudaSuccess) {
    return access.status();
  }
  const dim3 block(16, 16);
  const dim3 grid((params.width + 15) / 16, (params.height + 15) / 16);
  raycast_kernel<<<grid, block, 0, stream>>>(a);
  return access.finish(cudaGetLastError());
}

VolumeData TsdfVolume::download() const {
  const std::uint64_t voxels = voxel_count();
  VolumeData data;
  data.tsdf.resize(voxels);
  data.weight.resize(voxels);
  std::vector<uchar4> rgba(config_.color ? voxels : 0U);
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
    throw std::runtime_error("TsdfVolume::download: cudaStreamCreate failed");
  }
  cudaError_t status = cudaSuccess;
  {
    Access access(*this, stream, &TsdfVolume::begin_access, &TsdfVolume::end_access);
    status = access.status();
    if (status == cudaSuccess) {
      status = cudaMemcpyAsync(data.tsdf.data(), impl_->tsdf, voxels * sizeof(float),
                               cudaMemcpyDeviceToHost, stream);
      if (status == cudaSuccess) {
        status = cudaMemcpyAsync(data.weight.data(), impl_->weight, voxels * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream);
      }
      if (status == cudaSuccess && config_.color) {
        status = cudaMemcpyAsync(rgba.data(), impl_->rgba, voxels * sizeof(uchar4),
                                 cudaMemcpyDeviceToHost, stream);
      }
      status = access.finish(status);
    }
  }
  const cudaError_t synced = cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  if (status == cudaSuccess) {
    status = synced;
  }
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("TsdfVolume::download: ") + cudaGetErrorString(status));
  }
  if (config_.color) {
    data.rgb.resize(3U * voxels);
    for (std::uint64_t i = 0; i < voxels; ++i) {
      data.rgb[3 * i] = rgba[i].x;
      data.rgb[3 * i + 1] = rgba[i].y;
      data.rgb[3 * i + 2] = rgba[i].z;
    }
  }
  return data;
}

void TsdfVolume::export_ply(const std::string& path) const {
  const auto fail = [](const std::string& what, cudaError_t status) {
    throw std::runtime_error("TsdfVolume::export_ply: " + what + ": " + cudaGetErrorString(status));
  };
  cudaStream_t stream = nullptr;
  if (const cudaError_t status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      status != cudaSuccess) {
    fail("cudaStreamCreate", status);
  }
  struct StreamGuard {
    cudaStream_t stream;
    ~StreamGuard() {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  } guard{stream};
  // A first guess of the size; when the mesh is larger, extract it again at
  // its size.
  std::uint32_t max_vertices = 1U << 18;
  std::uint32_t max_triangles = 1U << 19;
  for (int attempt = 0; attempt < 2; ++attempt) {
    MeshExtractor extractor(max_vertices, max_triangles);
    detail::DeviceBuffer vertices;
    detail::DeviceBuffer normals;
    detail::DeviceBuffer colors;
    detail::DeviceBuffer triangles;
    detail::DeviceBuffer counts;
    for (auto [buffer, bytes] :
         {std::pair{&vertices, std::size_t{max_vertices} * 12U},
          std::pair{&normals, std::size_t{max_vertices} * 12U},
          std::pair{&colors, config_.color ? std::size_t{max_vertices} * 3U : 0U},
          std::pair{&triangles, std::size_t{max_triangles} * 12U},
          std::pair{&counts, std::size_t{kMeshCountsSize} * 4U}}) {
      if (const cudaError_t status = buffer->reserve(bytes, stream); status != cudaSuccess) {
        fail("cudaMalloc", status);
      }
    }
    const MeshBuffers out{vertices.as<float>(), normals.as<float>(),
                          config_.color ? colors.as<std::uint8_t>() : nullptr,
                          triangles.as<std::uint32_t>(), counts.as<std::uint32_t>()};
    if (const cudaError_t status = extractor.extract(*this, out, stream); status != cudaSuccess) {
      fail("mesh extraction", status);
    }
    std::uint32_t host_counts[kMeshCountsSize] = {};
    if (const cudaError_t status = cudaMemcpyAsync(host_counts, out.counts, sizeof(host_counts),
                                                   cudaMemcpyDeviceToHost, stream);
        status != cudaSuccess) {
      fail("cudaMemcpyAsync", status);
    }
    if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
      fail("cudaStreamSynchronize", status);
    }
    const std::uint32_t need_v = host_counts[kCountNeededVertices];
    const std::uint32_t need_t = host_counts[kCountNeededTriangles];
    if (need_v > max_vertices || need_t > max_triangles) {
      if (need_v > kMaxMeshCapacity || need_t > kMaxMeshCapacity) {
        throw std::runtime_error("TsdfVolume::export_ply: the mesh is too large");
      }
      max_vertices = std::max(max_vertices, need_v);
      max_triangles = std::max(max_triangles, need_t);
      continue;
    }
    detail::PlyMesh mesh;
    mesh.vertices.resize(std::size_t{need_v} * 3U);
    mesh.normals.resize(std::size_t{need_v} * 3U);
    mesh.colors.resize(config_.color ? std::size_t{need_v} * 3U : 0U);
    mesh.triangles.resize(std::size_t{need_t} * 3U);
    for (auto [host, device, bytes] :
         {std::tuple{static_cast<void*>(mesh.vertices.data()),
                     static_cast<const void*>(out.vertices), mesh.vertices.size() * 4U},
          std::tuple{static_cast<void*>(mesh.normals.data()), static_cast<const void*>(out.normals),
                     mesh.normals.size() * 4U},
          std::tuple{static_cast<void*>(mesh.colors.data()), static_cast<const void*>(out.colors),
                     mesh.colors.size()},
          std::tuple{static_cast<void*>(mesh.triangles.data()),
                     static_cast<const void*>(out.triangles), mesh.triangles.size() * 4U}}) {
      if (bytes == 0U) {
        continue;
      }
      if (const cudaError_t status =
              cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, stream);
          status != cudaSuccess) {
        fail("cudaMemcpyAsync", status);
      }
    }
    if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
      fail("cudaStreamSynchronize", status);
    }
    detail::write_ply(path, mesh);
    return;
  }
  throw std::runtime_error("TsdfVolume::export_ply: the mesh did not fit after resizing");
}

}  // namespace holoscan::tsdf
