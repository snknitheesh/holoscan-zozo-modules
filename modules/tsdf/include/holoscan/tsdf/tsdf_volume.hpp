// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// TsdfVolume: a dense, fixed-size truncated signed distance volume on the GPU,
// with optional color, and what is done with it: integrating depth images
// from known camera poses, ray casting depth, normal and color images from a
// pose, and saving its mesh as a PLY file. Nothing here needs a Holoscan
// graph; TsdfIntegrateOp and TsdfMeshOp share one volume through a
// std::shared_ptr.
//
// Frames: a pose maps camera-frame points into the world (world_from_camera),
// row-major 4x4, the camera frame being the optical frame (x right, y down,
// z forward). Voxel (x, y, z) samples the world point
// origin + voxel_size * (x, y, z).
//
// Each voxel holds the signed distance to the nearest surface along the
// camera's z axis, as seen by the cameras, divided by the truncation distance
// and clamped to [-1, 1]: positive in front of the surface, negative behind
// it. Voxels farther than the truncation distance behind every surface seen
// are never written and keep weight 0. Each depth image adds weight 1 to the
// voxels it updates, as a running average, up to `weight_cap`; past the cap the
// average keeps following new images with a fixed weight.
//
// Ordering: every operation is enqueued on the stream given to it and waits,
// on the GPU, for the previous operation on the volume from any stream, so
// that operators on different streams and threads can share a volume without
// host waits. The methods are thread-safe. Only export_ply() and download()
// wait on the host.

#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace holoscan::tsdf {

// Pinhole intrinsics in pixels; pixel centers are at integer coordinates, as
// in OpenCV and ROS CameraInfo.
struct CameraIntrinsics {
  float fx{};
  float fy{};
  float cx{};
  float cy{};
};

// A row-major 4x4 rigid transform. The last row must be 0 0 0 1.
using Matrix4 = std::array<float, 16>;

[[nodiscard]] constexpr Matrix4 identity_pose() {
  return {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
          0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};
}

// Whether `pose` is finite, has the last row 0 0 0 1 and a rotation within
// `tolerance` of orthonormal with determinant +1.
[[nodiscard]] bool valid_pose(const Matrix4& pose, float tolerance = 1e-3F);

// The inverse of a rigid transform, computed in double precision.
[[nodiscard]] Matrix4 inverse_pose(const Matrix4& pose);

// The largest volume, in voxels (2^30, 1024^3).
inline constexpr std::uint64_t kMaxVoxels = std::uint64_t{1} << 30;

struct TsdfVolumeConfig {
  // Voxels along x, y and z: each 2..1024, at most kMaxVoxels in all.
  int dims[3]{256, 256, 256};
  // Edge of a voxel in meters.
  float voxel_size{0.01F};
  // World position of voxel (0, 0, 0).
  float origin[3]{-1.28F, -1.28F, -1.28F};
  // Truncation distance in meters; at least one voxel, typically 3 to 5.
  float truncation{0.04F};
  // Largest weight a voxel accumulates, 1..65535.
  float weight_cap{64.0F};
  // Keep an RGB color per voxel (4 more bytes per voxel).
  bool color{false};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const TsdfVolumeConfig& config);

// Device memory a volume of `config` holds: 8 bytes per voxel, 12 with color.
[[nodiscard]] std::uint64_t volume_bytes(const TsdfVolumeConfig& config);

enum class DepthType : std::uint8_t { kUint16, kFloat32 };

// A depth image in device memory, packed rows.
struct DepthImage {
  const void* data{};
  DepthType type{DepthType::kFloat32};
  int width{};
  int height{};
  // Multiplies raw depth into meters: 0.001 for uint16 millimeters, 1.0 for
  // float32 meters.
  float scale{1.0F};
};

// A color image aligned with the depth image, in device memory, packed rows,
// of the depth image's size.
struct ColorImage {
  const std::uint8_t* data{};
  int channels{3};  // 3 or 4; a fourth channel is ignored.
  bool bgr{false};  // BGR(A) instead of RGB(A).
};

struct IntegrateParams {
  CameraIntrinsics intrinsics;
  // Depths outside [depth_min, depth_max] meters, zero and non-finite depths
  // are no measurement.
  float depth_min{0.1F};
  float depth_max{4.0F};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_params(const IntegrateParams& params);

struct RaycastParams {
  CameraIntrinsics intrinsics;
  int width{640};
  int height{480};
  // Rays start depth_min and end depth_max meters along the camera's z axis.
  float depth_min{0.1F};
  float depth_max{4.0F};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_params(const RaycastParams& params);

// Ray-cast images in device memory, width x height, packed rows. Any may be
// null to skip it; `colors` needs a volume with color.
struct RaycastImages {
  // float32 meters along the camera's z axis, 0 where a ray hits nothing.
  float* depth{};
  // float32 x, y, z unit normals in the camera frame, facing the camera; NaN
  // where a ray hits nothing.
  float* normals{};
  // RGB8 of the nearest voxel at the hit, black where a ray hits nothing.
  std::uint8_t* colors{};
};

// The voxels of a volume, on the host: x fastest, then y, then z.
struct VolumeData {
  std::vector<float> tsdf;
  std::vector<float> weight;
  std::vector<std::uint8_t> rgb;  // 3 per voxel; empty without color.
};

class TsdfVolume {
 public:
  // Allocates the volume on the current CUDA device and resets it.
  // @throws std::invalid_argument if `config` is not valid.
  // @throws std::runtime_error if the GPU memory cannot be allocated; the
  // message says how much was asked for and how much was free.
  explicit TsdfVolume(TsdfVolumeConfig config);
  ~TsdfVolume();
  TsdfVolume(const TsdfVolume&) = delete;
  TsdfVolume& operator=(const TsdfVolume&) = delete;

  [[nodiscard]] const TsdfVolumeConfig& config() const noexcept { return config_; }
  [[nodiscard]] int device() const noexcept { return device_; }
  [[nodiscard]] std::uint64_t voxel_count() const noexcept;
  // Device memory the volume holds.
  [[nodiscard]] std::uint64_t bytes() const noexcept { return volume_bytes(config_); }

  // Clears every voxel to distance 1, weight 0 and black.
  cudaError_t reset(cudaStream_t stream);

  // Integrates a depth image seen from `world_from_camera`, with its color
  // when given (and the volume keeps color). Only the voxels inside the
  // camera's view between depth_min and depth_max are visited.
  cudaError_t integrate(const DepthImage& depth, const ColorImage* color,
                        const Matrix4& world_from_camera, const IntegrateParams& params,
                        cudaStream_t stream);

  // Ray casts the volume from `world_from_camera`: each pixel's ray stops at
  // the first crossing from in front of a surface to behind it, between
  // voxels that have been observed.
  cudaError_t raycast(const Matrix4& world_from_camera, const RaycastParams& params,
                      const RaycastImages& images, cudaStream_t stream);

  // Copies the voxels to the host and waits.
  // @throws std::runtime_error on a CUDA failure.
  [[nodiscard]] VolumeData download() const;

  // Extracts the mesh (see marching_cubes.hpp) and writes it as a binary PLY
  // file: x, y, z, nx, ny, nz (and red, green, blue with color) per vertex,
  // and the triangles. Waits for the GPU. For debugging and inspection.
  // @throws std::runtime_error naming the path or the CUDA failure.
  void export_ply(const std::string& path) const;

 private:
  friend class MeshExtractor;
  struct Impl;

  // The voxel arrays on the device, for the mesh extraction.
  struct DeviceView {
    const float* tsdf{};
    const float* weight{};
    const std::uint8_t* rgba{};  // 4 bytes per voxel; null without color.
  };
  [[nodiscard]] DeviceView view() const noexcept;
  // Makes `stream` wait for the previous operation on the volume and locks
  // the volume until end_access(), on the same thread.
  cudaError_t begin_access(cudaStream_t stream) const;
  // Records the end of the work enqueued on `stream` since begin_access(),
  // which the next operation waits for, and unlocks.
  cudaError_t end_access(cudaStream_t stream) const;

  TsdfVolumeConfig config_;
  int device_{};
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::tsdf
