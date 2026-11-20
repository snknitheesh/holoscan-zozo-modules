// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// TsdfMeshOp: on each camera pose it receives, ray casts a TsdfVolume from
// that pose through the camera that a CameraInfo describes, and on every
// mesh_every-th pose extracts the volume's mesh by marching cubes
// (marching_cubes.hpp), on the GPU.
//
//   pose              (input) float32 or float64 [4, 4] world_from_camera,
//                     row-major; host memory (device memory is copied to the
//                     host, which waits for the GPU). TsdfIntegrateOp's
//                     integrated_pose, or any pose to look from.
//   camera_info       (input, with `raycast`, mandatory) the CameraInfo
//                     (holoscan-messages) of the camera to ray cast through:
//                     its size and intrinsics; undistorted, without skew.
//   vertices          (with `mesh`) float32 [max_vertices, 3], world meters.
//   normals           (with `mesh`) float32 [max_vertices, 3] unit normals.
//   colors            (with `mesh` and a color volume) uint8 [max_vertices, 3].
//   triangles         (with `mesh`) uint32 [max_triangles, 3] vertex indices.
//   mesh_counts       (with `mesh`) uint32 [kMeshCountsSize]: the vertices
//                     and triangles written, and those the surface needs
//                     (see marching_cubes.hpp).
//   raycast_depth     (with `raycast`) MONO32F [H, W] meters along the
//                     camera's z axis, 0 where a ray hits nothing.
//   raycast_cloud     (with `raycast`) PointCloud (holoscan-messages),
//                     organized [H, W, 3]: the hits in the camera's frame
//                     (NaN where a ray hits nothing), their unit normals
//                     facing the camera (NaN likewise), and with a color
//                     volume their RGB8 colors (black likewise).
//
// The mesh outputs are plain tensors of this module (the messages module has
// no mesh message). They have a fixed size, so the operator never waits for
// the GPU to learn the mesh's: only the first mesh_counts[0] vertices and
// mesh_counts[1] triangles are valid. When the surface needs more than the
// capacity, both counts are 0 and the last two say how much it needs.
//
// Every output is in device memory and carries the pose's capture time and
// frame id; the ray cast's header has the camera info's frame id. The mesh
// outputs go out only on the poses that are meshed, the ray cast on every
// pose. With `raycast`, poses before the first camera info are skipped
// (neither ray cast nor meshed, nor counted for mesh_every). Meshing on
// demand: send a pose when a mesh is wanted, with mesh_every 1, or use
// MeshExtractor directly.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/tsdf/marching_cubes.hpp>
#include <holoscan/tsdf/tsdf_volume.hpp>
#include <holoscan/tsdf/volume_op.hpp>

namespace holoscan::tsdf {

struct TsdfMeshConfig {
  // The volume, shared with TsdfIntegrateOp. Required.
  std::shared_ptr<TsdfVolume> volume;
  // Declare the mesh outputs, and mesh every mesh_every-th pose (the
  // mesh_every-th, the twice mesh_every-th, ...).
  bool mesh{true};
  std::uint32_t mesh_every{1};
  // The mesh's capacity; they size the outputs and the scratch.
  std::uint32_t max_vertices{std::uint32_t{1} << 20};
  std::uint32_t max_triangles{std::uint32_t{1} << 21};
  // Declare the ray-cast outputs and the camera_info input.
  bool raycast{false};
  // Rays start depth_min and end depth_max meters along the camera's z axis.
  float raycast_depth_min{0.1F};
  float raycast_depth_max{4.0F};
  // The largest ray cast, which sizes its outputs; a camera info for larger
  // images fails the pose.
  int max_width{1920};
  int max_height{1080};
  // How the camera info arrives: kLatest (the default) for camera info at any
  // rate, kEveryFrame for camera info sent with every pose, with its capture
  // time.
  holoscan::messages::CameraInfoInputOptions camera_info;
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const TsdfMeshConfig& config);

class TsdfMeshOp final : public detail::VolumeOp {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit TsdfMeshOp(TsdfMeshConfig config);
  ~TsdfMeshOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> pose;
  holoscan::Input<holoscan::messages::schema::CameraInfoT> camera_info;
  holoscan::Output<holoscan::Tensor> vertices;
  holoscan::Output<holoscan::Tensor> normals;
  holoscan::Output<holoscan::Tensor> colors;
  holoscan::Output<holoscan::Tensor> triangles;
  holoscan::Output<holoscan::Tensor> mesh_counts;
  holoscan::Output<holoscan::schema::ImageT> raycast_depth;
  holoscan::Output<holoscan::messages::schema::PointCloudT> raycast_cloud;

 private:
  [[nodiscard]] bool color() const noexcept;

  TsdfMeshConfig config_;
  std::unique_ptr<MeshExtractor> extractor_;
  std::uint64_t poses_{};
  holoscan::messages::LatestCameraInfo<holoscan::messages::schema::CameraInfoT> info_;
  // The intrinsics of info_'s camera, once checked that the kernels can use
  // them.
  std::optional<CameraIntrinsics> intrinsics_;
};

}  // namespace holoscan::tsdf
