// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// TsdfMeshOp: on each camera pose it receives, ray casts depth, normal and
// color images of a TsdfVolume from that pose, and on every mesh_every-th
// pose extracts the volume's mesh by marching cubes (marching_cubes.hpp),
// on the GPU.
//
//   pose              (input) float32 or float64 [4, 4] world_from_camera,
//                     row-major; host memory (device memory is copied to the
//                     host, which waits for the GPU). TsdfIntegrateOp's
//                     integrated_pose, or any pose to look from.
//   vertices          (with `mesh`) float32 [max_vertices, 3], world meters.
//   normals           (with `mesh`) float32 [max_vertices, 3] unit normals.
//   colors            (with `mesh` and a color volume) uint8 [max_vertices, 3].
//   triangles         (with `mesh`) uint32 [max_triangles, 3] vertex indices.
//   mesh_counts       (with `mesh`) uint32 [kMeshCountsSize]: the vertices
//                     and triangles written, and those the surface needs
//                     (see marching_cubes.hpp).
//   raycast_depth     (with `raycast`) MONO32F [H, W] meters along the
//                     camera's z axis, 0 where a ray hits nothing.
//   raycast_normals   (with `raycast`) float32 [H, W, 3] unit normals in the
//                     camera frame, NaN where a ray hits nothing.
//   raycast_colors    (with `raycast` and a color volume) RGB8 [H, W, 3].
//
// The mesh outputs have a fixed size, so the operator never waits for the GPU
// to learn the mesh's: only the first mesh_counts[0] vertices and
// mesh_counts[1] triangles are valid. When the surface needs more than the
// capacity, both counts are 0 and the last two say how much it needs. Every
// output is in device memory and carries the pose's capture time and frame
// id. The mesh outputs go out only on the poses that are meshed; the
// ray-cast outputs on every pose. Meshing on demand: send a pose when a mesh
// is wanted, with mesh_every 1, or use MeshExtractor directly.

#pragma once

#include <cstdint>
#include <memory>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/tsdf/marching_cubes.hpp>
#include <holoscan/tsdf/tsdf_volume.hpp>
#include <holoscan/tsdf/volume_op.hpp>

namespace holoscan::tsdf {

struct TsdfMeshConfig {
  // The volume, shared with TsdfIntegrateOp. Required.
  std::shared_ptr<TsdfVolume> volume;
  // Declare the mesh outputs, and mesh every mesh_every-th pose (the
  // mesh_every-th, the 2 * mesh_every-th, ...).
  bool mesh{true};
  std::uint32_t mesh_every{1};
  // The mesh's capacity; they size the outputs and the scratch.
  std::uint32_t max_vertices{std::uint32_t{1} << 20};
  std::uint32_t max_triangles{std::uint32_t{1} << 21};
  // Declare the ray-cast outputs, with the camera they are seen with.
  bool raycast{false};
  RaycastParams raycast_params;
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const TsdfMeshConfig& config);

class TsdfMeshOp final : public VolumeOp {
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
  holoscan::Output<holoscan::Tensor> vertices;
  holoscan::Output<holoscan::Tensor> normals;
  holoscan::Output<holoscan::Tensor> colors;
  holoscan::Output<holoscan::Tensor> triangles;
  holoscan::Output<holoscan::Tensor> mesh_counts;
  holoscan::Output<holoscan::schema::ImageT> raycast_depth;
  holoscan::Output<holoscan::Tensor> raycast_normals;
  holoscan::Output<holoscan::schema::ImageT> raycast_colors;

 private:
  [[nodiscard]] bool color() const noexcept;

  TsdfMeshConfig config_;
  std::unique_ptr<MeshExtractor> extractor_;
  std::uint64_t poses_{};
};

}  // namespace holoscan::tsdf
