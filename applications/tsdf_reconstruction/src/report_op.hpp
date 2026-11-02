// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's ground-truth checks, against the tabletop scene's exact
// surfaces:
//
//   RaycastReportOp  each frame's ray cast from the camera's pose against the
//                    exact depth from that pose: the error and the share of
//                    the surfaces inside the volume it sees; and that every
//                    hit has a unit normal facing the camera, and nearly
//                    every hit a color.
//   MeshReportOp     each mesh's vertices against the exact surfaces: the
//                    distance of every vertex to the nearest surface; and
//                    that every triangle indexes written vertices and every
//                    vertex has a unit normal.
//
// Both write into one Report, and the one that completes it (every frame
// and every mesh checked) completes the run.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/tsdf/tsdf_volume.hpp>

namespace tsdf_reconstruction_app {

struct Report {
  std::mutex mutex;
  // Expected work, and the voxel size the errors are judged by.
  std::uint32_t expected_frames{};
  std::uint32_t expected_meshes{};
  std::uint32_t warmup_frames{10};  // Frames not judged while the volume fills.
  float voxel{};
  // The ray casts.
  std::uint32_t frames{};
  double worst_median_m{};  // The worst frame's median depth error.
  double worst_p95_m{};     // The worst frame's 95th percentile.
  // The worst frame's share of hits more than three voxels off: rays that
  // graze an edge and hit the surface beside the one the camera sees.
  double worst_outliers{};
  double worst_coverage{1.0};  // The worst frame's share of the surfaces seen.
  // Hits without a unit normal facing the camera, and frames with more than
  // 1% of hits black.
  std::uint64_t bad_hits{};
  // The meshes.
  std::uint32_t meshes{};
  std::uint32_t overflows{};      // Meshes beyond the capacity.
  std::uint64_t bad_triangles{};  // Triangles indexing vertices not written.
  std::uint64_t bad_normals{};    // Vertices without a unit normal.
  std::uint32_t last_vertices{};
  std::uint32_t last_triangles{};
  double last_mean_m{};
  double last_p95_m{};
  double last_max_m{};
  bool completed{};

  // Thresholds: the ray cast within half a voxel at the median and 1.5
  // voxels at the 95th percentile, under 2% of hits more than three voxels
  // off, seeing 95% of the surfaces; the last mesh within 0.35 voxels on
  // average and one voxel at the 95th percentile.
  [[nodiscard]] bool passed() const {
    return frames == expected_frames && meshes == expected_meshes && overflows == 0 &&
           bad_hits == 0 && bad_triangles == 0 && bad_normals == 0 &&
           worst_median_m < 0.5 * voxel && worst_p95_m < 1.5 * voxel && worst_outliers < 0.02 &&
           worst_coverage >= 0.95 &&
           (expected_meshes == 0 ||
            (last_triangles > 0 && last_mean_m < 0.35 * voxel && last_p95_m < 1.0 * voxel));
  }
};

struct ReportConfig {
  std::shared_ptr<Report> report;
  holoscan::tsdf::TsdfVolumeConfig volume;
  int width{};
  int height{};
  // Frames the ray-cast reporter has finished, for the camera.
  std::shared_ptr<std::atomic<std::uint32_t>> finished;
};

class RaycastReportOp final : public holoscan::Operator<> {
 public:
  explicit RaycastReportOp(ReportConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // MONO32F [H, W] meters on the GPU.
  holoscan::Input<holoscan::schema::ImageT> depth;
  // float32 [H, W, 3] camera-frame normals on the GPU.
  holoscan::Input<holoscan::Tensor> normals;
  // RGB8 [H, W, 3] on the GPU.
  holoscan::Input<holoscan::schema::ImageT> colors;
  // float32 [4, 4] world_from_camera, host memory.
  holoscan::Input<holoscan::Tensor> pose;

 private:
  ReportConfig config_;
  std::vector<float> pixels_;
  std::vector<float> normals_;
  std::vector<std::uint8_t> colors_;
};

class MeshReportOp final : public holoscan::Operator<> {
 public:
  explicit MeshReportOp(ReportConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // float32 [max_vertices, 3] on the GPU.
  holoscan::Input<holoscan::Tensor> vertices;
  // float32 [max_vertices, 3] on the GPU.
  holoscan::Input<holoscan::Tensor> normals;
  // uint8 [max_vertices, 3] on the GPU.
  holoscan::Input<holoscan::Tensor> colors;
  // uint32 [max_triangles, 3] on the GPU.
  holoscan::Input<holoscan::Tensor> triangles;
  // uint32 [4] on the GPU.
  holoscan::Input<holoscan::Tensor> counts;

 private:
  ReportConfig config_;
  std::vector<float> points_;
  std::vector<float> normals_;
  std::vector<std::uint32_t> triangles_;
};

}  // namespace tsdf_reconstruction_app
