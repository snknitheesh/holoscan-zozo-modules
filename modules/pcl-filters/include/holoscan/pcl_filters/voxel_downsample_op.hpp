// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// VoxelDownsampleOp: one point per occupied voxel, the voxels' centroids or
// first points, with their mean colors, normals and intensities, as a packed
// cloud in a deterministic order. See cloud_filter_op.hpp for the ports and VoxelGrid in
// filters.hpp for the grid.

#pragma once

#include <cstdint>
#include <memory>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/pcl_filters/cloud_filter_op.hpp>
#include <holoscan/pcl_filters/filters.hpp>

namespace holoscan::pcl_filters {

struct VoxelDownsampleConfig {
  // The voxels' side in meters, positive.
  float voxel_size{0.05F};
  VoxelMode mode{VoxelMode::kCentroid};
  // The output's layout (kDense or kPadded: a voxel has no place in the
  // input's shape), size and fields.
  CloudSettings cloud{.layout = OutputLayout::kDense};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const VoxelDownsampleConfig& config);

// The output is packed: the voxels in ascending voxel index order, as
// [voxels, 3] (kDense) or followed by removed points up to the input's size
// (kPadded).
class VoxelDownsampleOp final : public CloudFilterOp {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit VoxelDownsampleOp(VoxelDownsampleConfig config);
  ~VoxelDownsampleOp() override;

  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

 private:
  VoxelDownsampleConfig config_;
  std::unique_ptr<VoxelGrid> grid_;
};

}  // namespace holoscan::pcl_filters
