// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// VoxelDownsampleOp: one point per occupied voxel, the voxels' centroids or
// first points, with their mean colors, as a dense cloud in a deterministic
// order. See cloud_filter_op.hpp for the ports and VoxelGrid in filters.hpp
// for the grid.

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
  // Declare the `colors` input and `filtered_colors` output.
  bool color{false};
  // Largest cloud accepted; see CloudSettings.
  std::uint32_t max_points{1920U * 1080U};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const VoxelDownsampleConfig& config);

// The output is always dense: [voxels, 3], in ascending voxel index order.
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
