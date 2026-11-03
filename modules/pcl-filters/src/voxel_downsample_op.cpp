// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/pcl_filters/voxel_downsample_op.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace holoscan::pcl_filters {
void validate_config(const VoxelDownsampleConfig& config) {
  validate_settings(config.cloud);
  if (config.cloud.layout == OutputLayout::kKeep) {
    throw std::invalid_argument("layout must be kDense or kPadded: a voxel has no input position");
  }
  if (!(config.voxel_size > 0.0F) || !std::isfinite(config.voxel_size)) {
    throw std::invalid_argument("voxel_size must be finite and positive");
  }
  if (config.mode != VoxelMode::kCentroid && config.mode != VoxelMode::kFirstPoint) {
    throw std::invalid_argument("mode must be kCentroid or kFirstPoint");
  }
}

VoxelDownsampleOp::VoxelDownsampleOp(VoxelDownsampleConfig config)
    : CloudFilterOp("VoxelDownsampleOp", config.cloud),
      config_(config),
      grid_(std::make_unique<VoxelGrid>()) {
  validate_config(config_);
}

VoxelDownsampleOp::~VoxelDownsampleOp() = default;

void VoxelDownsampleOp::stop() {
  grid_ = std::make_unique<VoxelGrid>();
  CloudFilterOp::stop();
}

holoscan::expected<void, holoscan::Error> VoxelDownsampleOp::compute(
    holoscan::ExecutionContext& context) {
  auto cloud = receive(context);
  if (!cloud) {
    return holoscan::make_unexpected(std::move(cloud).error());
  }
  if (const cudaError_t status =
          grid_->group(cloud->fields.xyz, cloud->count, config_.voxel_size, cloud->stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("voxel grid", status));
  }
  return emit_packed(*cloud, grid_->device_count(), [&](const CloudOutput& out) {
    return grid_->write(cloud->fields, config_.mode, out, cloud->stream);
  });
}

}  // namespace holoscan::pcl_filters
