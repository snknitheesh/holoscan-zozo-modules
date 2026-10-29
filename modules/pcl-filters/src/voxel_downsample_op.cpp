// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/pcl_filters/voxel_downsample_op.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace holoscan::pcl_filters {
namespace {

CloudSettings settings_of(const VoxelDownsampleConfig& config) {
  return CloudSettings{
      .color = config.color, .layout = OutputLayout::kDense, .max_points = config.max_points};
}

}  // namespace

void validate_config(const VoxelDownsampleConfig& config) {
  if (!(config.voxel_size > 0.0F) || !std::isfinite(config.voxel_size)) {
    throw std::invalid_argument("voxel_size must be finite and positive");
  }
  if (config.mode != VoxelMode::kCentroid && config.mode != VoxelMode::kFirstPoint) {
    throw std::invalid_argument("mode must be kCentroid or kFirstPoint");
  }
}

VoxelDownsampleOp::VoxelDownsampleOp(VoxelDownsampleConfig config)
    : CloudFilterOp("VoxelDownsampleOp", settings_of(config)),
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
          grid_->group(cloud->xyz, cloud->count, config_.voxel_size, cloud->stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("voxel grid", status));
  }
  return emit_dense(*cloud, grid_->device_count(), [&](float* xyz, std::uint8_t* rgb) {
    return grid_->write(cloud->xyz, cloud->rgb, config_.mode, xyz, rgb, cloud->stream);
  });
}

}  // namespace holoscan::pcl_filters
