// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/pcl_filters/outlier_removal_op.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace holoscan::pcl_filters {
namespace {

CloudSettings settings_of(const OutlierRemovalConfig& config) {
  return CloudSettings{
      .color = config.color, .layout = config.layout, .max_points = config.max_points};
}

}  // namespace

void validate_config(const OutlierRemovalConfig& config) {
  const OutlierParams& p = config.outliers;
  if (p.mode == OutlierMode::kStatistical) {
    if (p.k < 1U || p.k > kMaxNeighbors) {
      throw std::invalid_argument("k must be 1.." + std::to_string(kMaxNeighbors));
    }
    if (!(p.search_radius > 0.0F) || !std::isfinite(p.search_radius)) {
      throw std::invalid_argument("search_radius must be finite and positive");
    }
    if (!std::isfinite(p.std_multiplier)) {
      throw std::invalid_argument("std_multiplier must be finite");
    }
  } else if (p.mode == OutlierMode::kRadius) {
    if (!(p.radius > 0.0F) || !std::isfinite(p.radius)) {
      throw std::invalid_argument("radius must be finite and positive");
    }
  } else {
    throw std::invalid_argument("mode must be kStatistical or kRadius");
  }
}

OutlierRemovalOp::OutlierRemovalOp(OutlierRemovalConfig config)
    : CloudFilterOp("OutlierRemovalOp", settings_of(config)),
      config_(config),
      filter_(std::make_unique<OutlierFilter>()) {
  validate_config(config_);
}

OutlierRemovalOp::~OutlierRemovalOp() = default;

void OutlierRemovalOp::stop() {
  filter_ = std::make_unique<OutlierFilter>();
  CloudFilterOp::stop();
}

holoscan::expected<void, holoscan::Error> OutlierRemovalOp::compute(
    holoscan::ExecutionContext& context) {
  auto cloud = receive(context);
  if (!cloud) {
    return holoscan::make_unexpected(std::move(cloud).error());
  }
  auto keep = mask(*cloud);
  if (!keep) {
    return holoscan::make_unexpected(std::move(keep).error());
  }
  if (const cudaError_t status =
          filter_->filter(cloud->xyz, cloud->count, config_.outliers, *keep, cloud->stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("outlier kernels", status));
  }
  return emit_kept(*cloud, *keep);
}

}  // namespace holoscan::pcl_filters
