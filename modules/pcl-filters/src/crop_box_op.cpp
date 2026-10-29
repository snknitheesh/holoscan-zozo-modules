// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/pcl_filters/crop_box_op.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace holoscan::pcl_filters {
namespace {

CloudSettings settings_of(const CropBoxConfig& config) {
  return CloudSettings{
      .color = config.color, .layout = config.layout, .max_points = config.max_points};
}

}  // namespace

void validate_config(const CropBoxConfig& config) {
  const CropBoxParams& box = config.box;
  for (int axis = 0; axis < 3; ++axis) {
    if (std::isnan(box.min[axis]) || std::isnan(box.max[axis]) || box.min[axis] > box.max[axis]) {
      throw std::invalid_argument("the box needs min <= max on every axis");
    }
  }
  if (!(box.min_distance >= 0.0F) || !(box.max_distance >= box.min_distance)) {
    throw std::invalid_argument("the distance range needs 0 <= min_distance <= max_distance");
  }
  if (box.use_transform &&
      !std::all_of(std::begin(box.transform.matrix), std::end(box.transform.matrix),
                   [](float value) { return std::isfinite(value); })) {
    throw std::invalid_argument("the transform must be finite");
  }
}

CropBoxOp::CropBoxOp(CropBoxConfig config)
    : CloudFilterOp("CropBoxOp", settings_of(config)), config_(config) {
  validate_config(config_);
}

holoscan::expected<void, holoscan::Error> CropBoxOp::compute(holoscan::ExecutionContext& context) {
  auto cloud = receive(context);
  if (!cloud) {
    return holoscan::make_unexpected(std::move(cloud).error());
  }
  auto keep = mask(*cloud);
  if (!keep) {
    return holoscan::make_unexpected(std::move(keep).error());
  }
  if (const cudaError_t status =
          launch_crop_box(cloud->xyz, cloud->count, config_.box, *keep, cloud->stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("crop box kernel", status));
  }
  return emit_kept(*cloud, *keep);
}

}  // namespace holoscan::pcl_filters
