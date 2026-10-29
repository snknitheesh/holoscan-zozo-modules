// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "filter_report_op.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/pcl_filters/filters.hpp>
#include <holoscan/pcl_filters/synthetic_scene.hpp>

namespace pcl_filters_app {
namespace {

using std::chrono_literals::operator""ns;
namespace pf = holoscan::pcl_filters;

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "FilterReportOp: " + message};
}

// The number of voxels of side `size` the valid points occupy, counted as
// VoxelDownsampleOp defines them.
std::size_t voxel_count(const std::vector<float>& xyz, float size) {
  std::vector<std::tuple<std::int64_t, std::int64_t, std::int64_t>> voxels;
  for (std::size_t i = 0; i + 2 < xyz.size(); i += 3) {
    std::array<std::int64_t, 3> index{};
    bool ok = true;
    for (int axis = 0; axis < 3; ++axis) {
      const float cell = std::floor(xyz[i + axis] / size);
      ok = ok && cell >= -static_cast<float>(pf::kVoxelIndexRange) &&
           cell < static_cast<float>(pf::kVoxelIndexRange);
      index[axis] = ok ? static_cast<std::int64_t>(cell) : 0;
    }
    if (ok) {
      voxels.emplace_back(index[0], index[1], index[2]);
    }
  }
  std::sort(voxels.begin(), voxels.end());
  return static_cast<std::size_t>(std::unique(voxels.begin(), voxels.end()) - voxels.begin());
}

}  // namespace

FilterReportOp::FilterReportOp(FilterReportConfig config) : config_(std::move(config)) {
  if (!config_.report) {
    throw std::invalid_argument("FilterReportOp needs a report");
  }
}

void FilterReportOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(points, "points").queue_depth(2U);
  spec.input(plane, "plane").queue_depth(2U);
  spec.input(voxels, "voxels").queue_depth(2U);
  spec.input(voxel_colors, "voxel_colors").queue_depth(2U);
}

holoscan::Contract FilterReportOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{std::vector<holoscan::PortRef>{
                                         points, plane, voxels, voxel_colors}},
                                     .window = 0ns});
  result.input(points, holoscan::Exact{});
  result.input(plane, holoscan::Exact{});
  result.input(voxels, holoscan::Exact{});
  result.input(voxel_colors, holoscan::Exact{});
  return result;
}

holoscan::expected<void, holoscan::Error> FilterReportOp::compute(
    holoscan::ExecutionContext& context) {
  auto cloud = points.receive();
  auto coefficients = plane.receive();
  auto voxel_cloud = voxels.receive();
  auto voxel_rgb = voxel_colors.receive();
  if (!cloud) {
    return holoscan::make_unexpected(std::move(cloud).error());
  }
  if (!coefficients) {
    return holoscan::make_unexpected(std::move(coefficients).error());
  }
  if (!voxel_cloud) {
    return holoscan::make_unexpected(std::move(voxel_cloud).error());
  }
  if (!voxel_rgb) {
    return holoscan::make_unexpected(std::move(voxel_rgb).error());
  }
  const std::span<const std::int64_t> shape = cloud->data.shape_span();
  if (shape.size() != 3 || shape[0] != config_.height || shape[1] != config_.width ||
      shape[2] != 3) {
    return holoscan::make_unexpected(invalid("expected the organized [H, W, 3] cloud"));
  }
  const std::span<const std::int64_t> voxel_shape = voxel_cloud->data.shape_span();
  if (voxel_shape.size() != 2 || voxel_shape[1] != 3) {
    return holoscan::make_unexpected(invalid("expected dense [M, 3] voxels"));
  }
  xyz_.resize(static_cast<std::size_t>(config_.width) * config_.height * 3U);
  if (auto copied =
          cloud->data.copy_to_host(xyz_.data(), xyz_.size() * sizeof(float), context.cuda_stream());
      !copied) {
    return copied;
  }
  std::array<float, 4> got{};
  if (auto copied = coefficients->data.copy_to_host(got.data(), sizeof(got), context.cuda_stream());
      !copied) {
    return copied;
  }

  FilterReport& report = *config_.report;
  const std::uint64_t frame_id = cloud->metadata.frame_id;
  report.frame_gaps += frame_id != last_frame_id_ + 1U ? 1U : 0U;
  last_frame_id_ = frame_id;
  const pf::SyntheticScene scene =
      pf::synthetic_scene(config_.width, config_.height, static_cast<std::uint32_t>(frame_id - 1U));

  // The plane against the true ground.
  float truth[4];
  pf::synthetic_ground_plane(scene, truth);
  const double dot = static_cast<double>(got[0]) * truth[0] +
                     static_cast<double>(got[1]) * truth[1] +
                     static_cast<double>(got[2]) * truth[2];
  const double angle = std::acos(std::clamp(dot, -1.0, 1.0)) * 180.0 / std::numbers::pi;
  const double offset = std::abs(static_cast<double>(got[3]) - truth[3]);
  if (!(angle <= kMaxPlaneAngleDeg && offset <= kMaxPlaneOffsetM)) {
    ++report.wrong_planes;
  }
  report.max_plane_angle_deg =
      std::max(report.max_plane_angle_deg, std::isnan(angle) ? 180.0 : angle);
  report.max_plane_offset_m =
      std::max(report.max_plane_offset_m, std::isnan(offset) ? 1e9 : offset);

  // What each pixel showed, and whether its point is left.
  for (int v = 0; v < config_.height; ++v) {
    for (int u = 0; u < config_.width; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * config_.width + u;
      const float* p = xyz_.data() + 3 * i;
      const bool left = std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
      switch (pf::synthetic_scene_pixel(scene, u, v).label) {
        case pf::SceneLabel::kGround:
          ++report.ground;
          report.ground_left += left ? 1U : 0U;
          break;
        case pf::SceneLabel::kObject:
          ++report.objects;
          report.objects_kept += left ? 1U : 0U;
          break;
        case pf::SceneLabel::kOutlier:
          ++report.outliers;
          report.outliers_left += left ? 1U : 0U;
          break;
        case pf::SceneLabel::kNone:
          break;
      }
    }
  }

  // The voxels of exactly this cloud.
  const auto count = static_cast<std::size_t>(voxel_shape[0]);
  const std::span<const std::int64_t> rgb_shape = voxel_rgb->data.shape_span();
  if (rgb_shape.size() != 2 || rgb_shape[0] != voxel_shape[0] || rgb_shape[1] != 3) {
    return holoscan::make_unexpected(invalid("voxel colors must match the voxels"));
  }
  rgb_.resize(count * 3U);
  if (auto copied = voxel_rgb->data.copy_to_host(rgb_.data(), rgb_.size(), context.cuda_stream());
      !copied) {
    return copied;
  }
  for (std::size_t i = 0; i < count; ++i) {
    report.black_voxels += rgb_[3 * i] == 0U && rgb_[3 * i + 1] == 0U && rgb_[3 * i + 2] == 0U;
  }
  report.voxels += count;
  if (count != voxel_count(xyz_, config_.voxel_size)) {
    ++report.wrong_voxel_counts;
  }

  ++report.frames;
  if (config_.finished) {
    config_.finished->fetch_add(1U, std::memory_order_release);
  }
  if (!finished_ && report.frames >= config_.expected_frames) {
    finished_ = true;
    context.request_completion();
  }
  return {};
}

}  // namespace pcl_filters_app
