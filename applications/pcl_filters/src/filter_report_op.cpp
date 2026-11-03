// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "filter_report_op.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/pcl_filters/filters.hpp>
#include <holoscan/pcl_filters/synthetic_scene.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "scene_source_op.hpp"

namespace pcl_filters_app {
namespace {

using std::chrono_literals::operator""ns;
namespace hm = holoscan::messages;
namespace pf = holoscan::pcl_filters;

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
// The side of the square each voxel is drawn as, in pixels.
constexpr int kVoxelPixels = 5;

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

bool finite_point(const float* p) {
  return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

// Copies `field` to `host`, resized to fit.
template <typename T>
holoscan::expected<void, holoscan::Error> to_host(const holoscan::Tensor& field,
                                                  std::vector<T>& host,
                                                  holoscan::ExecutionContext& context) {
  host.resize(static_cast<std::size_t>(field.size()));
  if (host.empty()) {
    return {};
  }
  return field.copy_to_host(host.data(), host.size() * sizeof(T), context.cuda_stream());
}

}  // namespace

FilterReportOp::FilterReportOp(FilterReportConfig config) : config_(std::move(config)) {
  if (!config_.report) {
    throw std::invalid_argument("FilterReportOp needs a report");
  }
}

void FilterReportOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(ground_cloud, "ground_cloud").queue_depth(2U);
  spec.input(plane, "plane").queue_depth(2U);
  spec.input(voxels, "voxels").queue_depth(2U);
  if (config_.view_output) {
    spec.output(view, "view")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kHost,
                    .dtype = kUInt8,
                    .rank = 3U,
                    .strides = holoscan::TensorStridePolicy::kContiguousOnly,
                },
            .bounds = holoscan::tensor_bounds(std::uint64_t{3U} * config_.width * config_.height),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        })
        .payload_options(holoscan::PayloadOptions{
            .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
        });
  }
}

holoscan::Contract FilterReportOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnAligned{
      .ports = holoscan::PortList{std::vector<holoscan::PortRef>{ground_cloud, plane, voxels}},
      .window = 0ns});
  result.input(ground_cloud, holoscan::Exact{});
  result.input(plane, holoscan::Exact{});
  result.input(voxels, holoscan::Exact{});
  return result;
}

holoscan::expected<void, holoscan::Error> FilterReportOp::compute(
    holoscan::ExecutionContext& context) {
  auto cloud = ground_cloud.receive();
  if (!cloud) {
    return holoscan::make_unexpected(std::move(cloud).error());
  }
  auto coefficients = plane.receive();
  if (!coefficients) {
    return holoscan::make_unexpected(std::move(coefficients).error());
  }
  auto voxel_cloud = voxels.receive();
  if (!voxel_cloud) {
    return holoscan::make_unexpected(std::move(voxel_cloud).error());
  }
  for (const hm::schema::PointCloudT* message : {&cloud->data, &voxel_cloud->data}) {
    if (auto checked = hm::check_point_cloud(*message); !checked) {
      return checked;
    }
    if (!message->header || message->header->frame_id != kCameraFrame) {
      return holoscan::make_unexpected(invalid("a cloud lost the camera's frame id"));
    }
  }
  const std::span<const std::int64_t> shape = cloud->data.points->shape_span();
  if (!cloud->data.organized || shape[0] != config_.height || shape[1] != config_.width) {
    return holoscan::make_unexpected(invalid("expected the organized [H, W, 3] cloud"));
  }
  if (auto copied = to_host(*cloud->data.points, xyz_, context); !copied) {
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
      const bool left = finite_point(xyz_.data() + 3 * i);
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

  if (auto checked = check_voxels(voxel_cloud->data, context); !checked) {
    return checked;
  }
  if (config_.view_output) {
    // Best effort: the display keeps only the newest view and never holds this back.
    (void)send_view(voxel_xyz_.size() / 3, holoscan::sensor_io::derive_from(*cloud));
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

holoscan::expected<void, holoscan::Error> FilterReportOp::check_voxels(
    const hm::schema::PointCloudT& cloud, holoscan::ExecutionContext& context) {
  if (!hm::is_present(cloud.colors) || cloud.organized) {
    return holoscan::make_unexpected(invalid("expected unorganized voxels with colors"));
  }
  std::vector<float> points;
  if (auto copied = to_host(*cloud.points, points, context); !copied) {
    return copied;
  }
  if (auto copied = to_host(*cloud.colors, voxel_rgb_, context); !copied) {
    return copied;
  }
  // The voxels come first; with kPadded only removed points follow them.
  const std::size_t slots = points.size() / 3;
  std::size_t count = 0;
  while (count < slots && finite_point(points.data() + 3 * count)) {
    ++count;
  }
  bool laid_out = true;
  for (std::size_t i = count; i < slots; ++i) {
    laid_out = laid_out && std::isnan(points[3 * i]) && std::isnan(points[3 * i + 1]) &&
               std::isnan(points[3 * i + 2]);
  }
  if (config_.voxel_layout == pf::OutputLayout::kDense) {
    laid_out = laid_out && count == slots && cloud.valid_count == static_cast<std::int64_t>(count);
  } else {
    laid_out = laid_out && slots == xyz_.size() / 3 && cloud.valid_count == -1;
  }
  FilterReport& report = *config_.report;
  for (std::size_t i = 0; i < count; ++i) {
    report.black_voxels +=
        voxel_rgb_[3 * i] == 0U && voxel_rgb_[3 * i + 1] == 0U && voxel_rgb_[3 * i + 2] == 0U;
  }
  report.voxels += count;
  if (!laid_out || count != voxel_count(xyz_, config_.voxel_size)) {
    ++report.wrong_voxel_counts;
  }
  points.resize(3 * count);
  voxel_xyz_ = std::move(points);
  return {};
}

holoscan::expected<void, holoscan::Error> FilterReportOp::send_view(
    std::size_t voxel_count, const holoscan::EmitOptions& emit) {
  const int w = config_.width;
  const int h = config_.height;
  const pf::SyntheticScene scene = pf::synthetic_scene(w, h, 0);
  auto loan = view.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = std::array<std::int64_t, 3>{h, w, 3}, .dtype = kUInt8});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto committed = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                   -> holoscan::expected<void, holoscan::Error> {
    auto* image = writer.data_as<std::uint8_t>();
    std::memset(image, 0, std::size_t{3U} * w * h);
    std::vector<float> nearest(static_cast<std::size_t>(w) * h,
                               std::numeric_limits<float>::infinity());
    for (std::size_t i = 0; i < voxel_count; ++i) {
      const float* p = voxel_xyz_.data() + 3 * i;
      if (!(p[2] > 0.0F)) {
        continue;
      }
      const int u = static_cast<int>(std::lround(scene.fx * p[0] / p[2] + scene.cx));
      const int v = static_cast<int>(std::lround(scene.fy * p[1] / p[2] + scene.cy));
      for (int y = v - kVoxelPixels / 2; y <= v + kVoxelPixels / 2; ++y) {
        for (int x = u - kVoxelPixels / 2; x <= u + kVoxelPixels / 2; ++x) {
          if (x < 0 || y < 0 || x >= w || y >= h) {
            continue;
          }
          const std::size_t pixel = static_cast<std::size_t>(y) * w + x;
          if (p[2] < nearest[pixel]) {
            nearest[pixel] = p[2];
            std::copy_n(voxel_rgb_.data() + 3 * i, 3, image + 3 * pixel);
          }
        }
      }
    }
    return {};
  });
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  holoscan::schema::ImageT descriptor;
  descriptor.width = w;
  descriptor.height = h;
  descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
  return view.emit_tensor(std::move(*loan), descriptor, emit);
}

}  // namespace pcl_filters_app
