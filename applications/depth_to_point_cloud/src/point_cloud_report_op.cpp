// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "point_cloud_report_op.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/point_cloud/synthetic_depth.hpp>

#include "image_io.hpp"

namespace depth_to_point_cloud_app {
namespace {

using holoscan::point_cloud::CameraIntrinsics;
using holoscan::point_cloud::synthetic_color;
using holoscan::point_cloud::synthetic_depth;
using holoscan::point_cloud::synthetic_depth_mm;
using holoscan::point_cloud::synthetic_frame;
using holoscan::point_cloud::SyntheticFrame;

namespace hm = holoscan::messages;

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "PointCloudReportOp: " + message};
}

// Copies `tensor` (the shape [H, W, 3] of element `T`) to `out`, waiting for
// its producer.
template <typename T>
holoscan::expected<void, holoscan::Error> fetch(const holoscan::Tensor& tensor,
                                                ::cuda::stream_ref stream, std::vector<T>& out,
                                                int& width, int& height) {
  const std::span<const std::int64_t> shape = tensor.shape_span();
  if (shape.size() != 3 || shape[2] != 3 ||
      tensor.nbytes() != static_cast<std::int64_t>(shape[0] * shape[1] * 3 * sizeof(T))) {
    return holoscan::make_unexpected(invalid("expected a packed [H, W, 3] tensor"));
  }
  height = static_cast<int>(shape[0]);
  width = static_cast<int>(shape[1]);
  out.resize(static_cast<std::size_t>(shape[0] * shape[1] * 3));
  return tensor.copy_to_host(out.data(), out.size() * sizeof(T), stream);
}

}  // namespace

PointCloudReportOp::PointCloudReportOp(PointCloudReportConfig config) : config_(std::move(config)) {
  if (!config_.report) {
    throw std::invalid_argument("PointCloudReportOp needs a report");
  }
}

void PointCloudReportOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots: one cloud in compute() while the next one arrives.
  spec.input(point_cloud, "point_cloud").queue_depth(2U);
}

holoscan::Contract PointCloudReportOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{point_cloud});
  return result;
}

// Each run counts its own clouds and frame ids from the start.
void PointCloudReportOp::start() { last_frame_id_ = 0; }

bool PointCloudReportOp::well_formed(const hm::schema::PointCloudT& cloud) const {
  return hm::check_point_cloud(cloud).has_value() && cloud.organized && cloud.valid_count == -1 &&
         cloud.header != nullptr && cloud.header->frame_id == config_.frame_id &&
         hm::is_present(cloud.colors) == config_.colors && !hm::is_present(cloud.normals) &&
         !hm::is_present(cloud.intensity);
}

holoscan::expected<void, holoscan::Error> PointCloudReportOp::compute(
    holoscan::ExecutionContext& context) {
  auto cloud = point_cloud.receive();
  if (!cloud) {
    return holoscan::make_unexpected(std::move(cloud).error());
  }
  CloudReport& report = *config_.report;
  if (!well_formed(cloud->data)) {
    return holoscan::make_unexpected(
        invalid("the cloud is not the operator's organized cloud of the camera's frame"));
  }
  int width = 0;
  int height = 0;
  if (auto fetched = fetch(*cloud->data.points, context.cuda_stream(), xyz_, width, height);
      !fetched) {
    return fetched;
  }
  const std::vector<std::uint8_t>* rgb = nullptr;
  if (config_.colors) {
    int color_width = 0;
    int color_height = 0;
    if (auto fetched =
            fetch(*cloud->data.colors, context.cuda_stream(), rgb_, color_width, color_height);
        !fetched) {
      return fetched;
    }
    rgb = &rgb_;
  }

  // Sources number their frames from 1; the cloud keeps its depth image's id.
  // The frames before the first cloud came before the first camera info.
  const std::uint64_t frame_id = cloud->metadata.frame_id;
  if (report.frames == 0) {
    report.first_frame_id = frame_id;
  } else if (frame_id != last_frame_id_ + 1U) {
    ++report.frame_gaps;
  }
  last_frame_id_ = frame_id;
  const std::size_t points = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  float lo = std::numeric_limits<float>::infinity();
  float hi = -std::numeric_limits<float>::infinity();
  std::uint64_t valid = 0;
  for (std::size_t i = 0; i < points; ++i) {
    const float z = xyz_[3 * i + 2];
    if (std::isfinite(z)) {
      ++valid;
      lo = std::min(lo, z);
      hi = std::max(hi, z);
    }
  }
  if (valid != 0) {
    report.min_depth = report.valid_points == 0 ? lo : std::min(report.min_depth, lo);
    report.max_depth = report.valid_points == 0 ? hi : std::max(report.max_depth, hi);
  }
  report.points += points;
  report.valid_points += valid;
  if (config_.truth) {
    verify(static_cast<std::uint32_t>(frame_id - 1U), xyz_, rgb, width, height);
  }
  if (!config_.ply_dir.empty()) {
    // Named by frame, from 0, as the synthetic camera numbers its frames.
    char name[32];
    // (Sources count at most 2^32 - 1 frames.)
    std::snprintf(name, sizeof(name), "frame_%06u.ply", static_cast<unsigned>(frame_id - 1U));
    try {
      write_ply(config_.ply_dir / name, xyz_, rgb);
    } catch (const std::exception& error) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, error.what()});
    }
    ++report.ply_files;
  }
  ++report.frames;
  if (config_.expected_frames != 0 && frame_id == config_.expected_frames) {
    context.request_completion();
  }
  return {};
}

void PointCloudReportOp::verify(std::uint32_t frame_number, const std::vector<float>& xyz,
                                const std::vector<std::uint8_t>* rgb, int width, int height) {
  CloudReport& report = *config_.report;
  const CloudTruth& truth = *config_.truth;
  report.verified = true;
  if (width != truth.width || height != truth.height) {
    report.wrong_points += static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
    return;
  }
  const SyntheticFrame frame = synthetic_frame(width, height, frame_number, truth.vary_intrinsics);
  const CameraIntrinsics& k = frame.intrinsics;
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * width + u;
      const float z = truth.millimeters
                          ? static_cast<float>(synthetic_depth_mm(frame, u, v)) * truth.depth_scale
                          : synthetic_depth(frame, u, v) * truth.depth_scale;
      const bool expect_valid = z > 0.0F && z >= truth.depth_min && z <= truth.depth_max;
      const float* p = xyz.data() + 3 * i;
      const bool got_valid = std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
      if (expect_valid != got_valid) {
        ++report.wrong_points;
        continue;
      }
      if (expect_valid) {
        const double ex = (u - k.cx) * z / k.fx;
        const double ey = (v - k.cy) * z / k.fy;
        const double error = std::hypot(p[0] - ex, p[1] - ey, p[2] - static_cast<double>(z));
        report.max_error_m = std::max(report.max_error_m, error);
        if (error > kMaxPointError) {
          ++report.wrong_points;
        }
      }
      if (rgb != nullptr) {
        std::uint8_t want[3];
        synthetic_color(frame, u, v, want);
        const std::uint8_t* got = rgb->data() + 3 * i;
        if (got[0] != want[0] || got[1] != want[1] || got[2] != want[2]) {
          ++report.wrong_colors;
        }
      }
    }
  }
}

}  // namespace depth_to_point_cloud_app
