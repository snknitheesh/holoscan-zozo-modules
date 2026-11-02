// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "report_op.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/normals/planes.hpp>
#include <holoscan/normals/synthetic_scene.hpp>

namespace normals_planes_app {
namespace {

using std::chrono_literals::operator""ns;
namespace hn = holoscan::normals;

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "ReportOp: " + message};
}

double angle_deg(const float* a, const float* b) {
  const double dot = static_cast<double>(a[0]) * b[0] + static_cast<double>(a[1]) * b[1] +
                     static_cast<double>(a[2]) * b[2];
  return std::acos(std::clamp(dot, -1.0, 1.0)) * 180.0 / std::numbers::pi;
}

double percentile(std::vector<double>& values, double q) {
  if (values.empty()) {
    return 0.0;
  }
  const auto at = std::min(values.size() - 1, static_cast<std::size_t>(q * values.size()));
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(at), values.end());
  return values[at];
}

}  // namespace

ReportOp::ReportOp(ReportConfig config) : config_(std::move(config)) {
  if (!config_.report) {
    throw std::invalid_argument("ReportOp needs a report");
  }
}

void ReportOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(normals, "normals").queue_depth(2U);
  spec.input(curvature, "curvature").queue_depth(2U);
  spec.input(labels, "labels").queue_depth(2U);
  spec.input(coefficients, "coefficients").queue_depth(2U);
  spec.input(centroids, "centroids").queue_depth(2U);
  spec.input(inliers, "inliers").queue_depth(2U);
}

holoscan::Contract ReportOp::contract() const {
  holoscan::Contract result;
  result.trigger(
      holoscan::OnAligned{.ports = holoscan::PortList{std::vector<holoscan::PortRef>{
                              normals, curvature, labels, coefficients, centroids, inliers}},
                          .window = 0ns});
  for (const auto* port : {&normals, &curvature, &labels, &coefficients, &centroids, &inliers}) {
    result.input(*port, holoscan::Exact{});
  }
  return result;
}

holoscan::expected<void, holoscan::Error> ReportOp::compute(holoscan::ExecutionContext& context) {
  auto normal_sample = normals.receive();
  auto curvature_sample = curvature.receive();
  auto label_sample = labels.receive();
  auto coefficient_sample = coefficients.receive();
  auto centroid_sample = centroids.receive();
  auto inlier_sample = inliers.receive();
  for (const auto* failed : {&normal_sample, &curvature_sample, &label_sample, &coefficient_sample,
                             &centroid_sample, &inlier_sample}) {
    if (!*failed) {
      return holoscan::make_unexpected(failed->error());
    }
  }
  const int width = config_.width;
  const int height = config_.height;
  const std::span<const std::int64_t> shape = normal_sample->data.shape_span();
  if (shape.size() != 3 || shape[0] != height || shape[1] != width || shape[2] != 3) {
    return holoscan::make_unexpected(invalid("expected organized [H, W, 3] normals"));
  }
  const std::size_t pixels = static_cast<std::size_t>(width) * height;
  const std::size_t planes = config_.max_planes;
  normals_.resize(3 * pixels);
  curvature_.resize(pixels);
  labels_.resize(pixels);
  std::vector<float> got(4 * planes);
  const auto stream = context.cuda_stream();
  for (auto copied : {normal_sample->data.copy_to_host(normals_.data(), 12 * pixels, stream),
                      curvature_sample->data.copy_to_host(curvature_.data(), 4 * pixels, stream),
                      label_sample->data.copy_to_host(labels_.data(), pixels, stream),
                      coefficient_sample->data.copy_to_host(got.data(), 16 * planes, stream)}) {
    if (!copied) {
      return copied;
    }
  }

  Report& report = *config_.report;
  const std::uint64_t frame_id = normal_sample->metadata.frame_id;
  report.frame_gaps += frame_id != last_frame_id_ + 1U ? 1U : 0U;
  last_frame_id_ = frame_id;
  const hn::SyntheticScene scene =
      hn::synthetic_scene(width, height, static_cast<std::uint32_t>(frame_id - 1U));
  std::vector<hn::ScenePixel> truth(pixels);
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      truth[static_cast<std::size_t>(v) * width + u] = hn::synthetic_scene_pixel(scene, u, v);
    }
  }

  // The normals on the inside of each surface, and the curvature there.
  const int r = config_.window_radius;
  std::vector<double> angles;
  std::vector<double> flat;
  std::vector<double> sphere;
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * width + u;
      const std::uint8_t surface = truth[i].surface;
      if (surface == hn::kSurfaceNone || !std::isfinite(normals_[3 * i])) {
        continue;
      }
      bool alone = true;
      for (int dv = -r; dv <= r && alone; ++dv) {
        for (int du = -r; du <= r && alone; ++du) {
          const int row = v + dv;
          const int column = u + du;
          if (row >= 0 && row < height && column >= 0 && column < width) {
            const std::uint8_t other =
                truth[static_cast<std::size_t>(row) * width + column].surface;
            alone = other == hn::kSurfaceNone || other == surface;
          }
        }
      }
      if (!alone) {
        continue;
      }
      const double angle = angle_deg(normals_.data() + 3 * i, truth[i].normal);
      report.flipped += angle > 90.0 ? 1U : 0U;
      angles.push_back(angle);
      float plane[4];
      (hn::synthetic_surface_plane(scene, surface, plane) ? flat : sphere).push_back(curvature_[i]);
    }
  }
  report.worst_median_deg = std::max(report.worst_median_deg, percentile(angles, 0.5));
  report.worst_p95_deg = std::max(report.worst_p95_deg, percentile(angles, 0.95));
  report.flat_sphere += percentile(sphere, 0.5) > percentile(flat, 0.5) ? 0U : 1U;

  // The large planes among those found.
  std::uint32_t found = 0;
  while (found < planes && std::isfinite(got[4 * found])) {
    ++found;
  }
  report.planes += found;
  for (const std::uint8_t surface :
       {hn::kSurfaceFloor, hn::kSurfaceBackWall, hn::kSurfaceSideWall, hn::kSurfaceTableTop}) {
    float want[4];
    hn::synthetic_surface_plane(scene, surface, want);
    int best = -1;
    double best_angle = 180.0;
    double best_offset = 1e30;
    for (std::uint32_t k = 0; k < found; ++k) {
      const double angle = angle_deg(got.data() + 4 * k, want);
      const double offset = std::fabs(got[4 * k + 3] - want[3]);
      if (angle < 5.0 && offset < best_offset) {
        best = static_cast<int>(k);
        best_angle = angle;
        best_offset = offset;
      }
    }
    std::size_t on_surface = 0;
    std::size_t labeled = 0;
    std::size_t both = 0;
    const auto label = static_cast<std::uint8_t>(best + 1);
    for (std::size_t i = 0; i < pixels && best >= 0; ++i) {
      on_surface += truth[i].surface == surface ? 1U : 0U;
      labeled += labels_[i] == label ? 1U : 0U;
      both += truth[i].surface == surface && labels_[i] == label ? 1U : 0U;
    }
    const double recall = on_surface == 0 ? 0.0 : static_cast<double>(both) / on_surface;
    const double precision = labeled == 0 ? 0.0 : static_cast<double>(both) / labeled;
    if (best < 0 || best_angle > kMaxPlaneAngleDeg || best_offset > kMaxPlaneOffsetM ||
        recall < kMinRecall || precision < kMinPrecision) {
      ++report.wrong_planes;
    }
    report.worst_plane_angle_deg = std::max(report.worst_plane_angle_deg, best_angle);
    report.worst_plane_offset_m =
        std::max(report.worst_plane_offset_m, best < 0 ? 1e9 : best_offset);
    report.worst_recall = std::min(report.worst_recall, recall);
    report.worst_precision = std::min(report.worst_precision, precision);
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

}  // namespace normals_planes_app
