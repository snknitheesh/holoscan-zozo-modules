// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "report_op.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/tsdf/marching_cubes.hpp>
#include <holoscan/tsdf/synthetic_scene.hpp>

namespace tsdf_reconstruction_app {
namespace {

namespace ht = holoscan::tsdf;
using std::chrono_literals::operator""ns;

holoscan::Error failure(const char* op, const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, std::string(op) + ": " + message};
}

// Completes the run once every frame and every mesh has been checked.
void maybe_complete(Report& report, holoscan::ExecutionContext& context) {
  if (!report.completed && report.frames >= report.expected_frames &&
      report.meshes >= report.expected_meshes) {
    report.completed = true;
    context.request_completion();
  }
}

double percentile(std::vector<double>& values, std::size_t percent) {
  if (values.empty()) {
    return 0.0;
  }
  const std::size_t at = std::min(values.size() - 1, values.size() * percent / 100);
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(at), values.end());
  return values[at];
}

}  // namespace

RaycastReportOp::RaycastReportOp(ReportConfig config) : config_(std::move(config)) {
  if (!config_.report) {
    throw std::invalid_argument("RaycastReportOp needs a report");
  }
}

void RaycastReportOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(depth, "depth");
  spec.input(normals, "normals");
  spec.input(colors, "colors");
  spec.input(pose, "pose");
}

holoscan::Contract RaycastReportOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnAligned{
      .ports = holoscan::PortList{std::vector<holoscan::PortRef>{depth, normals, colors, pose}},
      .window = 0ns});
  result.input(depth, holoscan::Exact{});
  result.input(normals, holoscan::Exact{});
  result.input(colors, holoscan::Exact{});
  result.input(pose, holoscan::Exact{});
  return result;
}

holoscan::expected<void, holoscan::Error> RaycastReportOp::compute(
    holoscan::ExecutionContext& context) {
  auto depth_sample = depth.receive();
  auto normal_sample = normals.receive();
  auto color_sample = colors.receive();
  auto pose_sample = pose.receive();
  if (!depth_sample) {
    return holoscan::make_unexpected(std::move(depth_sample).error());
  }
  if (!normal_sample) {
    return holoscan::make_unexpected(std::move(normal_sample).error());
  }
  if (!color_sample) {
    return holoscan::make_unexpected(std::move(color_sample).error());
  }
  if (!pose_sample) {
    return holoscan::make_unexpected(std::move(pose_sample).error());
  }
  const holoscan::schema::ImageT& image = depth_sample->data;
  const holoscan::schema::ImageT& color_image = color_sample->data;
  if (!image.data || image.width != config_.width || image.height != config_.height ||
      !color_image.data) {
    return holoscan::make_unexpected(failure("RaycastReportOp", "unexpected ray-cast image"));
  }
  const std::size_t count = static_cast<std::size_t>(config_.width) * config_.height;
  pixels_.resize(count);
  normals_.resize(3U * count);
  colors_.resize(3U * count);
  if (auto copied = image.data->copy_to_host(pixels_.data(), count * 4U, context.cuda_stream());
      !copied) {
    return copied;
  }
  if (auto copied =
          normal_sample->data.copy_to_host(normals_.data(), count * 12U, context.cuda_stream());
      !copied) {
    return copied;
  }
  if (auto copied =
          color_image.data->copy_to_host(colors_.data(), count * 3U, context.cuda_stream());
      !copied) {
    return copied;
  }
  const holoscan::Tensor& pose_tensor = pose_sample->data;
  if (pose_tensor.size() != 16 || pose_tensor.dtype().bits != 32) {
    return holoscan::make_unexpected(failure("RaycastReportOp", "expected a float32 [4, 4] pose"));
  }
  if (auto ready = pose_tensor.synchronize_for_host(); !ready) {
    return holoscan::make_unexpected(std::move(ready).error());
  }
  ht::Matrix4 world_from_camera{};
  std::memcpy(world_from_camera.data(), pose_tensor.data(), sizeof(world_from_camera));

  const ht::SyntheticScene scene = ht::synthetic_scene(ht::SceneKind::kTabletop);
  const ht::CameraIntrinsics k = ht::synthetic_intrinsics(config_.width, config_.height);
  const ht::TsdfVolumeConfig& c = config_.volume;
  std::vector<double> errors;
  std::size_t expected = 0;
  std::uint64_t bad_hits = 0;
  std::uint64_t hits = 0;
  std::uint64_t colored = 0;
  for (int v = 0; v < config_.height; ++v) {
    for (int u = 0; u < config_.width; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * config_.width + u;
      if (pixels_[i] > 0.0F) {
        const float* n = &normals_[3U * i];
        const float length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        const float facing = n[0] * (static_cast<float>(u) - k.cx) / k.fx +
                             n[1] * (static_cast<float>(v) - k.cy) / k.fy + n[2];
        bad_hits += std::abs(length - 1.0F) < 1e-3F && facing < 0.0F ? 0U : 1U;
        colored += colors_[3U * i] != 0U || colors_[3U * i + 1] != 0U || colors_[3U * i + 2] != 0U
                       ? 1U
                       : 0U;
        ++hits;
      }
      const float truth = ht::scene_depth(scene, world_from_camera.data(), k, static_cast<float>(u),
                                          static_cast<float>(v), 3.0F, nullptr);
      if (!(truth > 0.0F)) {
        continue;
      }
      // Only surfaces inside the volume, away from its faces, can be seen.
      const float ray[3] = {(static_cast<float>(u) - k.cx) / k.fx * truth,
                            (static_cast<float>(v) - k.cy) / k.fy * truth, truth};
      bool inside = true;
      for (int r = 0; r < 3; ++r) {
        const float p = world_from_camera[4 * r] * ray[0] + world_from_camera[4 * r + 1] * ray[1] +
                        world_from_camera[4 * r + 2] * ray[2] + world_from_camera[4 * r + 3];
        inside = inside && p > c.origin[r] + 3.0F * c.voxel_size &&
                 p < c.origin[r] + static_cast<float>(c.dims[r] - 4) * c.voxel_size;
      }
      if (!inside) {
        continue;
      }
      ++expected;
      const float got = pixels_[i];
      if (got > 0.0F) {
        errors.push_back(std::abs(static_cast<double>(got) - truth));
      }
    }
  }
  const double far = 3.0 * config_.volume.voxel_size;
  const auto outliers = static_cast<double>(
      std::count_if(errors.begin(), errors.end(), [far](double e) { return e > far; }));
  const double outlier_share = errors.empty() ? 0.0 : outliers / static_cast<double>(errors.size());
  const double coverage =
      expected == 0 ? 1.0 : static_cast<double>(errors.size()) / static_cast<double>(expected);
  const double p95 = percentile(errors, 95);
  const double median = percentile(errors, 50);

  Report& report = *config_.report;
  const std::lock_guard lock(report.mutex);
  report.bad_hits += bad_hits;
  // Nearly every hit has the color of the surface it hit.
  report.bad_hits += 100U * colored < 99U * hits ? 1U : 0U;
  if (report.frames >= report.warmup_frames) {
    report.worst_median_m = std::max(report.worst_median_m, median);
    report.worst_outliers = std::max(report.worst_outliers, outlier_share);
    report.worst_p95_m = std::max(report.worst_p95_m, p95);
    report.worst_coverage = std::min(report.worst_coverage, coverage);
  }
  ++report.frames;
  if (config_.finished) {
    config_.finished->fetch_add(1U, std::memory_order_release);
  }
  maybe_complete(report, context);
  return {};
}

MeshReportOp::MeshReportOp(ReportConfig config) : config_(std::move(config)) {
  if (!config_.report) {
    throw std::invalid_argument("MeshReportOp needs a report");
  }
}

void MeshReportOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(vertices, "vertices");
  spec.input(normals, "normals");
  spec.input(colors, "colors");
  spec.input(triangles, "triangles");
  spec.input(counts, "counts");
}

holoscan::Contract MeshReportOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{std::vector<holoscan::PortRef>{
                                         vertices, normals, colors, triangles, counts}},
                                     .window = 0ns});
  for (const auto* port : {&vertices, &normals, &colors, &triangles, &counts}) {
    result.input(*port, holoscan::Exact{});
  }
  return result;
}

holoscan::expected<void, holoscan::Error> MeshReportOp::compute(
    holoscan::ExecutionContext& context) {
  auto vertex_sample = vertices.receive();
  auto normal_sample = normals.receive();
  auto color_sample = colors.receive();
  auto triangle_sample = triangles.receive();
  auto count_sample = counts.receive();
  for (const auto* received :
       {&vertex_sample, &normal_sample, &color_sample, &triangle_sample, &count_sample}) {
    if (!*received) {
      return holoscan::make_unexpected(received->error());
    }
  }
  std::array<std::uint32_t, ht::kMeshCountsSize> mesh_counts{};
  if (auto copied = count_sample->data.copy_to_host(mesh_counts.data(), sizeof(mesh_counts),
                                                    context.cuda_stream());
      !copied) {
    return copied;
  }
  const std::span<const std::int64_t> shape = vertex_sample->data.shape_span();
  const std::span<const std::int64_t> triangle_shape = triangle_sample->data.shape_span();
  const std::uint32_t n = mesh_counts[ht::kCountVertices];
  const std::uint32_t t = mesh_counts[ht::kCountTriangles];
  if (shape.size() != 2 || shape[1] != 3 || n > shape[0] || triangle_shape.size() != 2 ||
      t > triangle_shape[0]) {
    return holoscan::make_unexpected(failure("MeshReportOp", "unexpected mesh shapes"));
  }
  // Only the counted vertices and triangles are copied.
  points_.resize(static_cast<std::size_t>(n) * 3U);
  normals_.resize(static_cast<std::size_t>(n) * 3U);
  triangles_.resize(static_cast<std::size_t>(t) * 3U);
  cudaStream_t stream = context.cuda_stream().get();
  if ((n != 0U &&
       (cudaMemcpyAsync(points_.data(), vertex_sample->data.data(), points_.size() * 4U,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(normals_.data(), normal_sample->data.data(), normals_.size() * 4U,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess)) ||
      (t != 0U &&
       cudaMemcpyAsync(triangles_.data(), triangle_sample->data.data(), triangles_.size() * 4U,
                       cudaMemcpyDeviceToHost, stream) != cudaSuccess) ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    return holoscan::make_unexpected(failure("MeshReportOp", "copying the mesh failed"));
  }
  std::uint64_t bad_triangles = 0;
  for (const std::uint32_t index : triangles_) {
    bad_triangles += index < n ? 0U : 1U;
  }
  std::uint64_t bad_normals = 0;
  for (std::uint32_t i = 0; i < n; ++i) {
    const float* v = &normals_[3U * i];
    bad_normals +=
        std::abs(std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) - 1.0F) < 1e-3F ? 0U : 1U;
  }
  const ht::SyntheticScene scene = ht::synthetic_scene(ht::SceneKind::kTabletop);
  std::vector<double> errors;
  errors.reserve(n);
  double sum = 0.0;
  for (std::uint32_t i = 0; i < n; ++i) {
    errors.push_back(std::abs(ht::scene_distance(scene, &points_[3U * i])));
    sum += errors.back();
  }
  Report& report = *config_.report;
  const std::lock_guard lock(report.mutex);
  ++report.meshes;
  report.bad_triangles += bad_triangles;
  report.bad_normals += bad_normals;
  report.overflows +=
      mesh_counts[ht::kCountTriangles] < mesh_counts[ht::kCountNeededTriangles] ? 1U : 0U;
  report.last_vertices = n;
  report.last_triangles = mesh_counts[ht::kCountTriangles];
  report.last_mean_m = n == 0U ? 0.0 : sum / n;
  report.last_max_m = errors.empty() ? 0.0 : *std::max_element(errors.begin(), errors.end());
  report.last_p95_m = percentile(errors, 95);
  maybe_complete(report, context);
  return {};
}

}  // namespace tsdf_reconstruction_app
