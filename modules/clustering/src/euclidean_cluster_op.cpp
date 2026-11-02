// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/clustering/euclidean_cluster_op.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace holoscan::clustering {
namespace {

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kInt32{kDLInt, 32U, 1U};
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

CloudSettings settings_of(const EuclideanClusterConfig& config) {
  return CloudSettings{
      .color = config.color, .layout = config.layout, .max_points = config.max_points};
}

}  // namespace

void validate_config(const EuclideanClusterConfig& config) {
  validate_settings(settings_of(config));
  const ClusterParams& p = config.clustering;
  if (!(p.tolerance > 0.0F) || !std::isfinite(p.tolerance)) {
    throw std::invalid_argument("tolerance must be finite and positive");
  }
  if (p.min_cluster_size < 1U || p.min_cluster_size > p.max_cluster_size) {
    throw std::invalid_argument("cluster sizes need 1 <= min_cluster_size <= max_cluster_size");
  }
  if (p.max_clusters < 1U || p.max_clusters > kMaxClustersLimit) {
    throw std::invalid_argument("max_clusters must be 1.." + std::to_string(kMaxClustersLimit));
  }
  if (p.connectivity != Connectivity::kEuclidean && p.connectivity != Connectivity::kImage) {
    throw std::invalid_argument("connectivity must be kEuclidean or kImage");
  }
  if (p.connectivity == Connectivity::kImage &&
      (p.image_window < 1 || p.image_window > kMaxImageWindow)) {
    throw std::invalid_argument("image_window must be 1.." + std::to_string(kMaxImageWindow));
  }
  if (config.color && !config.emit_points) {
    throw std::invalid_argument("color needs emit_points: the colors go out with the points");
  }
}

EuclideanClusterOp::EuclideanClusterOp(EuclideanClusterConfig config)
    : CloudOp("EuclideanClusterOp", settings_of(config)),
      config_(config),
      clusterer_(std::make_unique<EuclideanClusterer>()) {
  validate_config(config_);
}

EuclideanClusterOp::~EuclideanClusterOp() = default;

void EuclideanClusterOp::setup(holoscan::OperatorSpec& spec) {
  CloudOp::setup(spec);
  const std::uint64_t points_max = config_.max_points;
  spec.output(labels, "labels").produces_tensor(device_tensor_spec(kInt32, 2U, points_max * 4U));
  spec.output(clusters, "clusters")
      .produces_tensor(device_tensor_spec(
          kFloat32, 2U, std::uint64_t{config_.clustering.max_clusters} * sizeof(ClusterSummary)));
  // Optional outputs exist only when configured: an emit into an unconnected
  // output fails.
  if (config_.emit_points) {
    spec.output(clustered_points, "clustered_points")
        .produces_tensor(device_tensor_spec(kFloat32, 3U, points_max * 3U * sizeof(float)));
  }
  if (config_.color) {
    spec.output(clustered_colors, "clustered_colors")
        .produces_tensor(device_tensor_spec(kUInt8, 3U, points_max * 3U));
  }
}

void EuclideanClusterOp::stop() {
  clusterer_ = std::make_unique<EuclideanClusterer>();
  CloudOp::stop();
}

holoscan::expected<void, holoscan::Error> EuclideanClusterOp::compute(
    holoscan::ExecutionContext& context) {
  auto received = receive(context);
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  const Cloud& cloud = *received;
  if (config_.clustering.connectivity == Connectivity::kImage && cloud.width == 0U) {
    return holoscan::make_unexpected(
        invalid("image connectivity needs an organized [H, W, 3] cloud"));
  }
  std::uint32_t found = 0;
  if (const cudaError_t status = clusterer_->cluster(cloud.xyz, cloud.count, cloud.width,
                                                     config_.clustering, found, cloud.stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("clustering", status));
  }
  const bool dense = config_.layout == OutputLayout::kDense;
  std::uint32_t clustered = 0;
  if (dense) {
    if (const cudaError_t status = clusterer_->clustered_count(clustered, cloud.stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("counting the clustered points", status));
    }
  }
  const std::span<const std::int64_t> shape = cloud.points.data.shape_span();
  const std::vector<std::int64_t> keep_shape(shape.begin(), shape.end());
  const std::vector<std::int64_t> keep_label_shape(shape.begin(), shape.end() - 1);
  const std::array<std::int64_t, 1> dense_label_shape{clustered};
  const std::array<std::int64_t, 2> dense_shape{clustered, 3};

  if (auto sent = emit_tensor(labels, "labels", cloud,
                              dense ? std::span<const std::int64_t>(dense_label_shape)
                                    : std::span<const std::int64_t>(keep_label_shape),
                              kInt32,
                              [&](void* data) {
                                auto* out = static_cast<std::int32_t*>(data);
                                return dense
                                           ? clusterer_->write_grouped(cloud.xyz, nullptr, nullptr,
                                                                       nullptr, out, cloud.stream)
                                           : clusterer_->write_labels(out, cloud.stream);
                              });
      !sent) {
    return sent;
  }
  const std::array<std::int64_t, 2> summary_shape{found, kSummaryFields};
  if (auto sent = emit_tensor(clusters, "clusters", cloud, summary_shape, kFloat32,
                              [&](void* data) {
                                return clusterer_->write_summaries(
                                    cloud.xyz, config_.oriented_boxes, static_cast<float*>(data),
                                    cloud.stream);
                              });
      !sent) {
    return sent;
  }
  const std::span<const std::int64_t> points_shape =
      dense ? std::span<const std::int64_t>(dense_shape)
            : std::span<const std::int64_t>(keep_shape);
  if (config_.emit_points) {
    if (auto sent = emit_tensor(
            clustered_points, "clustered_points", cloud, points_shape, kFloat32,
            [&](void* data) {
              auto* out = static_cast<float*>(data);
              return dense ? clusterer_->write_grouped(cloud.xyz, nullptr, out, nullptr, nullptr,
                                                       cloud.stream)
                           : clusterer_->write_kept(cloud.xyz, nullptr, out, nullptr, cloud.stream);
            });
        !sent) {
      return sent;
    }
  }
  if (config_.color) {
    if (auto sent = emit_tensor(clustered_colors, "clustered_colors", cloud, points_shape, kUInt8,
                                [&](void* data) {
                                  auto* out = static_cast<std::uint8_t*>(data);
                                  return dense ? clusterer_->write_grouped(cloud.xyz, cloud.rgb,
                                                                           nullptr, out, nullptr,
                                                                           cloud.stream)
                                               : clusterer_->write_kept(cloud.xyz, cloud.rgb,
                                                                        nullptr, out, cloud.stream);
                                });
        !sent) {
      return sent;
    }
  }
  return {};
}

}  // namespace holoscan::clustering
