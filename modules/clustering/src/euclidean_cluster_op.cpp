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

#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>

namespace holoscan::clustering {
namespace {

namespace hm = holoscan::messages;

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kInt32{kDLInt, 32U, 1U};

CloudSettings settings_of(const EuclideanClusterConfig& config) {
  return CloudSettings{.layout = config.layout,
                       .max_points = config.max_points,
                       .fields = config.emit_points ? config.fields : hm::PointCloudContents{}};
}

// A device kernel's launch status as the loan's fill result.
holoscan::expected<void, holoscan::Error> launched(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure,
        std::string("EuclideanClusterOp: writing ") + what + ": " + cudaGetErrorString(status)});
  }
  return {};
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
  if ((config.fields.colors || config.fields.normals || config.fields.intensity) &&
      !config.emit_points) {
    throw std::invalid_argument(
        "fields need emit_points: they go out in the clustered point cloud");
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
  // An optional output exists only when configured: an emit into an
  // unconnected output fails.
  if (config_.emit_points) {
    hm::declare_point_cloud_output(spec, clustered_point_cloud, "clustered_point_cloud",
                                   {.max_points = points_max,
                                    .colors = config_.fields.colors,
                                    .normals = config_.fields.normals,
                                    .intensity = config_.fields.intensity});
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
  if (config_.clustering.connectivity == Connectivity::kImage && cloud.height == 0U) {
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
  const std::span<const std::int64_t> shape = cloud.sample.data.points->shape_span();
  const std::vector<std::int64_t> keep_label_shape(shape.begin(), shape.end() - 1);
  const std::array<std::int64_t, 1> dense_label_shape{clustered};

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
  if (config_.emit_points) {
    return emit_clustered_cloud(context, cloud, clustered);
  }
  return {};
}

holoscan::expected<void, holoscan::Error> EuclideanClusterOp::emit_clustered_cloud(
    holoscan::ExecutionContext& context, const Cloud& cloud, std::uint32_t clustered) {
  const bool dense = config_.layout == OutputLayout::kDense;
  const hm::PointCloudShape shape = dense ? hm::unorganized_cloud(clustered)
                                    : cloud.height != 0U
                                        ? hm::organized_cloud(cloud.height, cloud.width)
                                        : hm::unorganized_cloud(cloud.count);
  const std::uint32_t points = dense ? clustered : cloud.count;
  const hm::PointCloudContents contents{.colors = cloud.rgb != nullptr,
                                        .normals = cloud.normals != nullptr,
                                        .intensity = cloud.intensity != nullptr};
  auto loan =
      hm::allocate_point_cloud(clustered_point_cloud, shape, contents, context.cuda_stream());
  if (!loan) {
    return emit_failure("clustered_point_cloud", std::move(loan).error());
  }
  const auto fill = [&](holoscan::TensorFieldSelector<hm::schema::PointCloudT> field,
                        const char* what,
                        const auto& write) -> holoscan::expected<void, holoscan::Error> {
    return loan->write_and_commit(field, context.cuda_stream(),
                                  [&](const holoscan::TensorOutputWriteGuard& guard)
                                      -> holoscan::expected<void, holoscan::Error> {
                                    if (points == 0U) {
                                      return {};
                                    }
                                    return launched(write(guard), what);
                                  });
  };
  if (auto written = fill(hm::PointCloudFields::points, "clustered points",
                          [&](const holoscan::TensorOutputWriteGuard& guard) {
                            auto* out = guard.data_as<float>();
                            return dense ? clusterer_->write_grouped(cloud.xyz, nullptr, out,
                                                                     nullptr, nullptr, cloud.stream)
                                         : clusterer_->write_kept(cloud.xyz, nullptr, out, nullptr,
                                                                  cloud.stream);
                          });
      !written) {
    return written;
  }
  if (contents.colors) {
    if (auto written = fill(hm::PointCloudFields::colors, "clustered colors",
                            [&](const holoscan::TensorOutputWriteGuard& guard) {
                              auto* out = guard.data_as<std::uint8_t>();
                              return dense
                                         ? clusterer_->write_grouped(cloud.xyz, cloud.rgb, nullptr,
                                                                     out, nullptr, cloud.stream)
                                         : clusterer_->write_kept(cloud.xyz, cloud.rgb, nullptr,
                                                                  out, cloud.stream);
                            });
        !written) {
      return written;
    }
  }
  const auto values = [&](holoscan::TensorFieldSelector<hm::schema::PointCloudT> field,
                          const char* what, const float* in, int channels) {
    return fill(field, what, [&](const holoscan::TensorOutputWriteGuard& guard) {
      auto* out = guard.data_as<float>();
      return dense ? clusterer_->write_grouped_values(in, channels, out, cloud.stream)
                   : clusterer_->write_kept_values(in, channels, out, cloud.stream);
    });
  };
  if (contents.normals) {
    if (auto written = values(hm::PointCloudFields::normals, "clustered normals", cloud.normals, 3);
        !written) {
      return written;
    }
  }
  if (contents.intensity) {
    if (auto written =
            values(hm::PointCloudFields::intensity, "clustered intensity", cloud.intensity, 1);
        !written) {
      return written;
    }
  }
  // Packed by cluster, every point is valid; kept in place, they are not
  // counted (that would take another wait).
  const std::int64_t valid_count = dense ? std::int64_t{clustered} : -1;
  if (auto sent = hm::emit_point_cloud(clustered_point_cloud, std::move(*loan), cloud.header, shape,
                                       valid_count, holoscan::sensor_io::derive_from(cloud.sample));
      !sent) {
    return emit_failure("clustered_point_cloud", std::move(sent).error());
  }
  return {};
}

}  // namespace holoscan::clustering
