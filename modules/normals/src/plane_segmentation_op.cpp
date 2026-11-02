// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/normals/plane_segmentation_op.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/normals/normals.hpp>

namespace holoscan::normals {
namespace {

using std::chrono_literals::operator""ns;

holoscan::TensorOutputSpec planes_output(DLDataType dtype, std::uint32_t rank,
                                         std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = dtype,
              .rank = rank,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(bytes),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

}  // namespace

void validate_config(const PlaneSegmentationConfig& config) {
  const char* what = nullptr;
  if (!valid_params(config.planes, &what)) {
    throw std::invalid_argument(what);
  }
  if (config.max_points < 1U || config.max_points > kMaxPointsLimit) {
    throw std::invalid_argument("max_points must be 1.." + std::to_string(kMaxPointsLimit));
  }
}

PlaneSegmentationOp::PlaneSegmentationOp(PlaneSegmentationConfig config)
    : CloudOp("PlaneSegmentationOp", config.max_points),
      config_(config),
      segmenter_(std::make_unique<PlaneSegmenter>()) {
  validate_config(config_);
}

PlaneSegmentationOp::~PlaneSegmentationOp() = default;

void PlaneSegmentationOp::setup(holoscan::OperatorSpec& spec) {
  const std::uint64_t planes = config_.planes.max_planes;
  spec.input(points, "points").queue_depth(2U);
  spec.input(normals, "normals").queue_depth(2U);
  spec.output(labels, "labels").produces_tensor(device_output(kUInt8, config_.max_points));
  spec.output(plane_coefficients, "plane_coefficients")
      .produces_tensor(planes_output(kFloat32, 2U, planes * 4U * sizeof(float)));
  spec.output(plane_centroids, "plane_centroids")
      .produces_tensor(planes_output(kFloat32, 2U, planes * 3U * sizeof(float)));
  spec.output(plane_inliers, "plane_inliers")
      .produces_tensor(planes_output(kUInt32, 1U, planes * sizeof(std::uint32_t)));
}

holoscan::Contract PlaneSegmentationOp::contract() const {
  holoscan::Contract result;
  // A cloud and its normals carry the same capture time.
  result.trigger(holoscan::OnAligned{
      .ports = holoscan::PortList{std::vector<holoscan::PortRef>{points, normals}}, .window = 0ns});
  result.input(points, holoscan::Exact{});
  result.input(normals, holoscan::Exact{});
  return result;
}

void PlaneSegmentationOp::stop() {
  segmenter_ = std::make_unique<PlaneSegmenter>();
  CloudOp::stop();
}

holoscan::expected<void, holoscan::Error> PlaneSegmentationOp::compute(
    holoscan::ExecutionContext& context) {
  auto cloud = receive_cloud(points, context);
  if (!cloud) {
    return holoscan::make_unexpected(std::move(cloud).error());
  }
  std::optional<holoscan::Sample<holoscan::Tensor>> normals_sample;
  auto received = receive_matching(normals, "normals", kFloat32, *cloud, normals_sample, 0);
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  const auto planes = static_cast<std::int64_t>(config_.planes.max_planes);
  const std::vector<std::int64_t> label_shape = cloud->point_shape();
  const std::array<std::int64_t, 2> coefficient_shape{planes, 4};
  const std::array<std::int64_t, 2> centroid_shape{planes, 3};
  const std::array<std::int64_t, 1> inlier_shape{planes};
  OutputTensor out_labels;
  OutputTensor out_coefficients;
  OutputTensor out_centroids;
  OutputTensor out_inliers;
  if (auto made = allocate(out_labels, labels, label_shape, kUInt8, *cloud); !made) {
    return made;
  }
  if (auto made =
          allocate(out_coefficients, plane_coefficients, coefficient_shape, kFloat32, *cloud);
      !made) {
    return made;
  }
  if (auto made = allocate(out_centroids, plane_centroids, centroid_shape, kFloat32, *cloud);
      !made) {
    return made;
  }
  if (auto made = allocate(out_inliers, plane_inliers, inlier_shape, kUInt32, *cloud); !made) {
    return made;
  }
  const PlaneOutputs outputs{.labels = out_labels.data<std::uint8_t>(),
                             .coefficients = out_coefficients.data<float>(),
                             .centroids = out_centroids.data<float>(),
                             .inliers = out_inliers.data<std::uint32_t>()};
  if (const cudaError_t status =
          segmenter_->segment(cloud->xyz, static_cast<const float*>(*received), cloud->count,
                              config_.planes, outputs, cloud->stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("plane segmentation", status));
  }
  if (auto sent = send(out_labels, labels, "labels", *cloud); !sent) {
    return sent;
  }
  if (auto sent = send(out_coefficients, plane_coefficients, "plane_coefficients", *cloud); !sent) {
    return sent;
  }
  if (auto sent = send(out_centroids, plane_centroids, "plane_centroids", *cloud); !sent) {
    return sent;
  }
  return send(out_inliers, plane_inliers, "plane_inliers", *cloud);
}

}  // namespace holoscan::normals
