// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/normals/normal_estimation_op.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "compact.cuh"
#include "device_buffer.hpp"

namespace holoscan::normals {
namespace {

using std::chrono_literals::operator""ns;

}  // namespace

void validate_config(const NormalEstimationConfig& config) {
  const char* what = nullptr;
  if (!valid_params(config.normals, &what)) {
    throw std::invalid_argument(what);
  }
  if (config.layout != OutputLayout::kKeep && config.layout != OutputLayout::kDense) {
    throw std::invalid_argument("layout must be kKeep or kDense");
  }
  if (config.max_points < 1U || config.max_points > kMaxPointsLimit) {
    throw std::invalid_argument("max_points must be 1.." + std::to_string(kMaxPointsLimit));
  }
}

struct NormalEstimationOp::Impl {
  NormalEstimator estimator;
  detail::Compactor compactor;
  detail::DeviceBuffer normals;    // kDense: the normals before packing.
  detail::DeviceBuffer curvature;  // kDense: the curvature before packing.
};

NormalEstimationOp::NormalEstimationOp(NormalEstimationConfig config)
    : CloudOp("NormalEstimationOp", config.max_points),
      config_(config),
      impl_(std::make_unique<Impl>()) {
  validate_config(config_);
}

NormalEstimationOp::~NormalEstimationOp() = default;

void NormalEstimationOp::setup(holoscan::OperatorSpec& spec) {
  const std::uint64_t n = config_.max_points;
  // Two slots per input: one cloud in compute() while the next one arrives.
  spec.input(points, "points").queue_depth(2U);
  // Optional ports exist only when configured: an unconnected input would
  // never be satisfied, and an emit into an unconnected output fails.
  if (config_.color) {
    spec.input(colors, "colors").queue_depth(2U);
  }
  spec.output(filtered_points, "filtered_points")
      .produces_tensor(device_output(kFloat32, n * 3U * sizeof(float)));
  if (config_.color) {
    spec.output(filtered_colors, "filtered_colors").produces_tensor(device_output(kUInt8, n * 3U));
  }
  spec.output(normals, "normals").produces_tensor(device_output(kFloat32, n * 3U * sizeof(float)));
  spec.output(curvature, "curvature").produces_tensor(device_output(kFloat32, n * sizeof(float)));
}

holoscan::Contract NormalEstimationOp::contract() const {
  holoscan::Contract result;
  if (!config_.color) {
    result.trigger(holoscan::OnEach{points});
    return result;
  }
  // A cloud and its colors carry the same capture time.
  result.trigger(holoscan::OnAligned{
      .ports = holoscan::PortList{std::vector<holoscan::PortRef>{points, colors}}, .window = 0ns});
  result.input(points, holoscan::Exact{});
  result.input(colors, holoscan::Exact{});
  return result;
}

void NormalEstimationOp::stop() {
  impl_ = std::make_unique<Impl>();
  CloudOp::stop();
}

holoscan::expected<void, holoscan::Error> NormalEstimationOp::compute(
    holoscan::ExecutionContext& context) {
  auto cloud = receive_cloud(points, context);
  if (!cloud) {
    return holoscan::make_unexpected(std::move(cloud).error());
  }
  std::optional<holoscan::Sample<holoscan::Tensor>> colors_sample;
  const std::uint8_t* rgb = nullptr;
  if (config_.color) {
    auto received = receive_matching(colors, "colors", kUInt8, *cloud, colors_sample, 0);
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    rgb = static_cast<const std::uint8_t*>(*received);
  }
  NormalParams params = config_.normals;
  params.search = resolve_search(params.search, cloud->organized);
  if (params.search == NeighborSearch::kWindow && !cloud->organized) {
    return holoscan::make_unexpected(
        invalid("the window search needs an organized [H, W, 3] cloud"));
  }
  const std::uint32_t n = cloud->count;
  OutputTensor out_points;
  OutputTensor out_colors;
  OutputTensor out_normals;
  OutputTensor out_curvature;
  const auto allocate_all =
      [&](std::span<const std::int64_t> shape,
          std::span<const std::int64_t> point_shape) -> holoscan::expected<void, holoscan::Error> {
    if (auto made = allocate(out_points, filtered_points, shape, kFloat32, *cloud); !made) {
      return made;
    }
    if (config_.color) {
      if (auto made = allocate(out_colors, filtered_colors, shape, kUInt8, *cloud); !made) {
        return made;
      }
    }
    if (auto made = allocate(out_normals, normals, shape, kFloat32, *cloud); !made) {
      return made;
    }
    return allocate(out_curvature, curvature, point_shape, kFloat32, *cloud);
  };

  if (config_.layout == OutputLayout::kKeep) {
    const std::vector<std::int64_t> point_shape = cloud->point_shape();
    if (auto made = allocate_all(cloud->shape, point_shape); !made) {
      return made;
    }
    if (const cudaError_t status = impl_->estimator.estimate(
            cloud->xyz, n, cloud->width, cloud->height, params, out_normals.data<float>(),
            out_curvature.data<float>(), cloud->stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("normal estimation", status));
    }
    if (const cudaError_t status = detail::keep_with_normals(
            cloud->xyz, rgb, out_normals.data<float>(), n, out_points.data<float>(),
            out_colors.data<std::uint8_t>(), cloud->stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("keep_with_normals", status));
    }
  } else {
    const std::size_t points_count = std::max<std::size_t>(n, 1U);
    if (const cudaError_t status =
            impl_->normals.reserve(points_count * 3U * sizeof(float), cloud->stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
    }
    if (const cudaError_t status =
            impl_->curvature.reserve(points_count * sizeof(float), cloud->stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
    }
    float* normals_scratch = impl_->normals.as<float>();
    float* curvature_scratch = impl_->curvature.as<float>();
    if (const cudaError_t status =
            impl_->estimator.estimate(cloud->xyz, n, cloud->width, cloud->height, params,
                                      normals_scratch, curvature_scratch, cloud->stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("normal estimation", status));
    }
    if (const cudaError_t status = impl_->compactor.count(normals_scratch, n, cloud->stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("compaction", status));
    }
    auto kept = wait_for_count(impl_->compactor.device_count(), *cloud);
    if (!kept) {
      return holoscan::make_unexpected(std::move(kept).error());
    }
    const std::array<std::int64_t, 2> shape{*kept, 3};
    const std::array<std::int64_t, 1> point_shape{*kept};
    if (auto made = allocate_all(shape, point_shape); !made) {
      return made;
    }
    if (const cudaError_t status = impl_->compactor.write(
            cloud->xyz, rgb, normals_scratch, curvature_scratch, out_points.data<float>(),
            out_colors.data<std::uint8_t>(), out_normals.data<float>(), out_curvature.data<float>(),
            cloud->stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("compaction", status));
    }
  }

  if (auto sent = send(out_points, filtered_points, "filtered_points", *cloud); !sent) {
    return sent;
  }
  if (config_.color) {
    if (auto sent = send(out_colors, filtered_colors, "filtered_colors", *cloud); !sent) {
      return sent;
    }
  }
  if (auto sent = send(out_normals, normals, "normals", *cloud); !sent) {
    return sent;
  }
  return send(out_curvature, curvature, "curvature", *cloud);
}

}  // namespace holoscan::normals
