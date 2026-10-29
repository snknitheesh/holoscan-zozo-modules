// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/pcl_filters/ground_removal_op.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>

namespace holoscan::pcl_filters {
namespace {

CloudSettings settings_of(const GroundRemovalConfig& config) {
  return CloudSettings{
      .color = config.color, .layout = config.layout, .max_points = config.max_points};
}

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};

}  // namespace

void validate_config(const GroundRemovalConfig& config) {
  const GroundParams& g = config.ground;
  const float up = g.up[0] * g.up[0] + g.up[1] * g.up[1] + g.up[2] * g.up[2];
  if (!(up > 0.0F) || !std::isfinite(up)) {
    throw std::invalid_argument("up must be a finite, nonzero vector");
  }
  if (g.mode == GroundMode::kRansac) {
    if (!(g.distance_threshold >= 0.0F) || !std::isfinite(g.distance_threshold)) {
      throw std::invalid_argument("distance_threshold must be finite and not negative");
    }
    if (g.max_iterations < 1U || g.max_iterations > kMaxIterationsLimit) {
      throw std::invalid_argument("max_iterations must be 1.." +
                                  std::to_string(kMaxIterationsLimit));
    }
    if (!(g.max_tilt_deg >= 0.0F && g.max_tilt_deg <= 90.0F)) {
      throw std::invalid_argument("max_tilt_deg must be 0..90");
    }
  } else if (g.mode == GroundMode::kHeight) {
    if (!std::isfinite(g.height)) {
      throw std::invalid_argument("height must be finite");
    }
  } else {
    throw std::invalid_argument("mode must be kRansac or kHeight");
  }
}

GroundRemovalOp::GroundRemovalOp(GroundRemovalConfig config)
    : CloudFilterOp("GroundRemovalOp", settings_of(config)),
      config_(config),
      segmenter_(std::make_unique<GroundSegmenter>()) {
  validate_config(config_);
}

GroundRemovalOp::~GroundRemovalOp() {
  if (plane_scratch_ != nullptr) {
    cudaFree(plane_scratch_);
  }
}

void GroundRemovalOp::setup(holoscan::OperatorSpec& spec) {
  CloudFilterOp::setup(spec);
  if (!config_.emit_plane) {
    return;
  }
  spec.output(plane, "plane")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kCudaDevice,
                  .dtype = kFloat32,
                  .rank = 1U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(4U * sizeof(float)),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
}

void GroundRemovalOp::stop() {
  segmenter_ = std::make_unique<GroundSegmenter>();
  if (plane_scratch_ != nullptr) {
    cudaFree(plane_scratch_);
    plane_scratch_ = nullptr;
  }
  CloudFilterOp::stop();
}

holoscan::expected<void, holoscan::Error> GroundRemovalOp::compute(
    holoscan::ExecutionContext& context) {
  auto cloud = receive(context);
  if (!cloud) {
    return holoscan::make_unexpected(std::move(cloud).error());
  }
  auto keep = mask(*cloud);
  if (!keep) {
    return holoscan::make_unexpected(std::move(keep).error());
  }
  if (plane_scratch_ == nullptr) {
    if (const cudaError_t status = cudaMalloc(&plane_scratch_, 4U * sizeof(float));
        status != cudaSuccess) {
      plane_scratch_ = nullptr;
      return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
    }
  }
  if (const cudaError_t status = segmenter_->segment(cloud->xyz, cloud->count, config_.ground,
                                                     plane_scratch_, *keep, nullptr, cloud->stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("ground kernels", status));
  }
  if (auto sent = emit_kept(*cloud, *keep); !sent || !config_.emit_plane) {
    return sent;
  }
  auto loan = plane.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = std::array<std::int64_t, 1>{4}, .dtype = kFloat32});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(cloud->stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  if (const cudaError_t status =
          cudaMemcpyAsync(writer->data_as<float>(), plane_scratch_, 4U * sizeof(float),
                          cudaMemcpyDeviceToDevice, cloud->stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", status));
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  if (auto sent = plane.emit(std::move(*loan), holoscan::sensor_io::derive_from(cloud->points));
      !sent) {
    return emit_failure("plane", std::move(sent).error());
  }
  return {};
}

}  // namespace holoscan::pcl_filters
