// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/normals/normal_estimation_op.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/sensor_io/provenance.hpp>

#include "compact.cuh"
#include "device_buffer.hpp"

namespace holoscan::normals {
namespace {

namespace hm = holoscan::messages;

// Runs `launch` as a fill of the loan's write of `field`, turning a CUDA
// error into the fill's error.
template <typename Launch>
holoscan::expected<void, holoscan::Error> write_field(
    hm::PointCloudLoan& loan, holoscan::TensorFieldSelector<hm::schema::PointCloudT> field,
    cudaStream_t stream, const std::string& op, const char* what, Launch&& launch) {
  return loan.write_and_commit(
      field, stream,
      [&](const holoscan::TensorOutputWriteGuard& writer)
          -> holoscan::expected<void, holoscan::Error> {
        if (const cudaError_t status = launch(writer); status != cudaSuccess) {
          return holoscan::make_unexpected(holoscan::Error{
              holoscan::ErrorCode::kFailure, op + ": " + what + ": " + cudaGetErrorString(status)});
        }
        return {};
      });
}

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
  // Two slots: one cloud in compute() while the next one arrives, so that the
  // producer does not wait for this operator to take the previous cloud.
  spec.input(point_cloud, "point_cloud").queue_depth(2U);
  hm::declare_point_cloud_output(
      spec, point_cloud_with_normals, "point_cloud_with_normals",
      {.max_points = n, .colors = config_.colors, .normals = true, .intensity = config_.intensity});
  spec.output(curvature, "curvature").produces_tensor(device_output(kFloat32, n * sizeof(float)));
}

holoscan::Contract NormalEstimationOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{point_cloud});
  return result;
}

void NormalEstimationOp::stop() {
  impl_ = std::make_unique<Impl>();
  CloudOp::stop();
}

holoscan::expected<void, holoscan::Error> NormalEstimationOp::compute(
    holoscan::ExecutionContext& context) {
  auto cloud = receive_cloud(point_cloud, "point_cloud", context);
  if (!cloud) {
    return holoscan::make_unexpected(std::move(cloud).error());
  }
  if (cloud->has_colors && !config_.colors) {
    return holoscan::make_unexpected(
        invalid("the cloud carries colors; set NormalEstimationConfig::colors to carry them"));
  }
  if (cloud->has_intensity && !config_.intensity) {
    return holoscan::make_unexpected(invalid(
        "the cloud carries intensities; set NormalEstimationConfig::intensity to carry them"));
  }
  NormalParams params = config_.normals;
  params.search = resolve_search(params.search, cloud->organized);
  if (params.search == NeighborSearch::kWindow && !cloud->organized) {
    return holoscan::make_unexpected(
        invalid("the window search needs an organized [H, W, 3] cloud"));
  }
  const std::uint32_t n = cloud->count;
  cudaStream_t stream = cloud->stream;
  const hm::PointCloudContents contents{
      .colors = cloud->has_colors, .normals = true, .intensity = cloud->has_intensity};
  OutputTensor out_curvature;
  std::optional<hm::PointCloudLoan> loan;
  hm::PointCloudShape shape = cloud->cloud_shape();
  std::int64_t valid_count = -1;

  if (config_.layout == OutputLayout::kKeep) {
    const std::vector<std::int64_t> point_shape = cloud->point_shape();
    if (auto made = allocate(out_curvature, curvature, point_shape, kFloat32, *cloud); !made) {
      return made;
    }
    auto allocated = hm::allocate_point_cloud(point_cloud_with_normals, shape, contents, stream);
    if (!allocated) {
      return holoscan::make_unexpected(std::move(allocated).error());
    }
    loan.emplace(std::move(*allocated));
    // The normals first, straight into the output; the other fields keep the
    // points that got one.
    const float* normals = nullptr;
    if (auto written = write_field(
            *loan, hm::PointCloudFields::normals, stream, op_name(), "normal estimation",
            [&](const holoscan::TensorOutputWriteGuard& writer) {
              normals = writer.data_as<float>();
              return impl_->estimator.estimate(cloud->xyz, n, cloud->width, cloud->height, params,
                                               writer.data_as<float>(), out_curvature.data<float>(),
                                               stream);
            });
        !written) {
      return written;
    }
    if (auto written = write_field(
            *loan, hm::PointCloudFields::points, stream, op_name(), "keeping the points",
            [&](const holoscan::TensorOutputWriteGuard& writer) {
              return detail::keep_with_normals(cloud->xyz, 3, normals, n, writer.data_as<float>(),
                                               stream);
            });
        !written) {
      return written;
    }
    if (cloud->has_colors) {
      if (auto written = write_field(
              *loan, hm::PointCloudFields::colors, stream, op_name(), "keeping the colors",
              [&](const holoscan::TensorOutputWriteGuard& writer) {
                return detail::keep_with_normals(cloud->rgb, 3, normals, n,
                                                 writer.data_as<std::uint8_t>(), stream);
              });
          !written) {
        return written;
      }
    }
    if (cloud->has_intensity) {
      if (auto written = write_field(
              *loan, hm::PointCloudFields::intensity, stream, op_name(), "keeping the intensities",
              [&](const holoscan::TensorOutputWriteGuard& writer) {
                return detail::keep_with_normals(cloud->intensity, 1, normals, n,
                                                 writer.data_as<float>(), stream);
              });
          !written) {
        return written;
      }
    }
  } else {
    const std::size_t points_count = std::max<std::size_t>(n, 1U);
    if (const cudaError_t status =
            impl_->normals.reserve(points_count * 3U * sizeof(float), stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
    }
    if (const cudaError_t status = impl_->curvature.reserve(points_count * sizeof(float), stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
    }
    float* normals_scratch = impl_->normals.as<float>();
    float* curvature_scratch = impl_->curvature.as<float>();
    if (const cudaError_t status =
            impl_->estimator.estimate(cloud->xyz, n, cloud->width, cloud->height, params,
                                      normals_scratch, curvature_scratch, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("normal estimation", status));
    }
    if (const cudaError_t status = impl_->compactor.count(normals_scratch, n, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("compaction", status));
    }
    auto kept = wait_for_count(impl_->compactor.device_count(), *cloud);
    if (!kept) {
      return holoscan::make_unexpected(std::move(kept).error());
    }
    shape = hm::unorganized_cloud(*kept);
    valid_count = *kept;
    const std::array<std::int64_t, 1> point_shape{*kept};
    if (auto made = allocate(out_curvature, curvature, point_shape, kFloat32, *cloud); !made) {
      return made;
    }
    if (const cudaError_t status =
            impl_->compactor.pack(curvature_scratch, 1, out_curvature.data<float>(), stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("compaction", status));
    }
    auto allocated = hm::allocate_point_cloud(point_cloud_with_normals, shape, contents, stream);
    if (!allocated) {
      return holoscan::make_unexpected(std::move(allocated).error());
    }
    loan.emplace(std::move(*allocated));
    const auto pack = [&](holoscan::TensorFieldSelector<hm::schema::PointCloudT> field,
                          const auto* in, int channels) {
      return write_field(*loan, field, stream, op_name(), "compaction",
                         [&](const holoscan::TensorOutputWriteGuard& writer) {
                           using T = std::remove_cv_t<std::remove_pointer_t<decltype(in)>>;
                           return impl_->compactor.pack(in, channels, writer.data_as<T>(), stream);
                         });
    };
    if (auto written = pack(hm::PointCloudFields::points, cloud->xyz, 3); !written) {
      return written;
    }
    if (auto written =
            pack(hm::PointCloudFields::normals, static_cast<const float*>(normals_scratch), 3);
        !written) {
      return written;
    }
    if (cloud->has_colors) {
      if (auto written = pack(hm::PointCloudFields::colors, cloud->rgb, 3); !written) {
        return written;
      }
    }
    if (cloud->has_intensity) {
      if (auto written = pack(hm::PointCloudFields::intensity, cloud->intensity, 1); !written) {
        return written;
      }
    }
  }

  if (auto sent =
          hm::emit_point_cloud(point_cloud_with_normals, std::move(*loan), cloud->header, shape,
                               valid_count, holoscan::sensor_io::derive_from(cloud->sample));
      !sent) {
    return emit_failure("point_cloud_with_normals", std::move(sent).error());
  }
  return send(out_curvature, curvature, "curvature", *cloud);
}

}  // namespace holoscan::normals
