// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "cluster_view_op.hpp"

#include <array>
#include <chrono>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/clustering/clustering.hpp>
#include <holoscan/clustering/render.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>

namespace clustering_app {
namespace {

using std::chrono_literals::operator""ns;
namespace hc = holoscan::clustering;

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

holoscan::Error failure(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "ClusterViewOp: " + message};
}

}  // namespace

void ClusterViewOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(labels, "labels");
  spec.input(clusters, "clusters");
  const auto pixels =
      static_cast<std::uint64_t>(config_.width) * static_cast<std::uint64_t>(config_.height);
  spec.output(image, "image")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kCudaDevice,
                  .dtype = kUInt8,
                  .rank = 3U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(pixels * 3U),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      })
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
}

holoscan::Contract ClusterViewOp::contract() const {
  holoscan::Contract result;
  // The labels and clusters of one frame carry its capture time.
  result.trigger(holoscan::OnAligned{
      .ports = holoscan::PortList{std::vector<holoscan::PortRef>{labels, clusters}},
      .window = 0ns});
  result.input(labels, holoscan::Exact{});
  result.input(clusters, holoscan::Exact{});
  return result;
}

holoscan::expected<void, holoscan::Error> ClusterViewOp::compute(
    holoscan::ExecutionContext& context) {
  auto label_sample = labels.receive();
  auto cluster_sample = clusters.receive();
  if (!label_sample) {
    return holoscan::make_unexpected(std::move(label_sample).error());
  }
  if (!cluster_sample) {
    return holoscan::make_unexpected(std::move(cluster_sample).error());
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(failure("the runtime gave no CUDA stream"));
  }
  const std::span<const std::int64_t> label_shape = label_sample->data.shape_span();
  const std::span<const std::int64_t> summary_shape = cluster_sample->data.shape_span();
  if (label_shape.size() != 2 || label_shape[0] != config_.height ||
      label_shape[1] != config_.width || summary_shape.size() != 2 ||
      summary_shape[1] != hc::kSummaryFields) {
    return holoscan::make_unexpected(failure("expected [H, W] labels and [K, 26] clusters"));
  }
  auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::array<std::int64_t, 3>{config_.height, config_.width, 3}, .dtype = kUInt8});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  const auto pixels =
      static_cast<std::uint32_t>(config_.width) * static_cast<std::uint32_t>(config_.height);
  std::uint8_t* rgb = writer->data_as<std::uint8_t>();
  if (const cudaError_t status = hc::colorize_labels(
          static_cast<const std::int32_t*>(label_sample->data.data()), pixels, rgb, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(
        failure(std::string("colorize: ") + cudaGetErrorString(status)));
  }
  if (const cudaError_t status = hc::draw_cluster_boxes(
          static_cast<const float*>(cluster_sample->data.data()),
          static_cast<std::uint32_t>(summary_shape[0]), config_.fx, config_.fy, config_.cx,
          config_.cy, config_.width, config_.height, rgb, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(failure(std::string("boxes: ") + cudaGetErrorString(status)));
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  holoscan::schema::ImageT descriptor;
  descriptor.width = config_.width;
  descriptor.height = config_.height;
  descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
  return image.emit_tensor(std::move(*loan), descriptor,
                           holoscan::sensor_io::derive_from(*label_sample));
}

}  // namespace clustering_app
