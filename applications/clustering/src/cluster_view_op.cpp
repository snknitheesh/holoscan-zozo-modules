// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "cluster_view_op.hpp"

#include <array>
#include <span>
#include <string>
#include <utility>

#include <holoscan/clustering/clustering.hpp>
#include <holoscan/clustering/render.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>

namespace clustering_app {
namespace {

namespace hc = holoscan::clustering;
namespace hm = holoscan::messages;

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr DLDataType kInt32{kDLInt, 32U, 1U};
constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};

// The kernels read the inputs in place: packed tensors in device memory, of
// the element type EuclideanClusterOp sends. (Their rank, which the operator
// bounds rather than fixes, is checked in compute().)
holoscan::TensorInputSpec device_input(DLDataType dtype) {
  return holoscan::TensorInputSpec{.representation = holoscan::TensorRepresentation{
                                       .memory_kind = holoscan::MemoryKind::kCudaDevice,
                                       .dtype = dtype,
                                       .strides = holoscan::TensorStridePolicy::kContiguousOnly,
                                   }};
}

holoscan::Error failure(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "ClusterViewOp: " + message};
}

}  // namespace

void ClusterViewOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(labels, "labels").expects_tensor(device_input(kInt32));
  spec.input(clusters, "clusters").expects_tensor(device_input(kFloat32));
  spec.input(camera_info, "camera_info");
  const auto pixels = static_cast<std::uint64_t>(config_.max_width) *
                      static_cast<std::uint64_t>(config_.max_height);
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
  // The labels and clusters of one frame carry its capture time; the camera
  // info comes at its own rate.
  return hm::camera_info_contract(camera_info, labels, clusters);
}

// Each run starts without a camera; the SDK may start an operator again.
void ClusterViewOp::start() { info_.reset(); }

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
  auto info = info_.update(camera_info);
  if (!info) {
    return holoscan::make_unexpected(std::move(info).error());
  }
  if (*info == nullptr) {
    return {};  // No camera info yet: this frame is not drawn.
  }
  const hm::CameraInfo& camera = **info;
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(failure("the runtime gave no CUDA stream"));
  }
  const std::span<const std::int64_t> label_shape = label_sample->data.shape_span();
  const std::span<const std::int64_t> summary_shape = cluster_sample->data.shape_span();
  if (label_shape.size() != 2 || summary_shape.size() != 2 ||
      summary_shape[1] != hc::kSummaryFields) {
    return holoscan::make_unexpected(failure("expected [H, W] labels and [K, 26] clusters"));
  }
  const auto height = static_cast<int>(label_shape[0]);
  const auto width = static_cast<int>(label_shape[1]);
  if (width > config_.max_width || height > config_.max_height ||
      std::cmp_not_equal(width, camera.width) || std::cmp_not_equal(height, camera.height)) {
    return holoscan::make_unexpected(
        failure("the labels are " + std::to_string(width) + "x" + std::to_string(height) +
                ", but camera info '" + camera.header.frame_id + "' is for " +
                std::to_string(camera.width) + "x" + std::to_string(camera.height) +
                " images (and at most " + std::to_string(config_.max_width) + "x" +
                std::to_string(config_.max_height) + " are drawn)"));
  }
  if (camera.state == hm::CameraState::kRaw || camera.skew != 0.0) {
    return holoscan::make_unexpected(
        failure("the boxes are drawn through an undistorted pinhole camera without skew"));
  }
  auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::array<std::int64_t, 3>{height, width, 3}, .dtype = kUInt8});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  const auto pixels = static_cast<std::uint32_t>(width) * static_cast<std::uint32_t>(height);
  std::uint8_t* rgb = writer->data_as<std::uint8_t>();
  if (const cudaError_t status = hc::colorize_labels(
          static_cast<const std::int32_t*>(label_sample->data.data()), pixels, rgb, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(
        failure(std::string("colorize: ") + cudaGetErrorString(status)));
  }
  if (const cudaError_t status = hc::draw_cluster_boxes(
          static_cast<const float*>(cluster_sample->data.data()),
          static_cast<std::uint32_t>(summary_shape[0]), static_cast<float>(camera.fx),
          static_cast<float>(camera.fy), static_cast<float>(camera.cx),
          static_cast<float>(camera.cy), width, height, rgb, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(failure(std::string("boxes: ") + cudaGetErrorString(status)));
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  holoscan::schema::ImageT descriptor;
  descriptor.width = width;
  descriptor.height = height;
  descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
  return image.emit_tensor(std::move(*loan), descriptor,
                           holoscan::sensor_io::derive_from(*label_sample));
}

}  // namespace clustering_app
