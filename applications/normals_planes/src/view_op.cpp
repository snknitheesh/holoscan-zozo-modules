// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "view_op.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/normals/views.hpp>
#include <holoscan/sensor_io/provenance.hpp>

namespace normals_planes_app {
namespace {

using std::chrono_literals::operator""ns;
namespace hm = holoscan::messages;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};

// What an input takes: a tensor of `dtype` in device memory (its shape is
// checked in compute()).
holoscan::TensorInputSpec device_input(DLDataType dtype) {
  return holoscan::TensorInputSpec{
      .representation = holoscan::TensorRepresentation{
          .memory_kind = holoscan::MemoryKind::kCudaDevice, .dtype = dtype}};
}

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "ViewOp: " + message};
}

}  // namespace

ViewOp::ViewOp(ViewConfig config) : config_(config) {}

void ViewOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(point_cloud, "point_cloud")
      .tensor_field(hm::PointCloudFields::normals, device_input(kFloat32));
  spec.input(labels, "labels").expects_tensor(device_input(kUInt8));
  const auto bytes = static_cast<std::uint64_t>(config_.width) * 2U *
                     static_cast<std::uint64_t>(config_.height) * 3U;
  spec.output(image, "image")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kCudaDevice,
                  .dtype = kUInt8,
                  .rank = 3U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(bytes),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      })
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
}

holoscan::Contract ViewOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnAligned{
      .ports = holoscan::PortList{std::vector<holoscan::PortRef>{point_cloud, labels}},
      .window = 0ns});
  result.input(point_cloud, holoscan::Exact{});
  result.input(labels, holoscan::Exact{});
  return result;
}

holoscan::expected<void, holoscan::Error> ViewOp::compute(holoscan::ExecutionContext& context) {
  auto cloud_sample = point_cloud.receive();
  auto label_sample = labels.receive();
  if (!cloud_sample) {
    return holoscan::make_unexpected(std::move(cloud_sample).error());
  }
  if (!label_sample) {
    return holoscan::make_unexpected(std::move(label_sample).error());
  }
  const std::shared_ptr<holoscan::Tensor>& normals = cloud_sample->data.normals;
  if (!hm::is_present(normals)) {
    return holoscan::make_unexpected(invalid("the cloud carries no normals"));
  }
  const std::span<const std::int64_t> shape = normals->shape_span();
  const std::span<const std::int64_t> label_shape = label_sample->data.shape_span();
  if (shape.size() != 3 || shape[0] != config_.height || shape[1] != config_.width ||
      label_shape.size() != 2 || label_shape[0] != shape[0] || label_shape[1] != shape[1]) {
    return holoscan::make_unexpected(
        invalid("expected organized [H, W, 3] normals and [H, W] labels"));
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(invalid("the runtime gave no CUDA stream"));
  }
  auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::array<std::int64_t, 3>{config_.height, 2 * config_.width, 3}, .dtype = kUInt8});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  auto* rgb = writer->data_as<std::uint8_t>();
  const std::size_t row = static_cast<std::size_t>(config_.width) * 2U * 3U;
  if (const cudaError_t status =
          holoscan::normals::colorize_normals(static_cast<const float*>(normals->data()),
                                              config_.width, config_.height, rgb, row, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(
        invalid(std::string("colorizing the normals failed: ") + cudaGetErrorString(status)));
  }
  if (const cudaError_t status = holoscan::normals::colorize_labels(
          static_cast<const std::uint8_t*>(label_sample->data.data()), config_.width,
          config_.height, rgb + 3U * static_cast<std::size_t>(config_.width), row, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(
        invalid(std::string("colorizing the labels failed: ") + cudaGetErrorString(status)));
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  holoscan::schema::ImageT descriptor;
  descriptor.width = 2 * config_.width;
  descriptor.height = config_.height;
  descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
  return image.emit_tensor(std::move(*loan), descriptor,
                           holoscan::sensor_io::derive_from(*cloud_sample));
}

}  // namespace normals_planes_app
