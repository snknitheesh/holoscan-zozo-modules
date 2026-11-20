// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "view_op.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/tsdf/views.hpp>

namespace tsdf_reconstruction_app {
namespace {

using std::chrono_literals::operator""ns;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "ViewOp: " + message};
}

}  // namespace

ViewOp::ViewOp(ViewConfig config) : config_(config) {}

void ViewOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(depth, "depth");
  spec.input(cloud, "cloud");
  const auto bytes = static_cast<std::uint64_t>(config_.width) * 2U *
                     static_cast<std::uint64_t>(config_.height) * 3U;
  spec.output(image, "image")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kCudaDevice,
                  .dtype = kUInt8,
                  .rank = 3U,
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
      .ports = holoscan::PortList{std::vector<holoscan::PortRef>{depth, cloud}}, .window = 0ns});
  result.input(depth, holoscan::Exact{});
  result.input(cloud, holoscan::Exact{});
  return result;
}

holoscan::expected<void, holoscan::Error> ViewOp::compute(holoscan::ExecutionContext& context) {
  auto depth_sample = depth.receive();
  auto cloud_sample = cloud.receive();
  if (!depth_sample) {
    return holoscan::make_unexpected(std::move(depth_sample).error());
  }
  if (!cloud_sample) {
    return holoscan::make_unexpected(std::move(cloud_sample).error());
  }
  const holoscan::schema::ImageT& depth_image = depth_sample->data;
  const holoscan::messages::schema::PointCloudT& ray_cloud = cloud_sample->data;
  if (auto checked = holoscan::messages::check_point_cloud(ray_cloud); !checked) {
    return checked;
  }
  if (!depth_image.data) {
    return holoscan::make_unexpected(invalid("a depth image without pixels"));
  }
  const std::span<const std::int64_t> depth_shape = depth_image.data->shape_span();
  const std::vector<std::int64_t> organized{config_.height, config_.width, 3};
  if (depth_shape.size() < 2 || depth_shape[0] != config_.height ||
      depth_shape[1] != config_.width || !ray_cloud.organized ||
      ray_cloud.normals->shape() != organized ||
      (holoscan::messages::is_present(ray_cloud.colors) &&
       ray_cloud.colors->shape() != organized)) {
    return holoscan::make_unexpected(
        invalid("expected [H, W] depth and an organized [H, W] cloud with normals"));
  }
  const bool millimeters = depth_image.encoding == holoscan::schema::ImageEncoding_MONO16;
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
  if (holoscan::tsdf::colorize_depth(
          depth_image.data->data(),
          millimeters ? holoscan::tsdf::DepthType::kUint16 : holoscan::tsdf::DepthType::kFloat32,
          millimeters ? 0.001F : 1.0F, config_.width, config_.height, config_.depth_min,
          config_.depth_max, rgb, row, stream) != cudaSuccess ||
      holoscan::tsdf::shade_raycast(static_cast<const float*>(ray_cloud.normals->data()),
                                    holoscan::messages::is_present(ray_cloud.colors)
                                        ? static_cast<const std::uint8_t*>(ray_cloud.colors->data())
                                        : nullptr,
                                    config_.width, config_.height,
                                    rgb + 3U * static_cast<std::size_t>(config_.width), row,
                                    stream) != cudaSuccess) {
    return holoscan::make_unexpected(invalid("drawing failed"));
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  holoscan::schema::ImageT descriptor;
  descriptor.width = 2 * config_.width;
  descriptor.height = config_.height;
  descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
  return image.emit_tensor(std::move(*loan), descriptor,
                           holoscan::sensor_io::derive_from(*depth_sample));
}

}  // namespace tsdf_reconstruction_app
