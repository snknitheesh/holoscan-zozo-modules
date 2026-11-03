// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/messages/point_cloud.hpp"

#include <array>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/messages/conversion.hpp>

namespace holoscan::messages {
namespace {

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

using Loan = PointCloudLoan;
using Field = holoscan::TensorFieldSelector<schema::PointCloudT>;

holoscan::TensorOutputSpec field_spec(DLDataType dtype, std::uint64_t values,
                                      holoscan::MemoryKind memory_kind) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = memory_kind,
              .dtype = dtype,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      // Rank 3 for [H, W, 3]; [N, 3], [N], [H, W] and [0] fit as well.
      .bounds = holoscan::TensorBounds{.max_rank = 3U, .max_byte_span = values * (dtype.bits / 8U)},
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

ElementType element_type(const holoscan::Tensor& tensor) {
  const DLDataType dtype = tensor.dtype();
  if (dtype.lanes != 1) return ElementType::kOther;
  if (dtype.code == kDLFloat && dtype.bits == 32) return ElementType::kFloat32;
  if (dtype.code == kDLUInt && dtype.bits == 8) return ElementType::kUInt8;
  return ElementType::kOther;
}

TensorLayout field_layout(const std::shared_ptr<holoscan::Tensor>& tensor) {
  return TensorLayout{tensor->shape(), element_type(*tensor)};
}

holoscan::Error invalid(const std::string& what) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, what};
}

// An absent field has nothing to write; commit it so the cloud can be sent.
holoscan::expected<void, holoscan::Error> commit_empty(Loan& loan, Field field,
                                                       ::cuda::stream_ref stream) {
  const auto nothing =
      [](const holoscan::TensorOutputWriteGuard&) -> holoscan::expected<void, holoscan::Error> {
    return {};
  };
  if (loan.memory_kind(field) == holoscan::MemoryKind::kCudaDevice) {
    return loan.write_and_commit(field, stream, nothing);
  }
  return loan.write_host_and_commit(field, nothing);
}

}  // namespace

void declare_point_cloud_output(holoscan::OperatorSpec& spec,
                                holoscan::Output<schema::PointCloudT>& port, std::string_view name,
                                const PointCloudOutputSpec& output) {
  if (output.max_points == 0) {
    throw std::invalid_argument("declare_point_cloud_output: max_points must be > 0");
  }
  if (output.memory_kind != holoscan::MemoryKind::kCudaDevice &&
      output.memory_kind != holoscan::MemoryKind::kHost) {
    throw std::invalid_argument(
        "declare_point_cloud_output: the memory kind must be host or CUDA device memory");
  }
  const std::uint64_t points = output.max_points;
  // An absent field is [0]: no bytes, but a pool slot needs a positive size.
  const auto values = [points](bool present, std::uint64_t per_point) {
    return present ? points * per_point : 1U;
  };
  const holoscan::MemoryKind memory = output.memory_kind;
  spec.output(port, name)
      .tensor_field(PointCloudFields::points, field_spec(kFloat32, points * 3U, memory))
      .tensor_field(PointCloudFields::colors, field_spec(kUInt8, values(output.colors, 3U), memory))
      .tensor_field(PointCloudFields::normals,
                    field_spec(kFloat32, values(output.normals, 3U), memory))
      .tensor_field(PointCloudFields::intensity,
                    field_spec(kFloat32, values(output.intensity, 1U), memory));
}

holoscan::expected<Loan, holoscan::Error> allocate_point_cloud(
    holoscan::Output<schema::PointCloudT>& port, const PointCloudShape& shape,
    const PointCloudContents& contents, ::cuda::stream_ref stream) {
  if (shape.organized ? !(shape.height > 0 && shape.width > 0)
                      : !(shape.height == 0 && shape.width >= 0)) {
    return holoscan::make_unexpected(
        invalid("allocate_point_cloud: an organized cloud needs H, W > 0, an unorganized one "
                "N >= 0 points"));
  }
  const std::array<std::int64_t, 3> vectors =
      shape.organized ? std::array<std::int64_t, 3>{shape.height, shape.width, 3}
                      : std::array<std::int64_t, 3>{shape.width, 3, 0};
  const std::size_t rank = shape.organized ? 3U : 2U;
  const std::span<const std::int64_t> per_point_vector(vectors.data(), rank);
  const std::span<const std::int64_t> per_point_scalar(vectors.data(), rank - 1U);
  constexpr std::array<std::int64_t, 1> kAbsent{0};
  const auto pick = [&](bool present, std::span<const std::int64_t> present_shape) {
    return present ? present_shape : std::span<const std::int64_t>(kAbsent);
  };
  const std::array<holoscan::TensorFieldLoanRequest<schema::PointCloudT>, 4> requests{{
      {PointCloudFields::points, {.shape = per_point_vector, .dtype = kFloat32}},
      {PointCloudFields::colors,
       {.shape = pick(contents.colors, per_point_vector), .dtype = kUInt8}},
      {PointCloudFields::normals,
       {.shape = pick(contents.normals, per_point_vector), .dtype = kFloat32}},
      {PointCloudFields::intensity,
       {.shape = pick(contents.intensity, per_point_scalar), .dtype = kFloat32}},
  }};
  auto loan = port.allocate_tensors(
      std::span<const holoscan::TensorFieldLoanRequest<schema::PointCloudT>>(requests));
  if (!loan) return holoscan::make_unexpected(std::move(loan).error());
  const std::array<std::pair<bool, Field>, 3> optional{{
      {contents.colors, PointCloudFields::colors},
      {contents.normals, PointCloudFields::normals},
      {contents.intensity, PointCloudFields::intensity},
  }};
  for (const auto& [present, field] : optional) {
    if (present) continue;
    if (auto committed = commit_empty(*loan, field, stream); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
  }
  return loan;
}

holoscan::expected<void, holoscan::Error> emit_point_cloud(
    holoscan::Output<schema::PointCloudT>& port, Loan loan, const Header& header,
    const PointCloudShape& shape, std::int64_t valid_count, holoscan::EmitOptions options) {
  if (header.frame_id.size() > kMaxFrameIdBytes) {
    return holoscan::make_unexpected(invalid("emit_point_cloud: the frame id has more than " +
                                             std::to_string(kMaxFrameIdBytes) + " bytes"));
  }
  schema::PointCloudT metadata;
  metadata.header = std::make_shared<holoscan::schema::HeaderT>(to_message(header));
  metadata.organized = shape.organized;
  metadata.valid_count = valid_count;
  return port.emit_tensors(std::move(loan), metadata, options);
}

PointCloudLayout layout_of(const schema::PointCloudT& cloud) {
  PointCloudLayout layout;
  const auto read = [](const std::shared_ptr<holoscan::Tensor>& tensor) {
    return tensor ? field_layout(tensor) : TensorLayout{};
  };
  layout.points = read(cloud.points);
  layout.colors = read(cloud.colors);
  layout.normals = read(cloud.normals);
  layout.intensity = read(cloud.intensity);
  layout.organized = cloud.organized;
  layout.valid_count = cloud.valid_count;
  return layout;
}

holoscan::expected<void, holoscan::Error> check_point_cloud(const schema::PointCloudT& cloud) {
  if (!cloud.points || !cloud.colors || !cloud.normals || !cloud.intensity) {
    return holoscan::make_unexpected(
        invalid("point_cloud: every message carries points, colors, normals and intensity"));
  }
  if (cloud.header && cloud.header->frame_id.size() > kMaxFrameIdBytes) {
    return holoscan::make_unexpected(invalid("point_cloud.header.frame_id is too long"));
  }
  try {
    validate(layout_of(cloud));
  } catch (const std::invalid_argument& error) {
    return holoscan::make_unexpected(invalid(error.what()));
  }
  return {};
}

bool is_present(const std::shared_ptr<holoscan::Tensor>& field) noexcept {
  return field != nullptr && !(field->ndim() == 1 && field->size() == 0);
}

}  // namespace holoscan::messages
