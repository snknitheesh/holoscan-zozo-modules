// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/messages/point_cloud.hpp"

#include <array>
#include <limits>
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

// a * b, or throws for a product past 64 bits.
std::uint64_t checked_product(std::uint64_t a, std::uint64_t b) {
  std::uint64_t product = 0;
  if (__builtin_mul_overflow(a, b, &product)) {
    throw std::invalid_argument("declare_point_cloud_output: max_points is too large");
  }
  return product;
}

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
      .bounds = holoscan::TensorBounds{.max_rank = 3U,
                                       .max_byte_span = checked_product(values, dtype.bits / 8U)},
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

std::string shape_text(std::span<const std::int64_t> shape) {
  std::string text = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    text += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
  }
  return text + "]";
}

// Each element directly after the one before, in row-major order. Any stride
// is taken on an axis of extent 1, which no element steps along.
bool packed(const holoscan::Tensor& tensor) {
  const std::span<const std::int64_t> strides = tensor.dlpack_strides_span();
  if (strides.empty()) return true;  // DLPack's compact row-major layout.
  const std::span<const std::int64_t> shape = tensor.shape_span();
  std::int64_t expected = 1;  // In elements, as DLPack counts strides.
  for (std::size_t axis = shape.size(); axis-- > 0;) {
    if (shape[axis] > 1 && strides[axis] != expected) return false;
    expected *= shape[axis];
  }
  return true;
}

bool same_device(const DLDevice& a, const DLDevice& b) {
  return a.device_type == b.device_type && a.device_id == b.device_id;
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
      output.memory_kind != holoscan::MemoryKind::kHost &&
      output.memory_kind != holoscan::MemoryKind::kPinnedHost) {
    throw std::invalid_argument(
        "declare_point_cloud_output: the memory kind must be host, pinned host or CUDA device "
        "memory");
  }
  const std::uint64_t points = output.max_points;
  // An absent field is [0]: no bytes, but a pool slot needs a positive size.
  const auto values = [points](bool present, std::uint64_t per_point) {
    return present ? checked_product(points, per_point) : 1U;
  };
  const holoscan::MemoryKind memory = output.memory_kind;
  // Every size first, so that one too large declares nothing.
  const holoscan::TensorOutputSpec points_spec = field_spec(kFloat32, values(true, 3U), memory);
  const holoscan::TensorOutputSpec colors_spec =
      field_spec(kUInt8, values(output.colors, 3U), memory);
  const holoscan::TensorOutputSpec normals_spec =
      field_spec(kFloat32, values(output.normals, 3U), memory);
  const holoscan::TensorOutputSpec intensity_spec =
      field_spec(kFloat32, values(output.intensity, 1U), memory);
  spec.output(port, name)
      .tensor_field(PointCloudFields::points, points_spec)
      .tensor_field(PointCloudFields::colors, colors_spec)
      .tensor_field(PointCloudFields::normals, normals_spec)
      .tensor_field(PointCloudFields::intensity, intensity_spec);
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
  // The organization and the count describe the points allocated: a cloud
  // that says otherwise would be refused by every consumer.
  const std::span<const std::int64_t> allocated = loan.shape(PointCloudFields::points);
  const bool matches = shape.organized ? allocated.size() == 3 && allocated[0] == shape.height &&
                                             allocated[1] == shape.width
                                       : allocated.size() == 2 && allocated[0] == shape.width;
  if (!matches) {
    return holoscan::make_unexpected(invalid(
        "emit_point_cloud: the points were allocated as " + shape_text(allocated) + ", not as " +
        (shape.organized ? "the organized " + std::to_string(shape.height) + "x" +
                               std::to_string(shape.width) + " cloud"
                         : "an unorganized cloud of " + std::to_string(shape.width) + " points") +
        " given"));
  }
  const std::int64_t count = shape.organized ? shape.height * shape.width : shape.width;
  if (valid_count < -1 || valid_count > count) {
    return holoscan::make_unexpected(invalid("emit_point_cloud: valid_count must be -1 or 0.." +
                                             std::to_string(count) + ", not " +
                                             std::to_string(valid_count)));
  }
  schema::PointCloudT metadata;
  metadata.header = std::make_shared<holoscan::schema::HeaderT>(to_message(header));
  metadata.organized = shape.organized;
  metadata.valid_count = valid_count;
  return port.emit_tensors(std::move(loan), metadata, options);
}

PointCloudLayout layout_of(const schema::PointCloudT& cloud) {
  PointCloudLayout layout;
  // A missing tensor reads as [0], an absent field.
  const auto read = [](const std::shared_ptr<holoscan::Tensor>& tensor) {
    return tensor ? field_layout(tensor) : TensorLayout{{0}, ElementType::kOther};
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
  const DLDevice points_device = cloud.points->device();
  const std::array<std::pair<const char*, const holoscan::Tensor*>, 4> fields{{
      {"points", cloud.points.get()},
      {"colors", cloud.colors.get()},
      {"normals", cloud.normals.get()},
      {"intensity", cloud.intensity.get()},
  }};
  for (const auto& [name, tensor] : fields) {
    if (!packed(*tensor)) {
      return holoscan::make_unexpected(invalid(std::string("point_cloud.") + name +
                                               " must be packed: each value directly after "
                                               "the one before, row by row"));
    }
    // An absent field holds nothing, wherever its [0] is.
    const bool present = !(tensor->ndim() == 1 && tensor->size() == 0);
    if (present && !same_device(tensor->device(), points_device)) {
      return holoscan::make_unexpected(invalid(std::string("point_cloud.") + name +
                                               " must be in the same memory as the points"));
    }
  }
  return {};
}

bool is_present(const std::shared_ptr<holoscan::Tensor>& field) noexcept {
  return field != nullptr && !(field->ndim() == 1 && field->size() == 0);
}

}  // namespace holoscan::messages
