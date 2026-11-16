// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/messages/point_cloud_layout.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace holoscan::messages {
namespace {

std::string type_name(ElementType type) {
  switch (type) {
    case ElementType::kFloat32:
      return "float32";
    case ElementType::kUInt8:
      return "uint8";
    case ElementType::kOther:
      break;
  }
  return "another type";
}

std::string shape_text(const std::vector<std::int64_t>& shape) {
  std::string text = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    text += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
  }
  return text + "]";
}

std::string describe(const TensorLayout& field) {
  return type_name(field.element_type) + " " + shape_text(field.shape);
}

// The points' shape without its last axis: [N] or [H, W].
std::vector<std::int64_t> leading_shape(const PointCloudLayout& layout) {
  return {layout.points.shape.begin(), layout.points.shape.end() - 1};
}

void check_optional(const TensorLayout& field, ElementType type, bool per_point_vector,
                    const PointCloudLayout& layout, std::string_view name,
                    std::string_view member) {
  if (!is_present(field)) return;
  std::vector<std::int64_t> expected = leading_shape(layout);
  if (per_point_vector) expected.push_back(3);
  if (field.element_type != type || field.shape != expected) {
    throw std::invalid_argument(std::string(name) + "." + std::string(member) + " must be " +
                                type_name(type) + " " + shape_text(expected) +
                                " like points, or [0] when absent; it is " + describe(field));
  }
}

}  // namespace

bool is_present(const TensorLayout& field) noexcept {
  return !(field.shape.size() == 1 && field.shape.front() == 0);
}

std::int64_t point_count(const PointCloudLayout& layout) noexcept {
  const auto& shape = layout.points.shape;
  if (shape.size() == 2) return shape[0];
  std::int64_t count = 0;
  if (shape.size() == 3 && !__builtin_mul_overflow(shape[0], shape[1], &count)) return count;
  return 0;
}

void validate(const PointCloudLayout& layout, std::string_view name) {
  const std::string points = std::string(name) + ".points";
  const std::vector<std::int64_t>& shape = layout.points.shape;
  const std::size_t rank = layout.organized ? 3 : 2;
  bool extents_ok = shape.size() == rank && shape.back() == 3;
  for (const std::int64_t extent : shape) extents_ok = extents_ok && extent >= 0;
  if (layout.organized) extents_ok = extents_ok && shape[0] > 0 && shape[1] > 0;
  std::int64_t values = 1;
  for (const std::int64_t extent : shape) {
    extents_ok = extents_ok && !__builtin_mul_overflow(values, extent, &values);
  }
  if (layout.points.element_type != ElementType::kFloat32 || !extents_ok) {
    throw std::invalid_argument(points + " must be float32 " +
                                (layout.organized
                                     ? "[H, W, 3] with H, W > 0 (the cloud is organized)"
                                     : "[N, 3] (the cloud is not organized)") +
                                "; it is " + describe(layout.points));
  }
  check_optional(layout.colors, ElementType::kUInt8, true, layout, name, "colors");
  check_optional(layout.normals, ElementType::kFloat32, true, layout, name, "normals");
  check_optional(layout.intensity, ElementType::kFloat32, false, layout, name, "intensity");
  if (layout.valid_count < -1 || layout.valid_count > point_count(layout)) {
    throw std::invalid_argument(std::string(name) + ".valid_count must be -1 or 0.." +
                                std::to_string(point_count(layout)) + ", not " +
                                std::to_string(layout.valid_count));
  }
}

}  // namespace holoscan::messages
