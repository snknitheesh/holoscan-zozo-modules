// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The shapes and element types of a PointCloud message's tensors, and the
// rules they follow, without the Holoscan SDK. point_cloud.hpp reads a
// message's layout.
//
//   points     float32 [N, 3], or [H, W, 3] when organized
//   colors     uint8   [N, 3] or [H, W, 3]
//   normals    float32 [N, 3] or [H, W, 3]
//   intensity  float32 [N]    or [H, W]
//
// colors, normals and intensity are optional: a cloud without one carries the
// empty rank-1 tensor [0] in its place (every message has all four tensors).

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace holoscan::messages {

enum class ElementType : std::uint8_t {
  kFloat32 = 0,
  kUInt8 = 1,
  kOther = 2,
};

struct TensorLayout {
  std::vector<std::int64_t> shape;
  ElementType element_type{ElementType::kOther};
};

struct PointCloudLayout {
  TensorLayout points;
  TensorLayout colors;
  TensorLayout normals;
  TensorLayout intensity;
  bool organized{};
  // -1 when not counted, else 0 .. point_count().
  std::int64_t valid_count{-1};
};

// Whether an optional field holds values: it is not the empty [0] that
// stands for "absent". (In a cloud of no points, a present intensity is also
// [0]; there is nothing in it either way.)
[[nodiscard]] bool is_present(const TensorLayout& field) noexcept;

// N, or H * W; 0 for points that are not a cloud's.
[[nodiscard]] std::int64_t point_count(const PointCloudLayout& layout) noexcept;

// Throws std::invalid_argument naming the first field that breaks the rules
// above ("point_cloud.colors must be uint8 [4, 5, 3] like points, or [0] when
// absent; it is float32 [4, 5, 3]"); `name` begins the field's name.
void validate(const PointCloudLayout& layout, std::string_view name = "point_cloud");

}  // namespace holoscan::messages
