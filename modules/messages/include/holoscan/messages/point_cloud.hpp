// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Sending and receiving schema::PointCloudT, the point cloud message (see
// point_cloud_layout.hpp for its tensors).
//
// A producer declares its output, then per cloud allocates every tensor,
// fills the ones it carries and sends the cloud (namespace hm =
// holoscan::messages):
//
//   void setup(holoscan::OperatorSpec& spec) override {
//     hm::declare_point_cloud_output(spec, cloud, "cloud",
//                                    {.max_points = 640 * 480, .colors = true});
//   }
//   compute():
//     const auto shape = hm::organized_cloud(480, 640);
//     auto loan = hm::allocate_point_cloud(cloud, shape, {.colors = true}, stream);
//     if (!loan) return holoscan::make_unexpected(std::move(loan).error());
//     if (auto r = loan->write_and_commit(hm::PointCloudFields::points, stream, fill_points);
//         !r) return r;
//     if (auto r = loan->write_and_commit(hm::PointCloudFields::colors, stream, fill_colors);
//         !r) return r;
//     return hm::emit_point_cloud(cloud, std::move(*loan), header, shape, -1, options);
//
// A consumer declares a plain input (spec.input(cloud, "cloud")), and checks
// what it receives with check_point_cloud().

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/memory_kind.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/tensor_schema_output_loan.hpp>
#include <holoscan/messages/camera_info.hpp>
#include <holoscan/messages/point_cloud_layout.hpp>
#include <holoscan/messages/point_cloud_schema_traits.hpp>

namespace holoscan::messages {

// The tensor fields' selectors: points, colors, normals, intensity.
using PointCloudFields = schema::PointCloudTensorFields;
// One cloud's tensors, allocated and not yet sent.
using PointCloudLoan = holoscan::TensorSchemaOutputLoan<schema::PointCloudT>;

// What an output sends: the largest cloud, the optional fields it can carry,
// and where its tensors are.
struct PointCloudOutputSpec {
  // N or H * W, > 0.
  std::uint64_t max_points{};
  bool colors{};
  bool normals{};
  bool intensity{};
  // kCudaDevice, kHost or kPinnedHost.
  holoscan::MemoryKind memory_kind{holoscan::MemoryKind::kCudaDevice};
};

// Declares `port` as `name` with a runtime pool for each tensor, sized for
// `output` (a field it does not carry gets room for [0] only).
// @throws std::invalid_argument for max_points 0 or too large to size the
// pools, or a memory kind other than host, pinned host or CUDA device memory.
void declare_point_cloud_output(holoscan::OperatorSpec& spec,
                                holoscan::Output<schema::PointCloudT>& port, std::string_view name,
                                const PointCloudOutputSpec& output);

// [N, ...] or [H, W, ...].
struct PointCloudShape {
  bool organized{};
  // Organized: the rows H. Unorganized: 0.
  std::int64_t height{};
  // Organized: the columns W. Unorganized: the number of points N.
  std::int64_t width{};
};
[[nodiscard]] constexpr PointCloudShape unorganized_cloud(std::int64_t points) noexcept {
  return {.organized = false, .height = 0, .width = points};
}
[[nodiscard]] constexpr PointCloudShape organized_cloud(std::int64_t height,
                                                        std::int64_t width) noexcept {
  return {.organized = true, .height = height, .width = width};
}

// The optional fields a cloud carries.
struct PointCloudContents {
  bool colors{};
  bool normals{};
  bool intensity{};
};

// Allocates every tensor of one cloud on `port`: points and the fields
// `contents` names at `shape`, the others as [0], already committed (there is
// nothing to write in them). Fill the others with the loan's
// write_and_commit() (device memory, on `stream`) or write_host_and_commit()
// (host and pinned host memory), then send the loan with emit_point_cloud().
[[nodiscard]] holoscan::expected<PointCloudLoan, holoscan::Error> allocate_point_cloud(
    holoscan::Output<schema::PointCloudT>& port, const PointCloudShape& shape,
    const PointCloudContents& contents, ::cuda::stream_ref stream);

// Sends a cloud allocated by allocate_point_cloud(), with `header`, the
// organization of `shape` and `valid_count` (-1: not counted).
// @return The send's result, or kInvalidArgument for a frame id longer than
// kMaxFrameIdBytes, a `shape` other than the one the points were allocated
// with, or a valid_count other than -1 or 0 .. the number of points.
[[nodiscard]] holoscan::expected<void, holoscan::Error> emit_point_cloud(
    holoscan::Output<schema::PointCloudT>& port, PointCloudLoan loan, const Header& header,
    const PointCloudShape& shape, std::int64_t valid_count, holoscan::EmitOptions options = {});

// The shapes and element types of a received cloud's tensors; a missing
// tensor reads as [0], an absent field.
[[nodiscard]] PointCloudLayout layout_of(const schema::PointCloudT& cloud);

// Checks a received cloud: all four tensors there, their layout (see
// point_cloud_layout.hpp), each packed (row-major, without gaps), the fields
// it carries in the same memory as the points (host, pinned host or the same
// GPU's), and the frame id's length.
// @return kInvalidArgument naming the first problem.
[[nodiscard]] holoscan::expected<void, holoscan::Error> check_point_cloud(
    const schema::PointCloudT& cloud);

// Whether an optional tensor holds values (it is not the [0] of an absent field).
[[nodiscard]] bool is_present(const std::shared_ptr<holoscan::Tensor>& field) noexcept;

}  // namespace holoscan::messages
