// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The PointCloud message's tensor rules without the SDK: the layouts
// validate() accepts, and each one it refuses, naming the field.

#include "holoscan/messages/point_cloud_layout.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include "test_support.hpp"

namespace holoscan::messages {
namespace {

constexpr ElementType kF32 = ElementType::kFloat32;
constexpr ElementType kU8 = ElementType::kUInt8;
const TensorLayout kAbsent{{0}, kF32};

PointCloudLayout organized(bool optional_present) {
  PointCloudLayout layout;
  layout.organized = true;
  layout.points = {{4, 5, 3}, kF32};
  layout.colors = optional_present ? TensorLayout{{4, 5, 3}, kU8} : TensorLayout{{0}, kU8};
  layout.normals = optional_present ? TensorLayout{{4, 5, 3}, kF32} : kAbsent;
  layout.intensity = optional_present ? TensorLayout{{4, 5}, kF32} : kAbsent;
  return layout;
}

PointCloudLayout unorganized(std::int64_t n) {
  PointCloudLayout layout;
  layout.points = {{n, 3}, kF32};
  layout.colors = {{n, 3}, kU8};
  layout.normals = kAbsent;
  layout.intensity = {{n}, kF32};
  return layout;
}

template <typename Change>
bool refused(PointCloudLayout layout, Change change, std::string_view text) {
  change(layout);
  return test::refuses([&] { validate(layout); }, text);
}

void test_accepted() {
  validate(organized(true));
  validate(organized(false));
  validate(unorganized(7));
  validate(unorganized(0));
  PointCloudLayout counted = organized(true);
  counted.valid_count = 20;
  validate(counted);
  counted.valid_count = 0;
  validate(counted);
  VERIFY(point_count(organized(true)) == 20);
  VERIFY(point_count(unorganized(7)) == 7);
  VERIFY(is_present(organized(true).colors));
  VERIFY(!is_present(organized(false).colors));
  VERIFY(!is_present(kAbsent));
  VERIFY(is_present(unorganized(7).intensity));
}

void test_refused() {
  const PointCloudLayout base = organized(true);
  VERIFY(refused(
      base, [](PointCloudLayout& l) { l.points.element_type = ElementType::kOther; },
      "point_cloud.points must be float32 [H, W, 3]"));
  VERIFY(refused(
      base, [](PointCloudLayout& l) { l.points.shape = {20, 3}; },
      "point_cloud.points must be float32 [H, W, 3] with H, W > 0 (the cloud is organized); it is "
      "float32 [20, 3]"));
  VERIFY(refused(
      base, [](PointCloudLayout& l) { l.points.shape = {0, 5, 3}; },
      "point_cloud.points must be float32 [H, W, 3]"));
  VERIFY(refused(
      unorganized(3), [](PointCloudLayout& l) { l.points.shape = {3, 4}; },
      "point_cloud.points must be float32 [N, 3] (the cloud is not organized)"));
  VERIFY(refused(
      unorganized(3), [](PointCloudLayout& l) { l.organized = true; }, "the cloud is organized"));
  VERIFY(refused(
      base, [](PointCloudLayout& l) { l.colors.element_type = kF32; },
      "point_cloud.colors must be uint8 [4, 5, 3] like points, or [0] when absent; it is float32 "
      "[4, 5, 3]"));
  VERIFY(refused(
      base, [](PointCloudLayout& l) { l.colors.shape = {5, 4, 3}; },
      "point_cloud.colors must be uint8 [4, 5, 3]"));
  VERIFY(refused(
      base, [](PointCloudLayout& l) { l.normals.shape = {4, 5, 4}; },
      "point_cloud.normals must be float32 [4, 5, 3]"));
  VERIFY(refused(
      base, [](PointCloudLayout& l) { l.intensity.shape = {4, 5, 1}; },
      "point_cloud.intensity must be float32 [4, 5] like points"));
  VERIFY(refused(
      unorganized(3), [](PointCloudLayout& l) { l.intensity.shape = {2}; },
      "point_cloud.intensity must be float32 [3]"));
  VERIFY(refused(
      base, [](PointCloudLayout& l) { l.valid_count = 21; },
      "point_cloud.valid_count must be -1 or 0..20, not 21"));
  VERIFY(refused(base, [](PointCloudLayout& l) { l.valid_count = -2; }, "point_cloud.valid_count"));
  // More values than an int64 counts: refused, and counted as no points.
  PointCloudLayout huge = organized(false);
  huge.points.shape = {std::int64_t{1} << 40, std::int64_t{1} << 40, 3};
  VERIFY(point_count(huge) == 0);
  VERIFY(test::refuses([&] { validate(huge); }, "point_cloud.points must be float32 [H, W, 3]"));
}

}  // namespace
}  // namespace holoscan::messages

int main() {
  namespace m = holoscan::messages;
  return holoscan::messages::test::run({
      {"accepted layouts", m::test_accepted},
      {"refused layouts name the field", m::test_refused},
  });
}
