// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// PointCloudReportOp through the SDK's operator test harness: its frame-id
// check, which fails a run whose clouds skip a frame.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "point_cloud_report_op.hpp"
#include "test_support.hpp"

namespace depth_to_point_cloud_app {
namespace {

// A float32 tensor of `shape` holding `values`, in host memory.
holoscan::Tensor host_tensor(const std::vector<std::int64_t>& shape,
                             const std::vector<float>& values) {
  auto tensor = holoscan::make_tensor(shape, DLDataType{kDLFloat, 32, 1}, std::nullopt,
                                      holoscan::MemoryStorageType::kHost,
                                      std::make_shared<holoscan::UnboundedAllocator>());
  if (!tensor) {
    throw std::runtime_error(tensor.error().message);
  }
  auto destination = tensor->mutable_data();
  if (!destination) {
    throw std::runtime_error(destination.error().message);
  }
  std::copy(values.begin(), values.end(), static_cast<float*>(*destination));
  return std::move(*tensor);
}

// The reporter counts a cloud whose frame id does not follow the previous one.
void check_report_frame_gaps(const std::vector<std::uint64_t>& frame_ids,
                             std::uint32_t expected_gaps) {
  auto report = std::make_shared<CloudReport>();
  auto h = holoscan::testing::create_operator_test<PointCloudReportOp>(
      PointCloudReportConfig{.report = report});
  const std::vector<float> xyz(static_cast<std::size_t>(4) * 2 * 3, 1.0F);
  std::vector<holoscan::testing::TestSample<holoscan::Tensor>> samples;
  for (std::size_t i = 0; i < frame_ids.size(); ++i) {
    holoscan::testing::TestSample<holoscan::Tensor> sample = holoscan::testing::sample_at(
        host_tensor({2, 4, 3}, xyz), std::chrono::milliseconds(100 * (i + 1)));
    sample.metadata.frame_id = frame_ids[i];
    samples.push_back(std::move(sample));
  }
  h.push_samples("point_cloud", std::move(samples));
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  VERIFY(report->frames == frame_ids.size());
  VERIFY_MSG(report->frame_gaps == expected_gaps, std::to_string(report->frame_gaps) +
                                                      " frame gaps, expected " +
                                                      std::to_string(expected_gaps));
  VERIFY(report->passed() == (expected_gaps == 0));
}

}  // namespace
}  // namespace depth_to_point_cloud_app

int main() {
  namespace app = depth_to_point_cloud_app;
  return app::test::run({
      {"the reporter accepts consecutive frame ids",
       [] { app::check_report_frame_gaps({1, 2, 3}, 0); }},
      {"the reporter counts a gap in frame ids",
       [] { app::check_report_frame_gaps({1, 2, 4, 5}, 1); }},
  });
}
