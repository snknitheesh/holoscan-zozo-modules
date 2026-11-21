// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// FilterReportOp through the SDK's operator test harness: a view that fails
// to go out for another reason than a full display is reported, once.

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/pcl_filters/synthetic_scene.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "filter_report_op.hpp"
#include "scene_source_op.hpp"

namespace pcl_filters_app {
namespace {

namespace hm = holoscan::messages;
namespace pf = holoscan::pcl_filters;

int failures = 0;

void verify(bool condition, const std::string& what) {
  if (!condition) {
    ++failures;
    std::cerr << "check failed: " << what << '\n';
  }
}

constexpr int kWidth = 64;
constexpr int kHeight = 48;

// A tensor of `shape` holding `bytes` bytes from `data`, in host memory.
std::shared_ptr<holoscan::Tensor> host_tensor(const std::vector<std::int64_t>& shape,
                                              DLDataType dtype, const void* data,
                                              std::size_t bytes) {
  auto tensor =
      holoscan::make_tensor(shape, dtype, std::nullopt, holoscan::MemoryStorageType::kHost,
                            std::make_shared<holoscan::UnboundedAllocator>());
  if (!tensor) {
    throw std::runtime_error(tensor.error().message);
  }
  auto destination = tensor->mutable_data();
  if (!destination) {
    throw std::runtime_error(destination.error().message);
  }
  if (bytes != 0) {
    std::memcpy(*destination, data, bytes);
  }
  return std::make_shared<holoscan::Tensor>(std::move(*tensor));
}

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

hm::schema::PointCloudT cloud(const std::vector<std::int64_t>& leading,
                              const std::vector<float>& xyz, const std::vector<std::uint8_t>& rgb) {
  hm::schema::PointCloudT message;
  message.header = std::make_shared<holoscan::schema::HeaderT>(
      hm::to_message(hm::Header{.frame_id = kCameraFrame}));
  std::vector<std::int64_t> shape = leading;
  shape.push_back(3);
  message.points = host_tensor(shape, kFloat32, xyz.data(), xyz.size() * sizeof(float));
  message.colors = host_tensor(shape, kUInt8, rgb.data(), rgb.size());
  message.normals = host_tensor({0}, kFloat32, nullptr, 0);
  message.intensity = host_tensor({0}, kFloat32, nullptr, 0);
  message.organized = leading.size() == 2;
  message.valid_count = -1;
  return message;
}

// What the process writes to stdout and stderr, where the SDK logs, while
// `body` runs.
std::string capture_output(const std::function<void()>& body) {
  std::cout.flush();
  std::cerr.flush();
  std::fflush(nullptr);
  FILE* file = std::tmpfile();
  if (file == nullptr) {
    throw std::runtime_error("tmpfile failed");
  }
  const int saved_out = dup(STDOUT_FILENO);
  const int saved_err = dup(STDERR_FILENO);
  dup2(fileno(file), STDOUT_FILENO);
  dup2(fileno(file), STDERR_FILENO);
  struct Restore {
    int out;
    int err;
    ~Restore() {
      std::cout.flush();
      std::cerr.flush();
      std::fflush(nullptr);
      dup2(out, STDOUT_FILENO);
      dup2(err, STDERR_FILENO);
      close(out);
      close(err);
    }
  };
  {
    const Restore restore{saved_out, saved_err};
    body();
  }
  std::string output;
  std::rewind(file);
  char buffer[4096];
  for (std::size_t n; (n = std::fread(buffer, 1, sizeof(buffer), file)) > 0;) {
    output.append(buffer, n);
  }
  std::fclose(file);
  return output;
}

std::size_t count_of(const std::string& text, std::string_view needle) {
  std::size_t count = 0;
  for (std::size_t at = text.find(needle); at != std::string::npos;
       at = text.find(needle, at + needle.size())) {
    ++count;
  }
  return count;
}

// The view output exists but goes nowhere, so every send fails, which is no
// full display: the first failure is logged, and the frames still pass through.
void check_view_failure_logged_once() {
  holoscan::set_log_level(holoscan::LogLevel::INFO);
  constexpr std::uint32_t kFrames = 3;
  auto report = std::make_shared<FilterReport>();
  auto h = holoscan::testing::create_operator_test<FilterReportOp>(
      FilterReportConfig{.report = report,
                         .width = kWidth,
                         .height = kHeight,
                         .voxel_size = 0.05F,
                         .voxel_layout = pf::OutputLayout::kDense,
                         .expected_frames = kFrames,
                         .view_output = true});
  std::vector<float> xyz;
  std::vector<std::uint8_t> rgb;
  std::vector<pf::SceneLabel> labels;
  pf::synthetic_scene_cloud(pf::synthetic_scene(kWidth, kHeight, 0), xyz, rgb, labels);
  const std::vector<float> voxel{0.0F, 0.0F, 2.0F};
  const std::vector<std::uint8_t> voxel_rgb{10U, 20U, 30U};
  const std::vector<float> plane{0.0F, -1.0F, 0.0F, 1.0F};
  std::vector<holoscan::testing::TestSample<hm::schema::PointCloudT>> grounds;
  std::vector<holoscan::testing::TestSample<hm::schema::PointCloudT>> voxels;
  std::vector<holoscan::testing::TestSample<holoscan::Tensor>> planes;
  for (std::uint32_t frame = 0; frame < kFrames; ++frame) {
    const auto at = std::chrono::milliseconds(100 * (frame + 1));
    grounds.push_back(holoscan::testing::sample_at(cloud({kHeight, kWidth}, xyz, rgb), at));
    grounds.back().metadata.frame_id = frame + 1U;
    voxels.push_back(holoscan::testing::sample_at(cloud({1}, voxel, voxel_rgb), at));
    planes.push_back(holoscan::testing::sample_at(
        *host_tensor({4}, kFloat32, plane.data(), plane.size() * sizeof(float)), at));
  }
  h.push_samples("ground_cloud", std::move(grounds));
  h.push_samples("voxels", std::move(voxels));
  h.push_samples("plane", std::move(planes));
  bool ok = false;
  std::string failure;
  const std::string output = capture_output([&] {
    const auto result = h.run();
    ok = result.ok();
    failure = ok ? "" : result.describe_failure();
  });
  verify(ok, failure);
  verify(report->frames == kFrames, std::to_string(report->frames) + " frames reported");
  const std::size_t logged = count_of(output, "sending the view failed");
  verify(logged == 1,
         "the view failure was logged " + std::to_string(logged) + " times:\n" + output);
}

}  // namespace
}  // namespace pcl_filters_app

int main() {
  try {
    pcl_filters_app::check_view_failure_logged_once();
  } catch (const std::exception& error) {
    ++pcl_filters_app::failures;
    std::cerr << "unexpected exception: " << error.what() << '\n';
  }
  std::cout << (pcl_filters_app::failures == 0 ? "all passed" : "failed") << '\n';
  return pcl_filters_app::failures == 0 ? 0 : 1;
}
