// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// PointCloudReportOp through the SDK's operator test harness: its frame-id
// check, which accepts a first cloud after the first frame (the frames before
// the camera info) and fails a run whose clouds then skip a frame; the
// report's verdict on how many frames came before the camera info; the PLY
// files' names; and its check of the cloud message, which fails a run on a
// cloud that is not the operator's.
//
// Usage: depth_report_op_test DIR (a scratch folder for the files)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "point_cloud_report_op.hpp"
#include "test_support.hpp"

namespace depth_to_point_cloud_app {
namespace {

namespace hm = holoscan::messages;

std::filesystem::path g_directory;

// A tensor of `shape` holding `values`, in host memory.
template <typename T>
holoscan::Tensor host_tensor(const std::vector<std::int64_t>& shape, const std::vector<T>& values,
                             DLDataType dtype) {
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
  std::copy(values.begin(), values.end(), static_cast<T*>(*destination));
  return std::move(*tensor);
}

// An organized 2x4 cloud of the camera "camera" as the operator sends it,
// without colors.
hm::schema::PointCloudT cloud(bool organized = true) {
  constexpr DLDataType kFloat32{kDLFloat, 32, 1};
  const auto empty = [](DLDataType dtype) {
    return std::make_shared<holoscan::Tensor>(host_tensor<float>({0}, {}, dtype));
  };
  hm::schema::PointCloudT message;
  message.header = std::make_shared<holoscan::schema::HeaderT>();
  message.header->frame_id = "camera";
  message.points = std::make_shared<holoscan::Tensor>(host_tensor(
      {2, 4, 3}, std::vector<float>(static_cast<std::size_t>(2) * 4 * 3, 1.0F), kFloat32));
  message.colors = empty(DLDataType{kDLUInt, 8, 1});
  message.normals = empty(kFloat32);
  message.intensity = empty(kFloat32);
  message.organized = organized;
  return message;
}

// The reporter counts a cloud whose frame id does not follow the previous
// one; the first cloud may come after the first frame.
void check_report_frame_gaps(const std::vector<std::uint64_t>& frame_ids,
                             std::uint32_t expected_gaps) {
  auto report = std::make_shared<CloudReport>();
  auto h = holoscan::testing::create_operator_test<PointCloudReportOp>(
      PointCloudReportConfig{.report = report, .frame_id = "camera"});
  std::vector<holoscan::testing::TestSample<hm::schema::PointCloudT>> samples;
  for (std::size_t i = 0; i < frame_ids.size(); ++i) {
    holoscan::testing::TestSample<hm::schema::PointCloudT> sample =
        holoscan::testing::sample_at(cloud(), std::chrono::milliseconds(100 * (i + 1)));
    sample.metadata.frame_id = frame_ids[i];
    samples.push_back(std::move(sample));
  }
  h.push_samples("point_cloud", std::move(samples));
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  VERIFY(report->frames == frame_ids.size());
  VERIFY(report->first_frame_id == frame_ids.front());
  VERIFY_MSG(report->frame_gaps == expected_gaps, std::to_string(report->frame_gaps) +
                                                      " frame gaps, expected " +
                                                      std::to_string(expected_gaps));
  VERIFY(report->passed() == (expected_gaps == 0));
}

// Runs clouds of `frame_ids` through the reporter, writing PLY files into
// `ply_dir` when it is given, and returns its report.
std::shared_ptr<CloudReport> report_of(const std::vector<std::uint64_t>& frame_ids,
                                       const std::filesystem::path& ply_dir = {}) {
  auto report = std::make_shared<CloudReport>();
  auto h = holoscan::testing::create_operator_test<PointCloudReportOp>(
      PointCloudReportConfig{.report = report, .frame_id = "camera", .ply_dir = ply_dir});
  std::vector<holoscan::testing::TestSample<hm::schema::PointCloudT>> samples;
  for (std::size_t i = 0; i < frame_ids.size(); ++i) {
    holoscan::testing::TestSample<hm::schema::PointCloudT> sample =
        holoscan::testing::sample_at(cloud(), std::chrono::milliseconds(100 * (i + 1)));
    sample.metadata.frame_id = frame_ids[i];
    samples.push_back(std::move(sample));
  }
  h.push_samples("point_cloud", std::move(samples));
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  return report;
}

// A run of five frames is complete with a cloud for every frame from the
// first camera info on, and no more frames before it than allowed: one with
// a camera info source, none with camera info sent with every frame.
void check_report_complete() {
  VERIFY(report_of({1, 2, 3, 4, 5})->complete(5, 0));
  VERIFY(report_of({2, 3, 4, 5})->complete(5, 1));
  VERIFY(report_of({2, 3, 4, 5})->before_camera_info() == 1U);
  VERIFY_MSG(!report_of({2, 3, 4, 5})->complete(5, 0),
             "a frame before the camera info sent with every frame");
  VERIFY_MSG(!report_of({3, 4, 5})->complete(5, 1), "two frames before the first camera info");
  VERIFY_MSG(!report_of({5})->complete(5, 1), "only the last frame checked");
  VERIFY_MSG(!report_of({1, 2, 3, 4})->complete(5, 1), "the last frame missing");
  VERIFY(!CloudReport{}.complete(5, 1));
}

// PLY files are named by frame, from 0 for frame id 1, whatever frames came
// before the first cloud.
void check_report_names_ply_files_by_frame() {
  const std::filesystem::path folder = g_directory / "ply";
  std::filesystem::create_directories(folder);
  const auto report = report_of({2, 3}, folder);
  VERIFY(report->ply_files == 2U);
  VERIFY_MSG(std::filesystem::is_regular_file(folder / "frame_000001.ply") &&
                 std::filesystem::is_regular_file(folder / "frame_000002.ply") &&
                 !std::filesystem::exists(folder / "frame_000000.ply"),
             "frame ids 2 and 3 write frame_000001.ply and frame_000002.ply");
}

// A cloud that is not the operator's organized cloud fails the run.
void check_report_rejects_unorganized_cloud() {
  auto report = std::make_shared<CloudReport>();
  auto h = holoscan::testing::create_operator_test<PointCloudReportOp>(
      PointCloudReportConfig{.report = report, .frame_id = "camera"});
  h.push("point_cloud", cloud(false));
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), "an unorganized cloud fails the run");
  VERIFY(result.describe_failure().find("not the operator's organized cloud") != std::string::npos);
}

}  // namespace
}  // namespace depth_to_point_cloud_app

int main(int argc, char** argv) {
  namespace app = depth_to_point_cloud_app;
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " DIR\n";
    return 2;
  }
  app::g_directory = argv[1];
  std::filesystem::remove_all(app::g_directory);
  std::filesystem::create_directories(app::g_directory);
  return app::test::run({
      {"the reporter accepts consecutive frame ids",
       [] { app::check_report_frame_gaps({1, 2, 3}, 0); }},
      {"the reporter counts a gap in frame ids",
       [] { app::check_report_frame_gaps({1, 2, 4, 5}, 1); }},
      {"the first cloud may come after the first frame",
       [] { app::check_report_frame_gaps({2, 3, 4}, 0); }},
      {"the reporter rejects a cloud that is not the operator's",
       app::check_report_rejects_unorganized_cloud},
      {"a run is complete only with at most the allowed frames before the camera info",
       app::check_report_complete},
      {"PLY files are named by frame", app::check_report_names_ply_files_by_frame},
  });
}
