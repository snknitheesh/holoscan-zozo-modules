// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The demo's operators on their own: the file source with a consumer far
// slower than its clock, and the reporter's frame-id check, its completion and
// its truth settings, through the SDK's operator test harness.
//
// Usage: pcd_app_ops_test DIR (a scratch folder for the files)

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "cloud_io.hpp"
#include "depth_image_report_op.hpp"
#include "point_cloud_sources.hpp"
#include "test_support.hpp"

namespace point_cloud_to_depth_app {
namespace {

constexpr DLDataType kFloat32{kDLFloat, 32, 1};

std::filesystem::path g_directory;

// A tensor of `shape` holding `bytes`, in host or device memory.
holoscan::Tensor make(const std::vector<std::int64_t>& shape, DLDataType dtype, const void* bytes,
                      std::size_t size, bool device) {
  auto tensor = holoscan::make_tensor(
      shape, dtype, std::nullopt,
      device ? holoscan::MemoryStorageType::kDevice : holoscan::MemoryStorageType::kHost,
      std::make_shared<holoscan::UnboundedAllocator>());
  if (!tensor) {
    throw std::runtime_error(tensor.error().message);
  }
  auto destination = tensor->mutable_data();
  if (!destination) {
    throw std::runtime_error(destination.error().message);
  }
  if (size != 0 && cudaMemcpy(*destination, bytes, size, cudaMemcpyDefault) != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed");
  }
  return std::move(*tensor);
}

// The capture time of frame `frame` in the tests' frames.
std::chrono::nanoseconds capture_time(std::uint32_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

// Records the frame ids of the clouds it receives, and checks that each
// cloud's points and colors belong together, taking long enough over each
// that the source must wait for it.
struct SlowSinkLog {
  std::vector<std::uint64_t> frame_ids;
  int mismatched{};
};

class SlowCloudSinkOp final : public holoscan::Operator<> {
 public:
  explicit SlowCloudSinkOp(std::shared_ptr<SlowSinkLog> log) : log_(std::move(log)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(points, "points").queue_depth(1U);
    spec.input(colors, "colors").queue_depth(1U);
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAligned{
        .ports = holoscan::PortList{std::vector<holoscan::PortRef>{points, colors}},
        .window = std::chrono::nanoseconds{0}});
    result.input(points, holoscan::Exact{});
    result.input(colors, holoscan::Exact{});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto cloud = points.receive();
    auto rgb = colors.receive();
    if (!cloud || !rgb) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "SlowCloudSinkOp: missing input"});
    }
    // File i holds points of value i and colors of value i.
    float x = 0.0F;
    std::uint8_t c = 0U;
    std::memcpy(&x, cloud->data.data(), sizeof(x));
    std::memcpy(&c, rgb->data.data(), sizeof(c));
    log_->mismatched +=
        static_cast<int>(x) != c || cloud->metadata.frame_id != rgb->metadata.frame_id;
    log_->frame_ids.push_back(cloud->metadata.frame_id);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (log_->frame_ids.size() == kFrames) {
      context.request_completion();
    }
    return {};
  }

  static constexpr std::size_t kFrames = 12;
  holoscan::Input<holoscan::Tensor> points;
  holoscan::Input<holoscan::Tensor> colors;

 private:
  std::shared_ptr<SlowSinkLog> log_;
};

// A consumer twenty times slower than the file source's clock still gets
// every frame whole, once, in order. The runtime holds the source back, so
// its emits do not fail today; should one be refused as backpressured, the
// source sends the rest of the frame on its next tick, and this checks that
// the frame still arrives whole.
void check_file_source_waits_for_consumer() {
  const std::filesystem::path folder = g_directory / "slow_sink_clouds";
  std::filesystem::create_directories(folder);
  std::vector<std::filesystem::path> files;
  for (int i = 0; i < 4; ++i) {
    files.push_back(folder / ("cloud_" + std::to_string(i) + ".ply"));
    write_ply(files.back(), std::vector<float>(30, static_cast<float>(i)),
              std::vector<std::uint8_t>(30, static_cast<std::uint8_t>(i)));
  }
  holoscan::Graph graph{"file_source_test"};
  auto source = graph.op<CloudFileSourceOp>(
      "source", CloudFileSourceConfig{.files = files,
                                      .max_points = 10,
                                      .color = true,
                                      .count = static_cast<std::uint32_t>(SlowCloudSinkOp::kFrames),
                                      .loop = true,
                                      .period = std::chrono::milliseconds(1)});
  auto log = std::make_shared<SlowSinkLog>();
  auto sink = graph.op<SlowCloudSinkOp>("sink", log);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->points, sink->points, lossless);
  graph.add_flow(source->colors, sink->colors, lossless);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const auto plan = holoscan::compile(graph);
  VERIFY_MSG(plan.ok(), holoscan::format_diagnostics(plan.diagnostics(), plan.graph()));
  if (!plan.ok()) return;
  auto session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  if (session.is_running()) {
    session.request_stop();
  }
  session.wait();
  VERIFY_MSG(session.termination() == holoscan::RunTermination::kRequestedCompleted,
             "run completes: " + holoscan::format_diagnostics(session.diagnostics(), plan.graph()));
  std::vector<std::uint64_t> expected(SlowCloudSinkOp::kFrames);
  for (std::size_t i = 0; i < expected.size(); ++i) expected[i] = i + 1U;
  VERIFY_MSG(log->frame_ids == expected,
             std::to_string(log->frame_ids.size()) + " frames, in order from 1");
  VERIFY_MSG(log->mismatched == 0, std::to_string(log->mismatched) + " clouds mismatched");
}

// The reporter counts an image whose frame id does not follow the previous one.
void check_report_frame_gaps(const std::vector<std::uint64_t>& frame_ids,
                             std::uint32_t expected_gaps) {
  auto report = std::make_shared<DepthReport>();
  auto h = holoscan::testing::create_operator_test<DepthImageReportOp>(
      DepthImageReportConfig{.report = report});
  const std::vector<float> depth(8, 1.5F);
  std::vector<holoscan::testing::TestSample<holoscan::schema::ImageT>> samples;
  for (std::size_t i = 0; i < frame_ids.size(); ++i) {
    holoscan::schema::ImageT image;
    image.width = 4;
    image.height = 2;
    image.encoding = holoscan::schema::ImageEncoding_MONO32F;
    image.data = std::make_shared<holoscan::Tensor>(
        make({2, 4}, kFloat32, depth.data(), depth.size() * sizeof(float), false));
    holoscan::testing::TestSample<holoscan::schema::ImageT> sample =
        holoscan::testing::sample_at(std::move(image), capture_time(static_cast<std::uint32_t>(i)));
    sample.metadata.frame_id = frame_ids[i];
    samples.push_back(std::move(sample));
  }
  h.push_samples("depth", std::move(samples));
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  VERIFY(report->frames == frame_ids.size());
  VERIFY(report->filled_pixels == 8U * frame_ids.size());
  VERIFY_MSG(report->frame_gaps == expected_gaps, std::to_string(report->frame_gaps) +
                                                      " frame gaps, expected " +
                                                      std::to_string(expected_gaps));
  VERIFY(report->passed() == (expected_gaps == 0));
}

// A sample of a 4x2 depth image with frame id `frame_id`.
holoscan::testing::TestSample<holoscan::schema::ImageT> depth_sample(std::uint64_t frame_id) {
  const std::vector<float> depth(8, 1.5F);
  holoscan::schema::ImageT image;
  image.width = 4;
  image.height = 2;
  image.encoding = holoscan::schema::ImageEncoding_MONO32F;
  image.data = std::make_shared<holoscan::Tensor>(
      make({2, 4}, kFloat32, depth.data(), depth.size() * sizeof(float), false));
  holoscan::testing::TestSample<holoscan::schema::ImageT> sample = holoscan::testing::sample_at(
      std::move(image), capture_time(static_cast<std::uint32_t>(frame_id - 1U)));
  sample.metadata.frame_id = frame_id;
  return sample;
}

// The reporter completes after its own expected frames, whatever the shared
// report counted before it: a report already holding frames does not end the
// run at its first image.
void check_report_completes_on_its_own_count() {
  auto report = std::make_shared<DepthReport>();
  report->frames = 5U;
  auto h = holoscan::testing::create_operator_test<DepthImageReportOp>(
      DepthImageReportConfig{.report = report, .expected_frames = 3});
  std::vector<holoscan::testing::TestSample<holoscan::schema::ImageT>> samples;
  for (std::uint64_t id = 1; id <= 4; ++id) samples.push_back(depth_sample(id));
  h.push_samples("depth", std::move(samples));
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  VERIFY_MSG(report->frames == 8U, std::to_string(report->frames.load()) +
                                       " frames counted, expected 5 + 3: completion at the third");
}

// A truth that does not fit the images' format is refused up front.
void check_report_rejects_unfit_truth() {
  const auto rejects = [](DepthFormat format, SceneTruth scene, const std::string& what) {
    try {
      DepthImageReportOp op(DepthImageReportConfig{
          .report = std::make_shared<DepthReport>(),
          .depth_format = format,
          .truth = DepthImageTruth{.width = 4, .height = 2, .scene = scene}});
      VERIFY_MSG(false, what + " is rejected");
    } catch (const std::invalid_argument& error) {
      VERIFY_MSG(std::string(error.what()).find(what) != std::string::npos,
                 "'" + std::string(error.what()) + "' names '" + what + "'");
    }
  };
  const SceneTruth u16{.millimeter_depth = true,
                       .format = DepthFormat::kUint16,
                       .depth_scale = 0.001F,
                       .depth_min = 0.1F,
                       .depth_max = 10.0F};
  rejects(DepthFormat::kFloat32, u16, "the truth's depth format");
  SceneTruth meters = u16;
  meters.millimeter_depth = false;
  rejects(DepthFormat::kUint16, meters, "millimeter scene and a 0.001 scale");
  SceneTruth scaled = u16;
  scaled.depth_scale = 0.002F;
  rejects(DepthFormat::kUint16, scaled, "millimeter scene and a 0.001 scale");
  // The images' own scale must be the truth's.
  try {
    DepthImageReportOp op(
        DepthImageReportConfig{.report = std::make_shared<DepthReport>(),
                               .depth_format = DepthFormat::kUint16,
                               .depth_scale = 0.002F,
                               .truth = DepthImageTruth{.width = 4, .height = 2, .scene = u16}});
    VERIFY_MSG(false, "an image scale other than the truth's is rejected");
  } catch (const std::invalid_argument&) {
  }
  // A fitting truth is accepted.
  DepthImageReportOp accepted(
      DepthImageReportConfig{.report = std::make_shared<DepthReport>(),
                             .depth_format = DepthFormat::kUint16,
                             .truth = DepthImageTruth{.width = 4, .height = 2, .scene = u16}});
}

}  // namespace
}  // namespace point_cloud_to_depth_app

int main(int argc, char** argv) {
  namespace app = point_cloud_to_depth_app;
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " DIR\n";
    return 2;
  }
  app::g_directory = argv[1];
  std::filesystem::remove_all(app::g_directory);
  std::filesystem::create_directories(app::g_directory);
  return app::test::run({
      {"the file source keeps every frame for a slow consumer",
       app::check_file_source_waits_for_consumer},
      {"the reporter accepts consecutive frame ids",
       [] { app::check_report_frame_gaps({1, 2, 3}, 0); }},
      {"the reporter counts a gap in frame ids",
       [] { app::check_report_frame_gaps({1, 2, 4, 5}, 1); }},
      {"the reporter completes on its own count", app::check_report_completes_on_its_own_count},
      {"the reporter refuses a truth that does not fit", app::check_report_rejects_unfit_truth},
  });
}
