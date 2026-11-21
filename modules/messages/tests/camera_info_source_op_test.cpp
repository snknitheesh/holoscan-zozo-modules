// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// CameraInfoSourceOp: what it sends for one camera_info file and for a stereo
// pair's two (the files' calibration, stamped with the send time, `count`
// times on its period), to a consumer too slow for it, in a second run of the
// same plan, and what it refuses when constructed.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/messages/synthetic_calibration.hpp>

#include "graph_support.hpp"
#include "test_support.hpp"

namespace holoscan::messages {
namespace {

namespace fs = std::filesystem;
constexpr std::size_t kCount = 4;
constexpr auto kPeriod = std::chrono::milliseconds{15};

struct Folder {
  test::TemporaryFolder folder{"messages_source"};
  const fs::path& path = folder.path();
};

// The capture times are the clock's ticks, a whole number of periods apart
// (a tick the source ran late for is coalesced into the next), and equal the
// headers'.
template <typename Message>
void check_times(const std::vector<test::Received<Message>>& received,
                 std::chrono::nanoseconds period = kPeriod) {
  for (std::size_t i = 0; i < received.size(); ++i) {
    VERIFY(received[i].data.header->capture_timestamp_ns == received[i].capture_timestamp_ns);
    if (i == 0) continue;
    const std::int64_t step =
        received[i].capture_timestamp_ns - received[i - 1].capture_timestamp_ns;
    VERIFY_MSG(step > 0 && step % period.count() == 0, "step " + std::to_string(step) + " ns");
  }
}

template <typename Message>
std::vector<test::Received<Message>> run_source(CameraInfoSourceConfig config,
                                                std::size_t expected) {
  auto received = std::make_shared<std::vector<test::Received<Message>>>();
  holoscan::Graph graph{"source"};
  auto source = graph.op<CameraInfoSourceOp>("camera_info", std::move(config));
  auto sink = graph.op<test::CaptureOp<Message>>("sink", received, expected);
  if constexpr (std::is_same_v<Message, schema::CameraInfoT>) {
    graph.add_flow(source->camera_info, sink->input, test::lossless());
  } else {
    graph.add_flow(source->stereo_camera_info, sink->input, test::lossless());
  }
  const test::RunOutcome outcome = test::run_graph(graph);
  VERIFY_MSG(outcome.ok(), outcome.diagnostics);
  return *received;
}

void test_mono() {
  const Folder folder;
  const CameraInfo truth = synthetic_stereo_camera_info(DistortionModel::kPlumbBob, 640, 480).left;
  write_camera_info_yaml(truth, folder.path / "camera.yaml");
  const auto received = run_source<schema::CameraInfoT>(
      {.camera_info_path = folder.path / "camera.yaml", .period = kPeriod, .count = kCount},
      kCount);
  VERIFY(received.size() == kCount);
  for (const auto& message : received) {
    CameraInfo info = from_message(message.data);
    VERIFY(info.header.frame_id == "left" && info.header.device_sequence == 0);
    info.header.capture_timestamp_ns = truth.header.capture_timestamp_ns;
    VERIFY(info == truth);
  }
  check_times(received);
}

void test_stereo() {
  const Folder folder;
  const StereoCameraInfo truth =
      synthetic_stereo_camera_info(DistortionModel::kEquidistant, 640, 480);
  write_stereo_camera_info_yaml(truth, folder.path / "left.yaml", folder.path / "right.yaml");
  const StereoCameraInfo from_files =
      read_stereo_camera_info_yaml(folder.path / "left.yaml", folder.path / "right.yaml");
  // Unlimited: the sink ends the run.
  const auto received =
      run_source<schema::StereoCameraInfoT>({.camera_info_path = folder.path / "left.yaml",
                                             .right_camera_info_path = folder.path / "right.yaml",
                                             .period = kPeriod},
                                            kCount);
  VERIFY(received.size() == kCount);
  for (const auto& message : received) {
    const StereoCameraInfo info = from_message(message.data);
    VERIFY(same_calibration(info, from_files));
    VERIFY(info.header.frame_id == "left" && info.right.header.frame_id == "right");
    VERIFY(info.left.header.capture_timestamp_ns == message.capture_timestamp_ns &&
           info.right.header.capture_timestamp_ns == message.capture_timestamp_ns);
  }
  check_times(received);
}

// Takes one message at a time and holds each for 25 ms: on another worker
// than the source, which sends faster than it takes them, so that the source
// is refused (kBackpressured) while its one queue slot is full.
class SlowSinkOp final : public holoscan::Operator<> {
 public:
  SlowSinkOp(std::shared_ptr<std::vector<test::Received<schema::CameraInfoT>>> received,
             std::size_t count)
      : received_(std::move(received)), count_(count) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) return holoscan::make_unexpected(std::move(sample).error());
    received_->push_back({sample->data, sample->metadata.capture_timestamp_ns});
    std::this_thread::sleep_for(std::chrono::milliseconds{25});
    if (received_->size() == count_) context.request_completion();
    return {};
  }

  holoscan::Input<schema::CameraInfoT> input;

 private:
  std::shared_ptr<std::vector<test::Received<schema::CameraInfoT>>> received_;
  std::size_t count_;
};

void test_slow_consumer() {
  // Every 5 ms into a lossless queue of one, taken every 25 ms: a refused send
  // is no failure, and every one of the `count` is sent in the end.
  const Folder folder;
  const CameraInfo truth = synthetic_stereo_camera_info(DistortionModel::kPlumbBob, 640, 480).left;
  write_camera_info_yaml(truth, folder.path / "camera.yaml");
  constexpr std::size_t kSent = 8;
  constexpr auto kFast = std::chrono::milliseconds{5};
  auto received = std::make_shared<std::vector<test::Received<schema::CameraInfoT>>>();
  holoscan::Graph graph{"slow"};
  auto source = graph.op<CameraInfoSourceOp>(
      "camera_info",
      CameraInfoSourceConfig{
          .camera_info_path = folder.path / "camera.yaml", .period = kFast, .count = kSent});
  auto sink = graph.op<SlowSinkOp>("sink", received, kSent);
  graph.add_flow(source->camera_info, sink->input, test::lossless());
  // A worker each: one chain would otherwise run on one.
  graph.partition("source").add(source);
  graph.partition("sink").add(sink);
  const test::RunOutcome outcome = test::run_graph(graph);
  VERIFY_MSG(outcome.ok(), outcome.diagnostics);
  VERIFY(received->size() == kSent);
  check_times(*received, kFast);
  // Faster than it is taken: some ticks were refused and sent again later.
  if (received->size() == kSent) {
    VERIFY(received->back().capture_timestamp_ns - received->front().capture_timestamp_ns >
           static_cast<std::int64_t>(kSent - 1) * std::chrono::nanoseconds(kFast).count());
  }
}

// Counts what each run of a plan receives; ends a run at the `count`th.
class RunCountOp final : public holoscan::Operator<> {
 public:
  RunCountOp(std::shared_ptr<std::vector<std::size_t>> runs, std::size_t count)
      : runs_(std::move(runs)), count_(count) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(4U); }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }
  void start() override { runs_->push_back(0); }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) return holoscan::make_unexpected(std::move(sample).error());
    if (++runs_->back() == count_) context.request_completion();
    return {};
  }

  holoscan::Input<schema::CameraInfoT> input;

 private:
  std::shared_ptr<std::vector<std::size_t>> runs_;
  std::size_t count_;
};

void test_second_run() {
  // A limited count is per run: a second run of the same plan sends it all again.
  const Folder folder;
  write_camera_info_yaml(synthetic_stereo_camera_info(DistortionModel::kPlumbBob).left,
                         folder.path / "camera.yaml");
  auto runs = std::make_shared<std::vector<std::size_t>>();
  holoscan::Graph graph{"twice"};
  auto source = graph.op<CameraInfoSourceOp>(
      "camera_info",
      CameraInfoSourceConfig{
          .camera_info_path = folder.path / "camera.yaml", .period = kPeriod, .count = kCount});
  auto sink = graph.op<RunCountOp>("sink", runs, kCount);
  graph.add_flow(source->camera_info, sink->input, test::lossless());
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const auto plan = holoscan::compile(graph);
  VERIFY_MSG(plan.ok(), holoscan::format_diagnostics(plan.diagnostics(), plan.graph()));
  if (!plan.ok()) return;
  for (int run = 0; run < 2; ++run) {
    auto session = holoscan::run_async(plan);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
    while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds{5});
    }
    if (session.is_running()) session.request_stop();
    session.wait();
    VERIFY_MSG(session.termination() == holoscan::RunTermination::kRequestedCompleted,
               "run " + std::to_string(run + 1) + " completed");
  }
  VERIFY_MSG(*runs == (std::vector<std::size_t>{kCount, kCount}),
             "each run gets the count: " + std::to_string(runs->size()) + " runs");
}

void test_refusals() {
  const Folder folder;
  const fs::path missing = folder.path / "missing.yaml";
  VERIFY(test::refuses([&] { CameraInfoSourceOp op({.camera_info_path = missing}); },
                       missing.string() + ": cannot read"));
  VERIFY(test::refuses([&] { CameraInfoSourceOp op({}); }, "camera_info_path is required"));
  write_camera_info_yaml(synthetic_stereo_camera_info(DistortionModel::kPlumbBob).left,
                         folder.path / "camera.yaml");
  VERIFY(test::refuses(
      [&] {
        CameraInfoSourceOp op({.camera_info_path = folder.path / "camera.yaml",
                               .period = std::chrono::nanoseconds{0}});
      },
      "the period must be > 0"));
  {
    std::ofstream bad(folder.path / "bad.yaml");
    bad << "image_width: 640\nimage_height: 480\nfocal: 3\n";
  }
  VERIFY(
      test::refuses([&] { CameraInfoSourceOp op({.camera_info_path = folder.path / "bad.yaml"}); },
                    "'focal' (line 3) is not a camera_info key"));
  // A single camera's file is not a stereo pair: its P has no baseline.
  VERIFY(test::refuses(
      [&] {
        CameraInfoSourceOp op({.camera_info_path = folder.path / "camera.yaml",
                               .right_camera_info_path = folder.path / "camera.yaml"});
      },
      "carries no baseline"));
}

}  // namespace
}  // namespace holoscan::messages

int main() {
  namespace m = holoscan::messages;
  return holoscan::messages::test::run({
      {"one camera_info file", m::test_mono},
      {"a stereo pair's files, until stopped", m::test_stereo},
      {"a consumer slower than the period", m::test_slow_consumer},
      {"a second run of the same plan", m::test_second_run},
      {"refusals at construction", m::test_refusals},
  });
}
