// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The real-time path (edges.hpp) with a consumer ten times slower than the
// source: synthetic frames every 20 ms into the stub model's detector, whose
// detections go to the reporter, which spends 200 ms on each, and to a
// counter, which takes them all. The run must complete without a failure, the
// detector must keep the source's pace, not the reporter's, and the reporter
// must get the newest detections, each matching its own frame, ending with
// the last frame's.
//
// The checks count frames rather than time them: the source's period and the
// reporter's delay are ten to one, the bounds two or three to one.
//
// Usage: yolo26_slow_consumer_test, in the folder where the stub model's
// engine is cached.

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/yolo26/stub_model.hpp>
#include <holoscan/yolo26/yolo26_detection_op.hpp>

#include "detection_report_op.hpp"
#include "edges.hpp"
#include "image_sources.hpp"
#include "test_support.hpp"

namespace yolo26_app {
namespace {

namespace y26 = holoscan::yolo26;
using std::chrono_literals::operator""ms;

constexpr std::uint32_t kFrames = 50;
constexpr auto kPeriod = 20ms;
constexpr auto kSlow = 200ms;
constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr int kInputSize = 640;

// What the counter saw: the detections of how many frames, and whether their
// ids went up.
struct Counted {
  std::uint32_t frames{};
  std::uint64_t last_frame_id{};
  bool in_order{true};
};

// Counts each frame the detector finishes, by its `count` output. Its edge
// holds every frame of the run, so it never holds the detector back.
class CountingSinkOp final : public holoscan::Operator<> {
 public:
  explicit CountingSinkOp(std::shared_ptr<Counted> counted) : counted_(std::move(counted)) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.input(count, "count"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{count});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& /*context*/) override {
    auto sample = count.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    const std::uint64_t frame_id = sample->metadata.frame_id;
    counted_->in_order = counted_->in_order && frame_id > counted_->last_frame_id;
    counted_->last_frame_id = frame_id;
    ++counted_->frames;
    return {};
  }

  holoscan::Input<holoscan::Tensor> count;

 private:
  std::shared_ptr<Counted> counted_;
};

void check_slow_consumer() {
  holoscan::Graph graph{"slow_consumer"};
  auto source = graph.op<SyntheticImageSourceOp>(
      "source", SyntheticImageSourceConfig{
                    .width = kWidth, .height = kHeight, .count = kFrames, .period = kPeriod});
  const y26::Yolo26DetectionConfig detection{
      .model = y26::write_stub_model(std::filesystem::current_path(), kInputSize),
      .input_size = kInputSize,
      .num_classes = y26::kStubClasses};
  auto detector = graph.op<y26::Yolo26DetectionOp>("detector", detection);
  auto report = std::make_shared<DetectionReport>();
  auto reporter = graph.op<DetectionReportOp>(
      "report",
      DetectionReportConfig{
          .report = report,
          .expected_frames = kFrames,
          .realtime = true,
          .delay = kSlow,
          .truth = SyntheticTruth{.width = kWidth, .height = kHeight, .input_size = kInputSize}});
  auto counted = std::make_shared<Counted>();
  auto counter = graph.op<CountingSinkOp>("counter", counted);

  // The demo's real-time edges, as for the camera.
  graph.add_flow(source->image, detector->image, newest());
  connect_detections(graph, detector, reporter, newest());
  graph.add_flow(detector->count, counter->count,
                 holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kBlocking,
                                             .queue_depth = kFrames});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  for (const char* port : {"boxes", "scores", "class_ids", "count"}) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "detector", .output_port = port, .device = holoscan::DeviceId{0}});
  }
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "source", .output_port = "image", .device = holoscan::DeviceId{0}});
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  VERIFY_MSG(plan.ok(), holoscan::format_diagnostics(plan.diagnostics(), plan.graph()));
  if (!plan.ok()) {
    return;
  }
  y26::prepare_engine(detection);
  // Caller-managed signals: a test process leaves SIGINT and SIGTERM alone.
  auto session = holoscan::run_async(plan);
  session.wait();

  // No operator failed, a refused output (kBackpressured) among them: the
  // run completed when the reporter had the last frame.
  VERIFY_MSG(session.diagnostics().empty(),
             holoscan::format_diagnostics(session.diagnostics(), plan.graph()));
  VERIFY(session.termination() == holoscan::RunTermination::kRequestedCompleted);

  const std::uint32_t reported = report->frames.load();
  std::cout << "source " << kFrames << " frames, detector " << counted->frames << ", reporter "
            << reported << " (skipped " << report->skipped << ", last " << report->last_frame_id
            << ")\n";
  // The detector kept the source's pace: at the reporter's, it would have
  // finished about one frame in ten.
  VERIFY_MSG(counted->frames >= kFrames * 3U / 4U, "the detector finished " +
                                                       std::to_string(counted->frames) + " of " +
                                                       std::to_string(kFrames) + " frames");
  VERIFY(counted->in_order && counted->last_frame_id == kFrames);
  // The reporter was the slow one, and took the newest detections each time
  // rather than the next in line: it skipped most frames, and still got the
  // last.
  VERIFY_MSG(reported <= kFrames / 4U,
             "the reporter took " + std::to_string(reported) + " frames; it is not slow");
  VERIFY(report->skipped >= kFrames / 2U);
  VERIFY(report->frame_gaps == 0 && report->last_frame_id == kFrames &&
         reported + report->skipped == kFrames);
  // Every set it took was one frame's own detections, all four outputs of
  // the same frame.
  VERIFY(report->verified && report->missed == 0 && report->extra == 0);
}

}  // namespace
}  // namespace yolo26_app

int main() { return yolo26_app::test::run({{"slow_consumer", yolo26_app::check_slow_consumer}}); }
