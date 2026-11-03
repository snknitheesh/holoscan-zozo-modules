// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The recommended consumer contract, camera_info_contract() with
// LatestCameraInfo, in running graphs: frames of a stereo pair or of one
// camera, with camera info from CameraInfoSourceOp at its own rate, or sent
// with every frame before or after the frames; a calibration that changes; an
// invalid camera info; and a consumer whose camera info is not wired. The
// frames are bare SDK Headers: the contract does not look into them.

#include <unistd.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/messages/synthetic_calibration.hpp>
#include <holoscan/schemas/header_schema_traits.hpp>

#include "graph_support.hpp"
#include "test_support.hpp"

namespace holoscan::messages {
namespace {

namespace fs = std::filesystem;
using Frame = holoscan::schema::HeaderT;
constexpr auto kFramePeriod = std::chrono::milliseconds{5};
constexpr std::uint32_t kFrames = 40;

enum class PerFrameInfo { kNone, kBefore, kAfter };

// What a consumer saw.
struct Report {
  std::uint32_t frames{};         // Activations.
  std::uint32_t with_info{};      // Frames that had camera info.
  std::uint32_t matching_time{};  // Whose camera info has the frame's capture time.
  std::uint32_t changes{};        // Times changed() said so.
  std::uint32_t wrong_values{};   // Camera info other than the one sent.
  std::uint32_t first_with_info{UINT32_MAX};
};

template <typename Value>
struct FrameSourceConfig {
  bool stereo{};
  PerFrameInfo per_frame{PerFrameInfo::kNone};
  // Sent with every frame; from frame `change_at` on, `changed` instead.
  Value info;
  std::optional<Value> changed;
  std::uint32_t change_at{UINT32_MAX};
};

// kFrames frames (left and right, or one), each with its activation time as
// capture time, and the camera info with the same capture time if asked.
template <typename Value>
class FrameSourceOp final : public holoscan::Operator<> {
 public:
  using Message = decltype(to_message(std::declval<const Value&>()));

  explicit FrameSourceOp(FrameSourceConfig<Value> config) : config_(std::move(config)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(left, "left");
    if (config_.stereo) spec.output(right, "right");
    if (config_.per_frame != PerFrameInfo::kNone) spec.output(info, "info");
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kFramePeriod});
    return result;
  }
  void start() override {
    sent_ = 0;
    pending_.reset();
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (sent_ >= kFrames) return {};
    if (!pending_) {
      pending_ =
          holoscan::EmitOptions{.capture_time = context.activation_time(), .frame_id = sent_ + 1U};
      step_ = 0;
    }
    // The outputs of this frame, in order; a consumer without room makes the
    // rest wait for the next tick.
    std::vector<std::function<holoscan::expected<void, holoscan::Error>()>> steps;
    if (config_.per_frame == PerFrameInfo::kBefore)
      steps.emplace_back([this] { return send_info(); });
    steps.emplace_back([this] { return left.emit(frame(), *pending_); });
    if (config_.stereo) steps.emplace_back([this] { return right.emit(frame(), *pending_); });
    if (config_.per_frame == PerFrameInfo::kAfter)
      steps.emplace_back([this] { return send_info(); });
    for (; step_ < steps.size(); ++step_) {
      auto sent = steps[step_]();
      if (!sent) {
        return sent.error().code == holoscan::ErrorCode::kBackpressured
                   ? holoscan::expected<void, holoscan::Error>{}
                   : sent;
      }
    }
    pending_.reset();
    ++sent_;
    return {};
  }

  holoscan::Output<Frame> left;
  holoscan::Output<Frame> right;
  holoscan::Output<Message> info;

 private:
  [[nodiscard]] Frame frame() const {
    Frame frame;
    frame.capture_timestamp_ns = pending_->capture_time->timestamp_ns;
    frame.device_sequence = sent_ + 1U;
    return frame;
  }

  holoscan::expected<void, holoscan::Error> send_info() {
    Value value = sent_ >= config_.change_at ? *config_.changed : config_.info;
    value.header.capture_timestamp_ns = pending_->capture_time->timestamp_ns;
    return info.emit(to_message(value), *pending_);
  }

  FrameSourceConfig<Value> config_;
  std::uint32_t sent_{};
  // The frame being sent, and how many of its outputs went out.
  std::optional<holoscan::EmitOptions> pending_;
  std::size_t step_{};
};

template <typename Value>
struct ConsumerConfig {
  bool stereo{};
  CameraInfoInputOptions options;
  // The calibrations a consumer may get.
  std::vector<Value> expected;
  std::shared_ptr<Report> report;
};

// Counts every frame and checks its camera info; ends the run at the last frame.
template <typename Value>
class ConsumerOp final : public holoscan::Operator<> {
 public:
  using Message = decltype(to_message(std::declval<const Value&>()));

  explicit ConsumerOp(ConsumerConfig<Value> config) : config_(std::move(config)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(left, "left").queue_depth(4U);
    if (config_.stereo) spec.input(right, "right").queue_depth(4U);
    spec.input(camera_info, "camera_info").queue_depth(4U);
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    if (config_.stereo) return camera_info_contract(config_.options, camera_info, left, right);
    return camera_info_contract(config_.options, camera_info, left);
  }
  void start() override { info_.reset(); }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto frame = left.receive();
    if (!frame) return holoscan::make_unexpected(std::move(frame).error());
    auto info = info_.update(camera_info);
    if (!info) return holoscan::make_unexpected(std::move(info).error());
    Report& report = *config_.report;
    if (*info != nullptr) {
      const Value& value = **info;
      if (report.with_info++ == 0) report.first_with_info = report.frames;
      report.matching_time +=
          value.header.capture_timestamp_ns == frame->metadata.capture_timestamp_ns ? 1U : 0U;
      report.changes += info_.changed() ? 1U : 0U;
      bool known = false;
      for (const Value& expected : config_.expected)
        known = known || same_calibration(value, expected);
      report.wrong_values += known ? 0U : 1U;
    }
    if (++report.frames == kFrames) context.request_completion();
    return {};
  }

  holoscan::Input<Frame> left;
  holoscan::Input<Frame> right;
  holoscan::Input<Message> camera_info;

 private:
  ConsumerConfig<Value> config_;
  LatestCameraInfo<Message> info_;
};

// Sends `message` on every tick after skipping the first `skip` ticks: camera
// info that starts after the frames.
class LateInfoSourceOp final : public holoscan::Operator<> {
 public:
  LateInfoSourceOp(schema::StereoCameraInfoT message, std::uint32_t skip)
      : message_(std::move(message)), skip_(skip) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.output(info, "info"); }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kFramePeriod});
    return result;
  }
  void start() override { ticks_ = 0; }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (ticks_++ < skip_) return {};
    const holoscan::TimePoint now = context.activation_time();
    message_.header->capture_timestamp_ns = now.timestamp_ns;
    return info.emit_ref(message_, holoscan::EmitOptions{.capture_time = now});
  }

  holoscan::Output<schema::StereoCameraInfoT> info;

 private:
  schema::StereoCameraInfoT message_;
  std::uint32_t skip_;
  std::uint32_t ticks_{};
};

// A stereo pair's camera_info files in a folder of their own.
struct CalibrationFiles {
  CalibrationFiles()
      : dir(fs::temp_directory_path() / ("messages_contract_" + std::to_string(::getpid()))) {
    fs::create_directories(dir);
    write_stereo_camera_info_yaml(pair, dir / "left.yaml", dir / "right.yaml");
  }
  ~CalibrationFiles() { fs::remove_all(dir); }
  CalibrationFiles(const CalibrationFiles&) = delete;
  CalibrationFiles& operator=(const CalibrationFiles&) = delete;

  StereoCameraInfo pair = synthetic_stereo_camera_info(DistortionModel::kPlumbBob, 640, 480);
  fs::path dir;
};

template <typename Value>
test::RunOutcome run_consumer(FrameSourceConfig<Value> frames, CameraInfoInputOptions options,
                              const std::vector<Value>& expected, std::shared_ptr<Report> report,
                              const CalibrationFiles* periodic_info) {
  holoscan::Graph graph{"contract"};
  const bool stereo = frames.stereo;
  const PerFrameInfo per_frame = frames.per_frame;
  auto source = graph.op<FrameSourceOp<Value>>("frames", std::move(frames));
  auto consumer = graph.op<ConsumerOp<Value>>(
      "consumer",
      ConsumerConfig<Value>{
          .stereo = stereo, .options = options, .expected = expected, .report = std::move(report)});
  graph.add_flow(source->left, consumer->left, test::lossless());
  if (stereo) graph.add_flow(source->right, consumer->right, test::lossless());
  if (per_frame != PerFrameInfo::kNone) {
    graph.add_flow(source->info, consumer->camera_info, test::lossless());
  } else if (periodic_info != nullptr) {
    const bool is_stereo_info = std::is_same_v<Value, StereoCameraInfo>;
    auto info = graph.op<CameraInfoSourceOp>(
        "camera_info",
        CameraInfoSourceConfig{.camera_info_path = periodic_info->dir / "left.yaml",
                               .right_camera_info_path =
                                   is_stereo_info ? periodic_info->dir / "right.yaml" : fs::path{},
                               .period = std::chrono::milliseconds{20}});
    if constexpr (std::is_same_v<Value, StereoCameraInfo>) {
      graph.add_flow(info->stereo_camera_info, consumer->camera_info,
                     latest_camera_info_connection());
    } else {
      graph.add_flow(info->camera_info, consumer->camera_info, latest_camera_info_connection());
    }
  }
  return test::run_graph(graph);
}

// Runs one case and returns what the consumer saw.
template <typename Value>
Report consume(const std::string& what, FrameSourceConfig<Value> frames,
               CameraInfoInputOptions options, const std::vector<Value>& expected,
               const CalibrationFiles* periodic_info = nullptr) {
  auto report = std::make_shared<Report>();
  const test::RunOutcome outcome =
      run_consumer(std::move(frames), options, expected, report, periodic_info);
  VERIFY_MSG(outcome.ok(), what + ": " + outcome.diagnostics);
  std::cout << "  " << what << ": frames=" << report->frames << " with_info=" << report->with_info
            << " matching_time=" << report->matching_time << " changes=" << report->changes
            << " first_with_info=" << report->first_with_info << '\n';
  VERIFY_MSG(report->frames == kFrames, what + ": every frame is processed");
  VERIFY_MSG(report->wrong_values == 0, what + ": the camera info is the one sent");
  return *report;
}

void test_latest_periodic() {
  const CalibrationFiles files;
  const StereoCameraInfo stereo =
      read_stereo_camera_info_yaml(files.dir / "left.yaml", files.dir / "right.yaml");
  // Every frame from the first camera info on has it; at most the first
  // frames come before it, and it changes once.
  const Report pair = consume<StereoCameraInfo>("stereo, camera info every 20 ms", {.stereo = true},
                                                {}, {stereo}, &files);
  VERIFY(pair.first_with_info <= 1 && pair.with_info == kFrames - pair.first_with_info);
  VERIFY(pair.changes == 1);
  const CameraInfo mono = read_camera_info_yaml(files.dir / "left.yaml");
  const Report single =
      consume<CameraInfo>("one camera, camera info every 20 ms", {}, {}, {mono}, &files);
  VERIFY(single.first_with_info <= 1 && single.with_info == kFrames - single.first_with_info);
  VERIFY(single.changes == 1);
}

void test_latest_frames_before_camera_info() {
  // The frames before the first camera info come without it, and then every
  // one has it: nothing waits for camera info, so nothing stalls (a Skip or
  // ReuseLast policy would hold back the first frame for good).
  const StereoCameraInfo pair = synthetic_stereo_camera_info(DistortionModel::kPlumbBob, 320, 240);
  auto report = std::make_shared<Report>();
  holoscan::Graph graph{"late"};
  auto frames = graph.op<FrameSourceOp<StereoCameraInfo>>(
      "frames", FrameSourceConfig<StereoCameraInfo>{.stereo = true});
  auto info = graph.op<LateInfoSourceOp>("camera_info", to_message(pair), 8U);
  auto consumer = graph.op<ConsumerOp<StereoCameraInfo>>(
      "consumer",
      ConsumerConfig<StereoCameraInfo>{.stereo = true, .expected = {pair}, .report = report});
  graph.add_flow(frames->left, consumer->left, test::lossless());
  graph.add_flow(frames->right, consumer->right, test::lossless());
  graph.add_flow(info->info, consumer->camera_info, latest_camera_info_connection());
  const test::RunOutcome outcome = test::run_graph(graph);
  VERIFY_MSG(outcome.ok(), outcome.diagnostics);
  std::cout << "  camera info after 8 frames: frames=" << report->frames
            << " with_info=" << report->with_info << " first_with_info=" << report->first_with_info
            << '\n';
  VERIFY(report->frames == kFrames && report->wrong_values == 0);
  VERIFY(report->first_with_info >= 4 && report->first_with_info < kFrames);
  VERIFY(report->with_info == kFrames - report->first_with_info);
}

void test_latest_per_frame() {
  const StereoCameraInfo a = synthetic_stereo_camera_info(DistortionModel::kEquidistant, 320, 240);
  StereoCameraInfo b = a;
  b.right.fx *= 1.01;
  // Sent before the frames, each frame's own camera info is the newest.
  const Report report = consume<StereoCameraInfo>("kLatest, camera info sent before each frame",
                                                  {.stereo = true,
                                                   .per_frame = PerFrameInfo::kBefore,
                                                   .info = a,
                                                   .changed = b,
                                                   .change_at = 25},
                                                  {}, {a, b});
  VERIFY(report.with_info == kFrames && report.matching_time == kFrames);
  VERIFY(report.changes == 2);
}

void test_every_frame() {
  const StereoCameraInfo a = synthetic_stereo_camera_info(DistortionModel::kPlumbBob, 320, 240);
  StereoCameraInfo b = a;
  b.left.cx += 0.5;
  // Sent after the frames, kEveryFrame still pairs each frame with its own.
  const Report report = consume<StereoCameraInfo>(
      "kEveryFrame, camera info sent after each frame",
      {.stereo = true, .per_frame = PerFrameInfo::kAfter, .info = a, .changed = b, .change_at = 10},
      {.timing = CameraInfoTiming::kEveryFrame}, {a, b});
  VERIFY(report.with_info == kFrames && report.matching_time == kFrames);
  VERIFY(report.changes == 2);
  const CameraInfo mono = a.left;
  const Report single = consume<CameraInfo>("kEveryFrame, one camera",
                                            {.per_frame = PerFrameInfo::kAfter, .info = mono},
                                            {.timing = CameraInfoTiming::kEveryFrame}, {mono});
  VERIFY(single.with_info == kFrames && single.matching_time == kFrames && single.changes == 1);
}

void test_invalid_camera_info() {
  // A camera info that does not validate (raw, without distortion) fails the
  // consumer, and the run.
  holoscan::Graph graph{"invalid"};
  schema::CameraInfoT message;
  message.header = std::make_shared<holoscan::schema::HeaderT>();
  message.width = 320;
  message.height = 240;
  message.fx = message.fy = 100.0;
  message.distortion_model = schema::DistortionModel_PlumbBob;
  message.coefficients.assign(5, 0.0);
  auto source = graph.op<test::ListSourceOp<schema::CameraInfoT>>(
      "info", std::vector<schema::CameraInfoT>{message}, kFramePeriod);
  auto frames = graph.op<FrameSourceOp<CameraInfo>>("frames", FrameSourceConfig<CameraInfo>{});
  auto consumer = graph.op<ConsumerOp<CameraInfo>>(
      "consumer", ConsumerConfig<CameraInfo>{.report = std::make_shared<Report>()});
  graph.add_flow(frames->left, consumer->left, test::lossless());
  graph.add_flow(source->output, consumer->camera_info, latest_camera_info_connection());
  const test::RunOutcome outcome = test::run_graph(graph);
  VERIFY(outcome.compiled && outcome.termination == holoscan::RunTermination::kFailed);
  VERIFY_MSG(outcome.diagnostics.find("invalid camera info") != std::string::npos,
             outcome.diagnostics);
}

void test_camera_info_is_mandatory() {
  holoscan::Graph graph{"unwired"};
  auto frames = graph.op<FrameSourceOp<CameraInfo>>("frames", FrameSourceConfig<CameraInfo>{});
  auto consumer = graph.op<ConsumerOp<CameraInfo>>(
      "consumer", ConsumerConfig<CameraInfo>{.report = std::make_shared<Report>()});
  graph.add_flow(frames->left, consumer->left, test::lossless());
  const test::RunOutcome outcome = test::run_graph(graph);
  std::cout << "  unwired camera_info: " << outcome.diagnostics.substr(0, 300) << '\n';
  VERIFY(!outcome.compiled);
}

void test_contract_arguments() {
  holoscan::Input<Frame> frame;
  holoscan::Input<schema::CameraInfoT> info;
  VERIFY(test::refuses(
      [&] { (void)camera_info_contract({.max_age = std::chrono::nanoseconds{0}}, info, frame); },
      "max_age must be > 0"));
}

}  // namespace
}  // namespace holoscan::messages

int main() {
  namespace m = holoscan::messages;
  return holoscan::messages::test::run({
      {"kLatest with periodic camera info", m::test_latest_periodic},
      {"kLatest with frames before the first camera info",
       m::test_latest_frames_before_camera_info},
      {"kLatest with camera info before each frame", m::test_latest_per_frame},
      {"kEveryFrame with camera info after each frame", m::test_every_frame},
      {"an invalid camera info fails the run", m::test_invalid_camera_info},
      {"camera info must be wired", m::test_camera_info_is_mandatory},
      {"contract arguments", m::test_contract_arguments},
  });
}
