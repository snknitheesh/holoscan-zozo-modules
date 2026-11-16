// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The recommended consumer contract, camera_info_contract() with
// LatestCameraInfo, in running graphs, with the SDK's automatic partitioning
// and on one worker: frames of a stereo pair or of one camera, with camera
// info from CameraInfoSourceOp at its own rate, or sent with every frame
// before or after the frames; frames that reach the consumer later than the
// camera info's period; a calibration that changes; an invalid camera info;
// and a consumer whose camera info is not wired. The frames are bare SDK
// Headers: the contract does not look into them.

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
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
  // How long before its activation each frame was captured: the time it
  // would have taken to reach this point of a chain.
  std::chrono::nanoseconds latency{};
  std::uint32_t frames{kFrames};
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
    if (sent_ >= config_.frames) return {};
    if (!pending_) {
      holoscan::TimePoint capture = context.activation_time();
      capture.timestamp_ns -= config_.latency.count();
      pending_ = holoscan::EmitOptions{.capture_time = capture, .frame_id = sent_ + 1U};
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
  // How long LatestCameraInfo goes without camera info before it warns.
  std::chrono::nanoseconds warn_after{std::chrono::seconds{5}};
  // The run ends at this frame.
  std::uint32_t frames{kFrames};
};

// Counts every frame and checks its camera info; ends the run at the last frame.
template <typename Value>
class ConsumerOp final : public holoscan::Operator<> {
 public:
  using Message = decltype(to_message(std::declval<const Value&>()));

  explicit ConsumerOp(ConsumerConfig<Value> config)
      : config_(std::move(config)), info_(config_.warn_after) {}

  void setup(holoscan::OperatorSpec& spec) override {
    // Four: room for a few frames while the consumer is busy, so that the
    // source on another worker is rarely refused (it sends a refused one
    // again). The camera info takes it too when it comes with every frame on
    // a lossless edge; latest_camera_info_connection() sets its own depth.
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
    if (++report.frames == config_.frames) context.request_completion();
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
  CalibrationFiles() { write_stereo_camera_info_yaml(pair, dir / "left.yaml", dir / "right.yaml"); }

  test::TemporaryFolder folder{"messages_contract"};
  StereoCameraInfo pair = synthetic_stereo_camera_info(DistortionModel::kPlumbBob, 640, 480);
  const fs::path& dir = folder.path();
};

// Whatever the process writes to stdout and stderr (the SDK's log among it)
// from its construction to finish(), which returns it.
class CapturedOutput {
 public:
  explicit CapturedOutput(fs::path file) : file_(std::move(file)) {
    std::cout.flush();
    std::cerr.flush();
    std::fflush(nullptr);
    const int fd = ::open(file_.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) return;
    saved_out_ = ::dup(STDOUT_FILENO);
    saved_err_ = ::dup(STDERR_FILENO);
    ::dup2(fd, STDOUT_FILENO);
    ::dup2(fd, STDERR_FILENO);
    ::close(fd);
  }
  ~CapturedOutput() { (void)finish(); }
  CapturedOutput(const CapturedOutput&) = delete;
  CapturedOutput& operator=(const CapturedOutput&) = delete;

  std::string finish() {
    if (saved_out_ < 0) return {};
    std::cout.flush();
    std::cerr.flush();
    std::fflush(nullptr);
    ::dup2(saved_out_, STDOUT_FILENO);
    ::dup2(saved_err_, STDERR_FILENO);
    ::close(saved_out_);
    ::close(saved_err_);
    saved_out_ = saved_err_ = -1;
    std::ifstream in(file_);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
  }

 private:
  fs::path file_;
  int saved_out_{-1};
  int saved_err_{-1};
};

template <typename Value>
test::RunOutcome run_consumer(FrameSourceConfig<Value> frames, CameraInfoInputOptions options,
                              const std::vector<Value>& expected, std::shared_ptr<Report> report,
                              const CalibrationFiles* periodic_info, test::Layout layout) {
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
  return test::run_graph(graph, {}, layout);
}

// Runs one case and returns what the consumer saw.
template <typename Value>
Report consume(std::string what, test::Layout layout, FrameSourceConfig<Value> frames,
               CameraInfoInputOptions options, const std::vector<Value>& expected,
               const CalibrationFiles* periodic_info = nullptr) {
  what += std::string(", ") + test::layout_name(layout);
  auto report = std::make_shared<Report>();
  const test::RunOutcome outcome =
      run_consumer(std::move(frames), options, expected, report, periodic_info, layout);
  VERIFY_MSG(outcome.ok(), what + ": " + outcome.diagnostics);
  std::cout << "  " << what << ": frames=" << report->frames << " with_info=" << report->with_info
            << " matching_time=" << report->matching_time << " changes=" << report->changes
            << " first_with_info=" << report->first_with_info << '\n';
  VERIFY_MSG(report->frames == kFrames, what + ": every frame is processed");
  VERIFY_MSG(report->wrong_values == 0, what + ": the camera info is the one sent");
  return *report;
}

void test_latest_periodic(test::Layout layout) {
  const CalibrationFiles files;
  const StereoCameraInfo stereo =
      read_stereo_camera_info_yaml(files.dir / "left.yaml", files.dir / "right.yaml");
  // Every frame from the first camera info on has it; at most the first
  // frames come before it, and it changes once.
  const Report pair = consume<StereoCameraInfo>("stereo, camera info every 20 ms", layout,
                                                {.stereo = true}, {}, {stereo}, &files);
  VERIFY(pair.first_with_info <= 1 && pair.with_info == kFrames - pair.first_with_info);
  VERIFY(pair.changes == 1);
  const CameraInfo mono = read_camera_info_yaml(files.dir / "left.yaml");
  const Report single =
      consume<CameraInfo>("one camera, camera info every 20 ms", layout, {}, {}, {mono}, &files);
  VERIFY(single.first_with_info <= 1 && single.with_info == kFrames - single.first_with_info);
  VERIFY(single.changes == 1);
}

// Frames that reach the consumer 100 ms after their capture, and camera info
// every 10 ms from CameraInfoSourceOp, for 100 frames (500 ms). Keeping only
// the newest camera info, each one kept is newer than every frame (unless
// the camera info stalls for 90 ms), so none is taken, and LatestCameraInfo
// warns (after 300 ms); keeping sixteen (160 ms of them), every frame from
// the first camera info's capture time on (100 ms in) has one, and nothing
// warns.
void test_latest_late_frames(test::Layout layout) {
  constexpr std::uint32_t kLateFrames = 100;
  const CalibrationFiles files;
  const CameraInfo mono = read_camera_info_yaml(files.dir / "left.yaml");
  for (const std::uint32_t depth : {1U, 16U}) {
    const std::string what =
        "frames 100 ms late, depth " + std::to_string(depth) + ", " + test::layout_name(layout);
    auto report = std::make_shared<Report>();
    holoscan::Graph graph{"late_frames"};
    auto frames = graph.op<FrameSourceOp<CameraInfo>>(
        "frames", FrameSourceConfig<CameraInfo>{.latency = std::chrono::milliseconds{100},
                                                .frames = kLateFrames});
    auto info = graph.op<CameraInfoSourceOp>(
        "camera_info", CameraInfoSourceConfig{.camera_info_path = files.dir / "left.yaml",
                                              .period = std::chrono::milliseconds{10}});
    auto consumer = graph.op<ConsumerOp<CameraInfo>>(
        "consumer", ConsumerConfig<CameraInfo>{.expected = {mono},
                                               .report = report,
                                               .warn_after = std::chrono::milliseconds{300},
                                               .frames = kLateFrames});
    graph.add_flow(frames->left, consumer->left, test::lossless());
    graph.add_flow(
        info->camera_info, consumer->camera_info,
        depth == 1U ? latest_camera_info_connection() : latest_camera_info_connection(depth));
    std::string output;
    test::RunOutcome outcome;
    {
      CapturedOutput captured(files.dir / "output.txt");
      outcome = test::run_graph(graph, {}, layout);
      output = captured.finish();
    }
    std::cout << output;
    VERIFY_MSG(outcome.ok(), what + ": " + outcome.diagnostics);
    std::cout << "  " << what << ": frames=" << report->frames << " with_info=" << report->with_info
              << " first_with_info=" << report->first_with_info << '\n';
    VERIFY_MSG(report->frames == kLateFrames && report->wrong_values == 0, what);
    const bool warned = output.find("no camera info for any frame") != std::string::npos;
    if (depth == 1U) {
      VERIFY_MSG(report->with_info == 0 && warned, what + ": no camera info, and a warning");
    } else {
      VERIFY_MSG(report->first_with_info <= 40 &&
                     report->with_info == kLateFrames - report->first_with_info && !warned,
                 what + ": camera info from the first frame captured after it");
    }
  }
}

void test_latest_frames_before_camera_info(test::Layout layout) {
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
  const test::RunOutcome outcome = test::run_graph(graph, {}, layout);
  VERIFY_MSG(outcome.ok(), outcome.diagnostics);
  std::cout << "  camera info after 8 frames, " << test::layout_name(layout)
            << ": frames=" << report->frames << " with_info=" << report->with_info
            << " first_with_info=" << report->first_with_info << '\n';
  VERIFY(report->frames == kFrames && report->wrong_values == 0);
  VERIFY(report->first_with_info >= 4 && report->first_with_info < kFrames);
  VERIFY(report->with_info == kFrames - report->first_with_info);
}

void test_latest_per_frame(test::Layout layout) {
  const StereoCameraInfo a = synthetic_stereo_camera_info(DistortionModel::kEquidistant, 320, 240);
  StereoCameraInfo b = a;
  b.right.fx *= 1.01;
  // Sent before the frames, each frame's own camera info is the newest.
  const Report report =
      consume<StereoCameraInfo>("kLatest, camera info sent before each frame", layout,
                                {.stereo = true,
                                 .per_frame = PerFrameInfo::kBefore,
                                 .info = a,
                                 .changed = b,
                                 .change_at = 25},
                                {}, {a, b});
  VERIFY(report.with_info == kFrames && report.matching_time == kFrames);
  VERIFY(report.changes == 2);
}

void test_every_frame(test::Layout layout) {
  const StereoCameraInfo a = synthetic_stereo_camera_info(DistortionModel::kPlumbBob, 320, 240);
  StereoCameraInfo b = a;
  b.left.cx += 0.5;
  // Sent after the frames, kEveryFrame still pairs each frame with its own.
  const Report report = consume<StereoCameraInfo>(
      "kEveryFrame, camera info sent after each frame", layout,
      {.stereo = true, .per_frame = PerFrameInfo::kAfter, .info = a, .changed = b, .change_at = 10},
      {.timing = CameraInfoTiming::kEveryFrame}, {a, b});
  VERIFY(report.with_info == kFrames && report.matching_time == kFrames);
  VERIFY(report.changes == 2);
  const CameraInfo mono = a.left;
  const Report single = consume<CameraInfo>("kEveryFrame, one camera", layout,
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
  const auto depth = [](const holoscan::ConnectionOptions& options) {
    return options.queue_policy == holoscan::QueuePolicy::kDropOldest
               ? options.queue_depth.value_or(0U)
               : 0U;
  };
  VERIFY(depth(latest_camera_info_connection()) == 1U);
  VERIFY(depth(latest_camera_info_connection(6U)) == 6U);
}

void test_selection_errors() {
  // What update() returns for a camera info it cannot match to the frames.
  using holoscan::SelectionReason;
  const auto says = [](SelectionReason reason, std::string_view text) {
    const auto error = detail::selection_error(reason);
    return error && error->code == holoscan::ErrorCode::kInvalidArgument &&
           error->message.find(text) != std::string::npos;
  };
  VERIFY(says(SelectionReason::kCaptureTimeAbsent,
              "a frame or its camera info has no capture time, so they cannot be matched"));
  VERIFY(says(SelectionReason::kClockIncomparable, "in another clock domain than the frames'"));
  VERIFY(says(SelectionReason::kConversionUnavailable, "in another clock domain"));
  // The ordinary misses are no error: the operator keeps the camera info it has.
  for (const SelectionReason reason :
       {SelectionReason::kNone, SelectionReason::kNeverSeen, SelectionReason::kTooStale}) {
    VERIFY(!detail::selection_error(reason).has_value());
  }
}

}  // namespace
}  // namespace holoscan::messages

int main() {
  namespace m = holoscan::messages;
  std::vector<holoscan::messages::test::TestCase> cases;
  for (const auto layout : {m::test::Layout::kAutomatic, m::test::Layout::kOneWorker}) {
    const std::string on = std::string(", ") + m::test::layout_name(layout);
    cases.push_back(
        {"kLatest with periodic camera info" + on, [layout] { m::test_latest_periodic(layout); }});
    cases.push_back({"kLatest with frames later than the camera info's period" + on,
                     [layout] { m::test_latest_late_frames(layout); }});
    cases.push_back({"kLatest with frames before the first camera info" + on,
                     [layout] { m::test_latest_frames_before_camera_info(layout); }});
    cases.push_back({"kLatest with camera info before each frame" + on,
                     [layout] { m::test_latest_per_frame(layout); }});
    cases.push_back({"kEveryFrame with camera info after each frame" + on,
                     [layout] { m::test_every_frame(layout); }});
  }
  cases.push_back({"an invalid camera info fails the run", m::test_invalid_camera_info});
  cases.push_back({"camera info must be wired", m::test_camera_info_is_mandatory});
  cases.push_back({"contract arguments", m::test_contract_arguments});
  cases.push_back({"selection errors name the cause", m::test_selection_errors});
  return holoscan::messages::test::run(cases);
}
