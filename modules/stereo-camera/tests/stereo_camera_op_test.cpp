// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// StereoCameraOp in small graphs: a synthetic recording played back with and
// without rgb and depth, in host and device memory, every image checked pixel
// for pixel against the synthetic camera and every port's capture time, frame
// id, header and calibration message checked; a count, a loop and a slow
// consumer that must still get every frame; a live source of the test's own,
// whose frames a slow consumer may skip, and whose age moves the capture time;
// and the configurations and sources the operator refuses.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/stereo_camera/calibration_message.hpp>
#include <holoscan/stereo_camera/stereo_camera_op.hpp>
#include <holoscan/stereo_camera/synthetic.hpp>

#include "graph_support.hpp"
#include "test_support.hpp"

namespace holoscan::stereo_camera {
namespace {

using test::FrameSinkConfig;
using test::FrameSinkOp;
using test::Received;
using test::ReceivedFrame;
using test::ReceivedImage;

constexpr int kWidth = 160;
constexpr int kHeight = 120;
constexpr std::uint32_t kFrames = 5;

std::filesystem::path g_recording;  // NOLINT: set once in main().
std::filesystem::path g_plain;      // NOLINT: the same without rgb and depth.

const holoscan::ConnectionOptions kLossless{.queue_policy = holoscan::QueuePolicy::kBlocking};

holoscan::ConnectionOptions newest() {
  return {.queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 1U};
}

StereoCameraConfig playback(bool extras, std::uint64_t count = 0, bool loop = false) {
  StereoCameraConfig config;
  config.source = SourceConfig{
      .backend = Backend::kPlayback,
      .rgb = extras,
      .depth = extras,
      .playback = PlaybackOptions{.directory = extras ? g_recording : g_plain, .loop = loop}};
  config.count = count;
  config.period = std::chrono::milliseconds{5};
  return config;
}

bool same_calibration(const StereoCalibration& a, const StereoCalibration& b) {
  return format_calibration(a) == format_calibration(b);
}

// Checks a frame against synthetic frame `index`.
void check_image(const ReceivedImage& image, const Image& truth,
                 holoscan::schema::ImageEncoding encoding, const std::string& frame_id,
                 DLDeviceType device) {
  VERIFY(image.encoding == encoding);
  VERIFY(image.width == truth.width && image.height == truth.height);
  std::vector<std::int64_t> shape{truth.height, truth.width};
  if (channels(truth.format) == 3) shape.push_back(3);
  VERIFY(image.shape == shape);
  VERIFY_MSG(image.device == device, "image in device type " + std::to_string(image.device));
  VERIFY_MSG(image.frame_id == frame_id, "header frame id " + image.frame_id);
  VERIFY(same_pixels(image.view(truth.format), truth.view()));
}

void check_frames(const Received& received, bool extras, bool device, std::uint32_t frames,
                  std::uint32_t recorded) {
  VERIFY_MSG(received.frames.size() == frames,
             std::to_string(received.frames.size()) + " frames received");
  const StereoCalibration truth_calibration =
      synthetic_calibration(kWidth, kHeight, extras, extras);
  Image left;
  Image right;
  Image rgb;
  Image depth;
  std::optional<std::int64_t> last_capture;
  for (std::uint32_t n = 0; n < received.frames.size(); ++n) {
    const ReceivedFrame& frame = received.frames[n];
    const std::uint32_t index = n % recorded;
    render_synthetic_frame(kWidth, kHeight, index, left, right, &rgb, &depth);
    VERIFY(frame.consistent);
    VERIFY_MSG(frame.frame_id == n + 1U, "frame id " + std::to_string(frame.frame_id));
    VERIFY(!last_capture || frame.capture_ns > *last_capture);
    last_capture = frame.capture_ns;
    const DLDeviceType where = device ? kDLCUDA : kDLCPU;
    check_image(frame.left, left, holoscan::schema::ImageEncoding_MONO8, "stereo_camera_left",
                where);
    check_image(frame.right, right, holoscan::schema::ImageEncoding_MONO8, "stereo_camera_right",
                where);
    VERIFY(frame.left.header_capture_ns == frame.capture_ns);
    VERIFY(frame.left.device_sequence == index + 1U);
    VERIFY(frame.rgb.has_value() == extras && frame.depth.has_value() == extras);
    if (extras && frame.rgb && frame.depth) {
      check_image(*frame.rgb, rgb, holoscan::schema::ImageEncoding_RGB8, "stereo_camera_rgb",
                  where);
      check_image(*frame.depth, depth, holoscan::schema::ImageEncoding_MONO16,
                  "stereo_camera_depth", where);
    }
    const schema::StereoCalibrationT& message = frame.calibration;
    VERIFY(message.frame == index && message.timestamp_ns == synthetic_timestamp_ns(index));
    VERIFY(message.device_sequence == index + 1U && message.backend == "playback");
    VERIFY(same_calibration(from_message(message), truth_calibration));
    VERIFY(message.camera_name == "synthetic");
  }
}

void check_playback(bool extras, bool device) {
  auto received = std::make_shared<Received>();
  holoscan::Graph graph{"playback"};
  StereoCameraConfig config = playback(extras);
  if (device) config.memory = OutputMemory::kDevice;
  config.stats = std::make_shared<StereoCameraStats>();
  auto camera = graph.op<StereoCameraOp>("camera", config);
  VERIFY(camera->layout().rgb.has_value() == extras && !camera->layout().live);
  auto sink = graph.op<FrameSinkOp>(
      "sink",
      FrameSinkConfig{.rgb = extras, .depth = extras, .expected = kFrames, .received = received});
  test::connect(graph, camera, sink, extras, extras, kLossless);
  const auto outcome = test::run_graph(
      graph, device ? test::device_outputs(extras, extras) : holoscan::CompileOptions{});
  VERIFY_MSG(outcome.completed, outcome.diagnostics);
  check_frames(*received, extras, device, kFrames, kFrames);
  VERIFY(config.stats->sent == kFrames && config.stats->dropped == 0);
}

void check_without_calibration() {
  // Without its consumer, the calibration port is turned off.
  auto received = std::make_shared<Received>();
  holoscan::Graph graph{"no_calibration"};
  StereoCameraConfig config = playback(false);
  config.calibration = false;
  auto camera = graph.op<StereoCameraOp>("camera", config);
  auto sink = graph.op<FrameSinkOp>(
      "sink", FrameSinkConfig{.calibration = false, .expected = kFrames, .received = received});
  test::connect(graph, camera, sink, false, false, kLossless, false);
  const auto outcome = test::run_graph(graph);
  VERIFY_MSG(outcome.completed, outcome.diagnostics);
  VERIFY(received->frames.size() == kFrames);
  Image left;
  Image right;
  for (std::uint32_t i = 0; i < received->frames.size(); ++i) {
    render_synthetic_frame(kWidth, kHeight, i, left, right, nullptr, nullptr);
    VERIFY(received->frames[i].consistent && received->frames[i].frame_id == i + 1U);
    VERIFY(same_pixels(received->frames[i].right.view(PixelFormat::kMono8), right.view()));
  }
}

void check_count_and_loop() {
  auto received = std::make_shared<Received>();
  holoscan::Graph graph{"loop"};
  auto camera = graph.op<StereoCameraOp>("camera", playback(true, 12, true));
  auto sink = graph.op<FrameSinkOp>(
      "sink", FrameSinkConfig{.rgb = true, .depth = true, .expected = 12, .received = received});
  test::connect(graph, camera, sink, true, true, kLossless);
  const auto outcome = test::run_graph(graph);
  VERIFY_MSG(outcome.completed, outcome.diagnostics);
  check_frames(*received, true, false, 12, kFrames);
}

void check_slow_consumer_gets_every_frame() {
  // The sink takes 25 ms a frame, the source offers one every 5 ms: a recording
  // is lossless, so the source waits for room rather than drop.
  auto received = std::make_shared<Received>();
  holoscan::Graph graph{"slow"};
  auto camera = graph.op<StereoCameraOp>("camera", playback(true, 0, true));
  auto sink = graph.op<FrameSinkOp>("sink", FrameSinkConfig{.rgb = true,
                                                            .depth = true,
                                                            .expected = 3 * kFrames,
                                                            .delay = std::chrono::milliseconds{25},
                                                            .received = received});
  test::connect(graph, camera, sink, true, true, kLossless);
  const auto outcome = test::run_graph(graph);
  VERIFY_MSG(outcome.completed, outcome.diagnostics);
  check_frames(*received, true, false, 3 * kFrames, kFrames);
}

// A live camera of the test's own: a new synthetic frame on every grab.
class FakeLiveSource final : public StereoSource {
 public:
  explicit FakeLiveSource(std::shared_ptr<std::atomic<int>> opened) : opened_(std::move(opened)) {
    layout_.left = StreamLayout{kWidth, kHeight, PixelFormat::kMono8};
    layout_.right = layout_.left;
    layout_.live = true;
  }
  [[nodiscard]] Backend backend() const noexcept override { return Backend::kV4l2SideBySide; }
  [[nodiscard]] const SourceLayout& layout() const noexcept override { return layout_; }
  void open() override {
    ++*opened_;
    open_ = true;
  }
  [[nodiscard]] const StereoCalibration& calibration() const override { return calibration_; }
  [[nodiscard]] std::optional<StereoFrameView> grab() override {
    if (!open_) throw std::logic_error("grab() before open()");
    render_synthetic_frame(kWidth, kHeight, next_, left_, right_, nullptr, nullptr);
    StereoFrameView frame{.left = left_.view(),
                          .right = right_.view(),
                          .index = next_,
                          .timestamp_ns = synthetic_timestamp_ns(next_),
                          .device_sequence = next_ + 1U};
    ++next_;
    return frame;
  }
  void close() noexcept override { open_ = false; }

 private:
  std::shared_ptr<std::atomic<int>> opened_;
  SourceLayout layout_;
  StereoCalibration calibration_ = synthetic_calibration(kWidth, kHeight, false, false);
  bool open_{};
  std::uint64_t next_{};
  Image left_;
  Image right_;
};

void check_live_source() {
  auto opened = std::make_shared<std::atomic<int>>(0);
  auto received = std::make_shared<Received>();
  holoscan::Graph graph{"live"};
  StereoCameraConfig config;
  config.make_source = [opened] { return std::make_unique<FakeLiveSource>(opened); };
  config.stats = std::make_shared<StereoCameraStats>();
  auto camera = graph.op<StereoCameraOp>("camera", config);
  VERIFY(camera->layout().live);
  // A slow sink on drop-oldest edges: it gets the newest frame each time, and
  // never holds the camera back.
  auto sink = graph.op<FrameSinkOp>(
      "sink", FrameSinkConfig{
                  .expected = 10, .delay = std::chrono::milliseconds{20}, .received = received});
  test::connect(graph, camera, sink, false, false, newest());
  const auto outcome = test::run_graph(graph);
  VERIFY_MSG(outcome.completed, outcome.diagnostics);
  VERIFY(received->frames.size() == 10);
  VERIFY_MSG(opened->load() == 1, "the source opened " + std::to_string(opened->load()) + " times");
  std::uint64_t last_id = 0;
  int wrong = 0;
  Image left;
  Image right;
  for (const ReceivedFrame& frame : received->frames) {
    VERIFY(frame.consistent && frame.frame_id > last_id);
    last_id = frame.frame_id;
    // The frame the sink got is the one the source numbered, whatever it skipped.
    render_synthetic_frame(kWidth, kHeight, frame.calibration.frame, left, right, nullptr, nullptr);
    wrong += !same_pixels(frame.left.view(PixelFormat::kMono8), left.view());
    wrong += frame.calibration.frame + 1U != frame.frame_id;
  }
  VERIFY(wrong == 0);
  VERIFY(config.stats->sent >= last_id && config.stats->dropped == 0);
}

void check_rejections() {
  using test::throws;
  StereoCameraConfig config = playback(true);
  config.period = std::chrono::microseconds{10};
  VERIFY(throws<std::invalid_argument>([&] { validate_config(config); }));
  config = playback(true);
  config.frame_id.clear();
  VERIFY(throws<std::invalid_argument>([&] { validate_config(config); }));
  config = playback(true);
  config.source.playback.directory = g_plain;  // No rgb or depth there.
  VERIFY(throws<std::invalid_argument>([&] { StereoCameraOp op(config); }));
  StereoCameraConfig custom;
  custom.make_source = [] { return std::unique_ptr<StereoSource>(); };
  VERIFY(throws<std::invalid_argument>([&] { StereoCameraOp op(custom); }));

  // A source that does not open fails the run with its message.
  class Broken final : public StereoSource {
   public:
    Broken() {
      layout_.left = StreamLayout{kWidth, kHeight, PixelFormat::kMono8};
      layout_.right = layout_.left;
      layout_.live = true;
    }
    [[nodiscard]] Backend backend() const noexcept override { return Backend::kRealSense; }
    [[nodiscard]] const SourceLayout& layout() const noexcept override { return layout_; }
    void open() override { throw std::runtime_error("the camera is unplugged"); }
    [[nodiscard]] const StereoCalibration& calibration() const override {
      throw std::logic_error("not open");
    }
    [[nodiscard]] std::optional<StereoFrameView> grab() override { return std::nullopt; }
    void close() noexcept override {}

   private:
    SourceLayout layout_;
  };
  holoscan::Graph graph{"broken"};
  StereoCameraConfig broken;
  broken.make_source = [] { return std::make_unique<Broken>(); };
  auto camera = graph.op<StereoCameraOp>("camera", broken);
  auto received = std::make_shared<Received>();
  auto sink = graph.op<FrameSinkOp>("sink", FrameSinkConfig{.expected = 1, .received = received});
  test::connect(graph, camera, sink, false, false, newest());
  const auto outcome = test::run_graph(graph, {}, std::chrono::seconds{30});
  VERIFY(outcome.compiled && !outcome.completed);
  VERIFY(outcome.termination == holoscan::RunTermination::kFailed);
  VERIFY_MSG(outcome.diagnostics.find("the camera is unplugged") != std::string::npos,
             outcome.diagnostics);
}

void check_message_round_trip() {
  StereoCalibration calibration = synthetic_calibration(kWidth, kHeight, true, true);
  calibration.rgb->model.distortion_model = DistortionModel::kBrownConrady;
  calibration.rgb->model.distortion = {0.1, -0.01, 0.001, 0.0, 0.002};
  calibration.rgb->from_left.translation = {-0.03, 0.001, 0.0};
  calibration.depth->scale = 0.000452;
  calibration.camera.serial = "123";
  calibration.camera.firmware = "1.2.3";
  const schema::StereoCalibrationT message = to_message(calibration);
  VERIFY(same_calibration(from_message(message), calibration));
  schema::StereoCalibrationT broken = message;
  broken.right.reset();
  VERIFY(test::throws<std::invalid_argument>([&] { (void)from_message(broken); }));
  broken = message;
  broken.depth_scale = 0.0;
  VERIFY(test::throws<std::invalid_argument>([&] { (void)from_message(broken); }));
}

}  // namespace
}  // namespace holoscan::stereo_camera

int main(int argc, char** argv) {
  namespace sc = holoscan::stereo_camera;
  const std::filesystem::path work = argc > 1 ? argv[1] : std::filesystem::current_path();
  sc::g_recording = work / "op_recording";
  sc::g_plain = work / "op_recording_plain";
  std::filesystem::remove_all(sc::g_recording);
  std::filesystem::remove_all(sc::g_plain);
  sc::write_synthetic_recording(sc::g_recording, sc::kWidth, sc::kHeight, sc::kFrames, true, true);
  sc::write_synthetic_recording(sc::g_plain, sc::kWidth, sc::kHeight, sc::kFrames, false, false);
  return sc::test::run({
      {"playback, eyes only, host memory", [] { sc::check_playback(false, false); }},
      {"playback with rgb and depth, host memory", [] { sc::check_playback(true, false); }},
      {"playback with rgb and depth, device memory", [] { sc::check_playback(true, true); }},
      {"without the calibration port", sc::check_without_calibration},
      {"count and loop", sc::check_count_and_loop},
      {"a slow consumer gets every recorded frame", sc::check_slow_consumer_gets_every_frame},
      {"a live source with a slow consumer", sc::check_live_source},
      {"rejected configurations and a source that fails", sc::check_rejections},
      {"calibration message round trip", sc::check_message_round_trip},
  });
}
