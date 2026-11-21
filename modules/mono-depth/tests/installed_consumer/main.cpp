// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-mono-depth package: a source renders
// the synthetic scene, MonoDepthOp runs the stub model on it through TensorRT with metric
// depth from a scale and shift, CameraInfoSourceOp (holoscan-messages) sends the camera
// from a camera_info file written for the run, and a sink checks each frame's inverse and
// metric depth against the host reference of the whole pipeline, and its depth camera
// info against the camera. The graph is lossless, so it runs on one worker, which the
// consumer checks.

#include <stdlib.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/mono_depth/mono_depth_op.hpp>
#include <holoscan/mono_depth/reference.hpp>
#include <holoscan/mono_depth/stub_model.hpp>
#include <holoscan/mono_depth/synthetic_scene.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace hm = holoscan::messages;
namespace md = holoscan::mono_depth;

using std::chrono_literals::operator""ns;

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr int kInputSize = 518;
constexpr std::uint32_t kFrames = 5;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

std::vector<std::uint8_t> scene_pixels(std::uint32_t frame) {
  std::vector<std::uint8_t> pixels(std::size_t{kWidth} * kHeight * 3U);
  md::render_depth_scene(md::make_depth_scene(kWidth, kHeight, frame), md::PixelFormat::kRgb8,
                         pixels.data(), std::size_t{kWidth} * 3U);
  return pixels;
}

// kFrames RGB8 renderings of the synthetic scene in host memory, numbered from 1.
class SceneSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(image, "image")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kHost,
                    .dtype = kUInt8,
                    .rank = 3U,
                    .strides = holoscan::TensorStridePolicy::kContiguousOnly,
                },
            .bounds = holoscan::tensor_bounds(std::uint64_t{kWidth} * kHeight * 3U),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        })
        .payload_options(holoscan::PayloadOptions{
            .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{20}});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (emitted_ >= kFrames) {
      return {};
    }
    const std::array<std::int64_t, 3> shape{kHeight, kWidth, 3};
    auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
    if (!loan) {
      // Every slot still held by frames the consumers have not finished with:
      // backpressure too.
      return busy(loan.error()) ? holoscan::expected<void, holoscan::Error>{}
                                : holoscan::make_unexpected(std::move(loan).error());
    }
    const std::vector<std::uint8_t> pixels = scene_pixels(emitted_);
    auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                   -> holoscan::expected<void, holoscan::Error> {
      std::copy(pixels.begin(), pixels.end(), writer.data_as<std::uint8_t>());
      return {};
    });
    if (!written) {
      return holoscan::make_unexpected(std::move(written).error());
    }
    holoscan::schema::ImageT descriptor;
    descriptor.width = kWidth;
    descriptor.height = kHeight;
    descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
    auto emitted =
        image.emit_tensor(std::move(*loan), descriptor,
                          holoscan::EmitOptions{.capture_time = context.activation_time(),
                                                .frame_id = emitted_ + 1U});
    if (emitted) {
      ++emitted_;
      return {};
    }
    // A busy consumer refuses the frame; send it again on the next tick.
    return busy(emitted.error()) ? holoscan::expected<void, holoscan::Error>{} : emitted;
  }

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  // Whether `error` says the consumers are still busy: an emit refused with
  // kBackpressured, or an allocation with kResourceExhausted while every slot
  // of the output's pool is held. The frame goes again on the next tick.
  static bool busy(const holoscan::Error& error) {
    return error.code == holoscan::ErrorCode::kBackpressured ||
           error.code == holoscan::ErrorCode::kResourceExhausted;
  }

  std::uint32_t emitted_{};
};

// The camera of the frames: a pinhole that sees them undistorted.
hm::CameraInfo test_camera() {
  hm::CameraInfo info;
  info.header.frame_id = "consumer_camera";
  info.width = kWidth;
  info.height = kHeight;
  info.state = hm::CameraState::kUndistorted;
  info.fx = 480.0;
  info.fy = 480.0;
  info.cx = 319.5;
  info.cy = 239.5;
  return info;
}

struct Tally {
  std::uint32_t frames{};
  std::uint32_t frames_matching{};
  std::uint32_t cameras_matching{};
  std::uint64_t last_frame_id{};
  float max_error{};
};

// Copies each frame's inverse and metric depth to the host and compares them with the
// reference, and its camera with the file's; completes the run at frame kFrames.
class TallyOp final : public holoscan::Operator<> {
 public:
  TallyOp(std::shared_ptr<Tally> tally, md::MonoDepthConfig config)
      : tally_(std::move(tally)), config_(std::move(config)) {}

  // Two slots: one depth in compute() while the next one arrives, as MonoDepthOp's own
  // input declares.
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(depth, "depth").queue_depth(2U);
    spec.input(metric_depth, "metric_depth").queue_depth(2U);
    spec.input(camera_info, "camera_info").queue_depth(2U);
  }

  // One frame's outputs, by capture time.
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAligned{.ports = holoscan::ports(depth, metric_depth, camera_info),
                                       .window = 0ns});
    result.input(depth, holoscan::Exact{});
    result.input(metric_depth, holoscan::Exact{});
    result.input(camera_info, holoscan::Exact{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = depth.receive();
    if (!sample) return holoscan::make_unexpected(std::move(sample).error());
    if (!sample->data.data) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "depth without a tensor"});
    }
    auto metric = metric_depth.receive();
    if (!metric) return holoscan::make_unexpected(std::move(metric).error());
    auto camera = camera_info.receive();
    if (!camera) return holoscan::make_unexpected(std::move(camera).error());
    if (!metric->data.data) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "metric depth without a tensor"});
    }
    std::vector<float> values(std::size_t{kWidth} * kHeight);
    std::vector<float> meters(values.size());
    if (auto copied = sample->data.data->copy_to_host(values.data(), values.size() * sizeof(float),
                                                      context.cuda_stream());
        !copied) {
      return copied;
    }
    if (auto copied = metric->data.data->copy_to_host(meters.data(), meters.size() * sizeof(float),
                                                      context.cuda_stream());
        !copied) {
      return copied;
    }
    hm::CameraInfo want = test_camera();
    const hm::CameraInfo got = hm::from_message(camera->data);
    want.header = got.header;
    if (got == want && got.header.frame_id == "consumer_camera" &&
        got.header.capture_timestamp_ns == sample->metadata.capture_timestamp_ns) {
      ++tally_->cameras_matching;
    }
    // Sources number their frames from 1; the depth keeps its image's id.
    const std::uint64_t frame_id = sample->metadata.frame_id;
    const std::vector<std::uint8_t> pixels =
        scene_pixels(static_cast<std::uint32_t>(frame_id - 1U));
    const md::ImageFit fit = md::make_fit(kWidth, kHeight, kInputSize, kInputSize, kInputSize,
                                          kInputSize, config_.resize_mode);
    const std::vector<float> input =
        md::reference_preprocess(pixels.data(), std::size_t{kWidth} * 3U, md::PixelFormat::kRgb8,
                                 fit, config_.normalization);
    const md::ReferenceDepth expected = md::reference_postprocess(
        md::reference_stub(input, kInputSize, kInputSize), fit,
        md::ModelOutput::kRelativeInverseDepth, &*config_.metric_from_relative);
    float error = 0.0F;
    for (std::size_t i = 0; i < values.size(); ++i) {
      error = std::max(error, std::fabs(values[i] - expected.inverse[i]));
      error = std::max(error, std::fabs(meters[i] - expected.metric[i]) /
                                  std::max(1.0F, std::fabs(expected.metric[i])));
    }
    ++tally_->frames;
    tally_->last_frame_id = frame_id;
    tally_->max_error = std::max(tally_->max_error, error);
    if (frame_id != 0 && error < 1e-4F) {
      ++tally_->frames_matching;
    }
    // The last frame; a first one before the first camera info gives none.
    if (frame_id >= kFrames) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> depth;
  holoscan::Input<holoscan::schema::ImageT> metric_depth;
  holoscan::Input<hm::schema::CameraInfoT> camera_info;

 private:
  std::shared_ptr<Tally> tally_;
  md::MonoDepthConfig config_;
};

}  // namespace

int main() try {
  // Take over SIGINT and SIGTERM while this is the only thread: TensorRT starts threads of
  // its own, after which the SDK accepts no other way to manage the signals. Without them
  // (another thread already running), the run goes ahead unmanaged.
  auto signals = holoscan::ManagedShutdownSignals::create();
  if (!signals) {
    std::cerr << "mono_depth_consumer: warning: running without managed SIGINT/SIGTERM: "
              << holoscan::runtime_error_message(signals.error()) << '\n';
  }
  const md::MonoDepthConfig config{
      .model = md::write_stub_model(std::filesystem::current_path(), kInputSize, kInputSize),
      .input_width = kInputSize,
      .input_height = kInputSize,
      .metric_from_relative = md::InverseDepthAffine{.scale = 0.4F, .shift = 0.1F},
      .max_width = kWidth,
      .max_height = kHeight,
      .fp16 = false};
  // The camera's file, in a folder of its own that mkdtemp() names, removed at the end;
  // CameraInfoSourceOp reads it again when the run starts.
  std::string folder =
      (std::filesystem::temp_directory_path() / "mono_depth_consumer_XXXXXX").string();
  if (::mkdtemp(folder.data()) == nullptr) {
    std::cerr << "mono_depth_consumer: cannot make a temporary folder\n";
    return 1;
  }
  struct RemoveFolder {
    std::filesystem::path path;
    ~RemoveFolder() {
      std::error_code ignored;
      std::filesystem::remove_all(path, ignored);
    }
  } remove_folder{folder};
  const std::filesystem::path camera_file = std::filesystem::path(folder) / "camera_info.yaml";
  hm::write_camera_info_yaml(test_camera(), camera_file);

  holoscan::Graph graph{"mono_depth_consumer"};
  auto source = graph.op<SceneSourceOp>("source");
  auto camera = graph.op<hm::CameraInfoSourceOp>(
      "camera_info", hm::CameraInfoSourceConfig{.camera_info_path = camera_file});
  auto estimator = graph.op<md::MonoDepthOp>("depth", config);
  auto tally = std::make_shared<Tally>();
  auto sink = graph.op<TallyOp>("tally", tally, config);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->image, estimator->image, lossless);
  graph.add_flow(camera->camera_info, estimator->camera_info, hm::latest_camera_info_connection());
  graph.add_flow(estimator->depth, sink->depth, lossless);
  graph.add_flow(estimator->metric_depth, sink->metric_depth, lossless);
  graph.add_flow(estimator->depth_camera_info, sink->camera_info, lossless);
  // Lossless: every operator on one worker, so that no stage runs ahead of a busy
  // consumer (see holoscan-messages' README).
  graph.partition("mono_depth_consumer").include_all();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // The depth is in GPU memory, on device 0.
  holoscan::CompileOptions options;
  for (const char* port : {"depth", "metric_depth"}) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "depth", .output_port = port, .device = holoscan::DeviceId{0}});
  }
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
  }
  if (!plan.ok()) {
    return 1;
  }
  const std::uint32_t workers = plan.scheduler_plan().worker_count;
  // Build the TensorRT engine before the run, as the operator asks.
  md::prepare_engine(config);
  auto session = signals ? holoscan::run_async(plan, *signals) : holoscan::run_async(plan);
  session.wait();
  const holoscan::RunTermination termination = session.termination();
  if (!session.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(session.diagnostics(), plan.graph()) << '\n';
  }
  if (signals) {
    if (auto closed = signals->close(); !closed) {
      std::cerr << "mono_depth_consumer: warning: closing the signal domain failed: "
                << holoscan::runtime_error_message(closed.error()) << '\n';
    }
  }
  // Every frame, but for a first one before the first camera info.
  const bool all_frames = tally->last_frame_id == kFrames &&
                          (tally->frames == kFrames || tally->frames + 1U == kFrames);
  const bool passed = termination == holoscan::RunTermination::kRequestedCompleted &&
                      workers == 1 && all_frames && tally->frames_matching == tally->frames &&
                      tally->cameras_matching == tally->frames;
  std::cout << "mono_depth_consumer: workers=" << workers << " frames=" << tally->frames
            << " matching_reference=" << tally->frames_matching
            << " matching_camera=" << tally->cameras_matching << " max_error=" << tally->max_error
            << '\n'
            << "mono_depth_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
} catch (const std::exception& error) {
  std::cerr << "mono_depth_consumer: error: " << error.what() << '\n';
  return 1;
}
