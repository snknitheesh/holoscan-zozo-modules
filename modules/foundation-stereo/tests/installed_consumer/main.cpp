// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-foundation-stereo package: a source
// renders the synthetic scene's left and right views in host memory and sends the pair's
// stereo camera info with each, FoundationStereoOp runs the stub model on them through
// TensorRT with the depth and its camera info, and a sink checks each frame's disparity
// against the CPU reference of the operator's chain, its depth against fx * B / d, and the
// depth's camera info. The graph is lossless, so it runs on one worker.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
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
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/foundation_stereo/foundation_stereo_op.hpp>
#include <holoscan/foundation_stereo/reference.hpp>
#include <holoscan/foundation_stereo/stub_model.hpp>
#include <holoscan/foundation_stereo/synthetic_scene.hpp>
#include <holoscan/messages/camera_info.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using std::chrono_literals::operator""ns;
namespace fs = holoscan::foundation_stereo;
namespace hm = holoscan::messages;

constexpr int kWidth = 320;
constexpr int kHeight = 240;
constexpr int kModelWidth = 160;
constexpr int kModelHeight = 96;
constexpr std::uint32_t kFrames = 4;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

// The scene's two views, RGB8 on the host.
std::array<std::vector<std::uint8_t>, 2> render(std::uint32_t frame) {
  const fs::SyntheticScene scene = fs::make_scene(frame, kWidth, kHeight);
  std::array<std::vector<std::uint8_t>, 2> views;
  for (int side = 0; side < 2; ++side) {
    views[side].resize(std::size_t{kWidth} * kHeight * 3U);
    for (int v = 0; v < kHeight; ++v) {
      for (int u = 0; u < kWidth; ++u) {
        fs::scene_color(scene, side == 0 ? 0.0F : scene.baseline, static_cast<float>(u),
                        static_cast<float>(v),
                        views[side].data() + (std::size_t{kWidth} * v + u) * 3U);
      }
    }
  }
  return views;
}

// A refused send: the consumer is busy (kBackpressured), or its output's pool is still
// held (kResourceExhausted); the source sends the same part again on its next tick.
bool busy(const holoscan::Error& error) {
  return error.code == holoscan::ErrorCode::kBackpressured ||
         error.code == holoscan::ErrorCode::kResourceExhausted;
}

// kFrames stereo pairs of the synthetic scene in host memory, numbered from 1, each with
// the pair's camera info, sent first.
class PairSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(camera_info, "camera_info");
    for (auto* port : {&left, &right}) {
      spec.output(*port, port == &left ? "left" : "right")
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
    if (!capture_) {
      capture_ = context.activation_time();
      sent_ = 0;
    }
    const auto views = render(emitted_);
    const holoscan::EmitOptions emit{.capture_time = *capture_, .frame_id = emitted_ + 1U};
    if (sent_ == 0) {
      hm::schema::StereoCameraInfoT info =
          hm::to_message(fs::scene_camera_info(fs::make_scene(0, kWidth, kHeight)));
      info.header->capture_timestamp_ns = emit.capture_time ? emit.capture_time->timestamp_ns : 0;
      if (auto sent = camera_info.emit(std::move(info), emit); !sent) {
        return busy(sent.error()) ? holoscan::expected<void, holoscan::Error>{} : sent;
      }
      sent_ = 1;
    }
    for (int side = sent_ - 1; side < 2; ++side) {
      holoscan::Output<holoscan::schema::ImageT>& port = side == 0 ? left : right;
      auto loan = port.allocate_tensor(holoscan::TensorLoanRequest{
          .shape = std::array<std::int64_t, 3>{kHeight, kWidth, 3}, .dtype = kUInt8});
      if (!loan) {
        return busy(loan.error()) ? holoscan::expected<void, holoscan::Error>{}
                                  : holoscan::make_unexpected(std::move(loan).error());
      }
      auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                     -> holoscan::expected<void, holoscan::Error> {
        std::copy(views[side].begin(), views[side].end(), writer.data_as<std::uint8_t>());
        return {};
      });
      if (!written) {
        return holoscan::make_unexpected(std::move(written).error());
      }
      holoscan::schema::ImageT descriptor;
      descriptor.width = kWidth;
      descriptor.height = kHeight;
      descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
      auto emitted = port.emit_tensor(std::move(*loan), descriptor, emit);
      if (!emitted) {
        return busy(emitted.error()) ? holoscan::expected<void, holoscan::Error>{} : emitted;
      }
      sent_ = side + 2;
    }
    capture_.reset();
    ++emitted_;
    return {};
  }

  holoscan::Output<hm::schema::StereoCameraInfoT> camera_info;
  holoscan::Output<holoscan::schema::ImageT> left;
  holoscan::Output<holoscan::schema::ImageT> right;

 private:
  std::uint32_t emitted_{};
  std::optional<holoscan::TimePoint> capture_;
  int sent_{};  // Parts of the frame sent: the camera info, then the left and right images.
};

struct Tally {
  std::uint32_t frames{};
  std::uint32_t frames_matching{};
};

// Checks each frame's disparity against the reference, its depth against the formula and
// the depth's camera info against the left camera; completes the run after kFrames.
class TallyOp final : public holoscan::Operator<> {
 public:
  TallyOp(std::shared_ptr<Tally> tally, hm::StereoCameraInfo info)
      : tally_(std::move(tally)), info_(std::move(info)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(disparity, "disparity").queue_depth(2U);
    spec.input(depth, "depth").queue_depth(2U);
    spec.input(depth_camera_info, "depth_camera_info").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{std::vector<holoscan::PortRef>{
                                           disparity, depth, depth_camera_info}},
                                       .window = 0ns});
    result.input(disparity, holoscan::Exact{});
    result.input(depth, holoscan::Exact{});
    result.input(depth_camera_info, holoscan::Exact{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto d_sample = disparity.receive();
    auto z_sample = depth.receive();
    auto camera_sample = depth_camera_info.receive();
    if (!d_sample) return holoscan::make_unexpected(std::move(d_sample).error());
    if (!z_sample) return holoscan::make_unexpected(std::move(z_sample).error());
    if (!camera_sample) return holoscan::make_unexpected(std::move(camera_sample).error());
    // The depth's camera is the rectified left camera at the images' size.
    hm::CameraInfo camera = hm::from_message(camera_sample->data);
    std::size_t wrong = camera.width == info_.left.width && camera.height == info_.left.height &&
                                camera.fx == info_.left.fx && camera.cx == info_.left.cx &&
                                camera.state == hm::CameraState::kRectified
                            ? 0U
                            : 1U;
    std::vector<float> d(std::size_t{kWidth} * kHeight);
    std::vector<float> z(d.size());
    for (auto copied : {d_sample->data.data->copy_to_host(d.data(), d.size() * sizeof(float),
                                                          context.cuda_stream()),
                        z_sample->data.data->copy_to_host(z.data(), z.size() * sizeof(float),
                                                          context.cuda_stream())}) {
      if (!copied) {
        return copied;
      }
    }
    // Sources number their frames from 1; the outputs keep the left image's id.
    const std::uint64_t frame_id = d_sample->metadata.frame_id;
    const auto views = render(static_cast<std::uint32_t>(frame_id - 1U));
    const std::size_t pitch = std::size_t{kWidth} * 3U;
    const auto raw = fs::reference::rescale(
        fs::reference::stub(
            fs::reference::preprocess(views[0].data(), pitch, kWidth, kHeight,
                                      fs::PixelFormat::kRgb8, kModelWidth, kModelHeight),
            fs::reference::preprocess(views[1].data(), pitch, kWidth, kHeight,
                                      fs::PixelFormat::kRgb8, kModelWidth, kModelHeight),
            kModelWidth, kModelHeight),
        kModelWidth, kModelHeight, kWidth, kHeight);
    const auto fb = static_cast<float>(hm::fx_baseline(info_));
    for (std::size_t i = 0; i < raw.size(); ++i) {
      const int x = static_cast<int>(i % kWidth);
      const double limit = 1e-4 * std::max(1.0, std::abs(raw[i]));
      if (!fs::reference::near_limit(raw[i], x, limit)) {
        wrong += std::abs(d[i] - fs::reference::valid(raw[i], x, true)) > limit ? 1U : 0U;
      }
      wrong += z[i] != fs::depth_from_disparity(d[i], fb, 0.0F) ? 1U : 0U;
    }
    ++tally_->frames;
    if (frame_id != 0 && wrong == 0) {
      ++tally_->frames_matching;
    }
    if (tally_->frames == kFrames) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> disparity;
  holoscan::Input<holoscan::schema::ImageT> depth;
  holoscan::Input<hm::schema::CameraInfoT> depth_camera_info;

 private:
  std::shared_ptr<Tally> tally_;
  hm::StereoCameraInfo info_;
};

}  // namespace

int main() try {
  // Take over SIGINT and SIGTERM while this is the only thread: TensorRT starts threads of
  // its own, after which the SDK accepts no other way to manage the signals.
  auto signals = holoscan::ManagedShutdownSignals::create();
  if (!signals) {
    std::cerr << "foundation_stereo_consumer: cannot manage SIGINT and SIGTERM: "
              << holoscan::runtime_error_message(signals.error()) << '\n';
    return 1;
  }
  const hm::StereoCameraInfo info = fs::scene_camera_info(fs::make_scene(0, kWidth, kHeight));
  // The source sends the camera info with every pair: each frame takes its own.
  const fs::FoundationStereoConfig config{
      .model = fs::write_stub_model(std::filesystem::current_path(), kModelWidth, kModelHeight),
      .fp16 = false,
      .camera_info_input = {.timing = hm::CameraInfoTiming::kEveryFrame},
      .depth = true,
      .depth_camera_info = true,
      .max_width = kWidth,
      .max_height = kHeight};
  holoscan::Graph graph{"foundation_stereo_consumer"};
  auto source = graph.op<PairSourceOp>("source");
  auto stereo = graph.op<fs::FoundationStereoOp>("stereo", config);
  auto tally = std::make_shared<Tally>();
  auto sink = graph.op<TallyOp>("tally", tally, info);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->camera_info, stereo->camera_info, lossless);
  graph.add_flow(source->left, stereo->left, lossless);
  graph.add_flow(source->right, stereo->right, lossless);
  graph.add_flow(stereo->disparity, sink->disparity, lossless);
  graph.add_flow(stereo->depth, sink->depth, lossless);
  graph.add_flow(stereo->depth_camera_info, sink->depth_camera_info, lossless);
  // Lossless edges on one worker: on several, an operator whose consumer's queue is full
  // would fail on the refused emit (see the messages module's README).
  graph.partition("foundation_stereo_consumer").include_all();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // The outputs are in GPU memory, on device 0.
  holoscan::CompileOptions options;
  for (const char* port : {"disparity", "depth"}) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "stereo", .output_port = port, .device = holoscan::DeviceId{0}});
  }
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
    return 1;
  }
  if (const std::uint32_t workers = plan.scheduler_plan().worker_count; workers != 1U) {
    std::cerr << "foundation_stereo_consumer: the graph runs on " << workers
              << " workers, not one\n";
    return 1;
  }
  // Build the TensorRT engine before the run, as the operator asks; the run adopts the
  // signal domain.
  fs::prepare_engine(config);
  if (const int status = holoscan::run(plan); status != 0) {
    std::cerr << "foundation_stereo_consumer: run failed with " << status << '\n';
    return 1;
  }
  const bool passed = tally->frames == kFrames && tally->frames_matching == kFrames;
  std::cout << "foundation_stereo_consumer: frames=" << tally->frames
            << " matching=" << tally->frames_matching << '\n'
            << "foundation_stereo_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
} catch (const std::exception& error) {
  std::cerr << "foundation_stereo_consumer: error: " << error.what() << '\n';
  return 1;
}
