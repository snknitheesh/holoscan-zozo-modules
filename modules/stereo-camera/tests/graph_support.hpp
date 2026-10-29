// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Small graphs for the operator tests: StereoCameraOp into a sink that takes
// each frame's ports together, by capture time, copies everything to the host
// and ends the run after the frames it expects.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
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
#include <holoscan/stereo_camera/stereo_camera_op.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace holoscan::stereo_camera::test {

using std::chrono_literals::operator""ns;

// One image as the sink saw it.
struct ReceivedImage {
  holoscan::schema::ImageEncoding encoding{};
  int width{};
  int height{};
  std::vector<std::int64_t> shape;
  DLDeviceType device{};
  std::string frame_id;
  std::int64_t header_capture_ns{};
  std::uint64_t device_sequence{};
  std::vector<std::uint8_t> bytes;

  [[nodiscard]] ImageView view(PixelFormat format) const {
    return ImageView{.data = bytes.data(),
                     .width = width,
                     .height = height,
                     .pitch = static_cast<std::size_t>(width) * bytes_per_pixel(format),
                     .format = format};
  }
};

// One frame: every port's sample, and their envelopes.
struct ReceivedFrame {
  std::uint64_t frame_id{};
  std::int64_t capture_ns{};
  ReceivedImage left;
  ReceivedImage right;
  std::optional<ReceivedImage> rgb;
  std::optional<ReceivedImage> depth;
  schema::StereoCalibrationT calibration;
  // The envelopes of every port agreed on capture time and frame id.
  bool consistent{};
};

struct Received {
  std::mutex mutex;
  std::vector<ReceivedFrame> frames;
  std::atomic<std::uint32_t> count{0};
};

struct FrameSinkConfig {
  bool rgb{};
  bool depth{};
  bool calibration{true};
  // Frames after which the run completes; 0 never completes.
  std::uint32_t expected{};
  // Time each frame takes, to play a slow consumer.
  std::chrono::milliseconds delay{0};
  std::shared_ptr<Received> received;
};

class FrameSinkOp final : public holoscan::Operator<> {
 public:
  explicit FrameSinkOp(FrameSinkConfig config) : config_(std::move(config)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(left, "left").queue_depth(2U);
    spec.input(right, "right").queue_depth(2U);
    if (config_.rgb) spec.input(rgb, "rgb").queue_depth(2U);
    if (config_.depth) spec.input(depth, "depth").queue_depth(2U);
    if (config_.calibration) spec.input(calibration, "calibration").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    std::vector<holoscan::PortRef> ports{left, right};
    if (config_.rgb) ports.emplace_back(rgb);
    if (config_.depth) ports.emplace_back(depth);
    if (config_.calibration) ports.emplace_back(calibration);
    holoscan::Contract result;
    result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{ports}, .window = 0ns});
    result.input(left, holoscan::Exact{});
    result.input(right, holoscan::Exact{});
    if (config_.rgb) result.input(rgb, holoscan::Exact{});
    if (config_.depth) result.input(depth, holoscan::Exact{});
    if (config_.calibration) result.input(calibration, holoscan::Exact{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    ReceivedFrame frame;
    auto l = left.receive();
    auto r = right.receive();
    if (!l || !r) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "FrameSinkOp: a port had no sample"});
    }
    frame.frame_id = l->metadata.frame_id;
    frame.capture_ns = l->metadata.capture_timestamp_ns;
    frame.consistent = true;
    const auto agree = [&frame](const holoscan::SampleMetadata& metadata) {
      frame.consistent &=
          metadata.frame_id == frame.frame_id && metadata.capture_timestamp_ns == frame.capture_ns;
    };
    agree(r->metadata);
    if (config_.calibration) {
      auto c = calibration.receive();
      if (!c) return holoscan::make_unexpected(std::move(c).error());
      agree(c->metadata);
      frame.calibration = c->data;
    }
    if (auto copied = to_host(l->data, frame.left, context); !copied) {
      return copied;
    }
    if (auto copied = to_host(r->data, frame.right, context); !copied) {
      return copied;
    }
    if (config_.rgb) {
      auto sample = rgb.receive();
      if (!sample) return holoscan::make_unexpected(std::move(sample).error());
      agree(sample->metadata);
      if (auto copied = to_host(sample->data, frame.rgb.emplace(), context); !copied) {
        return copied;
      }
    }
    if (config_.depth) {
      auto sample = depth.receive();
      if (!sample) return holoscan::make_unexpected(std::move(sample).error());
      agree(sample->metadata);
      if (auto copied = to_host(sample->data, frame.depth.emplace(), context); !copied) {
        return copied;
      }
    }
    if (config_.delay.count() > 0) {
      std::this_thread::sleep_for(config_.delay);
    }
    {
      const std::lock_guard lock(config_.received->mutex);
      config_.received->frames.push_back(std::move(frame));
    }
    if (++config_.received->count == config_.expected) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> left;
  holoscan::Input<holoscan::schema::ImageT> right;
  holoscan::Input<holoscan::schema::ImageT> rgb;
  holoscan::Input<holoscan::schema::ImageT> depth;
  holoscan::Input<schema::StereoCalibrationT> calibration;

 private:
  static holoscan::expected<void, holoscan::Error> to_host(const holoscan::schema::ImageT& image,
                                                           ReceivedImage& out,
                                                           holoscan::ExecutionContext& context) {
    if (!image.data) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "FrameSinkOp: an image has no tensor"});
    }
    out.encoding = image.encoding;
    out.width = image.width;
    out.height = image.height;
    const std::span<const std::int64_t> shape = image.data->shape_span();
    out.shape.assign(shape.begin(), shape.end());
    out.device = image.data->device().device_type;
    if (image.header) {
      out.frame_id = image.header->frame_id;
      out.header_capture_ns = image.header->capture_timestamp_ns;
      out.device_sequence = image.header->device_sequence;
    }
    out.bytes.resize(image.data->nbytes());
    return image.data->copy_to_host(out.bytes.data(), out.bytes.size(), context.cuda_stream());
  }

  FrameSinkConfig config_;
};

// Every output of `camera` to `sink`, with `options`.
inline void connect(holoscan::Graph& graph, const holoscan::OperatorHandle<StereoCameraOp>& camera,
                    const holoscan::OperatorHandle<FrameSinkOp>& sink, bool rgb, bool depth,
                    const holoscan::ConnectionOptions& options, bool calibration = true) {
  graph.add_flow(camera->left, sink->left, options);
  graph.add_flow(camera->right, sink->right, options);
  if (rgb) graph.add_flow(camera->rgb, sink->rgb, options);
  if (depth) graph.add_flow(camera->depth, sink->depth, options);
  if (calibration) graph.add_flow(camera->calibration, sink->calibration, options);
}

// The ports of StereoCameraOp "camera" in device memory, on GPU 0.
inline holoscan::CompileOptions device_outputs(bool rgb, bool depth) {
  holoscan::CompileOptions options;
  std::vector<std::string> ports{"left", "right"};
  if (rgb) ports.emplace_back("rgb");
  if (depth) ports.emplace_back("depth");
  for (const std::string& port : ports) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "camera", .output_port = port, .device = holoscan::DeviceId{0}});
  }
  return options;
}

struct RunOutcome {
  bool compiled{};
  bool completed{};
  holoscan::RunTermination termination{};
  std::string diagnostics;
};

// Compiles and runs `graph`, stopping it after `timeout`.
inline RunOutcome run_graph(holoscan::Graph& graph, holoscan::CompileOptions options = {},
                            std::chrono::seconds timeout = std::chrono::seconds{120}) {
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  RunOutcome outcome;
  if (!plan.ok()) {
    outcome.diagnostics = holoscan::format_diagnostics(plan.diagnostics(), plan.graph());
    return outcome;
  }
  outcome.compiled = true;
  holoscan::RunSession session = holoscan::run_async(plan);
  if (!session.wait_for(timeout)) {
    session.request_stop();
  }
  session.wait();
  outcome.termination = session.termination();
  outcome.completed = outcome.termination == holoscan::RunTermination::kRequestedCompleted &&
                      session.diagnostics().empty();
  if (!session.diagnostics().empty()) {
    outcome.diagnostics = holoscan::format_diagnostics(session.diagnostics(), plan.graph());
  }
  return outcome;
}

}  // namespace holoscan::stereo_camera::test
