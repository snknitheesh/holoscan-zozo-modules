// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-mono-depth package: a source renders
// the synthetic scene, MonoDepthOp runs the stub model on it through TensorRT, and a
// sink checks each frame's depth against the host reference of the whole pipeline.

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
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/mono_depth/mono_depth_op.hpp>
#include <holoscan/mono_depth/reference.hpp>
#include <holoscan/mono_depth/stub_model.hpp>
#include <holoscan/mono_depth/synthetic_scene.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace md = holoscan::mono_depth;

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
      return holoscan::make_unexpected(std::move(loan).error());
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
    return emitted.error().code == holoscan::ErrorCode::kBackpressured
               ? holoscan::expected<void, holoscan::Error>{}
               : emitted;
  }

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  std::uint32_t emitted_{};
};

struct Tally {
  std::uint32_t frames{};
  std::uint32_t frames_matching{};
  float max_error{};
};

// Copies each frame's depth to the host and compares it with the reference;
// completes the run after kFrames.
class TallyOp final : public holoscan::Operator<> {
 public:
  TallyOp(std::shared_ptr<Tally> tally, md::MonoDepthConfig config)
      : tally_(std::move(tally)), config_(std::move(config)) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.input(depth, "depth").queue_depth(2U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{depth});
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
    std::vector<float> values(std::size_t{kWidth} * kHeight);
    if (auto copied = sample->data.data->copy_to_host(values.data(), values.size() * sizeof(float),
                                                      context.cuda_stream());
        !copied) {
      return copied;
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
    const md::ReferenceDepth expected =
        md::reference_postprocess(md::reference_stub(input, kInputSize, kInputSize), fit,
                                  md::ModelOutput::kRelativeInverseDepth, nullptr);
    float error = 0.0F;
    for (std::size_t i = 0; i < values.size(); ++i) {
      error = std::max(error, std::fabs(values[i] - expected.inverse[i]));
    }
    ++tally_->frames;
    tally_->max_error = std::max(tally_->max_error, error);
    if (frame_id != 0 && error < 1e-4F) {
      ++tally_->frames_matching;
    }
    if (tally_->frames == kFrames) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> depth;

 private:
  std::shared_ptr<Tally> tally_;
  md::MonoDepthConfig config_;
};

}  // namespace

int main() try {
  // Take over SIGINT and SIGTERM while this is the only thread: TensorRT starts threads of
  // its own, after which the SDK accepts no other way to manage the signals.
  auto signals = holoscan::ManagedShutdownSignals::create();
  if (!signals) {
    std::cerr << "mono_depth_consumer: cannot manage SIGINT and SIGTERM: "
              << holoscan::runtime_error_message(signals.error()) << '\n';
    return 1;
  }
  const md::MonoDepthConfig config{
      .model = md::write_stub_model(std::filesystem::current_path(), kInputSize, kInputSize),
      .input_width = kInputSize,
      .input_height = kInputSize,
      .max_width = kWidth,
      .max_height = kHeight,
      .fp16 = false};
  holoscan::Graph graph{"mono_depth_consumer"};
  auto source = graph.op<SceneSourceOp>("source");
  auto estimator = graph.op<md::MonoDepthOp>("depth", config);
  auto tally = std::make_shared<Tally>();
  auto sink = graph.op<TallyOp>("tally", tally, config);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->image, estimator->image, lossless);
  graph.add_flow(estimator->depth, sink->depth, lossless);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // The depth is in GPU memory, on device 0.
  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "depth", .output_port = "depth", .device = holoscan::DeviceId{0}});
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  // Build the TensorRT engine before the run, as the operator asks; the run adopts the
  // signal domain.
  md::prepare_engine(config);
  if (const int status = holoscan::run(plan); status != 0) {
    std::cerr << "mono_depth_consumer: run failed with " << status << '\n';
    return 1;
  }
  const bool passed = tally->frames == kFrames && tally->frames_matching == kFrames;
  std::cout << "mono_depth_consumer: frames=" << tally->frames
            << " matching_reference=" << tally->frames_matching << " max_error=" << tally->max_error
            << '\n'
            << "mono_depth_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
} catch (const std::exception& error) {
  std::cerr << "mono_depth_consumer: error: " << error.what() << '\n';
  return 1;
}
