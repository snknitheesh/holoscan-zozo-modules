// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-sam3 package: a source renders the
// synthetic scene, Sam3SegmentationOp runs the stub model on it through TensorRT with
// three prompts, Sam3OverlayOp draws the instances, and a sink checks each frame's
// overlay against the host reference of the whole pipeline, and its labels.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
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
#include <holoscan/sam3/reference.hpp>
#include <holoscan/sam3/sam3_overlay_op.hpp>
#include <holoscan/sam3/sam3_segmentation_op.hpp>
#include <holoscan/sam3/stub_model.hpp>
#include <holoscan/sam3/synthetic_scene.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using std::chrono_literals::operator""ns;
namespace s3 = holoscan::sam3;

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr std::uint32_t kFrames = 5;
constexpr std::uint8_t kFillAlpha = 100;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

std::vector<std::uint8_t> scene_pixels(std::uint32_t frame) {
  std::vector<std::uint8_t> pixels(std::size_t{kWidth} * kHeight * 3U);
  s3::render_shape_scene(s3::make_shape_scene(kWidth, kHeight, frame), s3::PixelFormat::kRgb8,
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
  std::size_t worst_pixels{};
};

// Copies each frame's overlay and labels to the host and compares them with the
// reference; completes the run after kFrames.
class TallyOp final : public holoscan::Operator<> {
 public:
  TallyOp(std::shared_ptr<Tally> tally, s3::Sam3SegmentationConfig config, s3::StubShape shape)
      : tally_(std::move(tally)), config_(std::move(config)), shape_(shape) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(overlay, "overlay").queue_depth(2U);
    spec.input(text, "text").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAligned{
        .ports = holoscan::PortList{std::vector<holoscan::PortRef>{overlay, text}}, .window = 0ns});
    result.input(overlay, holoscan::Exact{});
    result.input(text, holoscan::Exact{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto image = overlay.receive();
    auto labels = text.receive();
    if (!image) return holoscan::make_unexpected(std::move(image).error());
    if (!labels) return holoscan::make_unexpected(std::move(labels).error());
    if (!image->data.data) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "an overlay without a tensor"});
    }
    std::vector<std::uint8_t> rgba(std::size_t{kWidth} * kHeight * 4U);
    if (auto copied =
            image->data.data->copy_to_host(rgba.data(), rgba.size(), context.cuda_stream());
        !copied) {
      return copied;
    }
    // The reference: preprocessing, the stub, the instances and the overlay.
    const std::uint64_t frame_id = image->metadata.frame_id;
    const std::vector<std::uint8_t> pixels =
        scene_pixels(static_cast<std::uint32_t>(frame_id - 1U));
    const std::vector<float> input = s3::reference_preprocess(
        pixels.data(), std::size_t{kWidth} * 3U, s3::PixelFormat::kRgb8, kWidth, kHeight,
        shape_.input_size, shape_.input_size, s3::Normalization{});
    const std::vector<float> fpn0 = s3::reference_stub_encoder(input, shape_);
    const s3::PromptTable table = s3::read_prompt_table(config_.prompt_table);
    std::vector<s3::DecoderOutput> outputs;
    for (const s3::Sam3Prompt& prompt : config_.prompts) {
      outputs.push_back(s3::reference_stub_decoder(
          fpn0, s3::make_decoder_prompt(prompt, table, shape_.boxes), shape_));
    }
    const s3::ReferenceInstances expected =
        s3::reference_instances(outputs, config_.max_instances, config_.score_threshold,
                                s3::logit_of(config_.mask_threshold), kWidth, kHeight);
    const std::vector<std::uint8_t> reference =
        s3::reference_overlay(expected.labels, kWidth, kHeight, kFillAlpha);
    // TensorRT sums the stub's terms in its own order: a few edge pixels may differ.
    std::size_t differing = 0;
    for (std::size_t i = 0; i < rgba.size(); i += 4) {
      differing += std::memcmp(rgba.data() + i, reference.data() + i, 4) != 0 ? 1U : 0U;
    }
    // The first instance's label row: "red 0.90".
    std::vector<std::uint8_t> rows(labels->data.nbytes());
    if (auto copied = labels->data.copy_to_host(rows.data(), rows.size(), context.cuda_stream());
        !copied) {
      return copied;
    }
    // Rows 0 and 1 are the first instance's shadow and label, each a text run header
    // and then kOverlayLabelBytes of text.
    const std::size_t row = rows.size() / (2U * static_cast<std::size_t>(config_.max_instances));
    const std::size_t header = row - s3::kOverlayLabelBytes;
    const std::string first(reinterpret_cast<const char*>(rows.data() + row + header), 8U);
    ++tally_->frames;
    tally_->worst_pixels = std::max(tally_->worst_pixels, differing);
    if (frame_id != 0 && expected.count == 3 && differing <= 16U && first == "red 0.90") {
      ++tally_->frames_matching;
    }
    if (tally_->frames == kFrames) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> overlay;
  holoscan::Input<holoscan::Tensor> text;

 private:
  std::shared_ptr<Tally> tally_;
  s3::Sam3SegmentationConfig config_;
  s3::StubShape shape_;
};

}  // namespace

int main() try {
  // Take over SIGINT and SIGTERM while this is the only thread: TensorRT starts threads of
  // its own, after which the SDK accepts no other way to manage the signals.
  auto signals = holoscan::ManagedShutdownSignals::create();
  if (!signals) {
    std::cerr << "sam3_consumer: cannot manage SIGINT and SIGTERM: "
              << holoscan::runtime_error_message(signals.error()) << '\n';
    return 1;
  }
  const s3::StubShape shape{};
  const s3::StubModel stub = s3::write_stub_model(std::filesystem::current_path(), shape);
  const s3::ShapeScene scene = s3::make_shape_scene(kWidth, kHeight, 0);
  const s3::Sam3SegmentationConfig config{
      .encoder = stub.encoder,
      .decoder = stub.decoder,
      .prompt_table = stub.prompt_table,
      .prompts = {{.text = "red"},
                  {.text = "blue"},
                  {.boxes = {s3::PromptBox{.xyxy =
                                               s3::shape_box(scene, s3::SceneShape::kTriangle)}}}},
      .max_instances = 4,
      .output_masks = false,
      .max_width = kWidth,
      .max_height = kHeight,
      .fp16 = false};
  holoscan::Graph graph{"sam3_consumer"};
  auto source = graph.op<SceneSourceOp>("source");
  auto segmenter = graph.op<s3::Sam3SegmentationOp>("segment", config);
  auto overlay = graph.op<s3::Sam3OverlayOp>(
      "overlay", s3::Sam3OverlayConfig{.max_width = kWidth,
                                       .max_height = kHeight,
                                       .max_instances = config.max_instances,
                                       .prompt_names = {"red", "blue", "box"},
                                       .fill_alpha = kFillAlpha});
  auto tally = std::make_shared<Tally>();
  auto sink = graph.op<TallyOp>("tally", tally, config, shape);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->image, segmenter->image, lossless);
  graph.add_flow(segmenter->labels, overlay->labels, lossless);
  graph.add_flow(segmenter->scores, overlay->scores, lossless);
  graph.add_flow(segmenter->boxes, overlay->boxes, lossless);
  graph.add_flow(segmenter->prompt_ids, overlay->prompt_ids, lossless);
  graph.add_flow(segmenter->count, overlay->count, lossless);
  graph.add_flow(overlay->overlay, sink->overlay, lossless);
  graph.add_flow(overlay->text, sink->text, lossless);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // The segmenter's outputs and the overlay are in GPU memory, on device 0.
  holoscan::CompileOptions options;
  for (auto [op, port] : {std::pair{"segment", "labels"}, std::pair{"segment", "scores"},
                          std::pair{"segment", "boxes"}, std::pair{"segment", "prompt_ids"},
                          std::pair{"segment", "count"}, std::pair{"overlay", "overlay"}}) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
  }
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  // Build the TensorRT engines before the run, as the operator asks; the run adopts the
  // signal domain.
  s3::prepare_engines(config);
  if (const int status = holoscan::run(plan); status != 0) {
    std::cerr << "sam3_consumer: run failed with " << status << '\n';
    return 1;
  }
  const bool passed = tally->frames == kFrames && tally->frames_matching == kFrames;
  std::cout << "sam3_consumer: frames=" << tally->frames
            << " matching_reference=" << tally->frames_matching
            << " worst_differing_pixels=" << tally->worst_pixels << '\n'
            << "sam3_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
} catch (const std::exception& error) {
  std::cerr << "sam3_consumer: error: " << error.what() << '\n';
  return 1;
}
