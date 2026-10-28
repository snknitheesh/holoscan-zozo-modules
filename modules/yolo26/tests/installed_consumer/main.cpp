// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-yolo26 package: a source renders
// synthetic scenes, Yolo26DetectionOp runs the stub model on them through TensorRT, and
// a sink checks each frame's detections against the scene's two known boxes.

#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
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
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/yolo26/letterbox.hpp>
#include <holoscan/yolo26/stub_model.hpp>
#include <holoscan/yolo26/synthetic_scene.hpp>
#include <holoscan/yolo26/yolo26_detection_op.hpp>

namespace {

using std::chrono_literals::operator""ns;
namespace y26 = holoscan::yolo26;

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr int kInputSize = 640;
constexpr int kCapacity = 16;
constexpr std::uint32_t kFrames = 5;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

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
    auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                   -> holoscan::expected<void, holoscan::Error> {
      y26::render_scene(y26::make_scene(kWidth, kHeight, emitted_), y26::PixelFormat::kRgb8,
                        writer.data_as<std::uint8_t>(), std::size_t{kWidth} * 3U);
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
  std::uint32_t frames_matching_truth{};
};

// Pairs each frame's four outputs by capture time, copies them to the host and
// matches them against the frame's scene; completes the run after kFrames.
class TallyOp final : public holoscan::Operator<> {
 public:
  explicit TallyOp(std::shared_ptr<Tally> tally) : tally_(std::move(tally)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(boxes, "boxes").queue_depth(2U);
    spec.input(scores, "scores").queue_depth(2U);
    spec.input(class_ids, "class_ids").queue_depth(2U);
    spec.input(count, "count").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{std::vector<holoscan::PortRef>{
                                           boxes, scores, class_ids, count}},
                                       .window = 0ns});
    result.input(boxes, holoscan::Exact{});
    result.input(scores, holoscan::Exact{});
    result.input(class_ids, holoscan::Exact{});
    result.input(count, holoscan::Exact{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto count_sample = count.receive();
    auto boxes_sample = boxes.receive();
    auto scores_sample = scores.receive();
    auto class_ids_sample = class_ids.receive();
    if (!count_sample) return holoscan::make_unexpected(std::move(count_sample).error());
    if (!boxes_sample) return holoscan::make_unexpected(std::move(boxes_sample).error());
    if (!scores_sample) return holoscan::make_unexpected(std::move(scores_sample).error());
    if (!class_ids_sample) return holoscan::make_unexpected(std::move(class_ids_sample).error());
    std::int32_t n = 0;
    std::vector<float> box_values(std::size_t{kCapacity} * 4U);
    std::vector<float> score_values(kCapacity);
    std::vector<std::int32_t> class_values(kCapacity);
    for (auto copied :
         {count_sample->data.copy_to_host(&n, sizeof(n), context.cuda_stream()),
          boxes_sample->data.copy_to_host(box_values.data(), box_values.size() * sizeof(float),
                                          context.cuda_stream()),
          scores_sample->data.copy_to_host(score_values.data(), score_values.size() * sizeof(float),
                                           context.cuda_stream()),
          class_ids_sample->data.copy_to_host(class_values.data(),
                                              class_values.size() * sizeof(std::int32_t),
                                              context.cuda_stream())}) {
      if (!copied) {
        return copied;
      }
    }
    std::vector<y26::BoxDetection> detections;
    for (std::int32_t i = 0; i < n && i < kCapacity; ++i) {
      const auto k = static_cast<std::size_t>(i);
      detections.push_back(y26::BoxDetection{.x1 = box_values[4 * k],
                                             .y1 = box_values[4 * k + 1],
                                             .x2 = box_values[4 * k + 2],
                                             .y2 = box_values[4 * k + 3],
                                             .score = score_values[k],
                                             .class_id = class_values[k]});
    }
    // Sources number their frames from 1; the detections keep their image's id.
    const std::uint64_t frame_id = count_sample->metadata.frame_id;
    const y26::SyntheticScene scene =
        y26::make_scene(kWidth, kHeight, static_cast<std::uint32_t>(frame_id - 1U));
    const float scale = y26::make_letterbox(kWidth, kHeight, kInputSize).scale;
    const y26::TruthMatch match =
        y26::match_truth(detections, y26::scene_truth(scene), 2.0F / scale + 1.0F, 1e-3F);
    ++tally_->frames;
    if (frame_id != 0 && match.missed == 0 && match.extra == 0) {
      ++tally_->frames_matching_truth;
    }
    if (tally_->frames == kFrames) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> boxes;
  holoscan::Input<holoscan::Tensor> scores;
  holoscan::Input<holoscan::Tensor> class_ids;
  holoscan::Input<holoscan::Tensor> count;

 private:
  std::shared_ptr<Tally> tally_;
};

}  // namespace

int main() try {
  // Take over SIGINT and SIGTERM while this is the only thread: TensorRT starts threads of
  // its own, after which the SDK accepts no other way to manage the signals.
  auto signals = holoscan::ManagedShutdownSignals::create();
  if (!signals) {
    std::cerr << "yolo26_consumer: cannot manage SIGINT and SIGTERM: "
              << holoscan::runtime_error_message(signals.error()) << '\n';
    return 1;
  }
  const y26::Yolo26DetectionConfig config{
      .model = y26::write_stub_model(std::filesystem::current_path(), kInputSize),
      .input_size = kInputSize,
      .max_detections = kCapacity,
      .num_classes = y26::kStubClasses};
  holoscan::Graph graph{"yolo26_consumer"};
  auto source = graph.op<SceneSourceOp>("source");
  auto detector = graph.op<y26::Yolo26DetectionOp>("detector", config);
  auto tally = std::make_shared<Tally>();
  auto sink = graph.op<TallyOp>("tally", tally);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->image, detector->image, lossless);
  graph.add_flow(detector->boxes, sink->boxes, lossless);
  graph.add_flow(detector->scores, sink->scores, lossless);
  graph.add_flow(detector->class_ids, sink->class_ids, lossless);
  graph.add_flow(detector->count, sink->count, lossless);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // The detector's outputs are in GPU memory, on device 0.
  holoscan::CompileOptions options;
  for (const char* port : {"boxes", "scores", "class_ids", "count"}) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "detector", .output_port = port, .device = holoscan::DeviceId{0}});
  }
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  // Build the TensorRT engine before the run, as the operator asks; the run adopts the
  // signal domain.
  y26::prepare_engine(config);
  if (const int status = holoscan::run(plan); status != 0) {
    std::cerr << "yolo26_consumer: run failed with " << status << '\n';
    return 1;
  }
  const bool passed = tally->frames == kFrames && tally->frames_matching_truth == kFrames;
  std::cout << "yolo26_consumer: frames=" << tally->frames
            << " matching_truth=" << tally->frames_matching_truth << '\n'
            << "yolo26_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
} catch (const std::exception& error) {
  std::cerr << "yolo26_consumer: error: " << error.what() << '\n';
  return 1;
}
