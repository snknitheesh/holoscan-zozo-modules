// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-apriltag package: a source renders
// synthetic tags, AprilTagDetectorOp finds them on every compiled backend, and a sink
// checks that each frame has the scene's four tags.

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/apriltag/apriltag_detector_op.hpp>
#include <holoscan/apriltag/synthetic_scene.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace at = holoscan::apriltag;

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr std::uint32_t kFrames = 5;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

// kFrames RGB8 renderings of the synthetic scene, then idle.
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
    const std::vector<std::uint8_t> pixels =
        at::render_scene(at::make_scene("tag36h11", emitted_, kWidth, kHeight), 3);
    const std::array<std::int64_t, 3> shape{kHeight, kWidth, 3};
    auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                   -> holoscan::expected<void, holoscan::Error> {
      std::memcpy(writer.data_as<std::uint8_t>(), pixels.data(), pixels.size());
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
                          holoscan::EmitOptions{.capture_time = context.activation_time()});
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
  std::string backend;
  std::uint32_t frames{};
  std::uint32_t frames_with_all_tags{};
};

// Counts the frames with all four tags; the last sink to finish ends the run.
class TallyOp final : public holoscan::Operator<> {
 public:
  TallyOp(std::shared_ptr<Tally> tally, std::shared_ptr<std::atomic<int>> remaining)
      : tally_(std::move(tally)), remaining_(std::move(remaining)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(detections, "detections").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{detections});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto message = detections.receive_data();
    if (!message) {
      return holoscan::make_unexpected(std::move(message).error());
    }
    tally_->backend = message->backend;
    ++tally_->frames;
    if (message->detections.size() == 4U) {
      ++tally_->frames_with_all_tags;
    }
    if (tally_->frames == kFrames && remaining_->fetch_sub(1) == 1) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<at::schema::AprilTagDetectionsT> detections;

 private:
  std::shared_ptr<Tally> tally_;
  std::shared_ptr<std::atomic<int>> remaining_;
};

}  // namespace

int main() {
  holoscan::Graph graph{"apriltag_consumer"};
  auto source = graph.op<SceneSourceOp>("source");
  const at::SyntheticScene scene = at::make_scene("tag36h11", 0, kWidth, kHeight);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking,
                                             .queue_depth = 2U};
  const std::vector<at::Backend> backends = at::available_backends();
  auto remaining = std::make_shared<std::atomic<int>>(static_cast<int>(backends.size()));
  std::vector<std::shared_ptr<Tally>> tallies;
  for (const at::Backend backend : backends) {
    const std::string name(at::backend_name(backend));
    auto detector = graph.op<at::AprilTagDetectorOp>(
        "detector_" + name,
        at::AprilTagDetectorConfig{
            .backend = backend, .intrinsics = scene.intrinsics, .tag_size = scene.tag_size});
    auto tally = tallies.emplace_back(std::make_shared<Tally>());
    auto sink = graph.op<TallyOp>("tally_" + name, tally, remaining);
    graph.add_flow(source->image, detector->image, lossless);
    graph.add_flow(detector->detections, sink->detections, lossless);
  }
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (const int status = holoscan::run(plan); status != 0) {
    std::cerr << "apriltag_consumer: run failed with " << status << '\n';
    return 1;
  }
  bool passed = !tallies.empty();
  for (const auto& tally : tallies) {
    std::cout << "apriltag_consumer: backend=" << tally->backend << " frames=" << tally->frames
              << " with_all_tags=" << tally->frames_with_all_tags << '\n';
    passed = passed && tally->frames == kFrames && tally->frames_with_all_tags == kFrames;
  }
  std::cout << "apriltag_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
