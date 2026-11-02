// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-recorder package: a source
// sends five 64x48 MONO8 images, RecorderOp records them, and ReplayerOp plays
// them back into a sink that checks every pixel and capture time.

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/recorder/recorder_op.hpp>
#include <holoscan/recorder/replayer_op.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace rec = holoscan::recorder;

constexpr int kWidth = 64;
constexpr int kHeight = 48;
constexpr std::uint32_t kFrames = 5;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

std::uint8_t pixel(std::uint64_t frame, int i) { return static_cast<std::uint8_t>(frame * 17 + i); }
std::int64_t capture_ns(std::uint64_t frame) { return 2'000'000'000LL + 33'000'000LL * frame; }

class SourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(image, "image")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kHost,
                    .dtype = kUInt8,
                    .rank = 2U,
                    .strides = holoscan::TensorStridePolicy::kContiguousOnly,
                },
            .bounds = holoscan::tensor_bounds(std::uint64_t{kWidth} * kHeight),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        })
        .payload_options(holoscan::PayloadOptions{
            .schema = holoscan::schema_identity<holoscan::schema::ImageT>()});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{10}})
        .complete_after(holoscan::ComputeCalls{.limit = kFrames * 20U});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (sent_ >= kFrames) return {};
    const std::array<std::int64_t, 2> shape{kHeight, kWidth};
    auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
    if (!loan) return holoscan::make_unexpected(std::move(loan).error());
    auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& w) {
      for (int i = 0; i < kWidth * kHeight; ++i) w.data_as<std::uint8_t>()[i] = pixel(sent_, i);
    });
    if (!written) return written;
    holoscan::schema::ImageT descriptor;
    descriptor.width = kWidth;
    descriptor.height = kHeight;
    descriptor.encoding = holoscan::schema::ImageEncoding_MONO8;
    auto emitted = image.emit_tensor(
        std::move(*loan), descriptor,
        holoscan::EmitOptions{.capture_time = context.clock().at_ns(capture_ns(sent_)),
                              .frame_id = sent_ + 1U});
    if (emitted) ++sent_;
    return emitted || emitted.error().code == holoscan::ErrorCode::kBackpressured
               ? holoscan::expected<void, holoscan::Error>{}
               : emitted;
  }

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  std::uint64_t sent_{};
};

struct Tally {
  std::atomic<std::uint32_t> frames{0};
  std::atomic<std::uint32_t> wrong{0};
};

class CheckOp final : public holoscan::Operator<> {
 public:
  explicit CheckOp(std::shared_ptr<Tally> tally) : tally_(std::move(tally)) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.input(image, "image").queue_depth(2U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{image});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = image.receive();
    if (!sample) return holoscan::make_unexpected(std::move(sample).error());
    const std::uint64_t frame = tally_->frames.load();
    std::vector<std::uint8_t> pixels(kWidth * kHeight);
    if (auto copied =
            sample->data.data->copy_to_host(pixels.data(), pixels.size(), context.cuda_stream());
        !copied) {
      return copied;
    }
    bool ok = sample->metadata.capture_timestamp_ns == capture_ns(frame) &&
              sample->metadata.frame_id == frame + 1U &&
              sample->data.encoding == holoscan::schema::ImageEncoding_MONO8;
    for (int i = 0; i < kWidth * kHeight; ++i) ok = ok && pixels[i] == pixel(frame, i);
    if (!ok) ++tally_->wrong;
    if (++tally_->frames == kFrames) context.request_completion();
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> image;

 private:
  std::shared_ptr<Tally> tally_;
};

int run(holoscan::Graph& graph, holoscan::CompileOptions options = {}) {
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  // Two runs in one process: the second would find the first's threads, which
  // managed signal handling does not admit.
  return holoscan::run(plan, holoscan::RunSignalPolicy::kCallerManaged);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string folder = argc > 1 ? argv[1] : "recording";
  std::filesystem::remove_all(folder);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};

  auto stats = std::make_shared<rec::RecorderStats>();
  {
    holoscan::Graph graph{"record"};
    auto source = graph.op<SourceOp>("source");
    auto recorder = graph.op<rec::RecorderOp>(
        "recorder",
        rec::RecorderConfig{
            .directory = folder, .streams = {{"camera", rec::StreamKind::kImage}}, .stats = stats});
    graph.add_flow(source->image, recorder->image("camera"), lossless);
    if (run(graph) != 0) {
      std::cerr << "recorder_consumer: the recording run failed\n";
      return 1;
    }
  }

  auto tally = std::make_shared<Tally>();
  {
    holoscan::Graph graph{"replay"};
    auto replayer = graph.op<rec::ReplayerOp>(
        "replayer", rec::ReplayerConfig{.directory = folder, .memory = rec::ReplayMemory::kDevice});
    auto check = graph.op<CheckOp>("check", tally);
    graph.add_flow(replayer->image("camera"), check->image, lossless);
    holoscan::CompileOptions options;
    replayer->bind_device(options, "replayer");
    if (run(graph, std::move(options)) != 0) {
      std::cerr << "recorder_consumer: the replay run failed\n";
      return 1;
    }
  }
  const bool passed = stats->recorded == kFrames && stats->dropped == 0U &&
                      tally->frames == kFrames && tally->wrong == 0U;
  std::cout << "recorder_consumer: recorded=" << stats->recorded << " dropped=" << stats->dropped
            << " replayed=" << tally->frames << " wrong=" << tally->wrong << '\n'
            << "recorder_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
