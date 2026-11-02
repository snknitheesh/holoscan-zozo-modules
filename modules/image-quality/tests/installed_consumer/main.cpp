// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-image-quality package: a source
// sends synthetic frames, the last two repeats of the one before, through
// ImageQualityOp, and a sink checks that exactly the repeats are flagged frozen.

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/image_quality/image_quality_op.hpp>
#include <holoscan/image_quality/synthetic.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace iq = holoscan::image_quality;
using std::chrono_literals::operator""ns;

constexpr int kWidth = 640;
constexpr int kHeight = 360;
// The scene's frame for each frame sent: the last two repeat frame 2.
constexpr std::array<std::uint32_t, 5> kScene{0, 1, 2, 2, 2};
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

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
    if (sent_ >= kScene.size()) {
      return {};
    }
    const iq::HostImage scene = iq::render_scene(kWidth, kHeight, kScene[sent_]);
    const std::array<std::int64_t, 3> shape{kHeight, kWidth, 3};
    auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                   -> holoscan::expected<void, holoscan::Error> {
      std::memcpy(writer.data_as<std::uint8_t>(), scene.pixels.data(), scene.pixels.size());
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
      ++sent_;
      return {};
    }
    // A busy consumer refuses the frame; send it again on the next tick.
    return emitted.error().code == holoscan::ErrorCode::kBackpressured
               ? holoscan::expected<void, holoscan::Error>{}
               : emitted;
  }

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  std::size_t sent_{};
};

// Each frame's flags, and whether its metrics came with them.
class FlagsSinkOp final : public holoscan::Operator<> {
 public:
  explicit FlagsSinkOp(std::shared_ptr<std::vector<std::uint32_t>> flags)
      : flags_(std::move(flags)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(metrics, "metrics").queue_depth(2U);
    spec.input(flags, "flags").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAligned{.ports = holoscan::ports(metrics, flags), .window = 0ns});
    result.input(metrics, holoscan::Exact{});
    result.input(flags, holoscan::Exact{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto values = metrics.receive();
    if (!values) {
      return holoscan::make_unexpected(std::move(values).error());
    }
    auto bits = flags.receive();
    if (!bits) {
      return holoscan::make_unexpected(std::move(bits).error());
    }
    if (values->data.size() != static_cast<std::int64_t>(iq::kMetricCount)) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "metrics of the wrong size"});
    }
    std::uint32_t value = 0;
    std::memcpy(&value, bits->data.data(), sizeof(value));
    flags_->push_back(value);
    if (flags_->size() == kScene.size()) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> metrics;
  holoscan::Input<holoscan::Tensor> flags;

 private:
  std::shared_ptr<std::vector<std::uint32_t>> flags_;
};

}  // namespace

int main() {
  holoscan::Graph graph{"image_quality_consumer"};
  auto source = graph.op<SceneSourceOp>("source");
  auto quality = graph.op<iq::ImageQualityOp>("quality", iq::ImageQualityConfig{});
  auto flags = std::make_shared<std::vector<std::uint32_t>>();
  auto sink = graph.op<FlagsSinkOp>("sink", flags);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking,
                                             .queue_depth = 2U};
  graph.add_flow(source->image, quality->image, lossless);
  graph.add_flow(quality->metrics, sink->metrics, lossless);
  graph.add_flow(quality->flags, sink->flags, lossless);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (const int status = holoscan::run(plan); status != 0) {
    std::cerr << "image_quality_consumer: run failed with " << status << '\n';
    return 1;
  }
  const std::vector<std::uint32_t> expected{0U, 0U, 0U, iq::kFrozen, iq::kFrozen};
  for (std::size_t i = 0; i < flags->size(); ++i) {
    std::cout << "image_quality_consumer: frame " << i << ": " << iq::flag_names((*flags)[i])
              << '\n';
  }
  const bool passed = *flags == expected;
  std::cout << "image_quality_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
