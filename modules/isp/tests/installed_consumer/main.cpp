// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-isp package: a source sends
// frames of the synthetic raw camera as 12-bit BAYER_RGGB in host memory,
// IspOp turns them into RGB8 on the GPU, one sink checks every image against
// the camera's truth (PSNR), another that each histogram row counts every pixel.
// IspOp cannot send a refused output again, so the source keeps one frame in
// flight: it sends the next once both sinks have counted the last.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
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
#include <holoscan/isp/isp_op.hpp>
#include <holoscan/isp/synthetic_raw.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace isp = holoscan::isp;

constexpr int kWidth = 320;
constexpr int kHeight = 240;
constexpr std::uint32_t kFrames = 5;
constexpr DLDataType kUInt16{kDLUInt, 16U, 1U};

isp::SyntheticRawConfig camera(std::uint32_t frame) {
  isp::SyntheticRawConfig config;
  config.width = kWidth;
  config.height = kHeight;
  config.vignetting = 0.3F;
  config.frame = frame;
  return config;
}

// kFrames raw frames, uint16 in host memory, then idle; each once the sinks
// have counted the one before (`counted` goes up by two per frame).
class RawSourceOp final : public holoscan::Operator<> {
 public:
  explicit RawSourceOp(std::shared_ptr<const std::atomic<std::uint32_t>> counted)
      : counted_(std::move(counted)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(raw, "raw")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kHost,
                    .dtype = kUInt16,
                    .rank = 2U,
                    .strides = holoscan::TensorStridePolicy::kContiguousOnly,
                },
            .bounds = holoscan::tensor_bounds(std::uint64_t{kWidth} * kHeight * 2U),
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
    if (emitted_ >= kFrames || counted_->load(std::memory_order_acquire) < 2U * emitted_) {
      return {};
    }
    const std::array<std::int64_t, 2> shape{kHeight, kWidth};
    auto loan = raw.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt16});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                   -> holoscan::expected<void, holoscan::Error> {
      isp::render_synthetic_raw(isp::make_synthetic_scene(camera(emitted_)),
                                writer.data_as<std::uint16_t>(), 2U * kWidth);
      return {};
    });
    if (!written) {
      return holoscan::make_unexpected(std::move(written).error());
    }
    holoscan::schema::ImageT descriptor;
    descriptor.width = kWidth;
    descriptor.height = kHeight;
    descriptor.encoding = holoscan::schema::ImageEncoding_BAYER_RGGB;
    descriptor.significant_bits = 12;
    auto emitted = raw.emit_tensor(std::move(*loan), descriptor,
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

  holoscan::Output<holoscan::schema::ImageT> raw;

 private:
  std::shared_ptr<const std::atomic<std::uint32_t>> counted_;
  std::uint32_t emitted_{};
};

struct Tally {
  std::atomic<std::uint32_t> counted{};  // Frames counted by either sink.
  std::uint32_t frames{};
  double worst_psnr{99.0};
  std::uint32_t stats_frames{};
  std::uint32_t bad_histograms{};
};

// The last of the sinks to see kFrames frames ends the run.
void finish(std::atomic<int>& remaining, holoscan::ExecutionContext& context) {
  if (remaining.fetch_sub(1) == 1) {
    context.request_completion();
  }
}

// PSNR of each RGB8 image against the camera's truth through the sRGB curve.
class ImageCheckOp final : public holoscan::Operator<> {
 public:
  ImageCheckOp(std::shared_ptr<Tally> tally, std::shared_ptr<std::atomic<int>> remaining)
      : tally_(std::move(tally)), remaining_(std::move(remaining)) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.input(image, "image").queue_depth(2U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{image});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = image.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    if (!sample->data.data || sample->data.encoding != holoscan::schema::ImageEncoding_RGB8) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "expected an RGB8 image"});
    }
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(kWidth) * kHeight * 3U);
    if (auto copied =
            sample->data.data->copy_to_host(rgb.data(), rgb.size(), context.cuda_stream());
        !copied) {
      return copied;
    }
    const isp::SyntheticRawScene scene = isp::make_synthetic_scene(
        camera(static_cast<std::uint32_t>(sample->metadata.frame_id - 1U)));
    double squared = 0.0;
    for (int y = 0; y < kHeight; ++y) {
      for (int x = 0; x < kWidth; ++x) {
        float linear[3];
        isp::synthetic_truth(scene, x, y, linear);
        for (int k = 0; k < 3; ++k) {
          const double v = linear[k];
          const double coded = v <= 0.0031308 ? 12.92 * v : 1.055 * std::pow(v, 1 / 2.4) - 0.055;
          const double d =
              rgb[(static_cast<std::size_t>(y) * kWidth + x) * 3U + k] - std::round(255.0 * coded);
          squared += d * d;
        }
      }
    }
    const double psnr = 10.0 * std::log10(255.0 * 255.0 * 3.0 * kWidth * kHeight / squared);
    tally_->worst_psnr = std::min(tally_->worst_psnr, psnr);
    tally_->counted.fetch_add(1U, std::memory_order_release);
    if (++tally_->frames == kFrames) {
      finish(*remaining_, context);
    }
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> image;

 private:
  std::shared_ptr<Tally> tally_;
  std::shared_ptr<std::atomic<int>> remaining_;
};

// Each histogram row counts every pixel once.
class HistogramCheckOp final : public holoscan::Operator<> {
 public:
  HistogramCheckOp(std::shared_ptr<Tally> tally, std::shared_ptr<std::atomic<int>> remaining)
      : tally_(std::move(tally)), remaining_(std::move(remaining)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(histogram, "histogram").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{histogram});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = histogram.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    std::vector<std::uint32_t> counts(isp::kHistogramChannels * isp::kHistogramBins);
    if (auto copied =
            sample->data.copy_to_host(counts.data(), counts.size() * 4U, context.cuda_stream());
        !copied) {
      return copied;
    }
    for (int row = 0; row < isp::kHistogramChannels; ++row) {
      std::uint64_t sum = 0;
      for (int b = 0; b < isp::kHistogramBins; ++b) sum += counts[row * isp::kHistogramBins + b];
      tally_->bad_histograms += sum == std::uint64_t{kWidth} * kHeight ? 0U : 1U;
    }
    tally_->counted.fetch_add(1U, std::memory_order_release);
    if (++tally_->stats_frames == kFrames) {
      finish(*remaining_, context);
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> histogram;

 private:
  std::shared_ptr<Tally> tally_;
  std::shared_ptr<std::atomic<int>> remaining_;
};

// Takes the statistics vector and drops it: every output of IspOp must be connected.
class DropOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(stats, "stats").queue_depth(2U); }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{stats});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = stats.receive();
    return sample ? holoscan::expected<void, holoscan::Error>{}
                  : holoscan::make_unexpected(std::move(sample).error());
  }
  holoscan::Input<holoscan::Tensor> stats;
};

}  // namespace

int main() {
  holoscan::Graph graph{"isp_consumer"};
  auto tally = std::make_shared<Tally>();
  auto source = graph.op<RawSourceOp>(
      "source", std::shared_ptr<const std::atomic<std::uint32_t>>(tally, &tally->counted));
  isp::IspOpConfig config;
  config.params = isp::synthetic_isp_params(camera(0));
  config.stats = true;
  config.max_width = kWidth;
  config.max_height = kHeight;
  auto processor = graph.op<isp::IspOp>("isp", config);
  auto remaining = std::make_shared<std::atomic<int>>(2);
  auto image_check = graph.op<ImageCheckOp>("image_check", tally, remaining);
  auto histogram_check = graph.op<HistogramCheckOp>("histogram_check", tally, remaining);
  auto drop = graph.op<DropOp>("drop");
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->raw, processor->raw, lossless);
  graph.add_flow(processor->image, image_check->image, lossless);
  graph.add_flow(processor->histogram, histogram_check->histogram, lossless);
  graph.add_flow(processor->stats, drop->stats, lossless);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // IspOp's outputs are in device memory, on GPU 0.
  holoscan::CompileOptions options;
  for (const char* port : {"image", "histogram", "stats"}) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "isp", .output_port = port, .device = holoscan::DeviceId{0}});
  }
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (const int status = holoscan::run(plan); status != 0) {
    std::cerr << "isp_consumer: run failed with " << status << '\n';
    return 1;
  }
  const bool passed = tally->frames == kFrames && tally->worst_psnr > 25.0 &&
                      tally->stats_frames == kFrames && tally->bad_histograms == 0;
  std::cout << "isp_consumer: frames=" << tally->frames << " worst_psnr=" << tally->worst_psnr
            << " histogram_frames=" << tally->stats_frames
            << " bad_histograms=" << tally->bad_histograms << '\n'
            << "isp_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
