// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-denoise package: a source
// sends noisy frames of the synthetic low-light scene and three exposure
// brackets of the HDR scene, all in host memory; DenoiseOp and ExposureFusionOp
// process them on the GPU; one sink checks that the denoised frames gain PSNR
// over the noisy ones, another that the fused image keeps the shadow and the
// window unclipped. The source keeps one frame in flight.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
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
#include <holoscan/denoise/denoise_op.hpp>
#include <holoscan/denoise/exposure_fusion_op.hpp>
#include <holoscan/denoise/synthetic_scene.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace dn = holoscan::denoise;

constexpr int kWidth = 320;
constexpr int kHeight = 240;
constexpr std::uint32_t kFrames = 8;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr dn::NoiseModel kNoise{.read_sigma = 6.0F, .shot_gain = 1.0F, .seed = 5};

const dn::LowLightScene kScene{.width = kWidth, .height = kHeight, .speed = 0.0F};
const dn::HdrScene kHdr{.width = kWidth, .height = kHeight};

holoscan::TensorOutputSpec host_rgb() {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kHost,
              .dtype = kUInt8,
              .rank = 3U,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(std::uint64_t{kWidth} * kHeight * 3U),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

// kFrames frames on each output, then idle; each once the sinks have counted the
// one before (`counted` goes up by two per frame).
class SourceOp final : public holoscan::Operator<> {
 public:
  explicit SourceOp(std::shared_ptr<const std::atomic<std::uint32_t>> counted)
      : counted_(std::move(counted)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    const holoscan::PayloadOptions image{.schema =
                                             holoscan::schema_identity<holoscan::schema::ImageT>()};
    spec.output(noisy, "noisy").produces_tensor(host_rgb()).payload_options(image);
    spec.output(bracket0, "bracket0").produces_tensor(host_rgb()).payload_options(image);
    spec.output(bracket1, "bracket1").produces_tensor(host_rgb()).payload_options(image);
    spec.output(bracket2, "bracket2").produces_tensor(host_rgb()).payload_options(image);
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
    const holoscan::EmitOptions options{.capture_time = context.activation_time(),
                                        .frame_id = emitted_ + 1U};
    const int frame = static_cast<int>(emitted_);
    if (auto sent = send(noisy, dn::render_noisy(kScene, kNoise, frame), options); !sent) {
      return sent;
    }
    std::array<holoscan::Output<holoscan::schema::ImageT>*, 3> brackets{&bracket0, &bracket1,
                                                                        &bracket2};
    for (int k = 0; k < 3; ++k) {
      if (auto sent =
              send(*brackets[k], dn::render_bracket(kHdr, 0, dn::kBracketExposures[k]), options);
          !sent) {
        return sent;
      }
    }
    ++emitted_;
    return {};
  }

  holoscan::Output<holoscan::schema::ImageT> noisy;
  holoscan::Output<holoscan::schema::ImageT> bracket0;
  holoscan::Output<holoscan::schema::ImageT> bracket1;
  holoscan::Output<holoscan::schema::ImageT> bracket2;

 private:
  static holoscan::expected<void, holoscan::Error> send(
      holoscan::Output<holoscan::schema::ImageT>& port, const std::vector<std::uint8_t>& rgb,
      const holoscan::EmitOptions& options) {
    const std::array<std::int64_t, 3> shape{kHeight, kWidth, 3};
    auto loan = port.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
    if (!loan) return holoscan::make_unexpected(std::move(loan).error());
    auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                   -> holoscan::expected<void, holoscan::Error> {
      std::memcpy(writer.data_as<std::uint8_t>(), rgb.data(), rgb.size());
      return {};
    });
    if (!written) return holoscan::make_unexpected(std::move(written).error());
    holoscan::schema::ImageT descriptor;
    descriptor.width = kWidth;
    descriptor.height = kHeight;
    descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
    return port.emit_tensor(std::move(*loan), descriptor, options);
  }

  std::shared_ptr<const std::atomic<std::uint32_t>> counted_;
  std::uint32_t emitted_{};
};

struct Tally {
  std::atomic<std::uint32_t> counted{};
  std::uint32_t denoised{};
  double noisy_psnr{};
  double last_psnr{};
  std::uint32_t fused{};
  std::uint32_t bad_fused{};
};

void finish(std::atomic<int>& remaining, holoscan::ExecutionContext& context) {
  if (remaining.fetch_sub(1) == 1) context.request_completion();
}

holoscan::expected<std::vector<std::uint8_t>, holoscan::Error> download(
    const holoscan::schema::ImageT& image, holoscan::ExecutionContext& context) {
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(kWidth) * kHeight * 3U);
  if (!image.data || image.encoding != holoscan::schema::ImageEncoding_RGB8) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "expected an RGB8 image"});
  }
  if (auto copied = image.data->copy_to_host(rgb.data(), rgb.size(), context.cuda_stream());
      !copied) {
    return holoscan::make_unexpected(std::move(copied).error());
  }
  return rgb;
}

// PSNR of each denoised frame against the clean scene.
class DenoisedCheckOp final : public holoscan::Operator<> {
 public:
  DenoisedCheckOp(std::shared_ptr<Tally> tally, std::shared_ptr<std::atomic<int>> remaining)
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
    if (!sample) return holoscan::make_unexpected(std::move(sample).error());
    auto rgb = download(sample->data, context);
    if (!rgb) return holoscan::make_unexpected(std::move(rgb).error());
    const int frame = static_cast<int>(sample->metadata.frame_id) - 1;
    const auto clean = dn::render_clean(kScene, frame);
    const auto noisy = dn::render_noisy(kScene, kNoise, frame);
    tally_->noisy_psnr = dn::psnr(noisy.data(), clean.data(), clean.size());
    tally_->last_psnr = dn::psnr(rgb->data(), clean.data(), clean.size());
    tally_->counted.fetch_add(1U, std::memory_order_release);
    if (++tally_->denoised == kFrames) finish(*remaining_, context);
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> image;

 private:
  std::shared_ptr<Tally> tally_;
  std::shared_ptr<std::atomic<int>> remaining_;
};

// The fused image keeps the shadow and the window: none of it clipped or black,
// 16 pixels from their edges and the image's, where the blend may leave a halo.
class FusedCheckOp final : public holoscan::Operator<> {
 public:
  FusedCheckOp(std::shared_ptr<Tally> tally, std::shared_ptr<std::atomic<int>> remaining)
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
    if (!sample) return holoscan::make_unexpected(std::move(sample).error());
    auto rgb = download(sample->data, context);
    if (!rgb) return holoscan::make_unexpected(std::move(rgb).error());
    constexpr int kMargin = 16;
    bool bad = false;
    for (std::size_t i = 0; i < rgb->size(); ++i) {
      const int x = static_cast<int>(i / 3 % kWidth);
      const int y = static_cast<int>(i / 3 / kWidth);
      const dn::HdrRegion region = dn::hdr_region(kHdr, x);
      if (region == dn::HdrRegion::kMidtones || y < kMargin || y + kMargin >= kHeight ||
          x < kMargin || x + kMargin >= kWidth || dn::hdr_region(kHdr, x - kMargin) != region ||
          dn::hdr_region(kHdr, x + kMargin) != region) {
        continue;
      }
      bad = bad || (*rgb)[i] >= 254 || (*rgb)[i] <= 1;
    }
    tally_->bad_fused += bad ? 1U : 0U;
    tally_->counted.fetch_add(1U, std::memory_order_release);
    if (++tally_->fused == kFrames) finish(*remaining_, context);
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> image;

 private:
  std::shared_ptr<Tally> tally_;
  std::shared_ptr<std::atomic<int>> remaining_;
};

}  // namespace

int main() {
  holoscan::Graph graph{"denoise_consumer"};
  auto tally = std::make_shared<Tally>();
  auto source = graph.op<SourceOp>(
      "source", std::shared_ptr<const std::atomic<std::uint32_t>>(tally, &tally->counted));
  auto denoiser = graph.op<dn::DenoiseOp>(
      "denoise", dn::DenoiseOpConfig{.max_width = kWidth, .max_height = kHeight});
  auto fusion = graph.op<dn::ExposureFusionOp>(
      "fusion", dn::ExposureFusionOpConfig{.max_width = kWidth, .max_height = kHeight});
  auto remaining = std::make_shared<std::atomic<int>>(2);
  auto denoised_check = graph.op<DenoisedCheckOp>("denoised_check", tally, remaining);
  auto fused_check = graph.op<FusedCheckOp>("fused_check", tally, remaining);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->noisy, denoiser->image, lossless);
  graph.add_flow(source->bracket0, fusion->bracket0, lossless);
  graph.add_flow(source->bracket1, fusion->bracket1, lossless);
  graph.add_flow(source->bracket2, fusion->bracket2, lossless);
  graph.add_flow(denoiser->denoised, denoised_check->image, lossless);
  graph.add_flow(fusion->image, fused_check->image, lossless);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // The operators' outputs are in device memory, on GPU 0.
  holoscan::CompileOptions options;
  for (const auto& [op, port] : {std::pair{"denoise", "denoised"}, std::pair{"fusion", "image"}}) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
  }
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (const int status = holoscan::run(plan); status != 0) {
    std::cerr << "denoise_consumer: run failed with " << status << '\n';
    return 1;
  }
  const double gain = tally->last_psnr - tally->noisy_psnr;
  const bool passed =
      tally->denoised == kFrames && gain > 8.0 && tally->fused == kFrames && tally->bad_fused == 0;
  std::cout << "denoise_consumer: denoised=" << tally->denoised << " gain_db=" << gain
            << " fused=" << tally->fused << " bad_fused=" << tally->bad_fused << '\n'
            << "denoise_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
