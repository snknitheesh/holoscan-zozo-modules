// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-codec package: it runs a graph on one worker,
// test video frames into VideoEncoderOp, its access units into VideoDecoderOp, and checks that
// every decoded image comes back, in order, with its header and close to its source. Exits 77
// (skipped) when no backend can encode and decode, except in CI (HOLOSCAN_CI_PLATFORM set), where
// that fails.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/codec/test_video.hpp>
#include <holoscan/codec/video_codec.hpp>
#include <holoscan/codec/video_decoder_op.hpp>
#include <holoscan/codec/video_encoder_op.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace hc = holoscan::codec;
using std::chrono_literals::operator""ms;
constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr std::uint32_t kFrames = 10;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

// kFrames frames of the test video as RGB8 images in host memory, frame n with device_sequence
// n + 1; one the encoder has no room for goes again on the next tick.
class FrameSourceOp final : public holoscan::Operator<> {
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
    result.trigger(holoscan::OnClock{.period = 10ms});
    return result;
  }
  void start() override { sent_ = 0; }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (sent_ >= kFrames) return {};
    const auto pixels = hc::render_test_frame(kWidth, kHeight, sent_);
    auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = std::array<std::int64_t, 3>{kHeight, kWidth, 3}, .dtype = kUInt8});
    if (!loan) return holoscan::make_unexpected(std::move(loan).error());
    auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                   -> holoscan::expected<void, holoscan::Error> {
      std::memcpy(writer.data_as<std::uint8_t>(), pixels.data(), pixels.size());
      return {};
    });
    if (!written) return holoscan::make_unexpected(std::move(written).error());
    holoscan::schema::ImageT descriptor;
    descriptor.width = kWidth;
    descriptor.height = kHeight;
    descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
    descriptor.header = std::make_shared<holoscan::schema::HeaderT>();
    descriptor.header->device_sequence = sent_ + 1U;
    descriptor.header->capture_timestamp_ns = context.activation_time().timestamp_ns;
    auto sent = image.emit_tensor(std::move(*loan), descriptor);
    if (!sent) {
      return sent.error().code == holoscan::ErrorCode::kBackpressured
                 ? holoscan::expected<void, holoscan::Error>{}
                 : sent;
    }
    ++sent_;
    return {};
  }
  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  std::uint32_t sent_{};
};

struct Result {
  std::uint32_t images{};
  bool in_order{true};
  double worst_psnr{1e9};
};

// Checks each decoded image against its source frame, and completes the run after the last.
class CheckOp final : public holoscan::Operator<> {
 public:
  explicit CheckOp(std::shared_ptr<Result> result) : result_(std::move(result)) {}
  void setup(holoscan::OperatorSpec& spec) override { spec.input(image, "image"); }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = image.receive();
    if (!sample) return holoscan::make_unexpected(std::move(sample).error());
    const holoscan::schema::ImageT& picture = sample->data;
    const std::uint64_t sequence = picture.header ? picture.header->device_sequence : 0U;
    result_->in_order = result_->in_order && sequence == result_->images + 1U &&
                        picture.width == kWidth && picture.height == kHeight;
    if (auto ready = picture.data->synchronize_for_host(); !ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
    std::vector<std::uint8_t> host(std::size_t{kWidth} * kHeight * 3);
    if (cudaMemcpy(host.data(), picture.data->data(), host.size(), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "CheckOp: cudaMemcpy"});
    }
    const auto source =
        hc::render_test_frame(kWidth, kHeight, static_cast<std::uint32_t>(sequence - 1U));
    result_->worst_psnr = std::min(result_->worst_psnr, hc::psnr(source, host, kWidth, kHeight, 3));
    if (++result_->images == kFrames) context.request_completion();
    return {};
  }
  holoscan::Input<holoscan::schema::ImageT> image;

 private:
  std::shared_ptr<Result> result_;
};

bool usable() {
  const auto any = [](const std::vector<hc::BackendStatus>& statuses) {
    return std::any_of(statuses.begin(), statuses.end(),
                       [](const hc::BackendStatus& status) { return status.available; });
  };
  return any(hc::probe_encoder_backends(hc::Codec::kH264)) &&
         any(hc::probe_decoder_backends(hc::Codec::kH264));
}

}  // namespace

int main() try {
  if (!usable()) {
    const bool ci = std::getenv("HOLOSCAN_CI_PLATFORM") != nullptr;
    std::cout << "codec_consumer: " << (ci ? "FAIL" : "SKIPPED") << ", no H.264 backend here\n";
    return ci ? 1 : 77;
  }
  auto result = std::make_shared<Result>();
  holoscan::Graph graph{"codec_consumer"};
  auto source = graph.op<FrameSourceOp>("source");
  hc::VideoEncoderConfig encoding{.max_width = kWidth, .max_height = kHeight};
  encoding.settings.backend = hc::Backend::kAuto;
  encoding.settings.rate_control = hc::RateControl::kConstQp;
  encoding.settings.qp = 22;
  auto encoder = graph.op<hc::VideoEncoderOp>("encoder", encoding);
  auto decoder = graph.op<hc::VideoDecoderOp>(
      "decoder", hc::VideoDecoderConfig{
                     .backend = hc::Backend::kAuto, .max_width = kWidth, .max_height = kHeight});
  auto check = graph.op<CheckOp>("check", result);
  // Lossless edges: every access unit is needed to decode the next.
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking,
                                             .queue_depth = 4U};
  graph.add_flow(source->image, encoder->image, lossless);
  graph.add_flow(encoder->bitstream, decoder->bitstream, lossless);
  graph.add_flow(decoder->image, check->image, lossless);
  // One worker for the whole lossless graph.
  graph.partition("codec_consumer").include_all();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "decoder", .output_port = "image", .device = holoscan::DeviceId{0}});
  const auto plan = holoscan::compile(graph, std::move(options));
  if (!plan.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
  }
  if (!plan.ok()) return 1;
  const auto workers = plan.scheduler_plan().worker_count;
  auto session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{120};
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(5ms);
  }
  if (session.is_running()) session.request_stop();
  session.wait();
  if (!session.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(session.diagnostics(), plan.graph()) << '\n';
  }
  const bool passed = session.termination() == holoscan::RunTermination::kRequestedCompleted &&
                      workers == 1 && result->images == kFrames && result->in_order &&
                      result->worst_psnr > 35.0;
  std::cout << "codec_consumer: workers=" << workers << " images=" << result->images << '/'
            << kFrames << " in_order=" << result->in_order
            << " worst_psnr_db=" << result->worst_psnr << ": " << (passed ? "PASS" : "FAIL")
            << '\n';
  return passed ? 0 : 1;
} catch (const std::exception& error) {
  std::cerr << "codec_consumer: " << error.what() << '\n';
  return 1;
}
