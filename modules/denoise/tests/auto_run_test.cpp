// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A graph with DenoiseOp on Backend::kAuto, run with a plain
// holoscan::run(plan) in a fresh process. That run takes over SIGINT and
// SIGTERM only while the process has one thread, so nothing may start one
// before it: building the graph must not probe VPI. Also checks the backend
// and engine auto chose: on a CI runner (HOLOSCAN_CI_PLATFORM), VPI's TNR on
// the VIC on a Jetson whose build has VPI, the cuda backend on x86_64.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/denoise/denoise_op.hpp>
#include <holoscan/denoise/synthetic_scene.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace dn = holoscan::denoise;

constexpr int kWidth = 320;
constexpr int kHeight = 240;
constexpr std::uint32_t kFrames = 6;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
const dn::LowLightScene kScene{.width = kWidth, .height = kHeight, .speed = 0.0F};

// kFrames noisy frames rendered on the GPU, one per tick; a frame a busy
// consumer refuses goes again on the next tick.
class SourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(image, "image")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kCudaDevice,
                    .dtype = kUInt8,
                    .rank = 3U,
                    .strides = holoscan::TensorStridePolicy::kContiguousOnly,
                },
            .bounds = holoscan::tensor_bounds(std::uint64_t{kWidth} * kHeight * 3U),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        })
        .payload_options(holoscan::PayloadOptions{
            .schema = holoscan::schema_identity<holoscan::schema::ImageT>()});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{10}});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (emitted_ >= kFrames) return {};
    const auto busy = [](const holoscan::Error& error) {
      return error.code == holoscan::ErrorCode::kBackpressured ||
             error.code == holoscan::ErrorCode::kResourceExhausted;
    };
    const std::array<std::int64_t, 3> shape{kHeight, kWidth, 3};
    auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
    if (!loan) {
      return busy(loan.error()) ? holoscan::expected<void, holoscan::Error>{}
                                : holoscan::make_unexpected(std::move(loan).error());
    }
    cudaStream_t stream = context.cuda_stream().get();
    auto writer = loan->write(stream);
    if (!writer) return holoscan::make_unexpected(std::move(writer).error());
    dn::ImageView view{.format = dn::PixelFormat::kRgb8, .width = kWidth, .height = kHeight};
    view.planes[0] = dn::Plane{.data = writer->data_as<std::uint8_t>(),
                               .pitch = static_cast<std::size_t>(kWidth) * 3U};
    if (dn::render_noisy(kScene, dn::NoiseModel{}, static_cast<int>(emitted_), view, stream) !=
        cudaSuccess) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "render_noisy failed"});
    }
    if (auto committed = std::move(*writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    holoscan::schema::ImageT descriptor;
    descriptor.width = kWidth;
    descriptor.height = kHeight;
    descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
    auto sent = image.emit_tensor(
        std::move(*loan), std::move(descriptor),
        holoscan::EmitOptions{.capture_time = context.activation_time(), .frame_id = emitted_ + 1});
    if (!sent) return busy(sent.error()) ? holoscan::expected<void, holoscan::Error>{} : sent;
    ++emitted_;
    return {};
  }

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  std::uint32_t emitted_{};
};

// Counts the denoised frames and completes the run at the last.
class SinkOp final : public holoscan::Operator<> {
 public:
  explicit SinkOp(std::shared_ptr<std::atomic<std::uint32_t>> count) : count_(std::move(count)) {}

  // Two slots: one frame in compute() while the next one arrives.
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
    if (count_->fetch_add(1U) + 1U == kFrames) context.request_completion();
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> image;

 private:
  std::shared_ptr<std::atomic<std::uint32_t>> count_;
};

}  // namespace

int main() {
  holoscan::Graph graph{"denoise_auto_run"};
  auto stats = std::make_shared<dn::DenoiseStats>();
  auto count = std::make_shared<std::atomic<std::uint32_t>>(0U);
  dn::DenoiseOpConfig config{.backend = dn::Backend::kAuto,
                             .max_width = kWidth,
                             .max_height = kHeight,
                             .stats = stats};
  // The settings VPI's TNR runs: auto may choose it.
  config.params.spatial = dn::SpatialFilter::kNone;
  auto source = graph.op<SourceOp>("source");
  auto denoiser = graph.op<dn::DenoiseOp>("denoise", config);
  auto sink = graph.op<SinkOp>("sink", count);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->image, denoiser->image, lossless);
  graph.add_flow(denoiser->denoised, sink->image, lossless);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  graph.partition("denoise_auto_run").include_all();

  holoscan::CompileOptions options;
  for (const auto& [op, port] : {std::pair{"source", "image"}, std::pair{"denoise", "denoised"}}) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
  }
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
  }
  if (!plan.ok()) return 1;
  // Still one thread: building and compiling the graph probed nothing.
  const int status = holoscan::run(plan);

  const dn::Backend backend = stats->backend.load();
  const std::string engine = stats->engine.load();
  std::cout << "denoise_auto_run: run=" << status << " frames=" << count->load()
            << " backend=" << dn::backend_name(backend) << " engine=" << engine << '\n';
  bool passed = status == 0 && count->load() == kFrames && stats->frames.load() == kFrames &&
                backend != dn::Backend::kAuto;
  const char* platform = std::getenv("HOLOSCAN_CI_PLATFORM");
  if (platform != nullptr) {
    const auto built = dn::available_backends();
    const bool vpi_built = std::find(built.begin(), built.end(), dn::Backend::kVpi) != built.end();
    const std::string_view expected =
        std::string_view(platform) == "jetson" && vpi_built ? "vpi-vic" : "cuda";
    std::cout << "denoise_auto_run: expected engine " << expected << " (" << platform << ")\n";
    passed = passed && engine == expected;
  }
  std::cout << "denoise_auto_run: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
