// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-stereo-disparity package: a
// source sends frames of the synthetic stereo scene as MONO8 pairs in host
// memory, one StereoDisparityOp per backend this platform has matches them,
// and a sink per backend checks each disparity map against the scene's truth.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
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
#include <holoscan/stereo_disparity/platform.hpp>
#include <holoscan/stereo_disparity/stereo_disparity_op.hpp>
#include <holoscan/stereo_disparity/synthetic_scene.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace sd = holoscan::stereo_disparity;

constexpr int kWidth = 320;
constexpr int kHeight = 240;
constexpr std::uint32_t kFrames = 3;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

// The scene's frame `frame` as MONO8 left and right images, on the host.
std::array<std::vector<std::uint8_t>, 2> render(std::uint32_t frame) {
  const std::size_t pixels = static_cast<std::size_t>(kWidth) * kHeight;
  std::array<std::vector<std::uint8_t>, 2> pair{std::vector<std::uint8_t>(pixels),
                                                std::vector<std::uint8_t>(pixels)};
  std::uint8_t* gpu = nullptr;
  if (cudaMalloc(&gpu, 2 * pixels) != cudaSuccess) return {};
  const bool ok =
      sd::render_scene(sd::make_scene(frame, kWidth, kHeight), 1, gpu, gpu + pixels, nullptr) ==
          cudaSuccess &&
      cudaMemcpy(pair[0].data(), gpu, pixels, cudaMemcpyDeviceToHost) == cudaSuccess &&
      cudaMemcpy(pair[1].data(), gpu + pixels, pixels, cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaFree(gpu);
  return ok ? pair : std::array<std::vector<std::uint8_t>, 2>{};
}

// kFrames stereo pairs, MONO8 in host memory, then idle.
class PairSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    for (auto* port : {&left, &right}) {
      spec.output(*port, port == &left ? "left" : "right")
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
              .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
          });
    }
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{50}});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (emitted_ >= kFrames) {
      return {};
    }
    // A frame keeps its capture time until both eyes are out.
    if (!capture_) {
      capture_ = context.activation_time();
      pair_ = render(emitted_);
      if (pair_[0].empty()) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, "rendering the scene failed"});
      }
    }
    for (int side = 0; side < 2; ++side) {
      if (sent_[side]) continue;
      holoscan::Output<holoscan::schema::ImageT>& port = side == 0 ? left : right;
      auto loan = port.allocate_tensor(holoscan::TensorLoanRequest{
          .shape = std::array<std::int64_t, 2>{kHeight, kWidth}, .dtype = kUInt8});
      if (!loan) {
        return holoscan::make_unexpected(std::move(loan).error());
      }
      auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                     -> holoscan::expected<void, holoscan::Error> {
        std::copy(pair_[side].begin(), pair_[side].end(), writer.data_as<std::uint8_t>());
        return {};
      });
      if (!written) {
        return written;
      }
      holoscan::schema::ImageT descriptor;
      descriptor.width = kWidth;
      descriptor.height = kHeight;
      descriptor.encoding = holoscan::schema::ImageEncoding_MONO8;
      auto emitted = port.emit_tensor(
          std::move(*loan), descriptor,
          holoscan::EmitOptions{.capture_time = *capture_, .frame_id = emitted_ + 1U});
      if (!emitted) {
        // A busy consumer refuses the frame; send it again on the next tick.
        return emitted.error().code == holoscan::ErrorCode::kBackpressured
                   ? holoscan::expected<void, holoscan::Error>{}
                   : emitted;
      }
      sent_[side] = true;
    }
    ++emitted_;
    capture_.reset();
    sent_ = {};
    return {};
  }

  holoscan::Output<holoscan::schema::ImageT> left;
  holoscan::Output<holoscan::schema::ImageT> right;

 private:
  std::uint32_t emitted_{};
  std::optional<holoscan::TimePoint> capture_;
  std::array<std::vector<std::uint8_t>, 2> pair_;
  std::array<bool, 2> sent_{};
};

struct Tally {
  std::uint32_t frames{};
  std::uint64_t truth_pixels{};
  std::uint64_t within_1px{};
};

// Checks each disparity map against the frame it came from; the last sink to
// see kFrames frames ends the run.
class CheckOp final : public holoscan::Operator<> {
 public:
  CheckOp(std::shared_ptr<Tally> tally, std::shared_ptr<std::atomic<int>> remaining)
      : tally_(std::move(tally)), remaining_(std::move(remaining)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(disparity, "disparity").queue_depth(2U);
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{disparity});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto image = disparity.receive();
    if (!image) {
      return holoscan::make_unexpected(std::move(image).error());
    }
    std::vector<float> values(static_cast<std::size_t>(kWidth) * kHeight);
    if (auto copied = image->data.data->copy_to_host(values.data(), values.size() * sizeof(float),
                                                     context.cuda_stream());
        !copied) {
      return copied;
    }
    const sd::SyntheticScene scene =
        sd::make_scene(static_cast<std::uint32_t>(image->metadata.frame_id - 1U), kWidth, kHeight);
    for (int v = 0; v < kHeight; ++v) {
      for (int u = 0; u < kWidth; ++u) {
        const float truth = sd::true_disparity(scene, static_cast<float>(u), static_cast<float>(v));
        if (truth < 0.0F) continue;
        ++tally_->truth_pixels;
        const float d = values[static_cast<std::size_t>(v) * kWidth + u];
        tally_->within_1px += d > 0.0F && std::abs(d - truth) <= 1.0F;
      }
    }
    if (++tally_->frames == kFrames && remaining_->fetch_sub(1) == 1) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> disparity;

 private:
  std::shared_ptr<Tally> tally_;
  std::shared_ptr<std::atomic<int>> remaining_;
};

}  // namespace

int main() {
  const std::vector<sd::Backend> backends = sd::available_backends();
  holoscan::Graph graph{"stereo_disparity_consumer"};
  auto source = graph.op<PairSourceOp>("source");
  auto remaining = std::make_shared<std::atomic<int>>(static_cast<int>(backends.size()));
  std::vector<std::shared_ptr<Tally>> tallies;
  holoscan::CompileOptions options;
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  for (sd::Backend backend : backends) {
    const std::string name(sd::backend_name(backend));
    auto matcher = graph.op<sd::StereoDisparityOp>(
        "stereo_" + name,
        sd::StereoDisparityConfig{.backend = backend, .max_width = kWidth, .max_height = kHeight});
    tallies.push_back(std::make_shared<Tally>());
    auto check = graph.op<CheckOp>("check_" + name, tallies.back(), remaining);
    graph.add_flow(source->left, matcher->left, lossless);
    graph.add_flow(source->right, matcher->right, lossless);
    graph.add_flow(matcher->disparity, check->disparity, lossless);
    // The disparity is in device memory, on GPU 0.
    options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = "stereo_" + name,
                                              .output_port = "disparity",
                                              .device = holoscan::DeviceId{0}});
  }
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  // VPI, asked for the platform's backends above, has started threads of its
  // own, so the SDK's managed signal handling would refuse: this short test
  // leaves signals to the process.
  if (const int status = holoscan::run(plan, holoscan::RunSignalPolicy::kCallerManaged);
      status != 0) {
    std::cerr << "stereo_disparity_consumer: run failed with " << status << '\n';
    return 1;
  }
  bool passed = true;
  for (std::size_t i = 0; i < backends.size(); ++i) {
    const Tally& tally = *tallies[i];
    const double within =
        tally.truth_pixels == 0 ? 0.0 : static_cast<double>(tally.within_1px) / tally.truth_pixels;
    const bool ok = tally.frames == kFrames && within >= 0.8;
    passed = passed && ok;
    std::cout << "stereo_disparity_consumer: backend=" << sd::backend_name(backends[i])
              << " frames=" << tally.frames << " within_1px=" << within << '\n';
  }
  std::cout << "stereo_disparity_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
