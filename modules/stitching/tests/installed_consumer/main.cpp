// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-stitching package: a source
// renders the synthetic world through the four surround cameras of a moving
// rig, in CUDA device memory; StitchOp makes a bird's-eye view of them; a sink
// checks every view against the directly rendered ground truth.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
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
#include <holoscan/stitching/stitch_op.hpp>
#include <holoscan/stitching/synthetic_world.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace st = holoscan::stitching;

constexpr int kWidth = 640;
constexpr int kHeight = 360;
constexpr std::uint32_t kFrames = 5;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

// kFrames frame sets of the surround rig, RGB8 on the GPU, then idle.
class RigSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    for (std::size_t c = 0; c < images.size(); ++c) {
      spec.output(images[c], names_[c])
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
              .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
          });
    }
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{20}});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (sent_ >= kFrames) {
      return {};
    }
    cudaStream_t stream = context.cuda_stream().get();
    const auto capture = context.activation_time();
    for (std::size_t c = 0; c < images.size(); ++c) {
      auto loan = images[c].allocate_tensor(holoscan::TensorLoanRequest{
          .shape = std::array<std::int64_t, 3>{kHeight, kWidth, 3}, .dtype = kUInt8});
      if (!loan) {
        return holoscan::make_unexpected(std::move(loan).error());
      }
      auto writer = loan->write(stream);
      if (!writer) {
        return holoscan::make_unexpected(std::move(writer).error());
      }
      if (const cudaError_t status =
              st::render_camera(world_, rig_.cameras[c], st::moving_pose(10U * sent_), 1.0F,
                                st::PixelFormat::kRgb8, writer->data(), kWidth * 3U, 3, stream);
          status != cudaSuccess) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, cudaGetErrorString(status)});
      }
      if (auto committed = std::move(*writer).commit(); !committed) {
        return holoscan::make_unexpected(std::move(committed).error());
      }
      holoscan::schema::ImageT descriptor;
      descriptor.width = kWidth;
      descriptor.height = kHeight;
      descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
      if (auto sent = images[c].emit_tensor(
              std::move(*loan), descriptor,
              holoscan::EmitOptions{.capture_time = capture, .frame_id = sent_ + 1U});
          !sent) {
        return sent;
      }
    }
    ++sent_;
    return {};
  }

  std::array<holoscan::Output<holoscan::schema::ImageT>, 4> images;

 private:
  st::SyntheticWorld world_;
  st::CameraRig rig_ = st::surround_rig(kWidth, kHeight);
  std::array<std::string, 4> names_{"front", "left", "rear", "right"};
  std::uint32_t sent_{};
};

struct Tally {
  std::uint32_t frames{};
  double worst_psnr{99.0};
  double worst_corner{};
  double worst_coverage{1.0};
};

// Checks each bird's-eye view against the truth of its frame.
class CheckOp final : public holoscan::Operator<> {
 public:
  CheckOp(std::shared_ptr<Tally> tally, st::OutputGeometry geometry)
      : tally_(std::move(tally)), geometry_(geometry) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(stitched, "stitched").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{stitched});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto image = stitched.receive();
    if (!image) {
      return holoscan::make_unexpected(std::move(image).error());
    }
    const std::size_t pitch = static_cast<std::size_t>(geometry_.width) * 4U;
    std::vector<std::uint8_t> got(pitch * geometry_.height);
    if (auto copied = image->data.data->copy_to_host(got.data(), got.size(), context.cuda_stream());
        !copied) {
      return copied;
    }
    const st::RigPose pose =
        st::moving_pose(10U * static_cast<std::uint32_t>(image->metadata.frame_id - 1U));
    void* truth_device = nullptr;
    std::vector<std::uint8_t> truth(got.size());
    if (cudaMalloc(&truth_device, truth.size()) != cudaSuccess ||
        st::render_truth(world_, geometry_, pose, truth_device, pitch, 3, nullptr) != cudaSuccess ||
        cudaMemcpy(truth.data(), truth_device, truth.size(), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
      cudaFree(truth_device);
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "rendering the truth failed"});
    }
    cudaFree(truth_device);
    const auto corners = st::checker_corners(world_, geometry_, pose);
    if (resolution_.empty()) {
      resolution_ =
          st::resolution_map(st::build_remap_tables(st::surround_rig(kWidth, kHeight), geometry_));
    }
    const st::Comparison c =
        st::compare_with_truth(got.data(), pitch, truth.data(), pitch, geometry_.width,
                               geometry_.height, corners, 2, resolution_);
    tally_->worst_psnr = std::min(tally_->worst_psnr, c.psnr_db);
    tally_->worst_corner = std::max(tally_->worst_corner, c.max_corner_error);
    tally_->worst_coverage = std::min(tally_->worst_coverage, c.coverage);
    if (++tally_->frames == kFrames) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> stitched;

 private:
  std::shared_ptr<Tally> tally_;
  st::OutputGeometry geometry_;
  st::SyntheticWorld world_;
  std::vector<float> resolution_;
};

}  // namespace

int main() {
  const st::OutputGeometry geometry = st::ground_view(256, 256, 5.5);
  holoscan::Graph graph{"stitching_consumer"};
  auto source = graph.op<RigSourceOp>("source");
  auto stitch = graph.op<st::StitchOp>(
      "stitch", st::StitchConfig{.rig = st::surround_rig(kWidth, kHeight), .output = geometry});
  auto tally = std::make_shared<Tally>();
  auto check = graph.op<CheckOp>("check", tally, geometry);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  for (std::size_t c = 0; c < 4; ++c) {
    graph.add_flow(source->images[c], stitch->images[c], lossless);
  }
  graph.add_flow(stitch->stitched, check->stitched, lossless);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  for (const char* port : {"front", "left", "rear", "right"}) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "source", .output_port = port, .device = holoscan::DeviceId{0}});
  }
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "stitch", .output_port = "stitched", .device = holoscan::DeviceId{0}});
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (const int status = holoscan::run(plan); status != 0) {
    std::cerr << "stitching_consumer: run failed with " << status << '\n';
    return 1;
  }
  const bool passed = tally->frames == kFrames && tally->worst_psnr >= 24.0 &&
                      tally->worst_corner <= 1.0 && tally->worst_coverage >= 0.9;
  std::cout << "stitching_consumer: frames=" << tally->frames << " worst_psnr=" << tally->worst_psnr
            << " dB worst_corner=" << tally->worst_corner
            << " px worst_coverage=" << tally->worst_coverage << '\n'
            << "stitching_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
