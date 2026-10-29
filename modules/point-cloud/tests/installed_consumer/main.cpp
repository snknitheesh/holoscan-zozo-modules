// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-point-cloud package: a source
// sends frames of the synthetic depth camera as uint16 millimeters in host
// memory, DepthToPointCloudOp deprojects them on the GPU, and a sink checks
// every point against the camera.

#include <array>
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
#include <holoscan/point_cloud/depth_to_point_cloud_op.hpp>
#include <holoscan/point_cloud/synthetic_depth.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace pc = holoscan::point_cloud;

constexpr int kWidth = 160;
constexpr int kHeight = 120;
constexpr std::uint32_t kFrames = 5;
constexpr DLDataType kUInt16{kDLUInt, 16U, 1U};
constexpr float kDepthScale = 0.001F;
constexpr float kDepthMin = 0.1F;
constexpr float kDepthMax = 10.0F;

// kFrames synthetic depth images, uint16 millimeters in host memory, then idle.
class DepthSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(depth, "depth")
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
    if (emitted_ >= kFrames) {
      return {};
    }
    const pc::SyntheticFrame frame = pc::synthetic_frame(kWidth, kHeight, emitted_, false);
    const std::array<std::int64_t, 2> shape{kHeight, kWidth};
    auto loan =
        depth.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt16});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                   -> holoscan::expected<void, holoscan::Error> {
      auto* pixels = writer.data_as<std::uint16_t>();
      for (int v = 0; v < kHeight; ++v) {
        for (int u = 0; u < kWidth; ++u) {
          pixels[v * kWidth + u] = pc::synthetic_depth_mm(frame, u, v);
        }
      }
      return {};
    });
    if (!written) {
      return holoscan::make_unexpected(std::move(written).error());
    }
    holoscan::schema::ImageT descriptor;
    descriptor.width = kWidth;
    descriptor.height = kHeight;
    descriptor.encoding = holoscan::schema::ImageEncoding_MONO16;
    auto emitted =
        depth.emit_tensor(std::move(*loan), descriptor,
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

  holoscan::Output<holoscan::schema::ImageT> depth;

 private:
  std::uint32_t emitted_{};
};

struct Tally {
  std::uint32_t frames{};
  std::uint64_t valid_points{};
  std::uint64_t wrong_points{};
};

// Checks every point of each cloud against the frame it came from, and ends
// the run after kFrames clouds.
class CheckOp final : public holoscan::Operator<> {
 public:
  explicit CheckOp(std::shared_ptr<Tally> tally) : tally_(std::move(tally)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(point_cloud, "point_cloud").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{point_cloud});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto cloud = point_cloud.receive();
    if (!cloud) {
      return holoscan::make_unexpected(std::move(cloud).error());
    }
    std::vector<float> xyz(static_cast<std::size_t>(kWidth) * kHeight * 3U);
    if (auto copied =
            cloud->data.copy_to_host(xyz.data(), xyz.size() * sizeof(float), context.cuda_stream());
        !copied) {
      return copied;
    }
    const pc::SyntheticFrame frame = pc::synthetic_frame(
        kWidth, kHeight, static_cast<std::uint32_t>(cloud->metadata.frame_id - 1U), false);
    const pc::CameraIntrinsics& k = frame.intrinsics;
    for (int v = 0; v < kHeight; ++v) {
      for (int u = 0; u < kWidth; ++u) {
        const float z = static_cast<float>(pc::synthetic_depth_mm(frame, u, v)) * kDepthScale;
        const float* p = xyz.data() + 3U * (static_cast<std::size_t>(v) * kWidth + u);
        const bool expect_valid = z > 0.0F && z >= kDepthMin && z <= kDepthMax;
        const bool valid = std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
        if (valid != expect_valid ||
            (valid && std::hypot(p[0] - (u - k.cx) * z / k.fx, p[1] - (v - k.cy) * z / k.fy,
                                 p[2] - z) > 1e-4)) {
          ++tally_->wrong_points;
        }
        tally_->valid_points += valid;
      }
    }
    if (++tally_->frames == kFrames) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> point_cloud;

 private:
  std::shared_ptr<Tally> tally_;
};

}  // namespace

int main() {
  holoscan::Graph graph{"point_cloud_consumer"};
  auto source = graph.op<DepthSourceOp>("source");
  auto deproject = graph.op<pc::DepthToPointCloudOp>(
      "deproject", pc::DepthToPointCloudConfig{
                       .intrinsics = pc::synthetic_frame(kWidth, kHeight, 0, false).intrinsics,
                       .depth_scale = kDepthScale,
                       .depth_min = kDepthMin,
                       .depth_max = kDepthMax,
                       .max_width = kWidth,
                       .max_height = kHeight});
  auto tally = std::make_shared<Tally>();
  auto check = graph.op<CheckOp>("check", tally);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->depth, deproject->depth, lossless);
  graph.add_flow(deproject->point_cloud, check->point_cloud, lossless);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // The cloud is in device memory, on GPU 0.
  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "deproject", .output_port = "point_cloud", .device = holoscan::DeviceId{0}});
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (const int status = holoscan::run(plan); status != 0) {
    std::cerr << "point_cloud_consumer: run failed with " << status << '\n';
    return 1;
  }
  const bool passed =
      tally->frames == kFrames && tally->wrong_points == 0 && tally->valid_points != 0;
  std::cout << "point_cloud_consumer: frames=" << tally->frames
            << " valid_points=" << tally->valid_points << " wrong_points=" << tally->wrong_points
            << '\n'
            << "point_cloud_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
