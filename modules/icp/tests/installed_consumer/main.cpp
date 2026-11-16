// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-icp package: a source
// renders clouds of the synthetic room along its camera path into device
// memory and sends them as PointCloud messages, IcpOp tracks them as depth
// odometry, and a sink checks each pose against the camera's true one. The
// chain is lossless, so the graph runs on one worker.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/icp/icp_op.hpp>
#include <holoscan/icp/synthetic_scene.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/time/time_point.hpp>

namespace {

namespace icp = holoscan::icp;
namespace hm = holoscan::messages;

constexpr int kWidth = 320;
constexpr int kHeight = 240;
constexpr std::uint32_t kFrames = 10;
// The path the frames are the start of, in frames.
constexpr std::uint32_t kPath = 150;
constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};

// kFrames organized clouds of the room in device memory, then idle.
class CloudSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    hm::declare_point_cloud_output(spec, points, "points",
                                   {.max_points = std::uint64_t{kWidth} * kHeight});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{30}});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (emitted_ >= kFrames) {
      return {};
    }
    const hm::PointCloudShape shape = hm::organized_cloud(kHeight, kWidth);
    auto loan = hm::allocate_point_cloud(points, shape, {}, context.cuda_stream());
    if (!loan) {
      return consumers_busy(std::move(loan).error());
    }
    const icp::DepthNoise noise{.sigma = 0.0005F, .sigma_per_m2 = 0.0005F, .seed = emitted_};
    if (auto written = loan->write_and_commit(
            hm::PointCloudFields::points, context.cuda_stream(),
            [&](const holoscan::TensorOutputWriteGuard& writer)
                -> holoscan::expected<void, holoscan::Error> {
              if (const cudaError_t status = icp::render_scene_cloud(
                      icp::synthetic_scene(icp::SceneKind::kRoom),
                      icp::synthetic_trajectory(emitted_, kPath),
                      icp::synthetic_camera_info(kWidth, kHeight), 8.0F, noise,
                      writer.data_as<float>(), context.cuda_stream().get());
                  status != cudaSuccess) {
                return holoscan::make_unexpected(
                    holoscan::Error{holoscan::ErrorCode::kFailure, cudaGetErrorString(status)});
              }
              return {};
            });
        !written) {
      return written;
    }
    const holoscan::TimePoint capture = context.activation_time();
    auto emitted = hm::emit_point_cloud(
        points, std::move(*loan),
        hm::Header{.frame_id = "synthetic_camera", .capture_timestamp_ns = capture.timestamp_ns},
        shape, -1, holoscan::EmitOptions{.capture_time = capture, .frame_id = emitted_ + 1U});
    if (emitted) {
      ++emitted_;
      return {};
    }
    return consumers_busy(std::move(emitted).error());
  }

  holoscan::Output<hm::schema::PointCloudT> points;

 private:
  // A consumer still busy with earlier clouds: the emit is refused
  // (kBackpressured), or every slot of the pool is still held
  // (kResourceExhausted). The same cloud is rendered again on the next tick.
  static holoscan::expected<void, holoscan::Error> consumers_busy(holoscan::Error error) {
    if (error.code == holoscan::ErrorCode::kBackpressured ||
        error.code == holoscan::ErrorCode::kResourceExhausted) {
      return {};
    }
    return holoscan::make_unexpected(std::move(error));
  }

  std::uint32_t emitted_{};
};

struct Tally {
  std::uint32_t frames{};
  std::uint32_t wrong_poses{};
  double max_error_m{};
};

// Checks each pose against the camera's true pose of its frame.
class CheckOp final : public holoscan::Operator<> {
 public:
  explicit CheckOp(std::shared_ptr<Tally> tally) : tally_(std::move(tally)) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.input(pose, "pose").queue_depth(2U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{pose});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = pose.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    icp::Matrix4 m{};
    if (auto copied = sample->data.copy_to_host(m.data(), sizeof(m)); !copied) {
      return copied;
    }
    const auto frame = static_cast<std::uint32_t>(sample->metadata.frame_id - 1U);
    const icp::TransformError error =
        icp::transform_error(m, icp::synthetic_trajectory(frame, kPath));
    tally_->max_error_m = std::max(tally_->max_error_m, error.translation);
    tally_->wrong_poses += error.translation > 0.01 || error.rotation_rad > 0.0035 ? 1U : 0U;
    if (++tally_->frames == kFrames) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> pose;

 private:
  std::shared_ptr<Tally> tally_;
};

}  // namespace

int main() {
  holoscan::Graph graph{"icp_consumer"};
  auto source = graph.op<CloudSourceOp>("source");
  icp::IcpConfig config;
  config.mode = icp::IcpMode::kOdometry;
  config.options.max_correspondence_distance = 0.10F;
  config.options.min_correspondence_distance = 0.03F;
  config.options.voxel_size = 0.02F;
  config.options.normal_radius = 0.06F;
  config.options.max_iterations = 100;
  config.max_points = std::uint32_t{kWidth} * kHeight;
  // Only the pose is connected.
  config.transform_output = false;
  config.stats_output = false;
  auto odometry = graph.op<icp::IcpOp>("odometry", config);
  auto tally = std::make_shared<Tally>();
  auto check = graph.op<CheckOp>("check", tally);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->points, odometry->source, lossless);
  graph.add_flow(odometry->pose, check->pose, lossless);
  // A lossless chain on one worker: every operator of the graph, one at a
  // time (see holoscan-messages' README).
  graph.partition("icp_consumer").include_all();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // The clouds are in device memory, on GPU 0.
  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "source", .output_port = "points", .device = holoscan::DeviceId{0}});
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  const std::uint32_t workers = plan.scheduler_plan().worker_count;
  std::cout << "icp_consumer: workers=" << workers << '\n';
  if (workers != 1U) {
    std::cerr << "icp_consumer: the graph should run on one worker\n";
    return 1;
  }
  if (const int status = holoscan::run(plan); status != 0) {
    std::cerr << "icp_consumer: run failed with " << status << '\n';
    return 1;
  }
  const bool passed = tally->frames == kFrames && tally->wrong_poses == 0U;
  std::cout << "icp_consumer: frames=" << tally->frames << " wrong_poses=" << tally->wrong_poses
            << " max_error_mm=" << tally->max_error_m * 1000.0 << '\n'
            << "icp_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
