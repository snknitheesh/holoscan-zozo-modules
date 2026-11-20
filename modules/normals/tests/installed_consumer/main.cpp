// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-normals package: a source
// sends organized PointCloud messages of the synthetic scene, with colors, in
// host memory; NormalEstimationOp adds their normals and PlaneSegmentationOp
// finds their planes. A sink checks that the floor and the back wall are
// found where the scene has them, and that every output came through. The
// graph is lossless and runs on one worker.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/normals/normal_estimation_op.hpp>
#include <holoscan/normals/plane_segmentation_op.hpp>
#include <holoscan/normals/synthetic_scene.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace hm = holoscan::messages;
namespace hn = holoscan::normals;

constexpr int kWidth = 160;
constexpr int kHeight = 120;
constexpr std::uint32_t kFrames = 5;
constexpr std::uint32_t kPlanes = 6;

// Whether `error` says the consumers are still busy with earlier frames: an
// emit refused before delivery, or every slot of the output's pool still held.
bool consumers_busy(const holoscan::Error& error) {
  return error.code == holoscan::ErrorCode::kBackpressured ||
         error.code == holoscan::ErrorCode::kResourceExhausted;
}

// kFrames organized clouds of the scene with their colors, then idle. A cloud
// the consumers cannot take yet is sent again on the next tick.
class CloudSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    hm::declare_point_cloud_output(
        spec, point_cloud, "point_cloud",
        {.max_points = kPixels, .colors = true, .memory_kind = holoscan::MemoryKind::kHost});
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
    std::vector<float> xyz;
    std::vector<std::uint8_t> rgb;
    std::vector<std::uint8_t> surfaces;
    std::vector<float> truth;
    hn::synthetic_scene_cloud(hn::synthetic_scene(kWidth, kHeight, emitted_), xyz, rgb, surfaces,
                              truth);
    // Both fields are allocated together, before anything is sent: a refusal
    // leaves nothing half sent.
    const hm::PointCloudShape shape = hm::organized_cloud(kHeight, kWidth);
    auto loan =
        hm::allocate_point_cloud(point_cloud, shape, {.colors = true}, context.cuda_stream());
    if (!loan) {
      return consumers_busy(loan.error()) ? holoscan::expected<void, holoscan::Error>{}
                                          : holoscan::make_unexpected(std::move(loan).error());
    }
    const auto copy = [](const void* data, std::size_t bytes) {
      return [data, bytes](const holoscan::TensorOutputWriteGuard& writer) {
        std::memcpy(writer.data_as<void>(), data, bytes);
      };
    };
    if (auto filled = loan->write_host_and_commit(hm::PointCloudFields::points,
                                                  copy(xyz.data(), xyz.size() * sizeof(float)));
        !filled) {
      return filled;
    }
    if (auto filled =
            loan->write_host_and_commit(hm::PointCloudFields::colors, copy(rgb.data(), rgb.size()));
        !filled) {
      return filled;
    }
    // A frame keeps its capture time until it is out.
    if (!capture_) {
      capture_ = context.activation_time();
    }
    auto sent = hm::emit_point_cloud(
        point_cloud, std::move(*loan), {.frame_id = "scene_optical"}, shape, -1,
        holoscan::EmitOptions{.capture_time = *capture_, .frame_id = emitted_ + 1U});
    if (!sent) {
      return consumers_busy(sent.error()) ? holoscan::expected<void, holoscan::Error>{} : sent;
    }
    capture_.reset();
    ++emitted_;
    return {};
  }

  holoscan::Output<hm::schema::PointCloudT> point_cloud;

 private:
  static constexpr std::uint64_t kPixels = std::uint64_t{kWidth} * kHeight;
  std::uint32_t emitted_{};
  std::optional<holoscan::TimePoint> capture_;
};

struct Tally {
  std::uint32_t frames{};
  std::uint32_t missed_planes{};
  std::uint32_t frame_gaps{};
  std::uint32_t without_colors{};
};

// Checks each frame's planes; takes every other output, which must be connected.
class CheckOp final : public holoscan::Operator<> {
 public:
  explicit CheckOp(std::shared_ptr<Tally> tally) : tally_(std::move(tally)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    // Two slots per input: one frame in compute() while the next arrives.
    spec.input(point_cloud, "point_cloud").queue_depth(2U);
    spec.input(curvature, "curvature").queue_depth(2U);
    spec.input(labels, "labels").queue_depth(2U);
    spec.input(coefficients, "coefficients").queue_depth(2U);
    spec.input(centroids, "centroids").queue_depth(2U);
    spec.input(inliers, "inliers").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    using std::chrono_literals::operator""ns;
    holoscan::Contract result;
    result.trigger(
        holoscan::OnAligned{.ports = holoscan::PortList{std::vector<holoscan::PortRef>{
                                point_cloud, curvature, labels, coefficients, centroids, inliers}},
                            .window = 0ns});
    result.input(point_cloud, holoscan::Exact{});
    for (const auto* port : {&curvature, &labels, &coefficients, &centroids, &inliers}) {
      result.input(*port, holoscan::Exact{});
    }
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto cloud = point_cloud.receive();
    if (!cloud) {
      return holoscan::make_unexpected(cloud.error());
    }
    auto coefficient_sample = coefficients.receive();
    for (auto* port : {&curvature, &labels, &centroids, &inliers}) {
      if (auto sample = port->receive(); !sample) {
        return holoscan::make_unexpected(sample.error());
      }
    }
    if (!coefficient_sample) {
      return holoscan::make_unexpected(coefficient_sample.error());
    }
    tally_->without_colors += hm::is_present(cloud->data.colors) ? 0U : 1U;
    std::array<float, 4 * kPlanes> got{};
    if (auto copied =
            coefficient_sample->data.copy_to_host(got.data(), sizeof(got), context.cuda_stream());
        !copied) {
      return copied;
    }
    const std::uint64_t frame_id = coefficient_sample->metadata.frame_id;
    tally_->frame_gaps += frame_id != tally_->frames + 1U ? 1U : 0U;
    const hn::SyntheticScene scene =
        hn::synthetic_scene(kWidth, kHeight, static_cast<std::uint32_t>(frame_id - 1U));
    for (const std::uint8_t surface : {hn::kSurfaceFloor, hn::kSurfaceBackWall}) {
      float truth[4];
      hn::synthetic_surface_plane(scene, surface, truth);
      bool found = false;
      for (std::uint32_t k = 0; k < kPlanes; ++k) {
        const float* p = got.data() + 4 * k;
        found = found || (p[0] * truth[0] + p[1] * truth[1] + p[2] * truth[2] > 0.9995F &&
                          std::abs(p[3] - truth[3]) < 0.02F);
      }
      tally_->missed_planes += found ? 0U : 1U;
    }
    if (++tally_->frames == kFrames) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<hm::schema::PointCloudT> point_cloud;
  holoscan::Input<holoscan::Tensor> curvature;
  holoscan::Input<holoscan::Tensor> labels;
  holoscan::Input<holoscan::Tensor> coefficients;
  holoscan::Input<holoscan::Tensor> centroids;
  holoscan::Input<holoscan::Tensor> inliers;

 private:
  std::shared_ptr<Tally> tally_;
};

}  // namespace

int main() {
  holoscan::Graph graph{"normals_consumer"};
  auto source = graph.op<CloudSourceOp>("source");
  constexpr std::uint32_t kPoints = std::uint32_t{kWidth} * kHeight;
  auto normals = graph.op<hn::NormalEstimationOp>(
      "normals", hn::NormalEstimationConfig{.colors = true, .max_points = kPoints});
  hn::PlaneSegmentationConfig planes{.max_points = kPoints};
  planes.planes.max_planes = kPlanes;
  planes.planes.min_inliers = 200;
  auto segmenter = graph.op<hn::PlaneSegmentationOp>("planes", planes);
  auto tally = std::make_shared<Tally>();
  auto check = graph.op<CheckOp>("check", tally);

  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->point_cloud, normals->point_cloud, lossless);
  graph.add_flow(normals->point_cloud_with_normals, segmenter->point_cloud, lossless);
  graph.add_flow(normals->point_cloud_with_normals, check->point_cloud, lossless);
  graph.add_flow(normals->curvature, check->curvature, lossless);
  graph.add_flow(segmenter->labels, check->labels, lossless);
  graph.add_flow(segmenter->plane_coefficients, check->coefficients, lossless);
  graph.add_flow(segmenter->plane_centroids, check->centroids, lossless);
  graph.add_flow(segmenter->plane_inliers, check->inliers, lossless);
  // A lossless graph on one worker: across workers a stage that falls behind
  // refuses its producer's emit, which fails the run.
  graph.partition("normals_consumer").include_all();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // Every output of both operators is in device memory, on GPU 0.
  holoscan::CompileOptions options;
  const auto on_gpu = [&options](const char* op, const char* port) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
  };
  for (const char* port : {"point_cloud_with_normals", "curvature"}) {
    on_gpu("normals", port);
  }
  for (const char* port : {"labels", "plane_coefficients", "plane_centroids", "plane_inliers"}) {
    on_gpu("planes", port);
  }
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  // Warnings on an accepted plan, as well as the reasons for a rejected one.
  if (!plan.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
  }
  if (!plan.ok()) {
    return 1;
  }
  const std::size_t workers = plan.execution_lanes().size();
  if (workers != 1U) {
    std::cerr << "normals_consumer: the graph runs on " << workers << " workers, not one\n";
    return 1;
  }
  if (const int status = holoscan::run(plan); status != 0) {
    std::cerr << "normals_consumer: run failed with " << status << '\n';
    return 1;
  }
  const bool passed = tally->frames == kFrames && tally->missed_planes == 0 &&
                      tally->frame_gaps == 0 && tally->without_colors == 0;
  std::cout << "normals_consumer: workers=" << workers << " frames=" << tally->frames
            << " missed_planes=" << tally->missed_planes << " frame_gaps=" << tally->frame_gaps
            << " without_colors=" << tally->without_colors << '\n'
            << "normals_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
