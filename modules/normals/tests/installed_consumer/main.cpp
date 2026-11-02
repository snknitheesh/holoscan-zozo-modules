// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-normals package: a source
// sends organized clouds of the synthetic scene, with colors, in host memory;
// NormalEstimationOp estimates their normals and PlaneSegmentationOp their
// planes. A sink checks that the floor and the back wall are found where the
// scene has them, and that every output came through.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/normals/normal_estimation_op.hpp>
#include <holoscan/normals/plane_segmentation_op.hpp>
#include <holoscan/normals/synthetic_scene.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace hn = holoscan::normals;

constexpr int kWidth = 160;
constexpr int kHeight = 120;
constexpr std::uint32_t kFrames = 5;
constexpr std::uint32_t kPlanes = 6;

holoscan::TensorOutputSpec host_tensor(DLDataType dtype, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kHost,
              .dtype = dtype,
              .rank = 3U,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(bytes),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

// kFrames organized clouds of the scene with their colors, then idle.
class CloudSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(points, "points").produces_tensor(host_tensor(hn::kFloat32, kPixels * 12U));
    spec.output(colors, "colors").produces_tensor(host_tensor(hn::kUInt8, kPixels * 3U));
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
    const std::array<std::int64_t, 3> shape{kHeight, kWidth, 3};
    auto points_loan =
        points.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = hn::kFloat32});
    auto colors_loan =
        colors.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = hn::kUInt8});
    if (!points_loan || !colors_loan) {
      return holoscan::make_unexpected(!points_loan ? points_loan.error() : colors_loan.error());
    }
    const auto fill = [](holoscan::TensorOutputLoan& loan, const void* data, std::size_t bytes) {
      return loan.write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                            -> holoscan::expected<void, holoscan::Error> {
        std::memcpy(writer.data_as<std::uint8_t>(), data, bytes);
        return {};
      });
    };
    if (auto filled = fill(*points_loan, xyz.data(), xyz.size() * sizeof(float)); !filled) {
      return filled;
    }
    if (auto filled = fill(*colors_loan, rgb.data(), rgb.size()); !filled) {
      return filled;
    }
    const holoscan::EmitOptions emit{.capture_time = context.activation_time(),
                                     .frame_id = emitted_ + 1U};
    if (auto sent = points.emit(std::move(*points_loan), emit); !sent) {
      return sent;
    }
    if (auto sent = colors.emit(std::move(*colors_loan), emit); !sent) {
      return sent;
    }
    ++emitted_;
    return {};
  }

  holoscan::Output<holoscan::Tensor> points;
  holoscan::Output<holoscan::Tensor> colors;

 private:
  static constexpr std::uint64_t kPixels = std::uint64_t{kWidth} * kHeight;
  std::uint32_t emitted_{};
};

struct Tally {
  std::uint32_t frames{};
  std::uint32_t missed_planes{};
  std::uint32_t frame_gaps{};
};

// Checks each frame's planes; takes every other output, which must be connected.
class CheckOp final : public holoscan::Operator<> {
 public:
  explicit CheckOp(std::shared_ptr<Tally> tally) : tally_(std::move(tally)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(colors, "colors").queue_depth(2U);
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
                                colors, curvature, labels, coefficients, centroids, inliers}},
                            .window = 0ns});
    for (const auto* port : {&colors, &curvature, &labels, &coefficients, &centroids, &inliers}) {
      result.input(*port, holoscan::Exact{});
    }
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto coefficient_sample = coefficients.receive();
    for (auto* port : {&colors, &curvature, &labels, &centroids, &inliers}) {
      if (auto sample = port->receive(); !sample) {
        return holoscan::make_unexpected(sample.error());
      }
    }
    if (!coefficient_sample) {
      return holoscan::make_unexpected(coefficient_sample.error());
    }
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

  holoscan::Input<holoscan::Tensor> colors;
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
      "normals", hn::NormalEstimationConfig{.color = true, .max_points = kPoints});
  hn::PlaneSegmentationConfig planes{.max_points = kPoints};
  planes.planes.max_planes = kPlanes;
  planes.planes.min_inliers = 200;
  auto segmenter = graph.op<hn::PlaneSegmentationOp>("planes", planes);
  auto tally = std::make_shared<Tally>();
  auto check = graph.op<CheckOp>("check", tally);

  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->points, normals->points, lossless);
  graph.add_flow(source->colors, normals->colors, lossless);
  graph.add_flow(normals->filtered_points, segmenter->points, lossless);
  graph.add_flow(normals->normals, segmenter->normals, lossless);
  graph.add_flow(normals->filtered_colors, check->colors, lossless);
  graph.add_flow(normals->curvature, check->curvature, lossless);
  graph.add_flow(segmenter->labels, check->labels, lossless);
  graph.add_flow(segmenter->plane_coefficients, check->coefficients, lossless);
  graph.add_flow(segmenter->plane_centroids, check->centroids, lossless);
  graph.add_flow(segmenter->plane_inliers, check->inliers, lossless);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // Every output of both operators is in device memory, on GPU 0.
  holoscan::CompileOptions options;
  const auto on_gpu = [&options](const char* op, const char* port) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
  };
  for (const char* port : {"filtered_points", "filtered_colors", "normals", "curvature"}) {
    on_gpu("normals", port);
  }
  for (const char* port : {"labels", "plane_coefficients", "plane_centroids", "plane_inliers"}) {
    on_gpu("planes", port);
  }
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (const int status = holoscan::run(plan); status != 0) {
    std::cerr << "normals_consumer: run failed with " << status << '\n';
    return 1;
  }
  const bool passed =
      tally->frames == kFrames && tally->missed_planes == 0 && tally->frame_gaps == 0;
  std::cout << "normals_consumer: frames=" << tally->frames
            << " missed_planes=" << tally->missed_planes << " frame_gaps=" << tally->frame_gaps
            << '\n'
            << "normals_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
