// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-clustering package: a source
// sends organized clouds of the synthetic scene, with colors, in host memory;
// EuclideanClusterOp splits them into objects and packs the clustered points
// by cluster. A sink checks the number of clusters against the scene's truth
// and that the labels, summaries, points and colors agree.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <holoscan/clustering/clustering.hpp>
#include <holoscan/clustering/euclidean_cluster_op.hpp>
#include <holoscan/clustering/synthetic_scene.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace hc = holoscan::clustering;

constexpr int kWidth = 320;
constexpr int kHeight = 240;
constexpr std::uint32_t kFrames = 5;
constexpr std::uint32_t kMinSize = 50;
constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

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

// The objects of frame `frame` that have at least kMinSize points.
std::uint32_t expected_clusters(std::uint32_t frame) {
  std::vector<float> xyz;
  std::vector<std::int32_t> groups;
  hc::synthetic_scene_cloud(hc::synthetic_scene(kWidth, kHeight, frame), xyz, groups);
  std::array<std::uint32_t, hc::kSceneGroups> count{};
  for (const std::int32_t g : groups) {
    if (g >= 0) {
      ++count[static_cast<std::size_t>(g)];
    }
  }
  return static_cast<std::uint32_t>(
      std::count_if(count.begin(), count.end(), [](std::uint32_t n) { return n >= kMinSize; }));
}

// kFrames organized clouds of the scene with a color per point, then idle.
class CloudSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(points, "points").produces_tensor(host_tensor(kFloat32, kPixels * 12U));
    spec.output(colors, "colors").produces_tensor(host_tensor(kUInt8, kPixels * 3U));
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
    std::vector<std::int32_t> groups;
    hc::synthetic_scene_cloud(hc::synthetic_scene(kWidth, kHeight, emitted_), xyz, groups);
    std::vector<std::uint8_t> rgb(xyz.size());
    for (std::size_t i = 0; i < groups.size(); ++i) {
      std::fill_n(rgb.begin() + 3 * i, 3, static_cast<std::uint8_t>(100 + groups[i]));
    }
    const std::array<std::int64_t, 3> shape{kHeight, kWidth, 3};
    auto points_loan =
        points.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kFloat32});
    auto colors_loan =
        colors.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
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
  std::uint32_t wrong_counts{};
  std::uint32_t inconsistent{};
  std::uint32_t frame_gaps{};
  std::uint64_t clusters{};
};

// Checks each frame's clusters.
class CheckOp final : public holoscan::Operator<> {
 public:
  explicit CheckOp(std::shared_ptr<Tally> tally) : tally_(std::move(tally)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(labels, "labels").queue_depth(2U);
    spec.input(clusters, "clusters").queue_depth(2U);
    spec.input(points, "points").queue_depth(2U);
    spec.input(colors, "colors").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    using std::chrono_literals::operator""ns;
    holoscan::Contract result;
    result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{std::vector<holoscan::PortRef>{
                                           labels, clusters, points, colors}},
                                       .window = 0ns});
    result.input(labels, holoscan::Exact{});
    result.input(clusters, holoscan::Exact{});
    result.input(points, holoscan::Exact{});
    result.input(colors, holoscan::Exact{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto label_sample = labels.receive();
    auto cluster_sample = clusters.receive();
    auto point_sample = points.receive();
    auto color_sample = colors.receive();
    if (!label_sample || !cluster_sample || !point_sample || !color_sample) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "a frame's outputs did not all arrive"});
    }
    const std::span<const std::int64_t> shape = cluster_sample->data.shape_span();
    const std::span<const std::int64_t> label_shape = label_sample->data.shape_span();
    if (shape.size() != 2 || shape[1] != hc::kSummaryFields || label_shape.size() != 1 ||
        point_sample->data.shape_span()[0] != label_shape[0] ||
        color_sample->data.shape_span()[0] != label_shape[0]) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "unexpected output shapes"});
    }
    std::vector<hc::ClusterSummary> summaries(static_cast<std::size_t>(shape[0]));
    std::vector<std::int32_t> ids(static_cast<std::size_t>(label_shape[0]));
    if (auto copied = cluster_sample->data.copy_to_host(
            summaries.data(), summaries.size() * sizeof(hc::ClusterSummary), context.cuda_stream());
        !copied) {
      return copied;
    }
    if (auto copied =
            label_sample->data.copy_to_host(ids.data(), ids.size() * 4U, context.cuda_stream());
        !copied) {
      return copied;
    }
    const std::uint64_t frame_id = label_sample->metadata.frame_id;
    tally_->frame_gaps += frame_id != tally_->frames + 1U ? 1U : 0U;
    if (summaries.size() != expected_clusters(static_cast<std::uint32_t>(frame_id - 1U))) {
      ++tally_->wrong_counts;
    }
    // The packed labels run cluster by cluster, as long as each summary says.
    std::size_t at = 0;
    bool consistent = true;
    for (std::size_t k = 0; k < summaries.size(); ++k) {
      const auto n = static_cast<std::size_t>(summaries[k].point_count);
      consistent = consistent && summaries[k].id == static_cast<float>(k) && n >= kMinSize &&
                   at + n <= ids.size() &&
                   std::all_of(ids.begin() + static_cast<std::ptrdiff_t>(at),
                               ids.begin() + static_cast<std::ptrdiff_t>(at + n),
                               [k](std::int32_t id) { return id == static_cast<std::int32_t>(k); });
      at += n;
    }
    tally_->inconsistent += consistent && at == ids.size() ? 0U : 1U;
    tally_->clusters += summaries.size();
    if (++tally_->frames == kFrames) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> labels;
  holoscan::Input<holoscan::Tensor> clusters;
  holoscan::Input<holoscan::Tensor> points;
  holoscan::Input<holoscan::Tensor> colors;

 private:
  std::shared_ptr<Tally> tally_;
};

}  // namespace

int main() {
  holoscan::Graph graph{"clustering_consumer"};
  auto source = graph.op<CloudSourceOp>("source");
  hc::EuclideanClusterConfig config{
      .clustering = {.tolerance = 0.03F, .min_cluster_size = kMinSize},
      .oriented_boxes = true,
      .emit_points = true,
      .color = true,
      .layout = hc::OutputLayout::kDense,
      .max_points = std::uint32_t{kWidth} * kHeight};
  auto cluster = graph.op<hc::EuclideanClusterOp>("cluster", config);
  auto tally = std::make_shared<Tally>();
  auto check = graph.op<CheckOp>("check", tally);

  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->points, cluster->points, lossless);
  graph.add_flow(source->colors, cluster->colors, lossless);
  graph.add_flow(cluster->labels, check->labels, lossless);
  graph.add_flow(cluster->clusters, check->clusters, lossless);
  graph.add_flow(cluster->clustered_points, check->points, lossless);
  graph.add_flow(cluster->clustered_colors, check->colors, lossless);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // The operator's outputs are in device memory, on GPU 0.
  holoscan::CompileOptions options;
  for (const char* port : {"labels", "clusters", "clustered_points", "clustered_colors"}) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "cluster", .output_port = port, .device = holoscan::DeviceId{0}});
  }
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (const int status = holoscan::run(plan); status != 0) {
    std::cerr << "clustering_consumer: run failed with " << status << '\n';
    return 1;
  }
  const bool passed = tally->frames == kFrames && tally->wrong_counts == 0 &&
                      tally->inconsistent == 0 && tally->frame_gaps == 0 && tally->clusters > 0;
  std::cout << "clustering_consumer: frames=" << tally->frames << " clusters=" << tally->clusters
            << " wrong_counts=" << tally->wrong_counts << " inconsistent=" << tally->inconsistent
            << " frame_gaps=" << tally->frame_gaps << '\n'
            << "clustering_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
