// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-pcl-filters package: a source
// sends organized clouds of the synthetic scene, with colors, in host memory;
// CropBoxOp keeps what is within 6 m, OutlierRemovalOp drops the flying
// pixels, GroundRemovalOp finds and removes the ground and VoxelDownsampleOp
// thins what is left. A sink checks the plane against the scene's truth and
// that the voxels and their colors came through.

#include <algorithm>
#include <array>
#include <atomic>
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
#include <holoscan/pcl_filters/crop_box_op.hpp>
#include <holoscan/pcl_filters/ground_removal_op.hpp>
#include <holoscan/pcl_filters/outlier_removal_op.hpp>
#include <holoscan/pcl_filters/synthetic_scene.hpp>
#include <holoscan/pcl_filters/voxel_downsample_op.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace pf = holoscan::pcl_filters;

constexpr int kWidth = 160;
constexpr int kHeight = 120;
constexpr std::uint32_t kFrames = 5;
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

// kFrames organized clouds of the scene with their colors, then idle.
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
    std::vector<std::uint8_t> rgb;
    std::vector<pf::SceneLabel> labels;
    pf::synthetic_scene_cloud(pf::synthetic_scene(kWidth, kHeight, emitted_), xyz, rgb, labels);
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
  std::uint64_t voxels{};
  std::uint32_t wrong_planes{};
  std::uint32_t frame_gaps{};
};

// Checks each frame's plane and voxels.
class CheckOp final : public holoscan::Operator<> {
 public:
  explicit CheckOp(std::shared_ptr<Tally> tally) : tally_(std::move(tally)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(voxels, "voxels").queue_depth(2U);
    spec.input(colors, "colors").queue_depth(2U);
    spec.input(plane, "plane").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    using std::chrono_literals::operator""ns;
    holoscan::Contract result;
    result.trigger(holoscan::OnAligned{
        .ports = holoscan::PortList{std::vector<holoscan::PortRef>{voxels, colors, plane}},
        .window = 0ns});
    result.input(voxels, holoscan::Exact{});
    result.input(colors, holoscan::Exact{});
    result.input(plane, holoscan::Exact{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto cloud = voxels.receive();
    auto rgb = colors.receive();
    auto coefficients = plane.receive();
    if (!cloud || !rgb || !coefficients) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "a frame's outputs did not all arrive"});
    }
    const std::span<const std::int64_t> shape = cloud->data.shape_span();
    if (shape.size() != 2 || shape[1] != 3 || rgb->data.shape_span()[0] != shape[0]) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "voxels must be a dense [M, 3] cloud"});
    }
    std::array<float, 4> got{};
    if (auto copied =
            coefficients->data.copy_to_host(got.data(), sizeof(got), context.cuda_stream());
        !copied) {
      return copied;
    }
    const std::uint64_t frame_id = cloud->metadata.frame_id;
    tally_->frame_gaps += frame_id != tally_->frames + 1U ? 1U : 0U;
    float truth[4];
    pf::synthetic_ground_plane(
        pf::synthetic_scene(kWidth, kHeight, static_cast<std::uint32_t>(frame_id - 1U)), truth);
    const float alignment = got[0] * truth[0] + got[1] * truth[1] + got[2] * truth[2];
    if (!(alignment > 0.9995F && std::abs(got[3] - truth[3]) < 0.01F)) {
      ++tally_->wrong_planes;
    }
    tally_->voxels += static_cast<std::uint64_t>(shape[0]);
    if (++tally_->frames == kFrames) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> voxels;
  holoscan::Input<holoscan::Tensor> colors;
  holoscan::Input<holoscan::Tensor> plane;

 private:
  std::shared_ptr<Tally> tally_;
};

}  // namespace

int main() {
  holoscan::Graph graph{"pcl_filters_consumer"};
  auto source = graph.op<CloudSourceOp>("source");
  constexpr std::uint32_t kPoints = std::uint32_t{kWidth} * kHeight;
  pf::CropBoxConfig crop{.color = true, .max_points = kPoints};
  crop.box.max_distance = 6.0F;
  auto cropper = graph.op<pf::CropBoxOp>("crop", crop);
  auto outliers = graph.op<pf::OutlierRemovalOp>(
      "outliers",
      pf::OutlierRemovalConfig{
          .outliers = {.k = 6, .search_radius = 0.3F}, .color = true, .max_points = kPoints});
  pf::GroundRemovalConfig ground{.color = true, .max_points = kPoints};
  float truth[4];
  pf::synthetic_ground_plane(pf::synthetic_scene(kWidth, kHeight, 0), truth);
  std::copy(truth, truth + 3, ground.ground.up);
  ground.ground.distance_threshold = 0.03F;
  auto grounder = graph.op<pf::GroundRemovalOp>("ground", ground);
  auto voxels = graph.op<pf::VoxelDownsampleOp>(
      "voxels",
      pf::VoxelDownsampleConfig{.voxel_size = 0.05F, .color = true, .max_points = kPoints});
  auto tally = std::make_shared<Tally>();
  auto check = graph.op<CheckOp>("check", tally);

  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->points, cropper->points, lossless);
  graph.add_flow(source->colors, cropper->colors, lossless);
  graph.add_flow(cropper->filtered_points, outliers->points, lossless);
  graph.add_flow(cropper->filtered_colors, outliers->colors, lossless);
  graph.add_flow(outliers->filtered_points, grounder->points, lossless);
  graph.add_flow(outliers->filtered_colors, grounder->colors, lossless);
  graph.add_flow(grounder->filtered_points, voxels->points, lossless);
  graph.add_flow(grounder->filtered_colors, voxels->colors, lossless);
  graph.add_flow(voxels->filtered_points, check->voxels, lossless);
  graph.add_flow(voxels->filtered_colors, check->colors, lossless);
  graph.add_flow(grounder->plane, check->plane, lossless);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // Every filter's outputs are in device memory, on GPU 0.
  holoscan::CompileOptions options;
  for (const char* op : {"crop", "outliers", "ground", "voxels"}) {
    for (const char* port : {"filtered_points", "filtered_colors"}) {
      options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
          .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
    }
  }
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "ground", .output_port = "plane", .device = holoscan::DeviceId{0}});
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (const int status = holoscan::run(plan); status != 0) {
    std::cerr << "pcl_filters_consumer: run failed with " << status << '\n';
    return 1;
  }
  const bool passed = tally->frames == kFrames && tally->wrong_planes == 0 &&
                      tally->frame_gaps == 0 && tally->voxels > 100U * kFrames;
  std::cout << "pcl_filters_consumer: frames=" << tally->frames << " voxels=" << tally->voxels
            << " wrong_planes=" << tally->wrong_planes << " frame_gaps=" << tally->frame_gaps
            << '\n'
            << "pcl_filters_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
