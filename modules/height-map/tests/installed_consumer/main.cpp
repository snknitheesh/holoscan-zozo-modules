// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-height-map package: a source
// sends clouds of a box on flat ground in host memory, with the sensor's pose
// moving along x; HeightMapOp maps them in a window that follows the sensor
// and TraversabilityOp finds the box lethal. A sink checks the box's height,
// the free ground around it, and that the window moved.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/height_map/height_map_op.hpp>
#include <holoscan/height_map/synthetic_terrain.hpp>
#include <holoscan/height_map/traversability_op.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace hm = holoscan::height_map;

constexpr std::uint32_t kFrames = 5;
constexpr int kSide = 60;             // Cells.
constexpr float kResolution = 0.05F;  // Meters.
constexpr std::uint64_t kMaxPoints = 120U * 120U;

hm::Terrain scene() {
  hm::Terrain terrain;
  hm::add_feature(terrain, hm::box(0.9F, 1.3F, -0.2F, 0.2F, 0.3F));
  return terrain;
}

holoscan::TensorOutputSpec host_tensor(DLDataType dtype, std::uint32_t rank, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kHost,
              .dtype = dtype,
              .rank = rank,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(bytes),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

// kFrames clouds, in the sensor's frame 1 m above the ground, and its poses.
class CloudSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(points, "points").produces_tensor(host_tensor(hm::kFloat32, 2U, kMaxPoints * 12U));
    spec.output(pose, "pose").produces_tensor(host_tensor(hm::kFloat64, 2U, 16U * 8U));
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
    // The sensor moves 10 cm a frame; it sees the ground 3 m around it.
    const double sx = 0.01 + 0.1 * emitted_;
    const hm::Terrain terrain = scene();
    std::vector<float> xyz;
    for (int j = 0; j < 120; ++j) {
      for (int i = 0; i < 120; ++i) {
        const float x = static_cast<float>(sx) - 1.5F + 0.025F * (static_cast<float>(i) + 0.5F);
        const float y = -1.5F + 0.025F * (static_cast<float>(j) + 0.5F);
        xyz.insert(xyz.end(),
                   {x - static_cast<float>(sx), y, hm::terrain_height(terrain, x, y) - 1.0F});
      }
    }
    const std::array<double, 16> matrix{1, 0, 0, sx, 0, 1, 0, 0, 0, 0, 1, 1.0, 0, 0, 0, 1};
    const std::array<std::int64_t, 2> cloud_shape{static_cast<std::int64_t>(xyz.size() / 3), 3};
    const std::array<std::int64_t, 2> pose_shape{4, 4};
    auto cloud_loan = points.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = cloud_shape, .dtype = hm::kFloat32});
    auto pose_loan = pose.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = pose_shape, .dtype = hm::kFloat64});
    if (!cloud_loan || !pose_loan) {
      return holoscan::make_unexpected(!cloud_loan ? cloud_loan.error() : pose_loan.error());
    }
    const auto fill = [](holoscan::TensorOutputLoan& loan, const void* data, std::size_t bytes) {
      return loan.write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                            -> holoscan::expected<void, holoscan::Error> {
        std::memcpy(writer.data_as<std::uint8_t>(), data, bytes);
        return {};
      });
    };
    if (auto filled = fill(*cloud_loan, xyz.data(), xyz.size() * sizeof(float)); !filled) {
      return filled;
    }
    if (auto filled = fill(*pose_loan, matrix.data(), sizeof(matrix)); !filled) {
      return filled;
    }
    const holoscan::EmitOptions emit{.capture_time = context.activation_time(),
                                     .frame_id = emitted_ + 1U};
    if (auto sent = points.emit(std::move(*cloud_loan), emit); !sent) {
      return sent;
    }
    if (auto sent = pose.emit(std::move(*pose_loan), emit); !sent) {
      return sent;
    }
    ++emitted_;
    return {};
  }

  holoscan::Output<holoscan::Tensor> points;
  holoscan::Output<holoscan::Tensor> pose;

 private:
  std::uint32_t emitted_{};
};

struct Tally {
  std::uint32_t frames{};
  std::uint32_t wrong{};
  double first_origin_x{};
  double last_origin_x{};
};

// Checks each map: the box's top, its lethal rim, and the free ground.
class CheckOp final : public holoscan::Operator<> {
 public:
  explicit CheckOp(std::shared_ptr<Tally> tally) : tally_(std::move(tally)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(elevation, "elevation").queue_depth(2U);
    spec.input(info, "info").queue_depth(2U);
    spec.input(cost, "cost").queue_depth(2U);
    spec.input(unknown, "unknown").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    using std::chrono_literals::operator""ns;
    holoscan::Contract result;
    result.trigger(holoscan::OnAligned{
        .ports = holoscan::PortList{std::vector<holoscan::PortRef>{elevation, info, cost, unknown}},
        .window = 0ns});
    for (const auto* port : {&elevation, &info, &cost, &unknown}) {
      result.input(*port, holoscan::Exact{});
    }
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto e = elevation.receive();
    auto i = info.receive();
    auto c = cost.receive();
    auto u = unknown.receive();
    if (!e || !i || !c || !u) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "CheckOp: an input is missing"});
    }
    std::vector<float> heights(kSide * kSide);
    std::vector<float> costs(kSide * kSide);
    std::array<double, hm::kMapInfoSize> grid{};
    for (auto copied :
         {e->data.copy_to_host(heights.data(), heights.size() * 4, context.cuda_stream()),
          c->data.copy_to_host(costs.data(), costs.size() * 4, context.cuda_stream()),
          i->data.copy_to_host(grid.data(), sizeof(grid), context.cuda_stream())}) {
      if (!copied) {
        return copied;
      }
    }
    const auto at = [&](double x, double y) {
      const int col = static_cast<int>(std::floor((x - grid[hm::kInfoOriginX]) / kResolution));
      const int row = static_cast<int>(std::floor((y - grid[hm::kInfoOriginY]) / kResolution));
      return static_cast<std::size_t>(row) * kSide + col;
    };
    tally_->wrong += std::fabs(heights[at(1.1, 0.0)] - 0.3F) < 1e-4F ? 0U : 1U;
    tally_->wrong += costs[at(0.88, 0.0)] == 1.0F ? 0U : 1U;  // The box's rim.
    tally_->wrong +=
        std::fabs(heights[at(0.2, 0.8)]) < 1e-4F && costs[at(0.2, 0.8)] == 0.0F ? 0U : 1U;
    (tally_->frames == 0 ? tally_->first_origin_x : tally_->last_origin_x) = grid[hm::kInfoOriginX];
    if (++tally_->frames == kFrames) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> elevation;
  holoscan::Input<holoscan::Tensor> info;
  holoscan::Input<holoscan::Tensor> cost;
  holoscan::Input<holoscan::Tensor> unknown;

 private:
  std::shared_ptr<Tally> tally_;
};

}  // namespace

int main() {
  holoscan::Graph graph{"height_map_consumer"};
  auto source = graph.op<CloudSourceOp>("source");
  hm::HeightMapConfig map_config;
  map_config.map.geometry = hm::MapGeometry{
      .resolution = kResolution, .rows = kSide, .cols = kSide, .origin_x = 0.0, .origin_y = 0.0};
  map_config.map.mode = hm::MapMode::kFollow;
  map_config.map.elevation = hm::ElevationMode::kMax;
  map_config.pose_input = true;
  map_config.outputs.variance = false;
  map_config.max_points = kMaxPoints;
  auto mapper = graph.op<hm::HeightMapOp>("map", map_config);
  auto traversability = graph.op<hm::TraversabilityOp>(
      "traversability", hm::TraversabilityConfig{.max_rows = kSide, .max_cols = kSide});
  auto tally = std::make_shared<Tally>();
  auto check = graph.op<CheckOp>("check", tally);

  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->points, mapper->points, lossless);
  graph.add_flow(source->pose, mapper->pose, lossless);
  graph.add_flow(mapper->elevation, traversability->elevation, lossless);
  graph.add_flow(mapper->map_info, traversability->map_info, lossless);
  graph.add_flow(mapper->elevation, check->elevation, lossless);
  graph.add_flow(mapper->map_info, check->info, lossless);
  graph.add_flow(traversability->cost, check->cost, lossless);
  graph.add_flow(traversability->unknown, check->unknown, lossless);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // The layers are in device memory, on GPU 0.
  holoscan::CompileOptions options;
  const auto on_gpu = [&options](const char* op, const char* port) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
  };
  on_gpu("map", "elevation");
  on_gpu("traversability", "cost");
  on_gpu("traversability", "unknown");
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (const int status = holoscan::run(plan); status != 0) {
    std::cerr << "height_map_consumer: run failed with " << status << '\n';
    return 1;
  }
  const bool moved = std::fabs(tally->last_origin_x - tally->first_origin_x - 0.4) < 1e-6;
  const bool passed = tally->frames == kFrames && tally->wrong == 0 && moved;
  std::cout << "height_map_consumer: frames=" << tally->frames << " wrong=" << tally->wrong
            << " window_moved=" << (moved ? "yes" : "no") << '\n'
            << "height_map_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
