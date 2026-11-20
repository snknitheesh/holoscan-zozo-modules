// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-height-map package: a source
// sends holoscan-messages PointCloud messages of a box on flat ground in host
// memory, with the sensor's pose moving along x; HeightMapOp maps them in a
// window that follows the sensor and TraversabilityOp finds the box lethal. A
// sink checks the box's height, the free ground around it, and that the
// window moved. The graph is lossless, so it runs on one worker, and the
// consumer checks that it does.

#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/height_map/height_map_op.hpp>
#include <holoscan/height_map/synthetic_terrain.hpp>
#include <holoscan/height_map/traversability_op.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace hm = holoscan::height_map;
namespace msg = holoscan::messages;

constexpr std::uint32_t kFrames = 5;
constexpr int kSide = 60;             // Cells.
constexpr float kResolution = 0.05F;  // Meters.
constexpr int kCloudSide = 120;       // Points along x and along y.
constexpr std::uint64_t kMaxPoints = std::uint64_t{kCloudSide} * kCloudSide;

hm::Terrain scene() {
  hm::Terrain terrain;
  hm::add_feature(terrain, hm::box(0.9F, 1.3F, -0.2F, 0.2F, 0.3F));
  return terrain;
}

// Whether a refused send or allocation is backpressure: the consumer has no
// room yet, and the same frame goes again on the next tick.
bool backpressure(const holoscan::Error& error) {
  return error.code == holoscan::ErrorCode::kBackpressured ||
         error.code == holoscan::ErrorCode::kResourceExhausted;
}

// kFrames clouds, in the sensor's frame 1 m above the ground, and its poses,
// each pair with the capture time of its first attempt. A cloud or pose that
// is refused goes again on the next tick; one that went is not sent again.
class CloudSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    msg::declare_point_cloud_output(
        spec, point_cloud, "point_cloud",
        {.max_points = kMaxPoints, .memory_kind = holoscan::MemoryKind::kHost});
    spec.output(pose, "pose")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kHost,
                    .dtype = hm::kFloat64,
                    .rank = 2U,
                    .strides = holoscan::TensorStridePolicy::kContiguousOnly,
                },
            .bounds = holoscan::tensor_bounds(16U * sizeof(double)),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{20}});
    return result;
  }

  void start() override {
    emitted_ = 0;
    capture_.reset();
    cloud_sent_ = false;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (emitted_ >= kFrames) {
      return {};
    }
    if (!capture_) {
      capture_ = context.activation_time();
    }
    const holoscan::EmitOptions emit{.capture_time = *capture_, .frame_id = emitted_ + 1U};
    // The sensor moves 10 cm a frame; it sees the ground 1.5 m around it.
    const double sx = 0.01 + 0.1 * emitted_;
    if (!cloud_sent_) {
      if (auto sent = send_cloud(context, sx, emit); !sent) {
        return backpressure(sent.error()) ? holoscan::expected<void, holoscan::Error>{} : sent;
      }
      cloud_sent_ = true;
    }
    const std::array<double, 16> matrix{1, 0, 0, sx, 0, 1, 0, 0, 0, 0, 1, 1.0, 0, 0, 0, 1};
    auto loan = pose.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = std::array<std::int64_t, 2>{4, 4}, .dtype = hm::kFloat64});
    if (!loan) {
      return backpressure(loan.error()) ? holoscan::expected<void, holoscan::Error>{}
                                        : holoscan::make_unexpected(std::move(loan).error());
    }
    if (auto written =
            loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& w)
                                            -> holoscan::expected<void, holoscan::Error> {
              std::memcpy(w.data_as<double>(), matrix.data(), sizeof(matrix));
              return {};
            });
        !written) {
      return written;
    }
    if (auto sent = pose.emit(std::move(*loan), emit); !sent) {
      return backpressure(sent.error()) ? holoscan::expected<void, holoscan::Error>{} : sent;
    }
    capture_.reset();
    cloud_sent_ = false;
    ++emitted_;
    return {};
  }

  holoscan::Output<msg::schema::PointCloudT> point_cloud;
  holoscan::Output<holoscan::Tensor> pose;

 private:
  holoscan::expected<void, holoscan::Error> send_cloud(holoscan::ExecutionContext& context,
                                                       double sx,
                                                       const holoscan::EmitOptions& emit) {
    const hm::Terrain terrain = scene();
    std::vector<float> xyz;
    xyz.reserve(kMaxPoints * 3U);
    for (int j = 0; j < kCloudSide; ++j) {
      for (int i = 0; i < kCloudSide; ++i) {
        const float x = static_cast<float>(sx) - 1.5F + 0.025F * (static_cast<float>(i) + 0.5F);
        const float y = -1.5F + 0.025F * (static_cast<float>(j) + 0.5F);
        xyz.insert(xyz.end(),
                   {x - static_cast<float>(sx), y, hm::terrain_height(terrain, x, y) - 1.0F});
      }
    }
    const msg::PointCloudShape shape =
        msg::unorganized_cloud(static_cast<std::int64_t>(kMaxPoints));
    auto loan = msg::allocate_point_cloud(point_cloud, shape, {}, context.cuda_stream());
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    if (auto written = loan->write_host_and_commit(
            msg::PointCloudFields::points,
            [&](const holoscan::TensorOutputWriteGuard& w)
                -> holoscan::expected<void, holoscan::Error> {
              std::memcpy(w.data_as<float>(), xyz.data(), xyz.size() * sizeof(float));
              return {};
            });
        !written) {
      return written;
    }
    const msg::Header header{.frame_id = "sensor",
                             .capture_timestamp_ns = emit.capture_time->timestamp_ns,
                             .device_sequence = emitted_ + 1U};
    return msg::emit_point_cloud(point_cloud, std::move(*loan), header, shape, -1, emit);
  }

  std::uint32_t emitted_{};
  std::optional<holoscan::TimePoint> capture_;
  bool cloud_sent_{};
};

struct Tally {
  std::uint32_t frames{};
  std::uint32_t wrong{};
  double first_origin_x{};
  double last_origin_x{};
};

// Checks each map: the box's top, its lethal rim, and the free ground. The
// layers are copied into one pinned host block on the operator's stream, and
// the host waits once for all of them.
class CheckOp final : public holoscan::Operator<> {
 public:
  explicit CheckOp(std::shared_ptr<Tally> tally) : tally_(std::move(tally)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    // Two slots per input: one map in compute() while the next one arrives.
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

  void start() override {
    if (host_ == nullptr && cudaMallocHost(&host_, kLayerBytes * 2U) != cudaSuccess) {
      host_ = nullptr;
    }
  }

  void stop() override {
    if (host_ != nullptr) {
      cudaFreeHost(host_);
      host_ = nullptr;
    }
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto e = elevation.receive();
    auto i = info.receive();
    auto c = cost.receive();
    auto u = unknown.receive();
    if (!e || !i || !c || !u || host_ == nullptr) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure, "CheckOp: an input or the pinned host block is missing"});
    }
    auto* heights = static_cast<float*>(host_);
    float* costs = heights + kSide * kSide;
    cudaStream_t stream = context.cuda_stream().get();
    for (const auto& [tensor, destination] :
         {std::pair{&e->data, heights}, std::pair{&c->data, costs}}) {
      if (auto copied = tensor->copy_to_host_async(destination, kLayerBytes, stream); !copied) {
        return copied;
      }
    }
    // map_info is in host memory: no GPU work to wait for.
    std::array<double, hm::kMapInfoSize> grid{};
    if (auto copied = i->data.copy_to_host(grid.data(), sizeof(grid)); !copied) {
      return copied;
    }
    if (stream != nullptr) {
      if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure,
            std::string("CheckOp: cudaStreamSynchronize: ") + cudaGetErrorString(status)});
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
  static constexpr std::size_t kLayerBytes = std::size_t{kSide} * kSide * sizeof(float);
  std::shared_ptr<Tally> tally_;
  void* host_{};
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
  map_config.cloud_frame_id = "sensor";
  map_config.outputs.variance = false;
  map_config.max_points = kMaxPoints;
  auto mapper = graph.op<hm::HeightMapOp>("map", map_config);
  auto traversability = graph.op<hm::TraversabilityOp>(
      "traversability", hm::TraversabilityConfig{.max_rows = kSide, .max_cols = kSide});
  auto tally = std::make_shared<Tally>();
  auto check = graph.op<CheckOp>("check", tally);

  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->point_cloud, mapper->point_cloud, lossless);
  graph.add_flow(source->pose, mapper->pose, lossless);
  graph.add_flow(mapper->elevation, traversability->elevation, lossless);
  graph.add_flow(mapper->map_info, traversability->map_info, lossless);
  graph.add_flow(mapper->elevation, check->elevation, lossless);
  graph.add_flow(mapper->map_info, check->info, lossless);
  graph.add_flow(traversability->cost, check->cost, lossless);
  graph.add_flow(traversability->unknown, check->unknown, lossless);
  // Every map must arrive: the whole graph on one worker, as holoscan-messages'
  // README advises for a lossless chain. Across workers an operator could run
  // while a consumer's queue is full and fail on the refused emit.
  graph.partition("height_map_consumer").include_all();
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
  // Warnings on an accepted plan, as well as the reasons for a rejected one.
  if (!plan.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
  }
  if (!plan.ok()) {
    std::cout << "height_map_consumer: FAIL (the plan was rejected)\n";
    return 1;
  }
  const std::uint32_t workers = plan.scheduler_plan().worker_count;
  if (workers != 1U) {
    std::cout << "height_map_consumer: workers=" << workers << ", expected 1\n"
              << "height_map_consumer: FAIL\n";
    return 1;
  }
  auto session = holoscan::run_async(plan);
  if (!session.wait_for(std::chrono::seconds{120})) {
    std::cerr << "height_map_consumer: timed out\n";
    session.request_stop();
  }
  session.wait();
  const bool completed = session.termination() == holoscan::RunTermination::kRequestedCompleted;
  if (!session.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(session.diagnostics(), plan.graph()) << '\n';
  }
  const bool moved = std::fabs(tally->last_origin_x - tally->first_origin_x - 0.4) < 1e-6;
  const bool passed = completed && tally->frames == kFrames && tally->wrong == 0 && moved;
  std::cout << "height_map_consumer: workers=" << workers << " frames=" << tally->frames
            << " wrong=" << tally->wrong << " window_moved=" << (moved ? "yes" : "no") << '\n'
            << "height_map_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
