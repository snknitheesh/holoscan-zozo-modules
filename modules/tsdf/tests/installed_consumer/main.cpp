// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-tsdf package: a source
// sends depth images of the synthetic tabletop scene with their camera poses,
// in host memory; TsdfIntegrateOp integrates them into a volume and
// TsdfMeshOp extracts its mesh after the last one. A sink checks that the
// mesh lies on the scene's surfaces, and the volume is saved as a PLY file.

#include <cuda_runtime_api.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/tsdf/synthetic_scene.hpp>
#include <holoscan/tsdf/tsdf_integrate_op.hpp>
#include <holoscan/tsdf/tsdf_mesh_op.hpp>

namespace {

namespace ht = holoscan::tsdf;

constexpr int kWidth = 160;
constexpr int kHeight = 120;
constexpr std::uint32_t kFrames = 8;

holoscan::TensorOutputSpec host_tensor(std::uint32_t rank, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kHost,
              .dtype = ht::kFloat32,
              .rank = rank,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(bytes),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

// kFrames depth images of the orbit and their poses, then idle.
class DepthSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(depth, "depth")
        .produces_tensor(host_tensor(2U, kPixels * 4U))
        .payload_options(holoscan::PayloadOptions{
            .schema = holoscan::schema_identity<holoscan::schema::ImageT>()});
    spec.output(pose, "pose").produces_tensor(host_tensor(2U, 64U));
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
    const ht::Matrix4 world_from_camera =
        ht::orbit_pose(ht::SceneKind::kTabletop, emitted_, kFrames);
    std::vector<float> meters(kPixels);
    float* device = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device), kPixels * 4U) != cudaSuccess) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "cudaMalloc failed"});
    }
    const cudaError_t rendered =
        ht::render_scene(ht::synthetic_scene(ht::SceneKind::kTabletop), world_from_camera,
                         ht::synthetic_intrinsics(kWidth, kHeight), kWidth, kHeight, 3.0F,
                         ht::DepthNoise{}, ht::DepthType::kFloat32, device, nullptr, nullptr);
    const cudaError_t copied =
        cudaMemcpy(meters.data(), device, kPixels * 4U, cudaMemcpyDeviceToHost);
    cudaFree(device);
    if (rendered != cudaSuccess || copied != cudaSuccess) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "rendering failed"});
    }
    auto depth_loan = depth.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = std::array<std::int64_t, 2>{kHeight, kWidth}, .dtype = ht::kFloat32});
    auto pose_loan = pose.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = std::array<std::int64_t, 2>{4, 4}, .dtype = ht::kFloat32});
    if (!depth_loan || !pose_loan) {
      return holoscan::make_unexpected(!depth_loan ? depth_loan.error() : pose_loan.error());
    }
    const auto fill = [](holoscan::TensorOutputLoan& loan, const void* data, std::size_t bytes) {
      return loan.write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                            -> holoscan::expected<void, holoscan::Error> {
        std::memcpy(writer.data_as<std::uint8_t>(), data, bytes);
        return {};
      });
    };
    if (auto filled = fill(*depth_loan, meters.data(), kPixels * 4U); !filled) {
      return filled;
    }
    if (auto filled = fill(*pose_loan, world_from_camera.data(), 64U); !filled) {
      return filled;
    }
    const holoscan::EmitOptions emit{.capture_time = context.activation_time(),
                                     .frame_id = emitted_ + 1U};
    holoscan::schema::ImageT descriptor;
    descriptor.width = kWidth;
    descriptor.height = kHeight;
    descriptor.encoding = holoscan::schema::ImageEncoding_MONO32F;
    if (auto sent = depth.emit_tensor(std::move(*depth_loan), descriptor, emit); !sent) {
      return sent;
    }
    if (auto sent = pose.emit(std::move(*pose_loan), emit); !sent) {
      return sent;
    }
    ++emitted_;
    return {};
  }

  holoscan::Output<holoscan::schema::ImageT> depth;
  holoscan::Output<holoscan::Tensor> pose;

 private:
  static constexpr std::uint64_t kPixels = std::uint64_t{kWidth} * kHeight;
  std::uint32_t emitted_{};
};

struct Tally {
  std::uint32_t meshes{};
  std::uint32_t triangles{};
  double mean_error_m{};
};

// Checks the mesh; takes every mesh output, which must be connected.
class CheckOp final : public holoscan::Operator<> {
 public:
  explicit CheckOp(std::shared_ptr<Tally> tally) : tally_(std::move(tally)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(vertices, "vertices");
    spec.input(normals, "normals");
    spec.input(triangles, "triangles");
    spec.input(counts, "counts");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    using std::chrono_literals::operator""ns;
    holoscan::Contract result;
    result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{std::vector<holoscan::PortRef>{
                                           vertices, normals, triangles, counts}},
                                       .window = 0ns});
    for (const auto* port : {&vertices, &normals, &triangles, &counts}) {
      result.input(*port, holoscan::Exact{});
    }
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto vertex_sample = vertices.receive();
    auto count_sample = counts.receive();
    for (auto* port : {&normals, &triangles}) {
      if (auto sample = port->receive(); !sample) {
        return holoscan::make_unexpected(sample.error());
      }
    }
    if (!vertex_sample || !count_sample) {
      return holoscan::make_unexpected(!vertex_sample ? vertex_sample.error()
                                                      : count_sample.error());
    }
    std::array<std::uint32_t, ht::kMeshCountsSize> mesh_counts{};
    if (auto copied = count_sample->data.copy_to_host(mesh_counts.data(), sizeof(mesh_counts),
                                                      context.cuda_stream());
        !copied) {
      return copied;
    }
    std::vector<float> points(static_cast<std::size_t>(vertex_sample->data.size()));
    if (auto copied = vertex_sample->data.copy_to_host(points.data(), points.size() * 4U,
                                                       context.cuda_stream());
        !copied) {
      return copied;
    }
    const ht::SyntheticScene scene = ht::synthetic_scene(ht::SceneKind::kTabletop);
    double sum = 0.0;
    for (std::uint32_t i = 0; i < mesh_counts[ht::kCountVertices]; ++i) {
      sum += std::abs(ht::scene_distance(scene, &points[3U * i]));
    }
    ++tally_->meshes;
    tally_->triangles = mesh_counts[ht::kCountTriangles];
    tally_->mean_error_m =
        mesh_counts[ht::kCountVertices] == 0U ? 1.0 : sum / mesh_counts[ht::kCountVertices];
    context.request_completion();
    return {};
  }

  holoscan::Input<holoscan::Tensor> vertices;
  holoscan::Input<holoscan::Tensor> normals;
  holoscan::Input<holoscan::Tensor> triangles;
  holoscan::Input<holoscan::Tensor> counts;

 private:
  std::shared_ptr<Tally> tally_;
};

}  // namespace

int main() {
  // Take over SIGINT and SIGTERM while this is the only thread, before CUDA
  // starts its own; run() adopts the domain.
  auto signals = holoscan::ManagedShutdownSignals::create();
  if (!signals) {
    std::cerr << "tsdf_consumer: no managed signal domain\n";
    return 1;
  }
  const ht::TsdfVolumeConfig volume_config = ht::scene_volume(ht::SceneKind::kTabletop, 96, false);
  auto volume = std::make_shared<ht::TsdfVolume>(volume_config);

  holoscan::Graph graph{"tsdf_consumer"};
  auto source = graph.op<DepthSourceOp>("source");
  auto integrate = graph.op<ht::TsdfIntegrateOp>(
      "integrate", ht::TsdfIntegrateConfig{.volume = volume,
                                           .intrinsics = ht::synthetic_intrinsics(kWidth, kHeight),
                                           .depth_scale = 1.0F,
                                           .depth_max = 3.0F,
                                           .max_width = kWidth,
                                           .max_height = kHeight});
  auto mesh = graph.op<ht::TsdfMeshOp>("mesh", ht::TsdfMeshConfig{.volume = volume,
                                                                  .mesh_every = kFrames,
                                                                  .max_vertices = 1U << 16,
                                                                  .max_triangles = 1U << 17});
  auto tally = std::make_shared<Tally>();
  auto check = graph.op<CheckOp>("check", tally);

  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->depth, integrate->depth, lossless);
  graph.add_flow(source->pose, integrate->pose, lossless);
  graph.add_flow(integrate->integrated_pose, mesh->pose, lossless);
  graph.add_flow(mesh->vertices, check->vertices, lossless);
  graph.add_flow(mesh->normals, check->normals, lossless);
  graph.add_flow(mesh->triangles, check->triangles, lossless);
  graph.add_flow(mesh->mesh_counts, check->counts, lossless);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // The mesh outputs are in device memory, on GPU 0.
  holoscan::CompileOptions options;
  for (const char* port : {"vertices", "normals", "triangles", "mesh_counts"}) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "mesh", .output_port = port, .device = holoscan::DeviceId{0}});
  }
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (const int status = holoscan::run(plan); status != 0) {
    std::cerr << "tsdf_consumer: run failed with " << status << '\n';
    return 1;
  }
  const std::filesystem::path ply = std::filesystem::temp_directory_path() /
                                    ("tsdf_consumer_" + std::to_string(::getpid()) + ".ply");
  volume->export_ply(ply.string());
  const bool saved = std::filesystem::file_size(ply) > 1000U;
  std::filesystem::remove(ply);
  const bool passed = tally->meshes == 1 && tally->triangles > 1000 &&
                      tally->mean_error_m < 0.5 * volume_config.voxel_size && saved;
  std::cout << "tsdf_consumer: meshes=" << tally->meshes << " triangles=" << tally->triangles
            << " mean_error_mm=" << 1000.0 * tally->mean_error_m << " ply=" << (saved ? 1 : 0)
            << '\n'
            << "tsdf_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
