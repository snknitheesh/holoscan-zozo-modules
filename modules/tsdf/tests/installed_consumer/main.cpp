// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A project built against the installed holoscan-tsdf package. add_tsdf() is
// the README's "Use it in your graph" snippet: a camera's depth images and
// poses, and its calibration from a camera_info file through
// CameraInfoSourceOp, into TsdfIntegrateOp and TsdfMeshOp. The live form
// (drop-oldest edges, automatic partitioning) is compiled; the lossless form
// (every frame, one worker) runs: a source sends depth images of the synthetic
// tabletop scene in host memory, the mesh after the last one must lie on the
// scene's surfaces, every ray cast must be a valid point cloud, and the volume
// is saved as a PLY file.

#include <cuda_runtime_api.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/tsdf/synthetic_scene.hpp>
#include <holoscan/tsdf/tsdf_integrate_op.hpp>
#include <holoscan/tsdf/tsdf_mesh_op.hpp>

namespace {

namespace hm = holoscan::messages;
namespace ht = holoscan::tsdf;
using std::chrono_literals::operator""ms;
using std::chrono_literals::operator""ns;

constexpr int kWidth = 160;
constexpr int kHeight = 120;
constexpr std::uint32_t kFrames = 8;
constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};

holoscan::TensorOutputSpec host_tensor(std::uint32_t rank, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kHost,
              .dtype = kFloat32,
              .rank = rank,
          },
      .bounds = holoscan::tensor_bounds(bytes),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

// kFrames depth images of the orbit and their poses, rendered once when the
// run starts, one per tick from the second (CameraInfoSourceOp sends when the
// run starts), then idle. An image the runtime has no room for is sent again
// on the next tick.
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
    result.trigger(holoscan::OnClock{.period = 20ms});
    return result;
  }

  void start() override {
    emitted_ = 0;
    ticks_ = 0;
    frames_.assign(kFrames, std::vector<float>(kPixels));
    float* device = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device), kPixels * 4U) != cudaSuccess) {
      throw std::runtime_error("DepthSourceOp: cudaMalloc failed");
    }
    for (std::uint32_t f = 0; f < kFrames; ++f) {
      const cudaError_t rendered =
          ht::render_scene(ht::synthetic_scene(ht::SceneKind::kTabletop),
                           ht::orbit_pose(ht::SceneKind::kTabletop, f, kFrames),
                           ht::synthetic_intrinsics(kWidth, kHeight), kWidth, kHeight, 3.0F,
                           ht::DepthNoise{}, ht::DepthType::kFloat32, device, nullptr, nullptr);
      if (rendered != cudaSuccess || cudaMemcpy(frames_[f].data(), device, kPixels * 4U,
                                                cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(device);
        throw std::runtime_error("DepthSourceOp: rendering failed");
      }
    }
    cudaFree(device);
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (ticks_++ == 0 || emitted_ >= kFrames) {
      return {};
    }
    const ht::Matrix4 world_from_camera =
        ht::orbit_pose(ht::SceneKind::kTabletop, emitted_, kFrames);
    auto depth_loan = depth.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = std::array<std::int64_t, 2>{kHeight, kWidth}, .dtype = kFloat32});
    auto pose_loan = pose.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = std::array<std::int64_t, 2>{4, 4}, .dtype = kFloat32});
    for (const auto* loan : {&depth_loan, &pose_loan}) {
      if (!*loan) {
        return loan->error().code == holoscan::ErrorCode::kResourceExhausted
                   ? holoscan::expected<void, holoscan::Error>{}
                   : holoscan::make_unexpected(loan->error());
      }
    }
    const auto fill = [](holoscan::TensorOutputLoan& loan, const void* data, std::size_t bytes) {
      return loan.write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                            -> holoscan::expected<void, holoscan::Error> {
        std::memcpy(writer.data_as<std::uint8_t>(), data, bytes);
        return {};
      });
    };
    if (auto filled = fill(*depth_loan, frames_[emitted_].data(), kPixels * 4U); !filled) {
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
    // The integration takes both or neither: a refused depth image is sent
    // again, with its pose, on the next tick.
    if (auto sent = depth.emit_tensor(std::move(*depth_loan), descriptor, emit); !sent) {
      return sent.error().code == holoscan::ErrorCode::kBackpressured
                 ? holoscan::expected<void, holoscan::Error>{}
                 : sent;
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
  std::vector<std::vector<float>> frames_;
  std::uint32_t emitted_{};
  std::uint32_t ticks_{};
};

struct Tally {
  std::uint32_t meshes{};
  std::uint32_t triangles{};
  double mean_error_m{};
  std::uint32_t clouds{};
  std::uint32_t bad_clouds{};
};

// Checks the mesh; takes every mesh output, which must be connected.
class MeshCheckOp final : public holoscan::Operator<> {
 public:
  explicit MeshCheckOp(std::shared_ptr<Tally> tally) : tally_(std::move(tally)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(vertices, "vertices");
    spec.input(normals, "normals");
    spec.input(triangles, "triangles");
    spec.input(counts, "counts");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAligned{
        .ports = holoscan::ports(vertices, normals, triangles, counts), .window = 0ns});
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

// Counts the ray casts, checking each cloud's layout.
class CloudCheckOp final : public holoscan::Operator<> {
 public:
  explicit CloudCheckOp(std::shared_ptr<Tally> tally) : tally_(std::move(tally)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(depth, "depth");
    spec.input(cloud, "cloud");
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAligned{.ports = holoscan::ports(depth, cloud), .window = 0ns});
    result.input(depth, holoscan::Exact{});
    result.input(cloud, holoscan::Exact{});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto depth_sample = depth.receive();
    auto cloud_sample = cloud.receive();
    if (!depth_sample || !cloud_sample) {
      return holoscan::make_unexpected(!depth_sample ? depth_sample.error() : cloud_sample.error());
    }
    ++tally_->clouds;
    const hm::schema::PointCloudT& message = cloud_sample->data;
    if (!hm::check_point_cloud(message) || !message.organized || !hm::is_present(message.normals) ||
        message.points->shape() != std::vector<std::int64_t>{kHeight, kWidth, 3}) {
      ++tally_->bad_clouds;
    }
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> depth;
  holoscan::Input<hm::schema::PointCloudT> cloud;

 private:
  std::shared_ptr<Tally> tally_;
};

// A folder for the camera_info file, which mkdtemp() names unpredictably and
// makes private; removed with everything in it, on every way out.
class TemporaryFolder {
 public:
  TemporaryFolder() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "tsdf_consumer_XXXXXX").string();
    if (::mkdtemp(pattern.data()) == nullptr) {
      throw std::system_error(errno, std::generic_category(), "mkdtemp");
    }
    path_ = pattern;
  }
  ~TemporaryFolder() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  TemporaryFolder(const TemporaryFolder&) = delete;
  TemporaryFolder& operator=(const TemporaryFolder&) = delete;
  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

// The README's "Use it in your graph": `camera_info_file` is the depth
// camera's ROS camera_info file. `live`: a live camera, whose newest frames
// matter (drop-oldest edges); else every frame (lossless edges, one worker).
// Returns TsdfMeshOp, whose outputs must all be connected.
template <typename Camera>
auto add_tsdf(holoscan::Graph& graph, Camera& camera, const std::filesystem::path& camera_info_file,
              const std::shared_ptr<ht::TsdfVolume>& volume, bool live) {
  auto info = graph.op<hm::CameraInfoSourceOp>(
      "camera_info", hm::CameraInfoSourceConfig{.camera_info_path = camera_info_file});
  auto integrate =
      graph.op<ht::TsdfIntegrateOp>("integrate", ht::TsdfIntegrateConfig{.volume = volume,
                                                                         .depth_scale = 1.0F,
                                                                         .depth_max = 3.0F,
                                                                         .max_width = kWidth,
                                                                         .max_height = kHeight});
  auto mesh = graph.op<ht::TsdfMeshOp>("mesh", ht::TsdfMeshConfig{.volume = volume,
                                                                  .mesh_every = kFrames,
                                                                  .max_vertices = 1U << 16,
                                                                  .max_triangles = 1U << 17,
                                                                  .raycast = true,
                                                                  .raycast_depth_max = 3.0F,
                                                                  .max_width = kWidth,
                                                                  .max_height = kHeight});
  const holoscan::ConnectionOptions edge =
      live ? holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                         .queue_depth = 1U}
           : holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(camera->depth, integrate->depth, edge);
  graph.add_flow(camera->pose, integrate->pose, edge);
  graph.add_flow(info->camera_info, integrate->camera_info, hm::latest_camera_info_connection());
  graph.add_flow(info->camera_info, mesh->camera_info, hm::latest_camera_info_connection());
  graph.add_flow(integrate->integrated_pose, mesh->pose, edge);
  return mesh;
}

// The graph with its checks; `live` as for add_tsdf().
holoscan::ExecutionPlan build(const std::filesystem::path& camera_info_file,
                              const std::shared_ptr<ht::TsdfVolume>& volume,
                              const std::shared_ptr<Tally>& tally, bool live) {
  holoscan::Graph graph{"tsdf_consumer"};
  auto source = graph.op<DepthSourceOp>("source");
  auto mesh = add_tsdf(graph, source, camera_info_file, volume, live);
  auto mesh_check = graph.op<MeshCheckOp>("mesh_check", tally);
  auto cloud_check = graph.op<CloudCheckOp>("cloud_check", tally);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(mesh->vertices, mesh_check->vertices, lossless);
  graph.add_flow(mesh->normals, mesh_check->normals, lossless);
  graph.add_flow(mesh->triangles, mesh_check->triangles, lossless);
  graph.add_flow(mesh->mesh_counts, mesh_check->counts, lossless);
  graph.add_flow(mesh->raycast_depth, cloud_check->depth, lossless);
  graph.add_flow(mesh->raycast_cloud, cloud_check->cloud, lossless);
  if (!live) {
    // Every frame must arrive: the whole graph on one worker, so that no
    // operator runs while its consumer's queue is full.
    graph.partition("tsdf_consumer").include_all();
  }
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // The mesh and ray-cast outputs are in device memory, on GPU 0.
  holoscan::CompileOptions options;
  for (const char* port :
       {"vertices", "normals", "triangles", "mesh_counts", "raycast_depth", "raycast_cloud"}) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "mesh", .output_port = port, .device = holoscan::DeviceId{0}});
  }
  return holoscan::compile(graph, std::move(options));
}

}  // namespace

int main() try {
  const TemporaryFolder folder;
  const std::filesystem::path camera_info_file = folder.path() / "depth.yaml";
  hm::write_camera_info_yaml(ht::synthetic_camera_info(kWidth, kHeight, "depth_optical"),
                             camera_info_file);
  const ht::TsdfVolumeConfig volume_config = ht::scene_volume(ht::SceneKind::kTabletop, 96, false);
  auto volume = std::make_shared<ht::TsdfVolume>(volume_config);
  auto tally = std::make_shared<Tally>();

  const holoscan::ExecutionPlan live = build(camera_info_file, volume, tally, true);
  if (!live.ok()) {
    std::cerr << "tsdf_consumer: the live graph does not compile:\n"
              << holoscan::format_diagnostics(live.diagnostics(), live.graph()) << '\n';
    return 1;
  }
  const holoscan::ExecutionPlan plan = build(camera_info_file, volume, tally, false);
  if (!plan.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
  }
  if (!plan.ok()) {
    return 1;
  }
  const std::uint32_t workers = plan.scheduler_plan().worker_count;
  if (workers != 1U) {
    std::cerr << "tsdf_consumer: the lossless graph runs on " << workers << " workers, not one\n";
    return 1;
  }
  auto session = holoscan::run_async(plan);
  if (!session.wait_for(std::chrono::seconds(60))) {
    std::cerr << "tsdf_consumer: timed out\n";
    session.request_stop();
  }
  session.wait();
  if (!session.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(session.diagnostics(), plan.graph()) << '\n';
  }
  const std::filesystem::path ply = folder.path() / "scene.ply";
  volume->export_ply(ply.string());
  const bool saved = std::filesystem::file_size(ply) > 1000U;
  const bool passed = session.termination() == holoscan::RunTermination::kRequestedCompleted &&
                      tally->meshes == 1 && tally->triangles > 1000 &&
                      tally->mean_error_m < 0.5 * volume_config.voxel_size &&
                      tally->clouds == kFrames && tally->bad_clouds == 0 && saved;
  std::cout << "tsdf_consumer: workers=" << workers << " meshes=" << tally->meshes
            << " triangles=" << tally->triangles
            << " mean_error_mm=" << 1000.0 * tally->mean_error_m << " clouds=" << tally->clouds
            << " bad_clouds=" << tally->bad_clouds << " ply=" << (saved ? 1 : 0) << '\n'
            << "tsdf_consumer: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
} catch (const std::exception& error) {
  std::cerr << "tsdf_consumer: " << error.what() << '\n';
  return 1;
}
