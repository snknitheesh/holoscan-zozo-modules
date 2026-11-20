// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The height map of the local recordings: for each clip in
// $HOLOSCAN_TEST_DATA/stereo-camera/<clip>/ with a depth.yaml, its first
// depth images (16-bit PGM) go through CameraInfoSourceOp (the clip's
// depth.yaml), DepthToPointCloudOp and HeightMapOp, on one worker, into a
// fused map in the camera's optical frame (x right, y down; the elevation is
// the depth). A checker maps each cloud the operator received with
// HeightMapper run directly, and every map must be the same, bit for bit. The
// scene is static, so the fused map settles: the median change of a cell from
// one map to the next falls, the last one no larger than the first and at
// most kMaxLastChangeM. Skipped
// (exit 77) when HOLOSCAN_TEST_DATA is not set, as in CI, whose tests make
// their own data.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/height_map/height_map_op.hpp>
#include <holoscan/height_map/height_mapper.hpp>
#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/point_cloud/depth_to_point_cloud_op.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

namespace fs = std::filesystem;
namespace hm = holoscan::height_map;
namespace msg = holoscan::messages;
namespace pc = holoscan::point_cloud;

constexpr DLDataType kUInt16{kDLUInt, 16U, 1U};
// The images of each clip that go through the operators.
constexpr std::size_t kFrames = 10;
// The recordings' depth unit, meters.
constexpr float kDepthScale = 0.000452F;
// A static scene: the last map's median change from the one before. Measured
// on the local recordings: 0.5 and 0.7 mm after ten frames of dense depth
// (from 8.8 mm after the first), 0 with sparse depth.
constexpr double kMaxLastChangeM = 0.002;

int failures = 0;

void check(bool condition, const std::string& what) {
  if (!condition) {
    ++failures;
    std::cerr << "check failed: " << what << '\n';
  }
}

// The map: 20 m square in the optical frame's x-y plane, 5 cm cells, fused.
hm::HeightMapParams map_params() {
  hm::HeightMapParams p;
  p.geometry = hm::MapGeometry{
      .resolution = 0.05F, .rows = 400, .cols = 400, .origin_x = -10.0, .origin_y = -10.0};
  p.elevation = hm::ElevationMode::kFused;
  return p;
}

struct DepthImage {
  int width{};
  int height{};
  std::vector<std::uint16_t> counts;
};

// A binary 16-bit PGM (P5, maxval above 255, big-endian samples).
DepthImage read_pgm16(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  std::string magic;
  int maxval = 0;
  DepthImage image;
  file >> magic >> image.width >> image.height >> maxval;
  file.get();  // The one whitespace after the header.
  if (!file || magic != "P5" || maxval <= 255 || image.width <= 0 || image.height <= 0) {
    throw std::runtime_error(path.string() + ": not a binary 16-bit PGM");
  }
  std::vector<unsigned char> bytes(static_cast<std::size_t>(image.width) * image.height * 2U);
  file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!file) {
    throw std::runtime_error(path.string() + ": truncated");
  }
  image.counts.resize(bytes.size() / 2U);
  for (std::size_t i = 0; i < image.counts.size(); ++i) {
    image.counts[i] = static_cast<std::uint16_t>((bytes[2 * i] << 8U) | bytes[2 * i + 1]);
  }
  return image;
}

// Sends the given images as MONO16 in host memory, one per tick, frame
// numbers from 1; one a consumer has no room for goes again on the next tick.
class ImageSourceOp final : public holoscan::Operator<> {
 public:
  explicit ImageSourceOp(std::shared_ptr<const std::vector<DepthImage>> images)
      : images_(std::move(images)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(depth, "depth")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kHost,
                    .dtype = kUInt16,
                    .rank = 2U,
                    .strides = holoscan::TensorStridePolicy::kContiguousOnly},
            .bounds = holoscan::tensor_bounds(images_->front().counts.size() * 2U),
            .storage = holoscan::TensorOutputStorage::kRuntimePool})
        .payload_options(holoscan::PayloadOptions{
            .schema = holoscan::schema_identity<holoscan::schema::ImageT>()});
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds(20)});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (next_ >= images_->size()) {
      return {};
    }
    const DepthImage& image = (*images_)[next_];
    auto loan = depth.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = std::array<std::int64_t, 2>{image.height, image.width}, .dtype = kUInt16});
    if (!loan) {
      return loan.error().code == holoscan::ErrorCode::kResourceExhausted
                 ? holoscan::expected<void, holoscan::Error>{}
                 : holoscan::make_unexpected(std::move(loan).error());
    }
    if (auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& w) {
          std::copy(image.counts.begin(), image.counts.end(), w.data_as<std::uint16_t>());
        });
        !written) {
      return written;
    }
    holoscan::schema::ImageT descriptor;
    descriptor.width = image.width;
    descriptor.height = image.height;
    descriptor.encoding = holoscan::schema::ImageEncoding_MONO16;
    auto sent = depth.emit_tensor(
        std::move(*loan), descriptor,
        holoscan::EmitOptions{.capture_time = context.activation_time(), .frame_id = next_ + 1U});
    if (!sent) {
      return sent.error().code == holoscan::ErrorCode::kBackpressured ? decltype(sent){} : sent;
    }
    ++next_;
    return {};
  }

  holoscan::Output<holoscan::schema::ImageT> depth;

 private:
  std::shared_ptr<const std::vector<DepthImage>> images_;
  std::uint64_t next_{};
};

struct Tally {
  std::uint64_t first_frame_id{};
  std::atomic<std::uint32_t> maps{};
  std::uint64_t wrong_cells{};
  std::uint32_t wrong_info{};
  std::uint64_t known_cells{};  // Of the last map.
  std::vector<float> previous_map;
  // The median change of the cells known in both, from each map to the next.
  std::vector<double> changes;
};

// The median of |after - before| over the cells known in both maps.
double median_change(const std::vector<float>& before, const std::vector<float>& after) {
  std::vector<double> changes;
  for (std::size_t i = 0; i < before.size() && i < after.size(); ++i) {
    if (std::isfinite(before[i]) && std::isfinite(after[i])) {
      changes.push_back(std::fabs(static_cast<double>(after[i]) - before[i]));
    }
  }
  if (changes.empty()) return std::nan("");
  std::nth_element(changes.begin(), changes.begin() + changes.size() / 2, changes.end());
  return changes[changes.size() / 2];
}

// Maps every cloud HeightMapOp received with HeightMapper run directly, and
// compares the two elevations bit for bit.
class CheckOp final : public holoscan::Operator<> {
 public:
  CheckOp(std::size_t frames, std::shared_ptr<Tally> tally)
      : frames_(frames), tally_(std::move(tally)), reference_(map_params()) {}
  void setup(holoscan::OperatorSpec& spec) override {
    // Two slots per input: one map in compute() while the next one arrives.
    spec.input(point_cloud, "point_cloud").queue_depth(2U);
    spec.input(elevation, "elevation").queue_depth(2U);
    spec.input(map_info, "map_info").queue_depth(2U);
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{std::vector<holoscan::PortRef>{
                                           point_cloud, elevation, map_info}},
                                       .window = std::chrono::nanoseconds{0}});
    result.input(point_cloud, holoscan::Exact{});
    result.input(elevation, holoscan::Exact{});
    result.input(map_info, holoscan::Exact{});
    return result;
  }
  void start() override {
    reference_.reset();
    const std::size_t bytes = map_params().geometry.cells() * sizeof(float);
    if (layer_ == nullptr && cudaMalloc(&layer_, bytes) != cudaSuccess) {
      layer_ = nullptr;
    }
  }
  void stop() override {
    if (layer_ != nullptr) {
      cudaFree(layer_);
      layer_ = nullptr;
    }
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const auto fail = [](const std::string& what) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "CheckOp: " + what});
    };
    auto cloud = point_cloud.receive();
    auto map = elevation.receive();
    auto grid = map_info.receive();
    if (!cloud || !map || !grid || layer_ == nullptr) {
      return fail("missing input or scratch");
    }
    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return fail("no CUDA stream");
    }
    const holoscan::Tensor& points = *cloud->data.points;
    if (points.device().device_type != kDLCUDA || points.ndim() != 3) {
      return fail("expected an organized cloud in device memory");
    }
    const auto count = static_cast<std::uint32_t>(points.shape_span()[0] * points.shape_span()[1]);
    if (const cudaError_t status =
            reference_.update(static_cast<const float*>(points.data()), count, hm::Transform{},
                              cloud->metadata.capture_timestamp_ns, stream);
        status != cudaSuccess) {
      return fail(std::string("update: ") + cudaGetErrorString(status));
    }
    if (const cudaError_t status = reference_.write_layers(
            hm::MapLayers{.elevation = static_cast<float*>(layer_)}, stream);
        status != cudaSuccess) {
      return fail(std::string("write_layers: ") + cudaGetErrorString(status));
    }
    const std::size_t cells = map_params().geometry.cells();
    std::vector<float> want(cells);
    std::vector<float> got(cells);
    if (const cudaError_t status = cudaMemcpyAsync(want.data(), layer_, cells * sizeof(float),
                                                   cudaMemcpyDeviceToHost, stream);
        status != cudaSuccess) {
      return fail(std::string("cudaMemcpyAsync: ") + cudaGetErrorString(status));
    }
    if (auto copied = map->data.copy_to_host(got.data(), cells * sizeof(float), stream); !copied) {
      return copied;
    }
    if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
      return fail(std::string("cudaStreamSynchronize: ") + cudaGetErrorString(status));
    }
    const hm::MapGeometry g = map_params().geometry;
    std::array<double, hm::kMapInfoSize> values{};
    if (auto copied = grid->data.copy_to_host(values.data(), sizeof(values)); !copied) {
      return copied;
    }
    tally_->wrong_info += values == std::array<double, hm::kMapInfoSize>{g.origin_x, g.origin_y,
                                                                         g.resolution, 400.0, 400.0}
                              ? 0U
                              : 1U;
    tally_->wrong_cells +=
        cells - static_cast<std::size_t>(
                    std::count_if(got.begin(), got.end(), [&, i = std::size_t{0}](float v) mutable {
                      return std::memcmp(&v, &want[i++], sizeof(float)) == 0;
                    }));
    if (tally_->maps.fetch_add(1U) == 0U) {
      tally_->first_frame_id = cloud->metadata.frame_id;
    } else {
      tally_->changes.push_back(median_change(tally_->previous_map, got));
    }
    tally_->previous_map = got;
    tally_->known_cells = static_cast<std::uint64_t>(
        std::count_if(got.begin(), got.end(), [](float v) { return std::isfinite(v); }));
    if (cloud->metadata.frame_id == frames_) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<msg::schema::PointCloudT> point_cloud;
  holoscan::Input<holoscan::Tensor> elevation;
  holoscan::Input<holoscan::Tensor> map_info;

 private:
  std::size_t frames_;
  std::shared_ptr<Tally> tally_;
  hm::HeightMapper reference_;
  void* layer_{};
};

// The clip's first kFrames depth images, in name order.
std::vector<DepthImage> first_images(const fs::path& folder) {
  std::vector<fs::path> files;
  for (const auto& entry : fs::directory_iterator(folder)) {
    if (entry.path().extension() == ".pgm") files.push_back(entry.path());
  }
  std::sort(files.begin(), files.end());
  files.resize(std::min(files.size(), kFrames));
  std::vector<DepthImage> images;
  for (const auto& file : files) images.push_back(read_pgm16(file));
  return images;
}

void check_clip(const fs::path& clip, const std::string& name) {
  const fs::path camera_file = clip / "depth.yaml";
  const msg::CameraInfo camera = msg::read_camera_info_yaml(camera_file);
  auto images = std::make_shared<const std::vector<DepthImage>>(first_images(clip / "depth"));
  check(!images->empty(), name + " has depth images");
  if (images->empty()) return;
  const int width = images->front().width;
  const int height = images->front().height;

  holoscan::Graph graph{"height_map_real_data"};
  auto source = graph.op<ImageSourceOp>("source", images);
  auto info = graph.op<msg::CameraInfoSourceOp>(
      "camera_info", msg::CameraInfoSourceConfig{.camera_info_path = camera_file,
                                                 .period = std::chrono::milliseconds(20)});
  auto deproject = graph.op<pc::DepthToPointCloudOp>(
      "deproject", pc::DepthToPointCloudConfig{.depth_scale = kDepthScale,
                                               .depth_min = 0.1F,
                                               .depth_max = 20.0F,
                                               .max_width = width,
                                               .max_height = height});
  hm::HeightMapConfig map_config;
  map_config.map = map_params();
  map_config.cloud_frame_id = camera.header.frame_id;
  map_config.outputs.variance = false;
  map_config.max_points = static_cast<std::uint32_t>(width) * static_cast<std::uint32_t>(height);
  auto mapper = graph.op<hm::HeightMapOp>("height_map", map_config);
  auto tally = std::make_shared<Tally>();
  auto checker = graph.op<CheckOp>("check", images->size(), tally);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->depth, deproject->depth, lossless);
  graph.add_flow(info->camera_info, deproject->camera_info, msg::latest_camera_info_connection());
  graph.add_flow(deproject->point_cloud, mapper->point_cloud, lossless);
  graph.add_flow(deproject->point_cloud, checker->point_cloud, lossless);
  graph.add_flow(mapper->elevation, checker->elevation, lossless);
  graph.add_flow(mapper->map_info, checker->map_info, lossless);
  graph.partition("height_map_real_data").include_all();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  holoscan::CompileOptions options;
  for (const auto& [op, port] :
       {std::pair{"deproject", "point_cloud"}, std::pair{"height_map", "elevation"}}) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
  }
  const auto plan = holoscan::compile(graph, std::move(options));
  check(plan.ok(), holoscan::format_diagnostics(plan.diagnostics(), plan.graph()));
  if (!plan.ok()) return;
  check(plan.scheduler_plan().worker_count == 1U, name + ": one worker");
  auto session = holoscan::run_async(plan);
  if (!session.wait_for(std::chrono::seconds(120))) session.request_stop();
  session.wait();
  check(session.termination() == holoscan::RunTermination::kRequestedCompleted,
        name + ": " + holoscan::format_diagnostics(session.diagnostics(), plan.graph()));
  // The frames before the first camera info give no cloud: at most the first,
  // as both sources start with the run. Every frame after it gives a map.
  const std::uint32_t maps = tally->maps.load();
  check(maps != 0U && tally->first_frame_id <= 2U &&
            tally->first_frame_id + maps - 1U == images->size(),
        name + ": " + std::to_string(maps) + " maps of " + std::to_string(images->size()) +
            " images, from frame " + std::to_string(tally->first_frame_id));
  check(tally->wrong_cells == 0,
        name + ": " + std::to_string(tally->wrong_cells) + " cells differ from HeightMapper's");
  check(tally->wrong_info == 0, name + ": map_info is the configured grid");
  check(tally->known_cells != 0, name + ": the map knows some cells");
  std::string changes;
  for (const double change : tally->changes) changes += " " + std::to_string(change);
  const double change = tally->changes.empty() ? std::nan("") : tally->changes.back();
  check(!tally->changes.empty() && change <= tally->changes.front() && change <= kMaxLastChangeM,
        name + ": the static scene's fused map settles; median changes:" + changes);
  std::cout << name << ": " << width << "x" << height << " maps=" << maps
            << " known_cells=" << tally->known_cells << " wrong_cells=" << tally->wrong_cells
            << " median_change_m=" << change << " changes=" << changes << '\n';
}

}  // namespace

int main() try {
  const char* root = std::getenv("HOLOSCAN_TEST_DATA");
  if (root == nullptr || *root == '\0') {
    std::cout << "skipped: HOLOSCAN_TEST_DATA is not set\n";
    return 77;
  }
  std::vector<fs::path> clips;
  const fs::path recordings = fs::path(root) / "stereo-camera";
  if (fs::is_directory(recordings)) {
    for (const auto& entry : fs::directory_iterator(recordings)) {
      if (fs::is_regular_file(entry.path() / "depth.yaml")) clips.push_back(entry.path());
    }
  }
  std::sort(clips.begin(), clips.end());
  if (clips.empty()) {
    std::cerr << "no stereo-camera/<clip>/depth.yaml in " << root << '\n';
    return 1;
  }
  // The clips are named by position, as their folder names are the recordings'.
  for (std::size_t k = 0; k < clips.size(); ++k) {
    check_clip(clips[k], "clip " + std::to_string(k + 1));
  }
  std::cout << (failures == 0 ? "all passed" : std::to_string(failures) + " check(s) failed")
            << '\n';
  return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
