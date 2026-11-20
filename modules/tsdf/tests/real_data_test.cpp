// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Both operators on the local recordings, which a static camera made of a
// static scene: for each clip in $HOLOSCAN_TEST_DATA/stereo-camera/<clip>/
// with a depth.yaml, its first depth images (16-bit PGM) go through
// TsdfIntegrateOp with the identity pose and that camera, sent by
// CameraInfoSourceOp, into a volume around what the camera sees, and
// TsdfMeshOp ray casts the volume from the same pose through the same camera.
// The last ray cast must see most surfaces inside the volume (over 80 %: the
// voxels are several pixels wide, and lose thin and broken structures), 95 %
// of them at the per-pixel median of the input depths within two voxels, and
// the ray-cast cloud's points must lie on their pixels' rays at the ray-cast
// depth. A clip with depth on fewer than 5 % of its pixels is left out; at
// least one must be checked.
// Skipped (exit 77) when HOLOSCAN_TEST_DATA is not set, as in CI, whose tests
// make their own data.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/tsdf/tsdf_integrate_op.hpp>
#include <holoscan/tsdf/tsdf_mesh_op.hpp>

#include "test_support.hpp"

namespace holoscan::tsdf {
namespace {

namespace fs = std::filesystem;
namespace hm = holoscan::messages;
using std::chrono_literals::operator""ns;

constexpr DLDataType kUInt16{kDLUInt, 16U, 1U};
constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
// The images of each clip that are integrated.
constexpr std::size_t kFrames = 20;
// Any scale works for a check against the input itself: depth counts as
// millimeters. The volume is sized from the depths.
constexpr float kDepthScale = 0.001F;
// The volume's longest side, in voxels.
constexpr int kDims = 512;

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

holoscan::TensorOutputSpec host_spec(DLDataType dtype, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation = holoscan::TensorRepresentation{.memory_kind = holoscan::MemoryKind::kHost,
                                                       .dtype = dtype,
                                                       .rank = 2U},
      .bounds = holoscan::tensor_bounds(bytes),
      .storage = holoscan::TensorOutputStorage::kRuntimePool};
}

// Sends the images as MONO16 in host memory with the identity pose, one per
// tick from the second (the camera info goes out when the run starts), frame
// ids from 1. A refused allocation or emit is sent again on the next tick.
class ImageSourceOp final : public holoscan::Operator<> {
 public:
  explicit ImageSourceOp(std::shared_ptr<const std::vector<DepthImage>> images)
      : images_(std::move(images)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(depth, "depth")
        .produces_tensor(host_spec(kUInt16, images_->front().counts.size() * 2U))
        .payload_options(holoscan::PayloadOptions{
            .schema = holoscan::schema_identity<holoscan::schema::ImageT>()});
    spec.output(pose, "pose").produces_tensor(host_spec(kFloat32, 64U));
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds(10)});
    return result;
  }
  void start() override {
    next_ = 0;
    ticks_ = 0;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (ticks_++ == 0 || next_ >= images_->size()) {
      return {};
    }
    const DepthImage& image = (*images_)[next_];
    auto depth_loan = depth.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = std::array<std::int64_t, 2>{image.height, image.width}, .dtype = kUInt16});
    auto pose_loan = pose.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = std::array<std::int64_t, 2>{4, 4}, .dtype = kFloat32});
    for (const auto* loan : {&depth_loan, &pose_loan}) {
      if (!*loan) {
        return loan->error().code == holoscan::ErrorCode::kResourceExhausted
                   ? holoscan::expected<void, holoscan::Error>{}
                   : holoscan::make_unexpected(loan->error());
      }
    }
    if (auto written =
            depth_loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& w) {
              std::copy(image.counts.begin(), image.counts.end(), w.data_as<std::uint16_t>());
              return holoscan::expected<void, holoscan::Error>{};
            });
        !written) {
      return written;
    }
    if (auto written =
            pose_loan->write_host_and_commit([](const holoscan::TensorOutputWriteGuard& w) {
              const Matrix4 identity = identity_pose();
              std::copy(identity.begin(), identity.end(), w.data_as<float>());
              return holoscan::expected<void, holoscan::Error>{};
            });
        !written) {
      return written;
    }
    holoscan::schema::ImageT descriptor;
    descriptor.width = image.width;
    descriptor.height = image.height;
    descriptor.encoding = holoscan::schema::ImageEncoding_MONO16;
    const holoscan::EmitOptions emit{.capture_time = context.activation_time(),
                                     .frame_id = next_ + 1U};
    if (auto sent = depth.emit_tensor(std::move(*depth_loan), descriptor, emit); !sent) {
      return sent.error().code == holoscan::ErrorCode::kBackpressured
                 ? holoscan::expected<void, holoscan::Error>{}
                 : sent;
    }
    if (auto sent = pose.emit(std::move(*pose_loan), emit); !sent) {
      return sent;
    }
    ++next_;
    return {};
  }

  holoscan::Output<holoscan::schema::ImageT> depth;
  holoscan::Output<holoscan::Tensor> pose;

 private:
  std::shared_ptr<const std::vector<DepthImage>> images_;
  std::uint32_t next_{};
  std::uint32_t ticks_{};
};

struct Tally {
  std::mutex mutex;
  std::uint32_t raycasts{};
  std::uint64_t last_frame_id{};
  std::vector<float> depth;  // The last ray cast.
  std::uint64_t wrong_points{};
};

// Keeps the last ray cast, checks every cloud against its depth image, and
// completes the run on the last frame's.
class CheckOp final : public holoscan::Operator<> {
 public:
  CheckOp(std::uint64_t last_frame_id, std::shared_ptr<Tally> tally)
      : last_frame_id_(last_frame_id), tally_(std::move(tally)) {}
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
      holoscan::ExecutionContext& context) override {
    auto depth_sample = depth.receive();
    if (!depth_sample) return holoscan::make_unexpected(std::move(depth_sample).error());
    auto cloud_sample = cloud.receive();
    if (!cloud_sample) return holoscan::make_unexpected(std::move(cloud_sample).error());
    if (auto checked = hm::check_point_cloud(cloud_sample->data); !checked) return checked;
    const holoscan::Tensor& image = *depth_sample->data.data;
    const std::size_t pixels = static_cast<std::size_t>(image.size());
    std::vector<float> values(pixels);
    std::vector<float> points(pixels * 3U);
    if (auto copied = image.copy_to_host(values.data(), pixels * 4U, context.cuda_stream());
        !copied) {
      return copied;
    }
    if (auto copied = cloud_sample->data.points->copy_to_host(points.data(), pixels * 12U,
                                                              context.cuda_stream());
        !copied) {
      return copied;
    }
    std::uint64_t wrong = 0;
    for (std::size_t i = 0; i < pixels; ++i) {
      const float z = points[3 * i + 2];
      wrong += values[i] > 0.0F ? (z == values[i] ? 0U : 1U) : (std::isnan(z) ? 0U : 1U);
    }
    const std::uint64_t frame_id = depth_sample->metadata.frame_id;
    const std::lock_guard lock(tally_->mutex);
    ++tally_->raycasts;
    tally_->last_frame_id = frame_id;
    tally_->wrong_points += wrong;
    tally_->depth = std::move(values);
    if (frame_id == last_frame_id_) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> depth;
  holoscan::Input<hm::schema::PointCloudT> cloud;

 private:
  std::uint64_t last_frame_id_;
  std::shared_ptr<Tally> tally_;
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

float percentile(std::vector<float> values, double fraction) {
  const auto at = static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1));
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(at), values.end());
  return values[at];
}

// Clips checked: one with too little depth (a recording without the
// camera's projector) is reported and left out.
std::atomic<int> g_checked{0};

void check_clip(const fs::path& clip, std::size_t number) {
  const std::string name = "clip " + std::to_string(number);
  const fs::path camera_file = clip / "depth.yaml";
  const hm::CameraInfo camera = hm::read_camera_info_yaml(camera_file);
  auto images = std::make_shared<const std::vector<DepthImage>>(first_images(clip / "depth"));
  VERIFY_MSG(!images->empty(), name + " has depth images");
  if (images->empty()) return;
  const int width = images->front().width;
  const int height = images->front().height;
  const std::size_t pixels = static_cast<std::size_t>(width) * height;

  // The per-pixel median of the input depths, where most images have one.
  std::vector<float> median(pixels, 0.0F);
  std::vector<float> valid;
  for (std::size_t i = 0; i < pixels; ++i) {
    std::vector<float> seen;
    for (const DepthImage& image : *images) {
      if (image.counts[i] != 0U) seen.push_back(static_cast<float>(image.counts[i]) * kDepthScale);
    }
    if (seen.size() * 2U > images->size()) {
      median[i] = percentile(seen, 0.5);
      valid.push_back(median[i]);
    }
  }
  if (valid.size() <= pixels / 20U) {
    std::cout << name << ": left out, " << 100.0 * static_cast<double>(valid.size()) / pixels
              << " % of the pixels have depth\n";
    return;
  }
  ++g_checked;

  // A volume around what the camera sees: the 2nd to 98th percentile of the
  // depths, as wide as the view there, kDims voxels on its longest side.
  const float near = 0.9F * percentile(valid, 0.02);
  const float far = 1.1F * percentile(valid, 0.98);
  const auto fx = static_cast<float>(camera.fx);
  const auto fy = static_cast<float>(camera.fy);
  const auto cx = static_cast<float>(camera.cx);
  const auto cy = static_cast<float>(camera.cy);
  const float lo[3] = {-cx / fx * far, -cy / fy * far, near};
  const float hi[3] = {(static_cast<float>(width - 1) - cx) / fx * far,
                       (static_cast<float>(height - 1) - cy) / fy * far, far};
  const float longest = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
  TsdfVolumeConfig volume_config;
  volume_config.voxel_size = longest / static_cast<float>(kDims - 1);
  for (int r = 0; r < 3; ++r) {
    volume_config.origin[r] = lo[r];
    volume_config.dims[r] =
        std::max(2, static_cast<int>(std::ceil((hi[r] - lo[r]) / volume_config.voxel_size)) + 1);
  }
  volume_config.truncation = 4.0F * volume_config.voxel_size;
  auto volume = std::make_shared<TsdfVolume>(volume_config);

  holoscan::Graph graph{"tsdf_real_data"};
  auto source = graph.op<ImageSourceOp>("source", images);
  auto info = graph.op<hm::CameraInfoSourceOp>(
      "camera_info", hm::CameraInfoSourceConfig{.camera_info_path = camera_file,
                                                .period = std::chrono::milliseconds(20)});
  auto integrate =
      graph.op<TsdfIntegrateOp>("integrate", TsdfIntegrateConfig{.volume = volume,
                                                                 .depth_scale = kDepthScale,
                                                                 .depth_min = 0.01F,
                                                                 .depth_max = 2.0F * far,
                                                                 .max_width = width,
                                                                 .max_height = height});
  auto mesh = graph.op<TsdfMeshOp>("raycast", TsdfMeshConfig{.volume = volume,
                                                             .mesh = false,
                                                             .raycast = true,
                                                             .raycast_depth_min = 0.01F,
                                                             .raycast_depth_max = 2.0F * far,
                                                             .max_width = width,
                                                             .max_height = height});
  auto tally = std::make_shared<Tally>();
  auto check = graph.op<CheckOp>("check", images->size(), tally);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->depth, integrate->depth, lossless);
  graph.add_flow(source->pose, integrate->pose, lossless);
  graph.add_flow(info->camera_info, integrate->camera_info, hm::latest_camera_info_connection());
  graph.add_flow(info->camera_info, mesh->camera_info, hm::latest_camera_info_connection());
  graph.add_flow(integrate->integrated_pose, mesh->pose, lossless);
  graph.add_flow(mesh->raycast_depth, check->depth, lossless);
  graph.add_flow(mesh->raycast_cloud, check->cloud, lossless);
  // A lossless chain: on one worker, so that no operator runs while its
  // consumer's queue is full.
  graph.partition("tsdf_real_data").include_all();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  holoscan::CompileOptions options;
  for (const char* port : {"raycast_depth", "raycast_cloud"}) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "raycast", .output_port = port, .device = holoscan::DeviceId{0}});
  }
  const auto plan = holoscan::compile(graph, std::move(options));
  VERIFY_MSG(plan.ok(), holoscan::format_diagnostics(plan.diagnostics(), plan.graph()));
  if (!plan.ok()) return;
  VERIFY_MSG(plan.scheduler_plan().worker_count == 1U, name + ": the graph runs on one worker");
  auto session = holoscan::run_async(plan);
  if (!session.wait_for(std::chrono::seconds(60))) {
    session.request_stop();
  }
  session.wait();
  VERIFY_MSG(session.termination() == holoscan::RunTermination::kRequestedCompleted,
             name + ": " + holoscan::format_diagnostics(session.diagnostics(), plan.graph()));

  const std::lock_guard lock(tally->mutex);
  // Every image after the camera info is integrated and ray cast; the source
  // waits a tick for the camera info, so that is every one.
  VERIFY_MSG(tally->raycasts == images->size(), name + ": " + std::to_string(tally->raycasts) +
                                                    " ray casts of " +
                                                    std::to_string(images->size()) + " images");
  VERIFY_MSG(tally->wrong_points == 0,
             name + ": " + std::to_string(tally->wrong_points) + " cloud points off their rays");
  if (tally->depth.size() != pixels) return;

  // The last ray cast against the median input, on the surfaces well inside
  // the volume.
  const float voxel = volume_config.voxel_size;
  std::size_t expected = 0;
  std::size_t hits = 0;
  std::size_t close = 0;
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * width + u;
      const float z = median[i];
      if (!(z > 0.0F)) continue;
      const float p[3] = {(static_cast<float>(u) - cx) / fx * z,
                          (static_cast<float>(v) - cy) / fy * z, z};
      bool inside = true;
      for (int r = 0; r < 3; ++r) {
        inside = inside && p[r] > lo[r] + 4.0F * voxel &&
                 p[r] < lo[r] + static_cast<float>(volume_config.dims[r] - 5) * voxel;
      }
      if (!inside) continue;
      ++expected;
      const float got = tally->depth[i];
      if (got > 0.0F) {
        ++hits;
        close += std::abs(got - z) <= 2.0F * voxel ? 1U : 0U;
      }
    }
  }
  const double coverage = expected == 0 ? 0.0 : static_cast<double>(hits) / expected;
  const double agree = hits == 0 ? 0.0 : static_cast<double>(close) / hits;
  std::cout << name << ": " << width << "x" << height << " images=" << images->size()
            << " raycasts=" << tally->raycasts << " volume=" << volume_config.dims[0] << "x"
            << volume_config.dims[1] << "x" << volume_config.dims[2]
            << " voxel=" << voxel << " surfaces=" << expected << " coverage=" << coverage
            << " within_2_voxels=" << agree << '\n';
  VERIFY_MSG(expected > valid.size() / 2U, name + ": surfaces inside the volume");
  VERIFY_MSG(coverage > 0.8, name + ": the ray cast sees the surfaces: " +
                                  std::to_string(coverage));
  VERIFY_MSG(agree > 0.95, name + ": the ray cast is within two voxels of the input: " +
                               std::to_string(agree));
}

}  // namespace
}  // namespace holoscan::tsdf

int main() {
  namespace ht = holoscan::tsdf;
  const char* root = std::getenv("HOLOSCAN_TEST_DATA");
  if (root == nullptr || *root == '\0') {
    std::cout << "skipped: HOLOSCAN_TEST_DATA is not set\n";
    return 77;
  }
  std::vector<std::filesystem::path> clips;
  const std::filesystem::path recordings = std::filesystem::path(root) / "stereo-camera";
  if (std::filesystem::is_directory(recordings)) {
    for (const auto& entry : std::filesystem::directory_iterator(recordings)) {
      if (std::filesystem::is_regular_file(entry.path() / "depth.yaml")) {
        clips.push_back(entry.path());
      }
    }
  }
  std::sort(clips.begin(), clips.end());
  if (clips.empty()) {
    std::cerr << "no stereo-camera/<clip>/depth.yaml in " << root << '\n';
    return 1;
  }
  std::vector<ht::test::TestCase> cases;
  for (std::size_t i = 0; i < clips.size(); ++i) {
    cases.push_back({"clip " + std::to_string(i), [clip = clips[i], i] { ht::check_clip(clip, i); }});
  }
  cases.push_back({"a clip with depth was checked", [] { VERIFY(ht::g_checked.load() > 0); }});
  return ht::test::run(cases);
}
