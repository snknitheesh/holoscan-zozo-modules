// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Both operators on the local recordings: for each clip in
// $HOLOSCAN_TEST_DATA/stereo-camera/<clip>/ with a depth.yaml, its first
// depth images (16-bit PGM) go through DepthToPointCloudOp with that camera,
// sent by CameraInfoSourceOp, and the clouds back through PointCloudToDepthOp
// into the same camera. Every point must be the host's deprojection of its
// pixel, and every depth must come back exactly. Skipped (exit 77) when
// HOLOSCAN_TEST_DATA is not set, as in CI, whose tests make their own data.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
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
#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/point_cloud/depth_to_point_cloud_op.hpp>
#include <holoscan/point_cloud/point_cloud_to_depth_op.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "test_support.hpp"

namespace holoscan::point_cloud {
namespace {

namespace fs = std::filesystem;
namespace hm = holoscan::messages;

constexpr DLDataType kUInt16{kDLUInt, 16U, 1U};
// The images of each clip that go through the operators.
constexpr std::size_t kFrames = 10;
// Any scale gives the counts back; this one keeps the recordings' depths in range.
constexpr float kDepthScale = 0.001F;
constexpr float kDepthMin = 0.0F;
constexpr float kDepthMax = 100.0F;

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

// Sends the given images as MONO16 in host memory, one per tick, frame ids
// from 1.
class ImageSourceOp final : public holoscan::Operator<> {
 public:
  explicit ImageSourceOp(std::shared_ptr<const std::vector<DepthImage>> images)
      : images_(std::move(images)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    const DepthImage& first = images_->front();
    spec.output(depth, "depth")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kHost,
                    .dtype = kUInt16,
                    .rank = 2U,
                    .strides = holoscan::TensorStridePolicy::kContiguousOnly},
            .bounds = holoscan::tensor_bounds(first.counts.size() * 2U),
            .storage = holoscan::TensorOutputStorage::kRuntimePool})
        .payload_options(holoscan::PayloadOptions{
            .schema = holoscan::schema_identity<holoscan::schema::ImageT>()});
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds(10)});
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
  std::atomic<std::uint32_t> clouds{};
  std::atomic<std::uint32_t> images{};
  std::uint64_t valid_points{};
  std::uint64_t wrong_points{};
  std::uint64_t filled_pixels{};
  std::uint64_t wrong_depths{};
};

// Whether a count is a depth in the operators' range.
bool in_range(std::uint16_t count) {
  const float z = static_cast<float>(count) * kDepthScale;
  return z > 0.0F && z >= kDepthMin && z <= kDepthMax;
}

// Checks every point of each cloud against the host's deprojection of its
// pixel, and every depth image against the image the cloud came from.
class CheckOp final : public holoscan::Operator<> {
 public:
  CheckOp(std::shared_ptr<const std::vector<DepthImage>> images, CameraIntrinsics k,
          std::shared_ptr<Tally> tally)
      : images_(std::move(images)), k_(k), tally_(std::move(tally)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(point_cloud, "point_cloud").queue_depth(2U);
    spec.input(depth, "depth").queue_depth(2U);
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAligned{
        .ports = holoscan::PortList{std::vector<holoscan::PortRef>{point_cloud, depth}},
        .window = std::chrono::nanoseconds{0}});
    result.input(point_cloud, holoscan::Exact{});
    result.input(depth, holoscan::Exact{});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto cloud = point_cloud.receive();
    auto image = depth.receive();
    if (!cloud || !image) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "CheckOp: missing input"});
    }
    const DepthImage& source = (*images_)[cloud->metadata.frame_id - 1U];
    std::vector<float> xyz(source.counts.size() * 3U);
    if (auto copied = cloud->data.points->copy_to_host(xyz.data(), xyz.size() * sizeof(float),
                                                       context.cuda_stream());
        !copied) {
      return copied;
    }
    std::vector<std::uint16_t> counts(source.counts.size());
    if (auto copied = image->data.data->copy_to_host(
            counts.data(), counts.size() * sizeof(std::uint16_t), context.cuda_stream());
        !copied) {
      return copied;
    }
    for (int v = 0; v < source.height; ++v) {
      for (int u = 0; u < source.width; ++u) {
        const std::size_t i = static_cast<std::size_t>(v) * source.width + u;
        const bool valid = in_range(source.counts[i]);
        const float z = static_cast<float>(source.counts[i]) * kDepthScale;
        const float* p = xyz.data() + 3U * i;
        const bool got = std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
        tally_->wrong_points +=
            got != valid ||
            (valid && std::hypot(p[0] - (u - k_.cx) * z / k_.fx, p[1] - (v - k_.cy) * z / k_.fy,
                                 p[2] - z) > 1e-4 * std::max(1.0F, z));
        tally_->valid_points += got;
        tally_->wrong_depths += counts[i] != (valid ? source.counts[i] : 0U);
        tally_->filled_pixels += counts[i] != 0U;
      }
    }
    tally_->clouds.fetch_add(1U);
    tally_->images.fetch_add(1U);
    if (cloud->metadata.frame_id == images_->size()) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<hm::schema::PointCloudT> point_cloud;
  holoscan::Input<holoscan::schema::ImageT> depth;

 private:
  std::shared_ptr<const std::vector<DepthImage>> images_;
  CameraIntrinsics k_;
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

void check_clip(const fs::path& clip) {
  const fs::path camera_file = clip / "depth.yaml";
  const hm::CameraInfo camera = hm::read_camera_info_yaml(camera_file);
  auto images = std::make_shared<const std::vector<DepthImage>>(first_images(clip / "depth"));
  VERIFY_MSG(!images->empty(), clip.string() + " has depth images");
  if (images->empty()) return;
  const int width = images->front().width;
  const int height = images->front().height;

  holoscan::Graph graph{"point_cloud_real_data"};
  auto source = graph.op<ImageSourceOp>("source", images);
  auto info = graph.op<hm::CameraInfoSourceOp>(
      "camera_info", hm::CameraInfoSourceConfig{.camera_info_path = camera_file,
                                                .period = std::chrono::milliseconds(20)});
  auto deproject =
      graph.op<DepthToPointCloudOp>("deproject", DepthToPointCloudConfig{.depth_scale = kDepthScale,
                                                                         .depth_min = kDepthMin,
                                                                         .depth_max = kDepthMax,
                                                                         .max_width = width,
                                                                         .max_height = height});
  auto project = graph.op<PointCloudToDepthOp>(
      "project", PointCloudToDepthConfig{.max_width = width,
                                         .max_height = height,
                                         .depth_min = kDepthMin,
                                         .depth_max = kDepthMax,
                                         .depth_format = DepthFormat::kUint16,
                                         .depth_scale = kDepthScale,
                                         .max_points = static_cast<std::uint32_t>(width) * height});
  auto tally = std::make_shared<Tally>();
  auto check = graph.op<CheckOp>(
      "check", images,
      CameraIntrinsics{static_cast<float>(camera.fx), static_cast<float>(camera.fy),
                       static_cast<float>(camera.cx), static_cast<float>(camera.cy)},
      tally);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->depth, deproject->depth, lossless);
  graph.add_flow(info->camera_info, deproject->camera_info, hm::latest_camera_info_connection());
  graph.add_flow(info->camera_info, project->camera_info, hm::latest_camera_info_connection());
  graph.add_flow(deproject->point_cloud, project->point_cloud, lossless);
  graph.add_flow(deproject->point_cloud, check->point_cloud, lossless);
  graph.add_flow(project->depth, check->depth, lossless);
  // One worker for the whole graph. Split across workers, as the automatic
  // partitioning does with two sources, an operator could run while the
  // checker's queue is full, and fail on the refused emit.
  graph.partition("point_cloud_real_data").include_all();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  holoscan::CompileOptions options;
  for (const auto& [op, port] : {std::pair{"deproject", "point_cloud"}, {"project", "depth"}}) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
  }
  const auto plan = holoscan::compile(graph, std::move(options));
  VERIFY_MSG(plan.ok(), holoscan::format_diagnostics(plan.diagnostics(), plan.graph()));
  if (!plan.ok()) return;
  auto session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  if (session.is_running()) session.request_stop();
  session.wait();
  VERIFY_MSG(
      session.termination() == holoscan::RunTermination::kRequestedCompleted,
      clip.string() + ": " + holoscan::format_diagnostics(session.diagnostics(), plan.graph()));
  // The frames before the first camera info give no cloud.
  VERIFY_MSG(tally->clouds.load() >= kFrames - 1U,
             std::to_string(tally->clouds.load()) + " clouds of " + std::to_string(images->size()));
  VERIFY(tally->valid_points != 0 && tally->filled_pixels == tally->valid_points);
  VERIFY_MSG(tally->wrong_points == 0, std::to_string(tally->wrong_points) + " points wrong");
  VERIFY_MSG(tally->wrong_depths == 0, std::to_string(tally->wrong_depths) + " depths wrong");
  std::cout << clip.filename().string() << ": " << width << "x" << height
            << " clouds=" << tally->clouds.load() << " valid_points=" << tally->valid_points
            << " wrong_points=" << tally->wrong_points << " wrong_depths=" << tally->wrong_depths
            << '\n';
}

}  // namespace
}  // namespace holoscan::point_cloud

int main() {
  namespace pc = holoscan::point_cloud;
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
  std::vector<pc::test::TestCase> cases;
  for (const auto& clip : clips) {
    cases.push_back({clip.filename().string(), [clip] { pc::check_clip(clip); }});
  }
  return pc::test::run(cases);
}
