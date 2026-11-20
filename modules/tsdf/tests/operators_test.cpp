// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The two operators through the SDK's operator test harness, on frames of the
// synthetic scene rendered at test time, with the camera as CameraInfo
// messages: TsdfIntegrateOp with uint16 and float32 depth in host and device
// memory, with RGB and BGRA color, float32, float64 and device poses, camera
// info at its own rate and with every frame, a frame before the first camera
// info, a change of calibration, and poses on their own clock (one too old),
// checked bit for bit against TsdfVolume::integrate() run directly; TsdfMeshOp's
// mesh every N poses and ray cast on every pose (depth image and organized
// point cloud), bit for bit against MeshExtractor and TsdfVolume::raycast(), and
// a pose before the first camera info; the capture time and frame id carried
// to every output; and the rejection of unusable inputs, cameras and settings.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#include <holoscan/tsdf/synthetic_scene.hpp>
#include <holoscan/tsdf/tsdf_integrate_op.hpp>
#include <holoscan/tsdf/tsdf_mesh_op.hpp>

#include "scene_support.hpp"
#include "test_support.hpp"

namespace holoscan::tsdf {
namespace {

namespace hm = holoscan::messages;
using detail::kFloat32;
using detail::kFloat64;
using detail::kUInt16;
using detail::kUInt8;
using std::chrono_literals::operator""ms;
using test::cuda_check;
using test::Device;

constexpr int kWidth = 96;
constexpr int kHeight = 72;
constexpr std::uint32_t kFrames = 3;
constexpr char kFrameId[] = "depth_optical";

std::chrono::nanoseconds capture_time(std::uint32_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

// A tensor of `shape` holding `size` bytes from `bytes`, in host or device memory.
holoscan::Tensor make(const std::vector<std::int64_t>& shape, DLDataType dtype, const void* bytes,
                      std::size_t size, bool device) {
  auto tensor = holoscan::make_tensor(
      shape, dtype, std::nullopt,
      device ? holoscan::MemoryStorageType::kDevice : holoscan::MemoryStorageType::kHost,
      std::make_shared<holoscan::UnboundedAllocator>());
  if (!tensor) {
    throw std::runtime_error(tensor.error().message);
  }
  auto destination = tensor->mutable_data();
  if (!destination) {
    throw std::runtime_error(destination.error().message);
  }
  if (size != 0 && cudaMemcpy(*destination, bytes, size, cudaMemcpyDefault) != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed");
  }
  return std::move(*tensor);
}

holoscan::schema::ImageT image(holoscan::Tensor tensor, int width, int height,
                               holoscan::schema::ImageEncoding encoding) {
  holoscan::schema::ImageT result;
  result.width = width;
  result.height = height;
  result.encoding = encoding;
  result.data = std::make_shared<holoscan::Tensor>(std::move(tensor));
  return result;
}

holoscan::Tensor pose_tensor(const Matrix4& pose, bool f64, bool device) {
  if (f64) {
    std::array<double, 16> values{};
    std::copy(pose.begin(), pose.end(), values.begin());
    return make({4, 4}, kFloat64, values.data(), sizeof(values), device);
  }
  return make({4, 4}, kFloat32, pose.data(), sizeof(pose), device);
}

// The synthetic camera of the test frames, and another one: 10 % longer
// focal lengths, as after a change of calibration.
hm::CameraInfo camera(bool changed = false) {
  hm::CameraInfo info = synthetic_camera_info(kWidth, kHeight, kFrameId);
  if (changed) {
    info.fx *= 1.1;
    info.fy *= 1.1;
  }
  return info;
}

CameraIntrinsics intrinsics_of(const hm::CameraInfo& info) {
  return CameraIntrinsics{static_cast<float>(info.fx), static_cast<float>(info.fy),
                          static_cast<float>(info.cx), static_cast<float>(info.cy)};
}

// One frame as the operator gets it, and as TsdfVolume::integrate() gets it.
struct OpFrame {
  Matrix4 pose{};
  std::vector<std::uint16_t> mm;
  std::vector<float> meters;
  std::vector<std::uint8_t> rgb;
};

std::vector<OpFrame> scene_frames() {
  std::vector<OpFrame> frames;
  test::Frame f(kWidth, kHeight);
  for (std::uint32_t i = 0; i < kFrames; ++i) {
    test::render(f, SceneKind::kTabletop, i * 5, 15, test::kNoise);
    OpFrame frame{f.pose, {}, f.host_depth, f.host_rgb};
    for (const float d : f.host_depth) {
      frame.mm.push_back(static_cast<std::uint16_t>(std::lround(d * 1000.0F)));
    }
    frames.push_back(std::move(frame));
  }
  return frames;
}

struct IntegrateCase {
  bool millimeters{};
  bool device{};
  std::optional<holoscan::schema::ImageEncoding> color;
  bool f64_pose{};
  bool device_pose{};
  // Camera info with every frame (kEveryFrame), else at its own rate
  // (kLatest): sent a millisecond before the first frame, and again when it
  // changes.
  bool every_frame{};
  // Frame 2 on has the other camera.
  bool change_camera{};
  // Frame 0 comes before any camera info, and is skipped.
  bool first_without_camera{};
  // Above 0: poses come this long before their depth images, with
  // pose_max_age twice that; frame 1's comes four times as long before, and
  // the frame is skipped.
  std::chrono::nanoseconds pose_offset{0};
};

void check_integrate_op(const IntegrateCase& c) {
  const std::vector<OpFrame> frames = scene_frames();
  const bool color = c.color.has_value();
  const bool pose_clock = c.pose_offset > std::chrono::nanoseconds::zero();
  const TsdfVolumeConfig volume_config = scene_volume(SceneKind::kTabletop, 64, color);
  auto volume = std::make_shared<TsdfVolume>(volume_config);
  TsdfIntegrateConfig config{.volume = volume,
                             .depth_scale = c.millimeters ? 0.001F : 1.0F,
                             .depth_min = 0.1F,
                             .depth_max = 3.0F,
                             .color = color,
                             .pose_max_age = 2 * c.pose_offset,
                             .max_width = kWidth,
                             .max_height = kHeight};
  if (c.every_frame) {
    config.camera_info.timing = hm::CameraInfoTiming::kEveryFrame;
  }
  const auto camera_for = [&c](std::uint32_t f) { return camera(c.change_camera && f >= 2); };
  // The frames that are integrated, in order.
  std::vector<std::uint32_t> integrated;
  for (std::uint32_t f = 0; f < kFrames; ++f) {
    const bool skipped = (c.first_without_camera && f == 0) || (pose_clock && f == 1);
    if (!skipped) {
      integrated.push_back(f);
    }
  }
  const bool bgra = c.color == holoscan::schema::ImageEncoding_BGRA8;
  auto h = holoscan::testing::create_operator_test<TsdfIntegrateOp>(config);
  for (std::uint32_t f = 0; f < kFrames; ++f) {
    const OpFrame& frame = frames[f];
    const std::size_t n = frame.meters.size();
    // In time order: the pose (when on its own clock) comes before the camera
    // info, which comes a millisecond before the frame.
    if (pose_clock) {
      const std::chrono::nanoseconds before = f == 1 ? 4 * c.pose_offset : c.pose_offset;
      h.push_frame(holoscan::testing::frame_at(
          capture_time(f) - before,
          holoscan::testing::input("pose", pose_tensor(frame.pose, c.f64_pose, c.device_pose))));
    }
    if (!c.every_frame) {
      const bool first = c.first_without_camera ? f == 1 : f == 0;
      if (first || (c.change_camera && f == 2)) {
        h.push_frame(holoscan::testing::frame_at(
            capture_time(f) - 1ms,
            holoscan::testing::input("camera_info", hm::to_message(camera_for(f)))));
      }
    }
    holoscan::testing::TestFrame test_frame = holoscan::testing::frame_at(
        capture_time(f),
        holoscan::testing::input(
            "depth",
            c.millimeters
                ? image(make({kHeight, kWidth}, kUInt16, frame.mm.data(), n * 2, c.device), kWidth,
                        kHeight, holoscan::schema::ImageEncoding_MONO16)
                : image(make({kHeight, kWidth, 1}, kFloat32, frame.meters.data(), n * 4, c.device),
                        kWidth, kHeight, holoscan::schema::ImageEncoding_MONO32F)));
    const auto append = [&test_frame](holoscan::testing::TestFrame extra) {
      for (auto& field : extra.fields) {
        test_frame.fields.push_back(std::move(field));
      }
    };
    if (!pose_clock) {
      append(holoscan::testing::frame_at(
          capture_time(f),
          holoscan::testing::input("pose", pose_tensor(frame.pose, c.f64_pose, c.device_pose))));
    }
    if (c.every_frame) {
      append(holoscan::testing::frame_at(
          capture_time(f), holoscan::testing::input("camera_info", hm::to_message(camera_for(f)))));
    }
    if (color) {
      std::vector<std::uint8_t> pixels;
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint8_t* rgb = &frame.rgb[3 * i];
        if (bgra) {
          pixels.insert(pixels.end(), {rgb[2], rgb[1], rgb[0], 255U});
        } else {
          pixels.insert(pixels.end(), {rgb[0], rgb[1], rgb[2]});
        }
      }
      const int channels = bgra ? 4 : 3;
      append(holoscan::testing::frame_at(
          capture_time(f),
          holoscan::testing::input("color", image(make({kHeight, kWidth, channels}, kUInt8,
                                                       pixels.data(), pixels.size(), c.device),
                                                  kWidth, kHeight, *c.color))));
    }
    h.push_frame(test_frame);
  }
  if (c.device) {
    h.input_source_device("depth", holoscan::DeviceId{0});
    if (color) {
      h.input_source_device("color", holoscan::DeviceId{0});
    }
  }
  if (c.device_pose) {
    h.input_source_device("pose", holoscan::DeviceId{0});
  }
  h.capture_snapshot<holoscan::Tensor>("integrated_pose");
  const std::string name =
      std::string("integrate") + (c.millimeters ? " uint16" : " float32") +
      (c.device ? " device" : " host") + (color ? " color" : "") + (bgra ? " bgra" : "") +
      (c.f64_pose ? " f64-pose" : "") + (c.device_pose ? " device-pose" : "") +
      (c.every_frame ? " camera-every-frame" : "") + (c.change_camera ? " camera-change" : "") +
      (c.first_without_camera ? " frame-before-camera" : "") + (pose_clock ? " pose-clock" : "");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), name + ": " + result.describe_failure());
  if (!result.ok()) {
    return;
  }

  // The same frames integrated directly, each with its camera.
  TsdfVolume direct(volume_config);
  for (const std::uint32_t f : integrated) {
    const OpFrame& frame = frames[f];
    const std::size_t n = frame.meters.size();
    Device<std::uint16_t> mm(n);
    Device<float> meters(n);
    Device<std::uint8_t> rgb(n * 3);
    mm.upload(frame.mm);
    meters.upload(frame.meters);
    rgb.upload(frame.rgb);
    const DepthImage depth =
        c.millimeters ? DepthImage{mm.get(), DepthType::kUint16, kWidth, kHeight, 0.001F}
                      : DepthImage{meters.get(), DepthType::kFloat32, kWidth, kHeight, 1.0F};
    const ColorImage colors{rgb.get(), 3, false};
    cuda_check(direct.integrate(depth, color ? &colors : nullptr, frame.pose,
                                IntegrateParams{intrinsics_of(camera_for(f)), 0.1F, 3.0F}, nullptr),
               "integrate");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  }
  const VolumeData got = volume->download();
  const VolumeData want = direct.download();
  VERIFY_MSG(std::memcmp(got.tsdf.data(), want.tsdf.data(), got.tsdf.size() * 4) == 0 &&
                 got.weight == want.weight,
             name + ": the volume matches TsdfVolume::integrate() bit for bit");
  VERIFY_MSG(got.rgb == want.rgb, name + ": the colors match");
  VERIFY_MSG(
      std::count_if(got.weight.begin(), got.weight.end(), [](float w) { return w > 0.0F; }) > 10000,
      name + ": the frames were integrated");

  const auto poses = result.snapshots<holoscan::Tensor>("integrated_pose");
  VERIFY_MSG(poses.size() == integrated.size(),
             name + ": " + std::to_string(poses.size()) + " integrated poses for " +
                 std::to_string(integrated.size()) + " integrated frames");
  for (std::size_t i = 0; i < std::min(poses.size(), integrated.size()); ++i) {
    const std::uint32_t f = integrated[i];
    VERIFY(poses[i].shape() == (std::vector<std::int64_t>{4, 4}));
    Matrix4 values{};
    if (poses[i].bytes().size() == sizeof(values)) {
      std::memcpy(values.data(), poses[i].bytes().data(), sizeof(values));
    }
    VERIFY_MSG(values == frames[f].pose, name + ": the integrated pose is the input pose");
    VERIFY_MSG(poses[i].sample_metadata().capture_timestamp_ns == capture_time(f).count(),
               name + ": the depth image's capture time is carried");
  }
}

// A volume with the whole tabletop orbit.
std::shared_ptr<TsdfVolume> integrated_volume(bool color) {
  auto volume = std::make_shared<TsdfVolume>(scene_volume(SceneKind::kTabletop, 64, color));
  test::integrate_orbit(*volume, SceneKind::kTabletop, 0, 12, 12, 160, 120, test::kNoise);
  return volume;
}

constexpr std::size_t kCaptures = 8;

using CloudSnapshot = holoscan::testing::Snapshot<hm::schema::PointCloudT>;

template <typename T>
std::vector<T> field(const CloudSnapshot& cloud,
                     holoscan::TensorFieldSelector<hm::schema::PointCloudT> selector) {
  const auto& tensor = cloud.tensor(selector);
  std::vector<T> values(tensor.bytes().size() / sizeof(T));
  std::memcpy(values.data(), tensor.bytes().data(), values.size() * sizeof(T));
  return values;
}

// The mesh every second pose and the ray cast on every pose, through the
// camera of a camera info sent at its own rate.
void check_mesh_op() {
  for (const bool color : {false, true}) {
    const std::string name = color ? "mesh color" : "mesh";
    auto volume = integrated_volume(color);
    constexpr std::uint32_t kMaxVertices = 1U << 16;
    constexpr std::uint32_t kMaxTriangles = 1U << 17;
    const TsdfMeshConfig config{.volume = volume,
                                .mesh = true,
                                .mesh_every = 2,
                                .max_vertices = kMaxVertices,
                                .max_triangles = kMaxTriangles,
                                .raycast = true,
                                .raycast_depth_min = 0.1F,
                                .raycast_depth_max = 3.0F,
                                .max_width = kWidth,
                                .max_height = kHeight};
    const RaycastParams raycast{.intrinsics = synthetic_intrinsics(kWidth, kHeight),
                                .width = kWidth,
                                .height = kHeight,
                                .depth_min = 0.1F,
                                .depth_max = 3.0F};
    auto h = holoscan::testing::create_operator_test<TsdfMeshOp>(config);
    std::vector<Matrix4> poses;
    h.push_frame(holoscan::testing::frame_at(
        capture_time(0) - 1ms, holoscan::testing::input("camera_info", hm::to_message(camera()))));
    for (std::uint32_t f = 0; f < 4; ++f) {
      poses.push_back(orbit_pose(SceneKind::kTabletop, 2 * f + 1, 12));
      auto sample =
          holoscan::testing::sample_at(pose_tensor(poses.back(), true, false), capture_time(f));
      sample.metadata.frame_id = 11U + f;
      h.push_samples("pose", {std::move(sample)});
    }
    std::vector<std::string> tensors{"vertices", "normals", "triangles", "mesh_counts"};
    if (color) {
      tensors.emplace_back("colors");
    }
    for (const std::string& port : tensors) {
      h.output_device(port, holoscan::DeviceId{0});
      h.capture_snapshot<holoscan::Tensor>(port, kCaptures);
    }
    h.output_device("raycast_depth", holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::schema::ImageT>("raycast_depth", kCaptures);
    h.output_device("raycast_cloud", holoscan::DeviceId{0});
    h.capture_snapshot<hm::schema::PointCloudT>("raycast_cloud", kCaptures);
    const auto result = h.run();
    VERIFY_MSG(result.ok(), name + ": " + result.describe_failure());
    if (!result.ok()) {
      continue;
    }
    const test::HostMesh want = test::extract(*volume, kMaxVertices, kMaxTriangles);
    VERIFY(want.counts[1] > 1000);
    const auto counts = result.snapshots<holoscan::Tensor>("mesh_counts");
    const auto vertices = result.snapshots<holoscan::Tensor>("vertices");
    const auto normals = result.snapshots<holoscan::Tensor>("normals");
    const auto triangles = result.snapshots<holoscan::Tensor>("triangles");
    VERIFY_MSG(
        counts.size() == 2 && vertices.size() == 2 && triangles.size() == 2 && normals.size() == 2,
        name + ": a mesh on every second pose");
    for (std::size_t m = 0; m < std::min<std::size_t>(counts.size(), 2); ++m) {
      std::uint32_t got[kMeshCountsSize] = {};
      std::memcpy(got, counts[m].bytes().data(), sizeof(got));
      VERIFY_MSG(std::equal(got, got + kMeshCountsSize, want.counts), name + ": mesh counts");
      VERIFY(vertices[m].shape() == (std::vector<std::int64_t>{kMaxVertices, 3}));
      VERIFY(triangles[m].shape() == (std::vector<std::int64_t>{kMaxTriangles, 3}));
      VERIFY_MSG(std::memcmp(vertices[m].bytes().data(), want.vertices.data(),
                             want.vertices.size() * 4) == 0 &&
                     std::memcmp(normals[m].bytes().data(), want.normals.data(),
                                 want.normals.size() * 4) == 0 &&
                     std::memcmp(triangles[m].bytes().data(), want.triangles.data(),
                                 want.triangles.size() * 4) == 0,
                 name + ": the mesh matches MeshExtractor bit for bit");
      if (color) {
        const auto colors = result.snapshots<holoscan::Tensor>("colors");
        VERIFY(colors.size() == 2 &&
               std::memcmp(colors[m].bytes().data(), want.colors.data(), want.colors.size()) == 0);
      }
      // The second and the fourth pose were meshed.
      VERIFY_MSG(counts[m].sample_metadata().capture_timestamp_ns ==
                         capture_time(2 * static_cast<std::uint32_t>(m) + 1).count() &&
                     counts[m].sample_metadata().frame_id == 12U + 2 * m,
                 name + ": the mesh carries its pose's capture time and frame id");
    }

    const auto depths = result.snapshots<holoscan::schema::ImageT>("raycast_depth");
    const auto clouds = result.snapshots<hm::schema::PointCloudT>("raycast_cloud");
    VERIFY_MSG(depths.size() == 4 && clouds.size() == 4, name + ": a ray cast on every pose");
    const std::size_t pixels = static_cast<std::size_t>(kWidth) * kHeight;
    for (std::size_t f = 0; f < std::min<std::size_t>({depths.size(), clouds.size(), 4}); ++f) {
      Device<float> depth(pixels);
      Device<float> normal(pixels * 3);
      Device<std::uint8_t> rgb(pixels * 3);
      Device<float> points(pixels * 3);
      cuda_check(volume->raycast(poses[f], raycast,
                                 RaycastImages{depth.get(), normal.get(),
                                               color ? rgb.get() : nullptr, points.get()},
                                 nullptr),
                 "raycast");
      const holoscan::schema::ImageT descriptor = depths[f].metadata();
      VERIFY(descriptor.encoding == holoscan::schema::ImageEncoding_MONO32F &&
             descriptor.width == kWidth && descriptor.height == kHeight);
      const auto& tensor = depths[f].tensor(holoscan::schema::ImageTensorFields::data);
      VERIFY(tensor.shape() == (std::vector<std::int64_t>{kHeight, kWidth}));
      const std::vector<float> want_depth = depth.host();
      VERIFY_MSG(tensor.bytes().size() == pixels * 4 &&
                     std::memcmp(tensor.bytes().data(), want_depth.data(), pixels * 4) == 0,
                 name + ": the ray-cast depth matches TsdfVolume::raycast()");
      VERIFY(std::count_if(want_depth.begin(), want_depth.end(), [](float d) { return d > 0.0F; }) >
             static_cast<std::int64_t>(pixels / 4));
      VERIFY_MSG(depths[f].sample_metadata().capture_timestamp_ns == capture_time(f).count() &&
                     depths[f].sample_metadata().frame_id == 11U + f,
                 name + ": the ray cast carries its pose's capture time and frame id");

      // The cloud: organized, the hits, their normals and colors.
      const CloudSnapshot& cloud = clouds[f];
      const hm::schema::PointCloudT message = cloud.metadata();
      VERIFY(message.organized && message.valid_count == -1);
      VERIFY(message.header != nullptr && message.header->frame_id == kFrameId &&
             message.header->capture_timestamp_ns == capture_time(f).count());
      const std::vector<std::int64_t> shape{kHeight, kWidth, 3};
      VERIFY(cloud.tensor(hm::PointCloudFields::points).shape() == shape);
      VERIFY(cloud.tensor(hm::PointCloudFields::normals).shape() == shape);
      VERIFY(cloud.tensor(hm::PointCloudFields::colors).shape() ==
             (color ? shape : std::vector<std::int64_t>{0}));
      VERIFY(cloud.tensor(hm::PointCloudFields::intensity).shape() == std::vector<std::int64_t>{0});
      const std::vector<float> want_points = points.host();
      const std::vector<float> want_normals = normal.host();
      const std::vector<float> got_points = field<float>(cloud, hm::PointCloudFields::points);
      const std::vector<float> got_normals = field<float>(cloud, hm::PointCloudFields::normals);
      VERIFY_MSG(got_points.size() == pixels * 3 &&
                     std::memcmp(got_points.data(), want_points.data(), pixels * 12) == 0,
                 name + ": the ray-cast points match TsdfVolume::raycast()");
      VERIFY_MSG(got_normals.size() == pixels * 3 &&
                     std::memcmp(got_normals.data(), want_normals.data(), pixels * 12) == 0,
                 name + ": the ray-cast normals match");
      if (color) {
        const std::vector<std::uint8_t> want_rgb = rgb.host();
        VERIFY_MSG(field<std::uint8_t>(cloud, hm::PointCloudFields::colors) == want_rgb,
                   name + ": the ray-cast colors match");
      }
      VERIFY(cloud.sample_metadata().capture_timestamp_ns == capture_time(f).count() &&
             cloud.sample_metadata().frame_id == 11U + f);
    }
  }
}

// With the ray cast, a pose before the first camera info is skipped: not ray
// cast, not meshed and not counted for mesh_every; the poses after it are.
void check_mesh_before_camera_info() {
  auto volume = integrated_volume(false);
  const TsdfMeshConfig config{.volume = volume,
                              .mesh_every = 2,
                              .max_vertices = 1U << 16,
                              .max_triangles = 1U << 17,
                              .raycast = true,
                              .max_width = kWidth,
                              .max_height = kHeight};
  auto h = holoscan::testing::create_operator_test<TsdfMeshOp>(config);
  for (std::uint32_t f = 0; f < 3; ++f) {
    if (f == 1) {
      h.push_frame(holoscan::testing::frame_at(
          capture_time(f) - 1ms,
          holoscan::testing::input("camera_info", hm::to_message(camera()))));
    }
    h.push_frame(holoscan::testing::frame_at(
        capture_time(f),
        holoscan::testing::input(
            "pose", pose_tensor(orbit_pose(SceneKind::kTabletop, f, 12), false, false))));
  }
  for (const char* port : {"vertices", "normals", "triangles", "mesh_counts"}) {
    h.output_device(port, holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::Tensor>(port, kCaptures);
  }
  h.output_device("raycast_depth", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::schema::ImageT>("raycast_depth", kCaptures);
  h.output_device("raycast_cloud", holoscan::DeviceId{0});
  h.capture_snapshot<hm::schema::PointCloudT>("raycast_cloud", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) {
    return;
  }
  const auto depths = result.snapshots<holoscan::schema::ImageT>("raycast_depth");
  const auto counts = result.snapshots<holoscan::Tensor>("mesh_counts");
  VERIFY_MSG(depths.size() == 2, std::to_string(depths.size()) +
                                     " ray casts of 2 poses after "
                                     "the camera info");
  VERIFY_MSG(counts.size() == 1, "the second pose after the camera info is meshed");
  if (counts.size() == 1) {
    VERIFY(counts[0].sample_metadata().capture_timestamp_ns == capture_time(2).count());
  }
  if (!depths.empty()) {
    VERIFY(depths[0].sample_metadata().capture_timestamp_ns == capture_time(1).count());
  }
}

// A mesh larger than the capacity: empty, with the counts it needs.
void check_mesh_overflow() {
  auto volume = integrated_volume(false);
  const test::HostMesh full = test::extract(*volume);
  const TsdfMeshConfig config{.volume = volume, .max_vertices = 100, .max_triangles = 100};
  auto h = holoscan::testing::create_operator_test<TsdfMeshOp>(config);
  h.push_samples("pose", {holoscan::testing::sample_at(pose_tensor(identity_pose(), false, false),
                                                       capture_time(0))});
  for (const char* port : {"vertices", "normals", "triangles", "mesh_counts"}) {
    h.output_device(port, holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::Tensor>(port, 2);
  }
  const auto result = h.run();
  VERIFY_MSG(result.ok(), "overflow: " + result.describe_failure());
  if (!result.ok()) {
    return;
  }
  const auto counts = result.snapshots<holoscan::Tensor>("mesh_counts");
  VERIFY(counts.size() == 1);
  if (counts.size() == 1) {
    std::uint32_t got[kMeshCountsSize] = {};
    std::memcpy(got, counts[0].bytes().data(), sizeof(got));
    VERIFY_MSG(got[0] == 0 && got[1] == 0 && got[2] == full.counts[0] && got[3] == full.counts[1],
               "a mesh beyond the capacity is empty and says what it needs");
  }
}

// Runs TsdfIntegrateOp on one frame with `change` applied to its inputs and
// checks that it fails naming `what`.
void expect_integrate_failure(
    const std::string& what,
    const std::function<void(holoscan::schema::ImageT&, holoscan::Tensor&, hm::CameraInfo&,
                             TsdfIntegrateConfig&)>& change) {
  const std::vector<OpFrame> frames = scene_frames();
  auto volume = std::make_shared<TsdfVolume>(scene_volume(SceneKind::kTabletop, 32, false));
  TsdfIntegrateConfig config{
      .volume = volume, .depth_scale = 1.0F, .max_width = kWidth, .max_height = kHeight};
  holoscan::schema::ImageT depth = image(make({kHeight, kWidth}, kFloat32, frames[0].meters.data(),
                                              frames[0].meters.size() * 4, false),
                                         kWidth, kHeight, holoscan::schema::ImageEncoding_MONO32F);
  holoscan::Tensor pose = pose_tensor(frames[0].pose, false, false);
  hm::CameraInfo info = camera();
  change(depth, pose, info, config);
  auto h = holoscan::testing::create_operator_test<TsdfIntegrateOp>(config);
  h.push_frame(holoscan::testing::frame_at(
      capture_time(0), holoscan::testing::input("camera_info", hm::to_message(info)),
      holoscan::testing::input("depth", depth), holoscan::testing::input("pose", pose)));
  h.capture_snapshot<holoscan::Tensor>("integrated_pose");
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), what + " is rejected");
  VERIFY_MSG(result.describe_failure().find(what) != std::string::npos,
             "failure names '" + what + "': " + result.describe_failure());
}

void check_rejections() {
  using Depth = holoscan::schema::ImageT;
  using Pose = holoscan::Tensor;
  expect_integrate_failure(
      "unsupported depth encoding", [](Depth& depth, Pose&, hm::CameraInfo&, TsdfIntegrateConfig&) {
        const std::vector<std::uint8_t> rgb(kWidth * kHeight * 3, 0U);
        depth = image(make({kHeight, kWidth, 3}, kUInt8, rgb.data(), rgb.size(), false), kWidth,
                      kHeight, holoscan::schema::ImageEncoding_RGB8);
      });
  expect_integrate_failure("exceeds the configured",
                           [](Depth&, Pose&, hm::CameraInfo&, TsdfIntegrateConfig& config) {
                             config.max_width = kWidth - 1;
                           });
  expect_integrate_failure("pose must be a packed float32 or float64 [4, 4]",
                           [](Depth&, Pose& pose, hm::CameraInfo&, TsdfIntegrateConfig&) {
                             const float values[12] = {};
                             pose = make({3, 4}, kFloat32, values, sizeof(values), false);
                           });
  expect_integrate_failure("pose is not a rigid transform",
                           [](Depth&, Pose& pose, hm::CameraInfo&, TsdfIntegrateConfig&) {
                             Matrix4 scaled = identity_pose();
                             scaled[0] = 2.0F;
                             pose = pose_tensor(scaled, false, false);
                           });
  // Camera info the kernels cannot use, or for images of another size.
  expect_integrate_failure("describes raw images",
                           [](Depth&, Pose&, hm::CameraInfo& info, TsdfIntegrateConfig&) {
                             info.state = hm::CameraState::kRaw;
                             info.distortion_model = hm::DistortionModel::kPlumbBob;
                             info.coefficients = {0.1, 0.0, 0.0, 0.0, 0.0};
                           });
  expect_integrate_failure("has a skew", [](Depth&, Pose&, hm::CameraInfo& info,
                                            TsdfIntegrateConfig&) { info.skew = 0.5; });
  expect_integrate_failure("but camera info 'depth_optical' is for 192x144 images",
                           [](Depth&, Pose&, hm::CameraInfo& info, TsdfIntegrateConfig&) {
                             info = synthetic_camera_info(2 * kWidth, 2 * kHeight, kFrameId);
                           });
  // Errors name the operator and its instance.
  expect_integrate_failure("TsdfIntegrateOp '", [](Depth&, Pose&, hm::CameraInfo& info,
                                                   TsdfIntegrateConfig&) { info.skew = 0.5; });

  // The ray cast refuses a camera larger than its outputs.
  {
    auto volume = integrated_volume(false);
    auto h = holoscan::testing::create_operator_test<TsdfMeshOp>(TsdfMeshConfig{
        .volume = volume, .mesh = false, .raycast = true, .max_width = 64, .max_height = 48});
    h.push_frame(holoscan::testing::frame_at(
        capture_time(0), holoscan::testing::input("camera_info", hm::to_message(camera())),
        holoscan::testing::input("pose", pose_tensor(identity_pose(), false, false))));
    h.output_device("raycast_depth", holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::schema::ImageT>("raycast_depth", 2);
    h.output_device("raycast_cloud", holoscan::DeviceId{0});
    h.capture_snapshot<hm::schema::PointCloudT>("raycast_cloud", 2);
    const auto result = h.run();
    VERIFY_MSG(
        !result.ok() &&
            result.describe_failure().find("larger than the configured 64x48") != std::string::npos,
        "a camera larger than the ray cast's outputs is refused: " + result.describe_failure());
  }

  // Unusable settings.
  auto volume = std::make_shared<TsdfVolume>(scene_volume(SceneKind::kTabletop, 16, false));
  const auto integrate_refused = [](TsdfIntegrateConfig config) {
    return test::throws<std::invalid_argument>([&config] { TsdfIntegrateOp op(config); });
  };
  VERIFY(integrate_refused(TsdfIntegrateConfig{}));
  VERIFY(integrate_refused(TsdfIntegrateConfig{.volume = volume, .color = true}));
  VERIFY(integrate_refused(
      TsdfIntegrateConfig{.volume = volume, .depth_min = 2.0F, .depth_max = 1.0F}));
  VERIFY(integrate_refused(TsdfIntegrateConfig{.volume = volume, .depth_scale = 0.0F}));
  VERIFY(integrate_refused(TsdfIntegrateConfig{.volume = volume, .pose_max_age = -1ms}));
  VERIFY(integrate_refused(TsdfIntegrateConfig{
      .volume = volume, .camera_info = {.max_age = std::chrono::nanoseconds{0}}}));
  VERIFY(!integrate_refused(TsdfIntegrateConfig{.volume = volume}));
  const auto mesh_refused = [](TsdfMeshConfig config) {
    return test::throws<std::invalid_argument>([&config] { TsdfMeshOp op(config); });
  };
  VERIFY(mesh_refused(TsdfMeshConfig{}));
  VERIFY(mesh_refused(TsdfMeshConfig{.volume = volume, .mesh = false}));
  VERIFY(mesh_refused(TsdfMeshConfig{.volume = volume, .mesh_every = 0}));
  VERIFY(mesh_refused(TsdfMeshConfig{.volume = volume, .max_vertices = 0}));
  VERIFY(
      mesh_refused(TsdfMeshConfig{.volume = volume, .raycast = true, .raycast_depth_min = 0.0F}));
  VERIFY(mesh_refused(TsdfMeshConfig{.volume = volume, .raycast = true, .max_width = 0}));
  VERIFY(!mesh_refused(TsdfMeshConfig{.volume = volume}));
  VERIFY(!mesh_refused(TsdfMeshConfig{.volume = volume, .raycast = true}));
}

}  // namespace

int run_all() {
  std::vector<test::TestCase> cases;
  for (const bool millimeters : {true, false}) {
    for (const bool device : {false, true}) {
      cases.push_back({std::string("TsdfIntegrateOp ") + (millimeters ? "uint16" : "float32") +
                           (device ? " device" : " host"),
                       [=] { check_integrate_op(IntegrateCase{millimeters, device}); }});
    }
  }
  cases.push_back(
      {"TsdfIntegrateOp RGB8, camera info with every frame, changing, float64 pose", [] {
         check_integrate_op(IntegrateCase{.millimeters = true,
                                          .device = true,
                                          .color = holoscan::schema::ImageEncoding_RGB8,
                                          .f64_pose = true,
                                          .every_frame = true,
                                          .change_camera = true});
       }});
  cases.push_back({"TsdfIntegrateOp BGRA8 host, device pose", [] {
                     check_integrate_op(
                         IntegrateCase{.millimeters = false,
                                       .device = false,
                                       .color = holoscan::schema::ImageEncoding_BGRA8,
                                       .device_pose = true});
                   }});
  cases.push_back({"TsdfIntegrateOp a frame before the first camera info, then a change", [] {
                     check_integrate_op(IntegrateCase{
                         .millimeters = true, .change_camera = true, .first_without_camera = true});
                   }});
  cases.push_back({"TsdfIntegrateOp poses on their own clock, one too old", [] {
                     check_integrate_op(IntegrateCase{.millimeters = true, .pose_offset = 5ms});
                   }});
  cases.push_back({"TsdfMeshOp mesh every second pose and ray cast", check_mesh_op});
  cases.push_back(
      {"TsdfMeshOp a pose before the first camera info", check_mesh_before_camera_info});
  cases.push_back({"TsdfMeshOp capacity", check_mesh_overflow});
  cases.push_back({"rejections", check_rejections});
  return test::run(cases);
}

}  // namespace holoscan::tsdf

int main() { return holoscan::tsdf::run_all(); }
