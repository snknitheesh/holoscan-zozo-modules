// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The two operators through the SDK's operator test harness, on frames of the
// synthetic scene rendered at test time: TsdfIntegrateOp with uint16 and
// float32 depth in host and device memory, with RGB and BGRA color, with
// configured and per-frame intrinsics and float32 and float64 poses, checked
// bit for bit against TsdfVolume::integrate() run directly; TsdfMeshOp's mesh
// every N poses and ray cast on every pose, checked bit for bit against
// MeshExtractor and TsdfVolume::raycast(); the capture time and frame id
// carried to every output; and the rejection of unusable inputs and settings.

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
#include <holoscan/testing/operator_test_harness.hpp>
#include <holoscan/tsdf/synthetic_scene.hpp>
#include <holoscan/tsdf/tsdf_integrate_op.hpp>
#include <holoscan/tsdf/tsdf_mesh_op.hpp>

#include "scene_support.hpp"
#include "test_support.hpp"

namespace holoscan::tsdf {
namespace {

using test::cuda_check;
using test::Device;

constexpr int kWidth = 96;
constexpr int kHeight = 72;
constexpr std::uint32_t kFrames = 3;

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

// One frame as the operator gets it, and as TsdfVolume::integrate() gets it.
struct OpFrame {
  Matrix4 pose{};
  CameraIntrinsics intrinsics;
  std::vector<std::uint16_t> mm;
  std::vector<float> meters;
  std::vector<std::uint8_t> rgb;
};

std::vector<OpFrame> scene_frames() {
  std::vector<OpFrame> frames;
  test::Frame f(kWidth, kHeight);
  for (std::uint32_t i = 0; i < kFrames; ++i) {
    test::render(f, SceneKind::kTabletop, i * 5, 15, test::kNoise);
    OpFrame frame{f.pose, f.intrinsics, {}, f.host_depth, f.host_rgb};
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
  bool intrinsics_input{};
  bool f64_pose{};
  bool device_pose{};
};

void check_integrate_op(const IntegrateCase& c) {
  const std::vector<OpFrame> frames = scene_frames();
  const bool color = c.color.has_value();
  const TsdfVolumeConfig volume_config = scene_volume(SceneKind::kTabletop, 64, color);
  auto volume = std::make_shared<TsdfVolume>(volume_config);
  TsdfIntegrateConfig config{.volume = volume,
                             .depth_scale = c.millimeters ? 0.001F : 1.0F,
                             .depth_min = 0.1F,
                             .depth_max = 3.0F,
                             .color = color,
                             .max_width = kWidth,
                             .max_height = kHeight};
  if (c.intrinsics_input) {
    config.intrinsics_input = true;
  } else {
    config.intrinsics = frames[0].intrinsics;
  }
  const bool bgra = c.color == holoscan::schema::ImageEncoding_BGRA8;
  auto h = holoscan::testing::create_operator_test<TsdfIntegrateOp>(config);
  for (std::uint32_t f = 0; f < kFrames; ++f) {
    const OpFrame& frame = frames[f];
    const std::size_t n = frame.meters.size();
    holoscan::testing::TestFrame test_frame = holoscan::testing::frame_at(
        capture_time(f),
        holoscan::testing::input(
            "depth",
            c.millimeters
                ? image(make({kHeight, kWidth}, kUInt16, frame.mm.data(), n * 2, c.device), kWidth,
                        kHeight, holoscan::schema::ImageEncoding_MONO16)
                : image(make({kHeight, kWidth, 1}, kFloat32, frame.meters.data(), n * 4, c.device),
                        kWidth, kHeight, holoscan::schema::ImageEncoding_MONO32F)),
        holoscan::testing::input("pose", pose_tensor(frame.pose, c.f64_pose, c.device_pose)));
    const auto append = [&test_frame](holoscan::testing::TestFrame extra) {
      for (auto& field : extra.fields) {
        test_frame.fields.push_back(std::move(field));
      }
    };
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
    if (c.intrinsics_input) {
      const float k[4] = {frame.intrinsics.fx, frame.intrinsics.fy, frame.intrinsics.cx,
                          frame.intrinsics.cy};
      append(holoscan::testing::frame_at(
          capture_time(f),
          holoscan::testing::input("intrinsics", make({4}, kFloat32, k, 16, false))));
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
  const std::string name = std::string("integrate") + (c.millimeters ? " uint16" : " float32") +
                           (c.device ? " device" : " host") + (color ? " color" : "") +
                           (bgra ? " bgra" : "") + (c.intrinsics_input ? " intrinsics-input" : "") +
                           (c.f64_pose ? " f64-pose" : "") + (c.device_pose ? " device-pose" : "");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), name + ": " + result.describe_failure());
  if (!result.ok()) {
    return;
  }

  // The same frames integrated directly.
  TsdfVolume direct(volume_config);
  for (const OpFrame& frame : frames) {
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
                                IntegrateParams{frame.intrinsics, 0.1F, 3.0F}, nullptr),
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
  VERIFY_MSG(poses.size() == kFrames, name + ": one integrated pose per frame");
  for (std::size_t f = 0; f < std::min<std::size_t>(poses.size(), kFrames); ++f) {
    VERIFY(poses[f].shape() == (std::vector<std::int64_t>{4, 4}));
    Matrix4 values{};
    if (poses[f].bytes().size() == sizeof(values)) {
      std::memcpy(values.data(), poses[f].bytes().data(), sizeof(values));
    }
    VERIFY_MSG(values == frames[f].pose, name + ": the integrated pose is the input pose");
    VERIFY_MSG(poses[f].sample_metadata().capture_timestamp_ns == capture_time(f).count(),
               name + ": the capture time is carried");
  }
}

// A volume with the whole tabletop orbit.
std::shared_ptr<TsdfVolume> integrated_volume(bool color) {
  auto volume = std::make_shared<TsdfVolume>(scene_volume(SceneKind::kTabletop, 64, color));
  test::integrate_orbit(*volume, SceneKind::kTabletop, 0, 12, 12, 160, 120, test::kNoise);
  return volume;
}

void check_mesh_op() {
  for (const bool color : {false, true}) {
    const std::string name = color ? "mesh color" : "mesh";
    auto volume = integrated_volume(color);
    constexpr std::uint32_t kMaxVertices = 1U << 16;
    constexpr std::uint32_t kMaxTriangles = 1U << 17;
    const RaycastParams raycast{.intrinsics = synthetic_intrinsics(kWidth, kHeight),
                                .width = kWidth,
                                .height = kHeight,
                                .depth_min = 0.1F,
                                .depth_max = 3.0F};
    const TsdfMeshConfig config{.volume = volume,
                                .mesh = true,
                                .mesh_every = 2,
                                .max_vertices = kMaxVertices,
                                .max_triangles = kMaxTriangles,
                                .raycast = true,
                                .raycast_params = raycast};
    auto h = holoscan::testing::create_operator_test<TsdfMeshOp>(config);
    std::vector<holoscan::testing::TestSample<holoscan::Tensor>> samples;
    std::vector<Matrix4> poses;
    for (std::uint32_t f = 0; f < 4; ++f) {
      poses.push_back(orbit_pose(SceneKind::kTabletop, 2 * f + 1, 12));
      auto sample =
          holoscan::testing::sample_at(pose_tensor(poses.back(), true, false), capture_time(f));
      sample.metadata.frame_id = 11U + f;
      samples.push_back(std::move(sample));
    }
    h.push_samples("pose", std::move(samples));
    std::vector<std::string> tensors{"vertices", "normals", "triangles", "mesh_counts",
                                     "raycast_normals"};
    if (color) {
      tensors.emplace_back("colors");
    }
    for (const std::string& port : tensors) {
      h.output_device(port, holoscan::DeviceId{0});
      h.capture_snapshot<holoscan::Tensor>(port, 8);
    }
    std::vector<std::string> images{"raycast_depth"};
    if (color) {
      images.emplace_back("raycast_colors");
    }
    for (const std::string& port : images) {
      h.output_device(port, holoscan::DeviceId{0});
      h.capture_snapshot<holoscan::schema::ImageT>(port, 8);
    }
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
    const auto ray_normals = result.snapshots<holoscan::Tensor>("raycast_normals");
    VERIFY_MSG(depths.size() == 4 && ray_normals.size() == 4, name + ": a ray cast on every pose");
    const std::size_t pixels = static_cast<std::size_t>(kWidth) * kHeight;
    for (std::size_t f = 0; f < std::min<std::size_t>(depths.size(), 4); ++f) {
      Device<float> depth(pixels);
      Device<float> normal(pixels * 3);
      Device<std::uint8_t> rgb(pixels * 3);
      cuda_check(
          volume->raycast(poses[f], raycast,
                          RaycastImages{depth.get(), normal.get(), color ? rgb.get() : nullptr},
                          nullptr),
          "raycast");
      const holoscan::schema::ImageT descriptor = depths[f].metadata();
      VERIFY(descriptor.encoding == holoscan::schema::ImageEncoding_MONO32F &&
             descriptor.width == kWidth && descriptor.height == kHeight);
      const auto& tensor = depths[f].tensor(holoscan::schema::ImageTensorFields::data);
      VERIFY(tensor.shape() == (std::vector<std::int64_t>{kHeight, kWidth}));
      const std::vector<float> want_depth = depth.host();
      const std::vector<float> want_normals = normal.host();
      VERIFY_MSG(tensor.bytes().size() == pixels * 4 &&
                     std::memcmp(tensor.bytes().data(), want_depth.data(), pixels * 4) == 0,
                 name + ": the ray-cast depth matches TsdfVolume::raycast()");
      VERIFY_MSG(std::memcmp(ray_normals[f].bytes().data(), want_normals.data(), pixels * 12) == 0,
                 name + ": the ray-cast normals match");
      VERIFY(std::count_if(want_depth.begin(), want_depth.end(), [](float d) { return d > 0.0F; }) >
             static_cast<std::int64_t>(pixels / 4));
      if (color) {
        const auto colors = result.snapshots<holoscan::schema::ImageT>("raycast_colors");
        const std::vector<std::uint8_t> want_rgb = rgb.host();
        VERIFY(
            colors.size() == 4 &&
            colors[f].metadata().encoding == holoscan::schema::ImageEncoding_RGB8 &&
            std::memcmp(colors[f].tensor(holoscan::schema::ImageTensorFields::data).bytes().data(),
                        want_rgb.data(), want_rgb.size()) == 0);
      }
      VERIFY_MSG(depths[f].sample_metadata().capture_timestamp_ns == capture_time(f).count() &&
                     depths[f].sample_metadata().frame_id == 11U + f,
                 name + ": the ray cast carries its pose's capture time and frame id");
    }
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
void expect_integrate_failure(const std::string& what,
                              const std::function<void(holoscan::schema::ImageT&, holoscan::Tensor&,
                                                       TsdfIntegrateConfig&)>& change) {
  const std::vector<OpFrame> frames = scene_frames();
  auto volume = std::make_shared<TsdfVolume>(scene_volume(SceneKind::kTabletop, 32, false));
  TsdfIntegrateConfig config{.volume = volume,
                             .intrinsics = frames[0].intrinsics,
                             .depth_scale = 1.0F,
                             .max_width = kWidth,
                             .max_height = kHeight};
  holoscan::schema::ImageT depth = image(make({kHeight, kWidth}, kFloat32, frames[0].meters.data(),
                                              frames[0].meters.size() * 4, false),
                                         kWidth, kHeight, holoscan::schema::ImageEncoding_MONO32F);
  holoscan::Tensor pose = pose_tensor(frames[0].pose, false, false);
  change(depth, pose, config);
  auto h = holoscan::testing::create_operator_test<TsdfIntegrateOp>(config);
  h.push_frame(holoscan::testing::frame_at(capture_time(0),
                                           holoscan::testing::input("depth", depth),
                                           holoscan::testing::input("pose", pose)));
  h.capture_snapshot<holoscan::Tensor>("integrated_pose");
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), what + " is rejected");
  VERIFY_MSG(result.describe_failure().find(what) != std::string::npos,
             "failure names '" + what + "': " + result.describe_failure());
}

void check_rejections() {
  expect_integrate_failure(
      "unsupported depth encoding",
      [](holoscan::schema::ImageT& depth, holoscan::Tensor&, TsdfIntegrateConfig&) {
        const std::vector<std::uint8_t> rgb(kWidth * kHeight * 3, 0U);
        depth = image(make({kHeight, kWidth, 3}, kUInt8, rgb.data(), rgb.size(), false), kWidth,
                      kHeight, holoscan::schema::ImageEncoding_RGB8);
      });
  expect_integrate_failure("exceeds the configured",
                           [](holoscan::schema::ImageT&, holoscan::Tensor&,
                              TsdfIntegrateConfig& config) { config.max_width = kWidth - 1; });
  expect_integrate_failure(
      "pose must be a packed float32 or float64 [4, 4]",
      [](holoscan::schema::ImageT&, holoscan::Tensor& pose, TsdfIntegrateConfig&) {
        const float values[12] = {};
        pose = make({3, 4}, kFloat32, values, sizeof(values), false);
      });
  expect_integrate_failure(
      "pose is not a rigid transform",
      [](holoscan::schema::ImageT&, holoscan::Tensor& pose, TsdfIntegrateConfig&) {
        Matrix4 scaled = identity_pose();
        scaled[0] = 2.0F;
        pose = pose_tensor(scaled, false, false);
      });

  // Unusable settings.
  auto volume = std::make_shared<TsdfVolume>(scene_volume(SceneKind::kTabletop, 16, false));
  const CameraIntrinsics k = synthetic_intrinsics(kWidth, kHeight);
  const auto integrate_refused = [](TsdfIntegrateConfig config) {
    return test::throws<std::invalid_argument>([&config] { TsdfIntegrateOp op(config); });
  };
  VERIFY(integrate_refused(TsdfIntegrateConfig{.intrinsics = k}));
  VERIFY(integrate_refused(TsdfIntegrateConfig{.volume = volume}));
  VERIFY(integrate_refused(
      TsdfIntegrateConfig{.volume = volume, .intrinsics = k, .intrinsics_input = true}));
  VERIFY(integrate_refused(TsdfIntegrateConfig{.volume = volume, .intrinsics = k, .color = true}));
  VERIFY(integrate_refused(TsdfIntegrateConfig{
      .volume = volume, .intrinsics = k, .depth_min = 2.0F, .depth_max = 1.0F}));
  VERIFY(integrate_refused(
      TsdfIntegrateConfig{.volume = volume, .intrinsics = k, .depth_scale = 0.0F}));
  VERIFY(!integrate_refused(TsdfIntegrateConfig{.volume = volume, .intrinsics = k}));
  const auto mesh_refused = [](TsdfMeshConfig config) {
    return test::throws<std::invalid_argument>([&config] { TsdfMeshOp op(config); });
  };
  VERIFY(mesh_refused(TsdfMeshConfig{}));
  VERIFY(mesh_refused(TsdfMeshConfig{.volume = volume, .mesh = false}));
  VERIFY(mesh_refused(TsdfMeshConfig{.volume = volume, .mesh_every = 0}));
  VERIFY(mesh_refused(TsdfMeshConfig{.volume = volume, .max_vertices = 0}));
  VERIFY(mesh_refused(TsdfMeshConfig{.volume = volume, .raycast = true}));  // No intrinsics.
  VERIFY(!mesh_refused(TsdfMeshConfig{.volume = volume}));
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
  cases.push_back({"TsdfIntegrateOp RGB8, intrinsics input, float64 pose", [] {
                     check_integrate_op(IntegrateCase{.millimeters = true,
                                                      .device = true,
                                                      .color = holoscan::schema::ImageEncoding_RGB8,
                                                      .intrinsics_input = true,
                                                      .f64_pose = true});
                   }});
  cases.push_back({"TsdfIntegrateOp BGRA8 host, device pose", [] {
                     check_integrate_op(
                         IntegrateCase{.millimeters = false,
                                       .device = false,
                                       .color = holoscan::schema::ImageEncoding_BGRA8,
                                       .device_pose = true});
                   }});
  cases.push_back({"TsdfMeshOp mesh every second pose and ray cast", check_mesh_op});
  cases.push_back({"TsdfMeshOp capacity", check_mesh_overflow});
  cases.push_back({"rejections", check_rejections});
  return test::run(cases);
}

}  // namespace holoscan::tsdf

int main() { return holoscan::tsdf::run_all(); }
