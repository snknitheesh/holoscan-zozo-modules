// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The two operators through the SDK's operator test harness, on clouds of the
// synthetic terrain made at test time, sent as holoscan-messages' PointCloud:
// organized and unorganized clouds in device, pinned host and pageable host
// memory, with a configured pose and with poses as an input (float32 and
// float64, host and device), fixed and following windows, every output
// checked bit for bit against HeightMapper and compute_traversability() run
// directly; map_info; the images and their headers; the capture time and
// frame id carried to every output; an empty cloud, a cloud without a capture
// time (the outputs then carry the activation time) and a cloud whose
// producer counted no valid point; and the rejection of unusable inputs (a
// cloud that is not packed or is in the wrong frame among them) and settings.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/height_map/height_map_op.hpp>
#include <holoscan/height_map/synthetic_terrain.hpp>
#include <holoscan/height_map/traversability_op.hpp>
#include <holoscan/height_map/views.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "tensor_checks.hpp"
#include "terrain_support.hpp"
#include "test_support.hpp"

namespace holoscan::height_map {
namespace {

using std::chrono_literals::operator""ms;
namespace ht = holoscan::testing;
namespace msg = holoscan::messages;
using holoscan::MemoryStorageType;

constexpr std::size_t kCaptures = 8;
constexpr const char* kSensorFrame = "sensor_optical";

// A tensor of `shape` holding `size` bytes from `bytes`, in `memory`: device,
// pinned host (kHost) or pageable host (kSystem) memory; `byte_strides` when
// given.
holoscan::Tensor make(const std::vector<std::int64_t>& shape, DLDataType dtype, const void* bytes,
                      std::size_t size, MemoryStorageType memory,
                      std::optional<std::vector<std::int64_t>> byte_strides = std::nullopt) {
  auto tensor = holoscan::make_tensor(shape, dtype, byte_strides, memory,
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

holoscan::Tensor make(const std::vector<std::int64_t>& shape, DLDataType dtype, const void* bytes,
                      std::size_t size, bool device) {
  return make(shape, dtype, bytes, size,
              device ? MemoryStorageType::kDevice : MemoryStorageType::kHost);
}

// The PointCloud message of `points`, float32 [N, 3] or [H, W, 3], in frame
// `frame`; its absent fields [0] in the same memory.
msg::schema::PointCloudT cloud_message(holoscan::Tensor points, MemoryStorageType memory,
                                       const std::string& frame = kSensorFrame,
                                       std::int64_t valid_count = -1) {
  msg::schema::PointCloudT message;
  message.header =
      std::make_shared<holoscan::schema::HeaderT>(msg::to_message(msg::Header{.frame_id = frame}));
  message.organized = points.ndim() == 3;
  message.points = std::make_shared<holoscan::Tensor>(std::move(points));
  message.colors = std::make_shared<holoscan::Tensor>(make({0}, kUInt8, nullptr, 0, memory));
  message.normals = std::make_shared<holoscan::Tensor>(make({0}, kFloat32, nullptr, 0, memory));
  message.intensity = std::make_shared<holoscan::Tensor>(make({0}, kFloat32, nullptr, 0, memory));
  message.valid_count = valid_count;
  return message;
}

const char* memory_name(MemoryStorageType memory) {
  return memory == MemoryStorageType::kDevice ? "device"
         : memory == MemoryStorageType::kHost ? "pinned"
                                              : "pageable";
}

std::chrono::nanoseconds capture_time(std::uint32_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

template <typename Snapshot>
std::vector<float> floats(const Snapshot& snapshot) {
  std::vector<float> values(snapshot.bytes().size() / sizeof(float));
  std::memcpy(values.data(), snapshot.bytes().data(), values.size() * sizeof(float));
  return values;
}

template <typename Snapshot>
std::vector<double> doubles(const Snapshot& snapshot) {
  std::vector<double> values(snapshot.bytes().size() / sizeof(double));
  std::memcpy(values.data(), snapshot.bytes().data(), values.size() * sizeof(double));
  return values;
}

template <typename Snapshot>
std::vector<std::uint8_t> bytes(const Snapshot& snapshot) {
  std::vector<std::uint8_t> values(snapshot.bytes().size());
  std::memcpy(values.data(), snapshot.bytes().data(), values.size());
  return values;
}

struct Frame {
  std::vector<float> xyz;  // In the sensor frame.
  Transform pose;
};

// Frames of the course from a sensor moving along x.
std::vector<Frame> course_frames(std::uint32_t count) {
  const Terrain terrain = course_terrain();
  std::vector<Frame> frames;
  for (std::uint32_t f = 0; f < count; ++f) {
    Frame frame;
    frame.pose = camera_pose(1.0 + 0.23 * f, 0.1, 1.0, 0.05 * f, 0.6);
    frame.xyz =
        test::transformed(test::terrain_cloud(terrain, 0.0F, 4.5F, -1.5F, 1.5F, 0.02F, 0.002F, f),
                          test::inverse(frame.pose));
    frames.push_back(std::move(frame));
  }
  return frames;
}

HeightMapConfig all_outputs(MapMode mode) {
  HeightMapConfig config;
  config.map.geometry =
      MapGeometry{.resolution = 0.05F, .rows = 60, .cols = 70, .origin_x = 0.5, .origin_y = -1.4};
  config.map.mode = mode;
  config.outputs = HeightMapOutputs{
      .variance = true, .count = true, .min_max = true, .time_since_update = true, .image = true};
  config.frame_id = "odom";
  config.max_points = 200000;
  return config;
}

const std::vector<std::string> kLayerPorts{
    "elevation", "variance", "count", "min_elevation", "max_elevation", "time_since_update",
};

void check_height_map_op() {
  const std::vector<Frame> frames = course_frames(3);
  for (const MapMode mode : {MapMode::kFixed, MapMode::kFollow}) {
    for (const MemoryStorageType memory :
         {MemoryStorageType::kDevice, MemoryStorageType::kHost, MemoryStorageType::kSystem}) {
      const bool device = memory == MemoryStorageType::kDevice;
      for (const bool organized : {false, true}) {
        // 0: configured pose; 1: float64 host pose; 2: float32 device pose.
        for (const int pose_kind : {0, 1, 2}) {
          HeightMapConfig config = all_outputs(mode);
          config.pose_input = pose_kind != 0;
          if (pose_kind == 0) {
            config.sensor_pose = frames[0].pose;
          }
          const std::string name = std::string(mode == MapMode::kFixed ? "fixed" : "follow") + " " +
                                   memory_name(memory) +
                                   (organized ? " organized" : " unorganized") + " pose " +
                                   std::to_string(pose_kind);
          auto h = ht::create_operator_test<HeightMapOp>(config);
          for (std::uint32_t f = 0; f < frames.size(); ++f) {
            const std::vector<float>& xyz = frames[f].xyz;
            const auto n = static_cast<std::int64_t>(xyz.size() / 3);
            // Organized: one row of points.
            const std::vector<std::int64_t> shape =
                organized ? std::vector<std::int64_t>{1, n, 3} : std::vector<std::int64_t>{n, 3};
            holoscan::Tensor points = make(shape, kFloat32, xyz.data(), xyz.size() * 4, memory);
            if (f == 0) {
              // The memory each case means to test.
              const DLDeviceType want = device                               ? kDLCUDA
                                        : memory == MemoryStorageType::kHost ? kDLCUDAHost
                                                                             : kDLCPU;
              VERIFY_MSG(points.device().device_type == want, name + ": the cloud's memory");
            }
            msg::schema::PointCloudT cloud = cloud_message(std::move(points), memory);
            if (pose_kind == 0) {
              auto sample = ht::sample_at(std::move(cloud), capture_time(f));
              sample.metadata.frame_id = 7U + f;
              h.push_samples("point_cloud", {std::move(sample)});
              continue;
            }
            const Transform& p = frames[f].pose;
            std::vector<double> m64(p.m, p.m + 12);
            m64.insert(m64.end(), {0.0, 0.0, 0.0, 1.0});
            std::vector<float> m32(m64.begin(), m64.end());
            holoscan::Tensor matrix = pose_kind == 1
                                          ? make({4, 4}, kFloat64, m64.data(), 128, false)
                                          : make({4, 4}, kFloat32, m32.data(), 64, true);
            h.push_frame(ht::frame_at(capture_time(f), ht::input("point_cloud", std::move(cloud)),
                                      ht::input("pose", matrix)));
          }
          if (device) {
            h.input_source_device("point_cloud", holoscan::DeviceId{0});
          }
          if (pose_kind == 2) {
            h.input_source_device("pose", holoscan::DeviceId{0});
          }
          for (const std::string& port : kLayerPorts) {
            h.output_device(port, holoscan::DeviceId{0});
            h.capture_snapshot<holoscan::Tensor>(port, kCaptures);
          }
          h.capture_snapshot<holoscan::Tensor>("map_info", kCaptures);
          h.output_device("elevation_image", holoscan::DeviceId{0});
          h.capture_snapshot<holoscan::schema::ImageT>("elevation_image", kCaptures);
          const auto result = h.run();
          VERIFY_MSG(result.ok(), name + ": " + result.describe_failure());
          if (!result.ok()) {
            continue;
          }
          // The mapper run directly, with the same clouds, poses and times.
          HeightMapper mapper(config.map);
          for (std::uint32_t f = 0; f < frames.size(); ++f) {
            const std::string at = name + " frame " + std::to_string(f);
            Transform pose = pose_kind == 0 ? frames[0].pose : frames[f].pose;
            if (pose_kind == 2) {
              // The operator received it in float32.
              for (double& value : pose.m) {
                value = static_cast<float>(value);
              }
            }
            test::update(mapper, frames[f].xyz, pose, capture_time(f).count());
            const test::HostLayers want = test::read_layers(mapper);
            const std::vector<const std::vector<float>*> wanted{
                &want.elevation,     &want.variance,      &want.count,
                &want.min_elevation, &want.max_elevation, &want.time};
            const MapGeometry g = mapper.geometry();
            for (std::size_t k = 0; k < kLayerPorts.size(); ++k) {
              const auto got = result.snapshots<holoscan::Tensor>(kLayerPorts[k]);
              if (got.size() <= f) {
                VERIFY_MSG(false, at + ": " + kLayerPorts[k] + " for every cloud");
                continue;
              }
              VERIFY_MSG(got[f].shape() == (std::vector<std::int64_t>{60, 70}),
                         at + ": " + kLayerPorts[k] + " shape");
              VERIFY_MSG(test::same_floats(floats(got[f]), *wanted[k]),
                         at + ": " + kLayerPorts[k] + " matches the mapper");
              VERIFY_MSG(got[f].sample_metadata().capture_timestamp_ns == capture_time(f).count(),
                         at + ": capture time");
              if (pose_kind == 0) {
                VERIFY_MSG(got[f].sample_metadata().frame_id == 7U + f, at + ": frame id");
              }
            }
            const auto info = result.snapshots<holoscan::Tensor>("map_info");
            if (info.size() > f) {
              const std::vector<double> values = doubles(info[f]);
              VERIFY_MSG(values == (std::vector<double>{g.origin_x, g.origin_y, 0.05F, 60.0, 70.0}),
                         at + ": map_info");
              VERIFY(info[f].sample_metadata().capture_timestamp_ns == capture_time(f).count());
            } else {
              VERIFY_MSG(false, at + ": map_info for every cloud");
            }
            const auto images = result.snapshots<holoscan::schema::ImageT>("elevation_image");
            if (images.size() > f) {
              const holoscan::schema::ImageT image = images[f].metadata();
              VERIFY(image.encoding == holoscan::schema::ImageEncoding_RGB8 && image.width == 70 &&
                     image.height == 60);
              VERIFY_MSG(image.header && image.header->frame_id == "odom" &&
                             image.header->capture_timestamp_ns == capture_time(f).count(),
                         at + ": image header");
              const auto& pixels = images[f].tensor(holoscan::schema::ImageTensorFields::data);
              VERIFY(pixels.shape() == (std::vector<std::int64_t>{60, 70, 3}));
              // Unknown cells gray, known ones colored.
              const std::vector<std::uint8_t> rgb = bytes(pixels);
              std::size_t mismatched = 0;
              for (std::size_t i = 0; i < want.elevation.size(); ++i) {
                const bool gray = rgb[3 * i] == kUnknownGray && rgb[3 * i + 1] == kUnknownGray &&
                                  rgb[3 * i + 2] == kUnknownGray;
                mismatched += gray == std::isnan(want.elevation[i]) ? 0U : 1U;
              }
              VERIFY_MSG(mismatched == 0, at + ": the image shows the known cells");
            } else {
              VERIFY_MSG(false, at + ": an image for every cloud");
            }
            if (f == 2) {
              const auto known = std::count_if(want.elevation.begin(), want.elevation.end(),
                                               [](float v) { return std::isfinite(v); });
              VERIFY_MSG(known > 1000, at + ": cells known");
            }
          }
          if (mode == MapMode::kFollow && pose_kind != 0) {
            const auto info = result.snapshots<holoscan::Tensor>("map_info");
            VERIFY_MSG(info.size() == 3 && doubles(info[2])[0] > doubles(info[0])[0],
                       name + ": the window follows the pose");
          }
        }
      }
    }
  }
}

// Only the configured outputs exist; an empty cloud gives an unknown map. The
// cloud has no capture time: the outputs carry the activation time instead,
// so that a consumer aligning them by capture time still can.
void check_minimal_outputs() {
  HeightMapConfig config;
  config.outputs = HeightMapOutputs{.variance = false};
  config.map.geometry = MapGeometry{.resolution = 0.1F, .rows = 10, .cols = 12};
  config.cloud_frame_id = kSensorFrame;
  auto h = ht::create_operator_test<HeightMapOp>(config);
  h.push("point_cloud",
         cloud_message(make({0, 3}, kFloat32, nullptr, 0, true), MemoryStorageType::kDevice));
  h.input_source_device("point_cloud", holoscan::DeviceId{0});
  h.output_device("elevation", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::Tensor>("elevation", kCaptures);
  h.capture_snapshot<holoscan::Tensor>("map_info", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), "minimal outputs: " + result.describe_failure());
  if (!result.ok()) {
    return;
  }
  const auto elevation = result.snapshots<holoscan::Tensor>("elevation");
  VERIFY(elevation.size() == 1);
  if (!elevation.empty()) {
    VERIFY_MSG(elevation[0].sample_metadata().source_clock_id.valid(),
               "an unstamped cloud's map carries the activation time");
    const std::vector<float> values = floats(elevation[0]);
    VERIFY(values.size() == 120U &&
           std::all_of(values.begin(), values.end(), [](float v) { return std::isnan(v); }));
  }
  const auto info = result.snapshots<holoscan::Tensor>("map_info");
  VERIFY(info.size() == 1 && doubles(info[0]) == (std::vector<double>{-5.0, -5.0, 0.1F, 10, 12}));
  if (!info.empty()) {
    VERIFY(info[0].sample_metadata().source_clock_id.valid());
  }
}

// A producer that counted no valid point: the map takes none of the points
// it carries, and is unknown.
void check_no_valid_points() {
  HeightMapConfig config;
  config.outputs = HeightMapOutputs{.variance = false};
  config.map.geometry = MapGeometry{.resolution = 0.1F, .rows = 10, .cols = 12};
  const std::vector<float> xyz(30, 0.0F);
  auto h = ht::create_operator_test<HeightMapOp>(config);
  h.push_at("point_cloud",
            cloud_message(make({10, 3}, kFloat32, xyz.data(), xyz.size() * 4, true),
                          MemoryStorageType::kDevice, kSensorFrame, 0),
            100ms);
  h.input_source_device("point_cloud", holoscan::DeviceId{0});
  h.output_device("elevation", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::Tensor>("elevation", kCaptures);
  h.capture_snapshot<holoscan::Tensor>("map_info", kCaptures);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), "no valid points: " + result.describe_failure());
  const auto elevation = result.snapshots<holoscan::Tensor>("elevation");
  VERIFY(elevation.size() == 1);
  if (!elevation.empty()) {
    const std::vector<float> values = floats(elevation[0]);
    VERIFY_MSG(std::all_of(values.begin(), values.end(), [](float v) { return std::isnan(v); }),
               "valid_count 0: every cell unknown");
  }
}

void check_traversability_op() {
  // An elevation layer of the course with holes, and its map_info.
  const Terrain terrain = course_terrain();
  const MapGeometry g{
      .resolution = 0.05F, .rows = 50, .cols = 64, .origin_x = 1.0, .origin_y = -1.3};
  std::vector<float> layer;
  for (int r = 0; r < g.rows; ++r) {
    for (int c = 0; c < g.cols; ++c) {
      layer.push_back((r * 7 + c * 3) % 23 == 0
                          ? std::numeric_limits<float>::quiet_NaN()
                          : terrain_height(terrain, static_cast<float>(test::cell_x(g, c)),
                                           static_cast<float>(test::cell_y(g, r))));
    }
  }
  const std::vector<double> info{g.origin_x, g.origin_y, g.resolution, 50.0, 64.0};
  for (const bool device : {true, false}) {
    TraversabilityConfig config;
    config.components = true;
    config.image = true;
    config.frame_id = "world";
    config.max_rows = 64;
    config.max_cols = 64;
    auto h = ht::create_operator_test<TraversabilityOp>(config);
    for (std::uint32_t f = 0; f < 2; ++f) {
      h.push_frame(ht::frame_at(
          capture_time(f),
          ht::input("elevation", make({50, 64}, kFloat32, layer.data(), layer.size() * 4, device)),
          ht::input("map_info", make({5}, kFloat64, info.data(), 40, false))));
    }
    if (device) {
      h.input_source_device("elevation", holoscan::DeviceId{0});
    }
    const std::vector<std::string> ports{"cost", "unknown", "slope", "step_height", "roughness"};
    for (const std::string& port : ports) {
      h.output_device(port, holoscan::DeviceId{0});
      h.capture_snapshot<holoscan::Tensor>(port, kCaptures);
    }
    h.output_device("cost_image", holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::schema::ImageT>("cost_image", kCaptures);
    const auto result = h.run();
    const std::string name = std::string("traversability") + (device ? " device" : " host");
    VERIFY_MSG(result.ok(), name + ": " + result.describe_failure());
    if (!result.ok()) {
      continue;
    }
    const std::size_t cells = layer.size();
    test::Scratch<float> in(layer);
    test::Scratch<float> cost(cells), slope(cells), step(cells), roughness(cells);
    test::Scratch<std::uint8_t> unknown(cells);
    VERIFY(compute_traversability(
               in.data, g.rows, g.cols, g.resolution, config.traversability,
               TraversabilityLayers{cost.data, unknown.data, slope.data, step.data, roughness.data},
               nullptr) == cudaSuccess);
    const std::vector<std::vector<float>> want{cost.host(), slope.host(), step.host(),
                                               roughness.host()};
    for (std::uint32_t f = 0; f < 2; ++f) {
      const std::string at = name + " frame " + std::to_string(f);
      const auto got_cost = result.snapshots<holoscan::Tensor>("cost");
      const auto got_unknown = result.snapshots<holoscan::Tensor>("unknown");
      if (got_cost.size() <= f || got_unknown.size() <= f) {
        VERIFY_MSG(false, at + ": one output per layer");
        continue;
      }
      VERIFY_MSG(test::same_floats(floats(got_cost[f]), want[0]), at + ": cost");
      VERIFY_MSG(bytes(got_unknown[f]) == unknown.host(), at + ": unknown");
      VERIFY(got_unknown[f].shape() == (std::vector<std::int64_t>{50, 64}));
      const char* components[] = {"slope", "step_height", "roughness"};
      for (int k = 0; k < 3; ++k) {
        const auto got = result.snapshots<holoscan::Tensor>(components[k]);
        VERIFY_MSG(got.size() > f && test::same_floats(floats(got[f]), want[k + 1]),
                   at + ": " + components[k]);
      }
      VERIFY(got_cost[f].sample_metadata().capture_timestamp_ns == capture_time(f).count());
      const auto images = result.snapshots<holoscan::schema::ImageT>("cost_image");
      VERIFY_MSG(images.size() > f && images[f].metadata().header &&
                     images[f].metadata().header->frame_id == "world",
                 at + ": cost image with its frame");
    }
  }
}

template <typename Op, typename Config>
void check_rejects(const Config& config, const std::vector<std::string>& inputs,
                   std::vector<holoscan::Tensor> tensors, const std::vector<std::string>& outputs,
                   const std::string& what) {
  auto h = ht::create_operator_test<Op>(config);
  if (inputs.size() == 1) {
    h.push(inputs[0], tensors[0]);
  } else {
    h.push_frame(
        ht::frame_at(100ms, ht::input(inputs[0], tensors[0]), ht::input(inputs[1], tensors[1])));
  }
  for (std::size_t k = 0; k < inputs.size(); ++k) {
    if (tensors[k].device().device_type == kDLCUDA) {
      h.input_source_device(inputs[k], holoscan::DeviceId{0});
    }
  }
  for (const std::string& output : outputs) {
    h.output_device(output, holoscan::DeviceId{0});
  }
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), what + " is rejected");
  VERIFY_MSG(result.describe_failure().find(what) != std::string::npos,
             "failure names '" + what + "': " + result.describe_failure());
}

// The operators' packing check, which the harness cannot reach (it stages
// only packed tensors; holoscan-messages' tests send a cloud with gaps
// through a graph): any stride on an axis of extent 1, none elsewhere.
void check_packed() {
  const std::vector<float> values(12, 1.0F);
  const auto device = MemoryStorageType::kDevice;
  VERIFY(detail::packed(make({4, 3}, kFloat32, values.data(), 48, device)));
  // Points 24 bytes apart, a gap of 12 after each: 36 bytes hold both.
  VERIFY(!detail::packed(
      make({2, 3}, kFloat32, values.data(), 36, device, std::vector<std::int64_t>{24, 4})));
  VERIFY(!detail::packed(
      make({3, 2}, kFloat32, values.data(), 24, device, std::vector<std::int64_t>{4, 12})));
  // One row: its stride is never stepped along.
  VERIFY(detail::packed(make({1, 4, 3}, kFloat32, values.data(), 48, device,
                             std::vector<std::int64_t>{4096, 12, 4})));
}

// HeightMapOp refuses `cloud` (and `pose`, with `pose_input`), naming `what`.
void check_map_rejects(const HeightMapConfig& config, msg::schema::PointCloudT cloud,
                       std::optional<holoscan::Tensor> pose, const std::string& what) {
  auto h = ht::create_operator_test<HeightMapOp>(config);
  const bool device = cloud.points->device().device_type == kDLCUDA;
  if (pose) {
    h.push_frame(
        ht::frame_at(100ms, ht::input("point_cloud", std::move(cloud)), ht::input("pose", *pose)));
  } else {
    h.push_at("point_cloud", std::move(cloud), 100ms);
  }
  if (device) {
    h.input_source_device("point_cloud", holoscan::DeviceId{0});
  }
  for (const char* output : {"elevation", "variance"}) {
    h.output_device(output, holoscan::DeviceId{0});
  }
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), what + " is rejected");
  VERIFY_MSG(result.describe_failure().find(what) != std::string::npos,
             "failure names '" + what + "': " + result.describe_failure());
}

void check_rejected_inputs() {
  const std::vector<float> xyz(12, 1.0F);
  const std::vector<double> xyz64(12, 1.0);
  const HeightMapConfig config;
  const auto device = MemoryStorageType::kDevice;
  const auto points = [&](const std::vector<std::int64_t>& shape) {
    return make(shape, kFloat32, xyz.data(), 48, device);
  };
  const std::string shape = "point_cloud.points must be float32";
  check_map_rejects(config, cloud_message(make({4, 3}, kFloat64, xyz64.data(), 96, device), device),
                    std::nullopt, shape);
  check_map_rejects(config, cloud_message(points({3, 4}), device), std::nullopt, shape);
  HeightMapConfig small = config;
  small.max_points = 3;
  check_map_rejects(small, cloud_message(points({4, 3}), device), std::nullopt,
                    "exceed the configured maximum of 3");
  HeightMapConfig framed = config;
  framed.cloud_frame_id = "lidar";
  check_map_rejects(framed, cloud_message(points({4, 3}), device), std::nullopt,
                    "the cloud is in frame '" + std::string(kSensorFrame) + "', not 'lidar'");
  HeightMapConfig posed = config;
  posed.pose_input = true;
  std::vector<double> sheared{1, 0.5, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  check_map_rejects(posed, cloud_message(points({4, 3}), device),
                    make({4, 4}, kFloat64, sheared.data(), 128, false),
                    "pose is not a rigid transform");
  check_map_rejects(posed, cloud_message(points({4, 3}), device),
                    make({3, 4}, kFloat64, sheared.data(), 96, false),
                    "pose must be float32 or float64 [4, 4]");

  const std::vector<float> layer(12, 0.0F);
  const std::vector<double> info{0.0, 0.0, 0.1, 3.0, 4.0};
  const std::vector<double> bad_info{0.0, 0.0, -0.1, 3.0, 4.0};
  const std::vector<std::string> cost_outputs{"cost", "unknown"};
  check_rejects<TraversabilityOp>(
      TraversabilityConfig{}, {"elevation", "map_info"},
      {make({4, 3}, kFloat32, layer.data(), 48, true), make({5}, kFloat64, info.data(), 40, false)},
      cost_outputs, "elevation must be float32 [rows, cols]");
  check_rejects<TraversabilityOp>(TraversabilityConfig{}, {"elevation", "map_info"},
                                  {make({3, 4}, kFloat32, layer.data(), 48, true),
                                   make({5}, kFloat64, bad_info.data(), 40, false)},
                                  cost_outputs, "map_info holds an unusable grid");
  check_rejects<TraversabilityOp>(
      TraversabilityConfig{.max_rows = 2}, {"elevation", "map_info"},
      {make({3, 4}, kFloat32, layer.data(), 48, true), make({5}, kFloat64, info.data(), 40, false)},
      cost_outputs, "exceeds the configured maximum of 2x1024");
}

void check_rejected_configs() {
  const auto rejects = [](const auto& construct, const std::string& what) {
    try {
      construct();
      VERIFY_MSG(false, what + " is rejected");
    } catch (const std::invalid_argument& error) {
      VERIFY_MSG(std::string(error.what()).find(what) != std::string::npos,
                 "'" + std::string(error.what()) + "' names '" + what + "'");
    }
  };
  rejects([] { HeightMapOp op(HeightMapConfig{.max_points = 0}); }, "max_points");
  rejects(
      [] {
        HeightMapConfig config;
        config.map.geometry.resolution = -1.0F;
        HeightMapOp op(config);
      },
      "resolution must be positive");
  rejects(
      [] {
        HeightMapConfig config;
        config.sensor_pose.m[0] = 2.0;
        HeightMapOp op(config);
      },
      "sensor_pose must be a rigid transform");
  rejects([] { TraversabilityOp op(TraversabilityConfig{.max_cols = 0}); },
          "max_rows and max_cols");
  rejects(
      [] {
        TraversabilityConfig config;
        config.traversability.max_slope_deg = 0.0F;
        TraversabilityOp op(config);
      },
      "max_slope_deg");
}

}  // namespace
}  // namespace holoscan::height_map

int main() {
  using holoscan::height_map::test::TestCase;
  namespace hm = holoscan::height_map;
  return holoscan::height_map::test::run({
      TestCase{"HeightMapOp", hm::check_height_map_op},
      TestCase{"HeightMapOp minimal outputs, empty cloud", hm::check_minimal_outputs},
      TestCase{"HeightMapOp cloud without valid points", hm::check_no_valid_points},
      TestCase{"TraversabilityOp", hm::check_traversability_op},
      TestCase{"rejected inputs", hm::check_rejected_inputs},
      TestCase{"packed tensors", hm::check_packed},
      TestCase{"rejected settings", hm::check_rejected_configs},
  });
}
