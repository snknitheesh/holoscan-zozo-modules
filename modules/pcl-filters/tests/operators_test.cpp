// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The four operators through the SDK's operator test harness, on PointCloud
// messages of the synthetic scene made at test time: organized and
// unorganized clouds in device, pinned and pageable memory, with no optional field and
// with colors, normals and intensity; every output layout, checked against
// the filters run directly and the CPU references, field for field; the
// header, the sample's capture time and frame number, organized and
// valid_count of every output; the plane output; empty and all-invalid
// clouds; and the rejection of unusable clouds and settings.

#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/messages/conversion.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/pcl_filters/crop_box_op.hpp>
#include <holoscan/pcl_filters/ground_removal_op.hpp>
#include <holoscan/pcl_filters/outlier_removal_op.hpp>
#include <holoscan/pcl_filters/synthetic_scene.hpp>
#include <holoscan/pcl_filters/voxel_downsample_op.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "reference.hpp"
#include "test_support.hpp"

namespace holoscan::pcl_filters {
namespace {

namespace hm = holoscan::messages;
using CloudMessage = hm::schema::PointCloudT;
using CloudSnapshot = holoscan::testing::Snapshot<CloudMessage>;

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr int kWidth = 96;
constexpr int kHeight = 72;
constexpr const char* kFrameId = "camera_optical";
// Outputs captured per port. The capture's queue holds this many, and the
// runtime sizes the output's pool for them, each of max_points.
constexpr std::size_t kCaptures = 4;

// Where a test cloud's tensors are.
enum class Memory : std::uint8_t {
  kDevice,
  kPinned,    // Host memory the GPU can read, which the operator copies asynchronously.
  kPageable,  // Plain host memory.
};

const char* memory_name(Memory memory) {
  switch (memory) {
    case Memory::kDevice:
      return " device";
    case Memory::kPinned:
      return " pinned";
    case Memory::kPageable:
      return " pageable";
  }
  return "";
}

// A tensor of `shape` holding `size` bytes from `bytes`, in `memory`.
std::shared_ptr<holoscan::Tensor> make(const std::vector<std::int64_t>& shape, DLDataType dtype,
                                       const void* bytes, std::size_t size, Memory memory) {
  const holoscan::MemoryStorageType storage =
      memory == Memory::kDevice   ? holoscan::MemoryStorageType::kDevice
      : memory == Memory::kPinned ? holoscan::MemoryStorageType::kHost
                                  : holoscan::MemoryStorageType::kSystem;
  auto tensor = holoscan::make_tensor(shape, dtype, std::nullopt, storage,
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
  return std::make_shared<holoscan::Tensor>(std::move(*tensor));
}

// A cloud on the host.
struct TestCloud {
  std::vector<float> xyz;
  std::vector<std::uint8_t> rgb;
  std::vector<float> normals;
  std::vector<float> intensity;
  // The optional fields it has, which a cloud of no points has no values of.
  hm::PointCloudContents contents;
  bool organized{};
  std::vector<std::int64_t> leading;  // [H, W] or [N].

  [[nodiscard]] std::size_t size() const { return xyz.size() / 3; }
  [[nodiscard]] std::vector<std::int64_t> shape(std::int64_t per_point) const {
    std::vector<std::int64_t> result = leading;
    if (per_point != 0) {
      result.push_back(per_point);
    }
    return result;
  }
};

enum class Fields : std::uint8_t { kPoints, kAll };

// The scene's cloud of frame `frame`, with made-up normals and intensities
// (some NaN) when `fields` asks for every field.
TestCloud scene_cloud(std::uint32_t frame, bool organized, Fields fields) {
  TestCloud cloud;
  std::vector<SceneLabel> labels;
  synthetic_scene_cloud(synthetic_scene(kWidth, kHeight, frame), cloud.xyz, cloud.rgb, labels);
  cloud.organized = organized;
  cloud.leading = organized ? std::vector<std::int64_t>{kHeight, kWidth}
                            : std::vector<std::int64_t>{kHeight * kWidth};
  if (fields == Fields::kPoints) {
    cloud.rgb.clear();
    return cloud;
  }
  cloud.contents = {.colors = true, .normals = true, .intensity = true};
  test::Random random(100U + frame);
  cloud.normals.resize(cloud.xyz.size());
  cloud.intensity.resize(cloud.size());
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    float v[3];
    for (float& value : v) {
      value = random.uniform(-1.0F, 1.0F);
    }
    const float length = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    for (int axis = 0; axis < 3; ++axis) {
      cloud.normals[3 * i + axis] =
          i % 17 == 3 ? std::numeric_limits<float>::quiet_NaN() : v[axis] / length;
    }
    cloud.intensity[i] =
        i % 19 == 4 ? std::numeric_limits<float>::quiet_NaN() : random.uniform(0.0F, 100.0F);
  }
  return cloud;
}

std::chrono::nanoseconds capture_time(std::uint32_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

// The message of `cloud`, its tensors in `memory`.
CloudMessage message_of(const TestCloud& cloud, Memory memory, std::uint32_t frame) {
  CloudMessage message;
  message.header = std::make_shared<holoscan::schema::HeaderT>(hm::to_message(hm::Header{
      .frame_id = kFrameId,
      .capture_timestamp_ns = capture_time(frame).count(),
      .device_sequence = 40U + frame,
  }));
  const auto field = [&](bool present, const auto& values, DLDataType dtype,
                         std::int64_t per_point) {
    using Value = typename std::decay_t<decltype(values)>::value_type;
    return present ? make(cloud.shape(per_point), dtype, values.data(),
                          values.size() * sizeof(Value), memory)
                   : make({0}, dtype, nullptr, 0, memory);
  };
  message.points = field(true, cloud.xyz, kFloat32, 3);
  message.colors = field(cloud.contents.colors, cloud.rgb, kUInt8, 3);
  message.normals = field(cloud.contents.normals, cloud.normals, kFloat32, 3);
  message.intensity = field(cloud.contents.intensity, cloud.intensity, kFloat32, 0);
  message.organized = cloud.organized;
  message.valid_count = -1;
  return message;
}

// The cloud a keep mask leaves in a layout, and its valid_count.
struct Expected {
  TestCloud cloud;
  std::int64_t valid_count{-1};
};

// Appends point `i` of `in` to `out`, or a removed point when `i` is null.
void append_point(const TestCloud& in, const std::size_t* i, TestCloud& out) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (int k = 0; k < 3; ++k) {
    out.xyz.push_back(i ? in.xyz[3 * *i + k] : nan);
    if (in.contents.colors) {
      out.rgb.push_back(i ? in.rgb[3 * *i + k] : std::uint8_t{0});
    }
    if (in.contents.normals) {
      out.normals.push_back(i ? in.normals[3 * *i + k] : nan);
    }
  }
  if (in.contents.intensity) {
    out.intensity.push_back(i ? in.intensity[*i] : nan);
  }
}

Expected expected(const TestCloud& in, const std::vector<std::uint8_t>& keep, OutputLayout layout) {
  Expected want;
  TestCloud& out = want.cloud;
  out.contents = in.contents;
  std::int64_t kept = 0;
  for (std::size_t i = 0; i < in.size(); ++i) {
    if (keep[i] != 0U) {
      append_point(in, &i, out);
      ++kept;
    } else if (layout == OutputLayout::kKeep) {
      append_point(in, nullptr, out);
    }
  }
  if (layout == OutputLayout::kKeep) {
    out.organized = in.organized;
    out.leading = in.leading;
    return want;
  }
  if (layout == OutputLayout::kPadded) {
    for (std::size_t i = static_cast<std::size_t>(kept); i < in.size(); ++i) {
      append_point(in, nullptr, out);
    }
  }
  out.leading = {static_cast<std::int64_t>(out.size())};
  want.valid_count = layout == OutputLayout::kDense ? kept : -1;
  return want;
}

// Bitwise equality, any NaN equal to any NaN.
bool same_floats(const std::vector<float>& a, const std::vector<std::byte>& bytes) {
  if (a.size() * sizeof(float) != bytes.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    float b = 0.0F;
    std::memcpy(&b, bytes.data() + 4 * i, 4);
    if (!(std::isnan(a[i]) && std::isnan(b)) && std::memcmp(&a[i], &b, 4) != 0) {
      return false;
    }
  }
  return true;
}

bool same_bytes(const std::vector<std::uint8_t>& a, const std::vector<std::byte>& bytes) {
  return a.size() == bytes.size() && std::memcmp(a.data(), bytes.data(), a.size()) == 0;
}

// Pushes `clouds` (frame f with frame number 7 + f), runs the operator and
// returns its outputs; `plane` adds the plane output.
struct Outputs {
  bool ok{};
  std::string failure;
  std::vector<CloudSnapshot> clouds;
  std::vector<holoscan::testing::Snapshot<holoscan::Tensor>> planes;
};

template <typename Op, typename Config>
Outputs run_op(const Config& config, const std::vector<TestCloud>& clouds, Memory memory,
               bool plane = false) {
  auto h = holoscan::testing::create_operator_test<Op>(config);
  std::vector<holoscan::testing::TestSample<CloudMessage>> samples;
  for (std::uint32_t frame = 0; frame < clouds.size(); ++frame) {
    holoscan::testing::TestSample<CloudMessage> sample =
        holoscan::testing::sample_at(message_of(clouds[frame], memory, frame), capture_time(frame));
    sample.metadata.frame_id = 7U + frame;
    samples.push_back(std::move(sample));
  }
  h.push_samples("point_cloud", std::move(samples));
  if (memory == Memory::kDevice) {
    h.input_source_device("point_cloud", holoscan::DeviceId{0});
  }
  h.output_device("filtered_point_cloud", holoscan::DeviceId{0});
  h.template capture_snapshot<CloudMessage>("filtered_point_cloud", kCaptures);
  if (plane) {
    h.output_device("plane", holoscan::DeviceId{0});
    h.template capture_snapshot<holoscan::Tensor>("plane", kCaptures);
  }
  const auto result = h.run();
  Outputs outputs;
  outputs.ok = result.ok();
  if (!outputs.ok) {
    outputs.failure = result.describe_failure();
    return outputs;
  }
  outputs.clouds = result.template snapshots<CloudMessage>("filtered_point_cloud");
  if (plane) {
    outputs.planes = result.template snapshots<holoscan::Tensor>("plane");
  }
  return outputs;
}

// Checks one output cloud against what it should be, field for field, and
// what it carries from its input frame.
void check_cloud(const std::string& at, const CloudSnapshot& got, const Expected& want,
                 std::uint32_t frame) {
  const TestCloud& cloud = want.cloud;
  const CloudMessage metadata = got.metadata();
  VERIFY_MSG(metadata.organized == cloud.organized, at + ": organized");
  VERIFY_MSG(metadata.valid_count == want.valid_count,
             at + ": valid_count " + std::to_string(metadata.valid_count) + ", expected " +
                 std::to_string(want.valid_count));
  const hm::Header header{.frame_id = kFrameId,
                          .capture_timestamp_ns = capture_time(frame).count(),
                          .device_sequence = 40U + frame};
  VERIFY_MSG(metadata.header && hm::from_message(*metadata.header) == header,
             at + ": the input's header");
  VERIFY_MSG(got.sample_metadata().capture_timestamp_ns == capture_time(frame).count(),
             at + ": capture time");
  VERIFY_MSG(got.sample_metadata().frame_id == 7U + frame, at + ": frame number");

  const auto check_field = [&](holoscan::TensorFieldSelector<CloudMessage> field, const char* name,
                               bool present, std::int64_t per_point, const auto& values) {
    const holoscan::testing::TensorSnapshot& tensor = got.tensor(field);
    if (!present) {
      VERIFY_MSG(tensor.shape() == std::vector<std::int64_t>{0},
                 at + ": " + name + " absent, as in the input");
      return;
    }
    VERIFY_MSG(tensor.shape() == cloud.shape(per_point), at + ": " + name + " shape");
    if constexpr (std::is_same_v<std::decay_t<decltype(values)>, std::vector<std::uint8_t>>) {
      VERIFY_MSG(same_bytes(values, tensor.bytes()), at + ": " + name);
    } else {
      VERIFY_MSG(same_floats(values, tensor.bytes()), at + ": " + name);
    }
  };
  check_field(hm::PointCloudFields::points, "points", true, 3, cloud.xyz);
  check_field(hm::PointCloudFields::colors, "colors", cloud.contents.colors, 3, cloud.rgb);
  check_field(hm::PointCloudFields::normals, "normals", cloud.contents.normals, 3, cloud.normals);
  check_field(hm::PointCloudFields::intensity, "intensity", cloud.contents.intensity, 0,
              cloud.intensity);
}

// Checks every output frame against the cloud `keep` leaves.
void check_outputs(const std::string& name, const Outputs& out, OutputLayout layout,
                   const std::vector<TestCloud>& clouds,
                   const std::vector<std::vector<std::uint8_t>>& keeps) {
  VERIFY_MSG(out.ok, name + ": " + out.failure);
  if (!out.ok) return;
  VERIFY_MSG(out.clouds.size() == clouds.size(), name + ": one output per cloud");
  for (std::size_t frame = 0; frame < out.clouds.size() && frame < clouds.size(); ++frame) {
    check_cloud(name + " frame " + std::to_string(frame), out.clouds[frame],
                expected(clouds[frame], keeps[frame], layout), static_cast<std::uint32_t>(frame));
  }
}

const char* layout_name(OutputLayout layout) {
  switch (layout) {
    case OutputLayout::kKeep:
      return " keep";
    case OutputLayout::kDense:
      return " dense";
    case OutputLayout::kPadded:
      return " padded";
  }
  return "";
}

// Every layout, memory and field combination of an operator whose mask
// `reference` gives.
template <typename Op, typename Config, typename Reference>
void check_masking_op(const std::string& name, Config config, Reference&& reference) {
  for (const bool organized : {true, false}) {
    for (const Memory memory : {Memory::kDevice, Memory::kPinned, Memory::kPageable}) {
      for (const Fields fields : {Fields::kPoints, Fields::kAll}) {
        for (const OutputLayout layout :
             {OutputLayout::kKeep, OutputLayout::kDense, OutputLayout::kPadded}) {
          config.cloud.layout = layout;
          const std::vector<TestCloud> clouds{scene_cloud(0, organized, fields),
                                              scene_cloud(1, organized, fields)};
          std::vector<std::vector<std::uint8_t>> keeps;
          for (const TestCloud& cloud : clouds) {
            keeps.push_back(reference(cloud));
          }
          const std::string what =
              name + (organized ? " organized" : " unorganized") + memory_name(memory) +
              (fields == Fields::kAll ? " all fields" : "") + layout_name(layout);
          check_outputs(what, run_op<Op>(config, clouds, memory), layout, clouds, keeps);
        }
      }
    }
  }
}

// The filters run directly on a host cloud.
template <typename Run>
std::vector<std::uint8_t> device_mask(const TestCloud& cloud, Run&& run) {
  const auto n = static_cast<std::uint32_t>(cloud.size());
  float* xyz = nullptr;
  std::uint8_t* keep = nullptr;
  std::vector<std::uint8_t> result(n);
  if (cudaMalloc(&xyz, cloud.xyz.size() * 4 + 4) != cudaSuccess ||
      cudaMalloc(&keep, n + 1U) != cudaSuccess) {
    throw std::runtime_error("cudaMalloc failed");
  }
  cudaMemcpy(xyz, cloud.xyz.data(), cloud.xyz.size() * 4, cudaMemcpyHostToDevice);
  const cudaError_t status = run(xyz, n, keep);
  cudaMemcpy(result.data(), keep, n, cudaMemcpyDeviceToHost);
  cudaFree(xyz);
  cudaFree(keep);
  if (status != cudaSuccess) {
    throw std::runtime_error("filter failed");
  }
  return result;
}

CropBoxConfig crop_config() {
  CropBoxConfig config;
  config.box.min[2] = 1.0F;
  config.box.max[2] = 3.0F;
  config.box.max_distance = 3.2F;
  return config;
}

void check_crop_op() {
  const CropBoxConfig config = crop_config();
  check_masking_op<CropBoxOp>("crop", config, [&config](const TestCloud& cloud) {
    return test::crop_reference(cloud.xyz, config.box);
  });
}

OutlierRemovalConfig outlier_config() {
  OutlierRemovalConfig config;
  config.outliers.k = 6;
  config.outliers.search_radius = 0.3F;
  config.outliers.std_multiplier = 1.5F;
  return config;
}

void check_outlier_op() {
  const OutlierRemovalConfig config = outlier_config();
  check_masking_op<OutlierRemovalOp>("outliers", config, [&config](const TestCloud& cloud) {
    OutlierFilter filter;
    return device_mask(cloud, [&](const float* xyz, std::uint32_t n, std::uint8_t* keep) {
      const cudaError_t status = filter.filter(xyz, n, config.outliers, keep, nullptr);
      return status == cudaSuccess ? cudaDeviceSynchronize() : status;
    });
  });
}

GroundRemovalConfig ground_config() {
  GroundRemovalConfig config;
  float truth[4];
  synthetic_ground_plane(synthetic_scene(kWidth, kHeight, 0), truth);
  std::copy_n(truth, 3, config.ground.up);
  config.ground.distance_threshold = 0.02F;
  return config;
}

std::vector<float> floats(const std::vector<std::byte>& bytes) {
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
  return values;
}

void check_ground_op() {
  const GroundRemovalConfig config = ground_config();
  const auto segment = [&config](const TestCloud& cloud, float plane[4]) {
    GroundSegmenter segmenter;
    float* device_plane = nullptr;
    if (cudaMalloc(&device_plane, 16) != cudaSuccess) {
      throw std::runtime_error("cudaMalloc failed");
    }
    const std::vector<std::uint8_t> keep =
        device_mask(cloud, [&](const float* xyz, std::uint32_t n, std::uint8_t* out) {
          const cudaError_t status =
              segmenter.segment(xyz, n, config.ground, device_plane, out, nullptr, nullptr);
          return status == cudaSuccess ? cudaDeviceSynchronize() : status;
        });
    cudaMemcpy(plane, device_plane, 16, cudaMemcpyDeviceToHost);
    cudaFree(device_plane);
    return keep;
  };
  // Without the plane output, which must be connected when it exists.
  GroundRemovalConfig without_plane = config;
  without_plane.emit_plane = false;
  check_masking_op<GroundRemovalOp>("ground", without_plane, [&segment](const TestCloud& cloud) {
    float plane[4];
    return segment(cloud, plane);
  });

  // The plane output: the filter's plane, close to the truth, with the
  // cloud's capture time and frame number; and the cloud as without it.
  const std::vector<TestCloud> clouds{scene_cloud(0, true, Fields::kAll),
                                      scene_cloud(1, true, Fields::kAll)};
  const Outputs out = run_op<GroundRemovalOp>(config, clouds, Memory::kDevice, true);
  VERIFY_MSG(out.ok, out.failure);
  VERIFY(out.planes.size() == 2);
  std::vector<std::vector<std::uint8_t>> keeps;
  for (std::size_t frame = 0; frame < out.planes.size(); ++frame) {
    float want[4];
    keeps.push_back(segment(clouds[frame], want));
    VERIFY(out.planes[frame].shape() == std::vector<std::int64_t>{4});
    const std::vector<float> got = floats(out.planes[frame].bytes());
    VERIFY_MSG(got.size() == 4 && std::memcmp(got.data(), want, 16) == 0,
               "the plane output is the filter's plane");
    float truth[4];
    synthetic_ground_plane(synthetic_scene(kWidth, kHeight, static_cast<std::uint32_t>(frame)),
                           truth);
    VERIFY(got.size() == 4 && std::abs(got[3] - truth[3]) < 0.01F &&
           got[0] * truth[0] + got[1] * truth[1] + got[2] * truth[2] > 0.9995F);
    const holoscan::SampleMetadata& metadata = out.planes[frame].sample_metadata();
    VERIFY(metadata.capture_timestamp_ns ==
           capture_time(static_cast<std::uint32_t>(frame)).count());
    VERIFY(metadata.frame_id == 7U + frame);
  }
  check_outputs("ground with plane", out, config.cloud.layout, clouds, keeps);
}

void check_voxel_op() {
  for (const bool organized : {true, false}) {
    for (const Memory memory : {Memory::kDevice, Memory::kPinned, Memory::kPageable}) {
      for (const Fields fields : {Fields::kPoints, Fields::kAll}) {
        for (const VoxelMode mode : {VoxelMode::kCentroid, VoxelMode::kFirstPoint}) {
          for (const OutputLayout layout : {OutputLayout::kDense, OutputLayout::kPadded}) {
            const VoxelDownsampleConfig config{
                .voxel_size = 0.04F, .mode = mode, .cloud = {.layout = layout}};
            const std::vector<TestCloud> clouds{scene_cloud(0, organized, fields),
                                                scene_cloud(1, organized, fields)};
            const Outputs out = run_op<VoxelDownsampleOp>(config, clouds, memory);
            const std::string name =
                std::string("voxels") + (organized ? " organized" : " unorganized") +
                memory_name(memory) + (fields == Fields::kAll ? " all fields" : "") +
                (mode == VoxelMode::kCentroid ? " centroid" : " first") + layout_name(layout);
            VERIFY_MSG(out.ok, name + ": " + out.failure);
            if (!out.ok) continue;
            VERIFY(out.clouds.size() == 2);
            for (std::size_t frame = 0; frame < out.clouds.size(); ++frame) {
              const TestCloud& in = clouds[frame];
              const test::VoxelResult voxels =
                  test::voxel_reference(in.xyz, in.rgb, 0.04F, mode, in.normals, in.intensity);
              Expected want;
              want.cloud.contents = in.contents;
              want.cloud.xyz = voxels.xyz;
              want.cloud.rgb = voxels.rgb;
              want.cloud.normals = voxels.normals;
              want.cloud.intensity = voxels.intensity;
              const auto count = static_cast<std::int64_t>(voxels.xyz.size() / 3);
              if (layout == OutputLayout::kPadded) {
                for (std::size_t i = static_cast<std::size_t>(count); i < in.size(); ++i) {
                  append_point(in, nullptr, want.cloud);
                }
              }
              want.cloud.leading = {static_cast<std::int64_t>(want.cloud.size())};
              want.valid_count = layout == OutputLayout::kDense ? count : -1;
              check_cloud(name + " frame " + std::to_string(frame), out.clouds[frame], want,
                          static_cast<std::uint32_t>(frame));
            }
          }
        }
      }
    }
  }
}

// An empty cloud gives empty outputs; an all-invalid one keeps nothing.
void check_empty_and_invalid() {
  TestCloud empty;
  empty.leading = {0};
  empty.contents = {.colors = true, .normals = true, .intensity = true};
  TestCloud invalid = scene_cloud(0, true, Fields::kAll);
  std::fill(invalid.xyz.begin(), invalid.xyz.end(), std::numeric_limits<float>::quiet_NaN());
  // One run each, as the harness stages one tensor layout per port.
  for (const TestCloud& cloud : {empty, invalid}) {
    const std::string kind = cloud.xyz.empty() ? " empty" : " all-invalid";
    const std::vector<TestCloud> clouds{cloud};
    const std::vector<std::vector<std::uint8_t>> keeps{std::vector<std::uint8_t>(cloud.size(), 0U)};
    for (const OutputLayout layout :
         {OutputLayout::kKeep, OutputLayout::kDense, OutputLayout::kPadded}) {
      const std::string name = std::string(kind) + layout_name(layout);
      CropBoxConfig crop = crop_config();
      crop.cloud.layout = layout;
      check_outputs("crop" + name, run_op<CropBoxOp>(crop, clouds, Memory::kDevice), layout, clouds,
                    keeps);
      OutlierRemovalConfig outliers = outlier_config();
      outliers.cloud.layout = layout;
      check_outputs("outliers" + name, run_op<OutlierRemovalOp>(outliers, clouds, Memory::kPinned),
                    layout, clouds, keeps);
      GroundRemovalConfig ground = ground_config();
      ground.cloud.layout = layout;
      const Outputs out = run_op<GroundRemovalOp>(ground, clouds, Memory::kPageable, true);
      check_outputs("ground" + name, out, layout, clouds, keeps);
      for (const auto& plane : out.planes) {
        const std::vector<float> values = floats(plane.bytes());
        VERIFY_MSG(values.size() == 4 && std::isnan(values[0]), "no plane" + name);
      }
    }
    for (const OutputLayout layout : {OutputLayout::kDense, OutputLayout::kPadded}) {
      const Outputs voxels = run_op<VoxelDownsampleOp>(
          VoxelDownsampleConfig{.cloud = {.layout = layout}}, clouds, Memory::kDevice);
      check_outputs("voxels" + kind + layout_name(layout), voxels, layout, clouds, keeps);
    }
  }
}

// Runs a crop box on one unusable cloud and checks the failure names `what`.
void check_rejects(const CropBoxConfig& config, const CloudMessage& cloud,
                   const std::string& what) {
  auto h = holoscan::testing::create_operator_test<CropBoxOp>(config);
  h.push("point_cloud", cloud);
  h.output_device("filtered_point_cloud", holoscan::DeviceId{0});
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), what + " is rejected");
  VERIFY_MSG(result.describe_failure().find(what) != std::string::npos,
             "failure names '" + what + "': " + result.describe_failure());
}

void check_rejected_inputs() {
  TestCloud four;
  four.xyz.assign(12, 1.0F);
  four.leading = {4};
  const CropBoxConfig config = crop_config();
  const std::vector<double> xyz64(12, 1.0);
  CloudMessage doubles = message_of(four, Memory::kPageable, 0);
  doubles.points = make({4, 3}, {kDLFloat, 64, 1}, xyz64.data(), 96, Memory::kPageable);
  check_rejects(config, doubles, "point_cloud.points must be float32 [N, 3]");
  CloudMessage organized_flat = message_of(four, Memory::kPageable, 0);
  organized_flat.organized = true;
  check_rejects(config, organized_flat, "point_cloud.points must be float32 [H, W, 3]");
  CropBoxConfig small = config;
  small.cloud.max_points = 3;
  check_rejects(small, message_of(four, Memory::kPageable, 0),
                "4 points exceed the configured maximum of 3");
  const std::vector<std::uint8_t> rgb(12, 9U);
  CloudMessage wrong_colors = message_of(four, Memory::kPageable, 0);
  wrong_colors.colors = make({2, 2, 3}, kUInt8, rgb.data(), 12, Memory::kPageable);
  check_rejects(config, wrong_colors, "point_cloud.colors must be uint8 [4, 3]");
  // A field the output has no room for.
  TestCloud with_normals = four;
  with_normals.normals.assign(12, 0.0F);
  with_normals.contents.normals = true;
  CropBoxConfig no_normals = config;
  no_normals.cloud.fields.normals = false;
  check_rejects(no_normals, message_of(with_normals, Memory::kPageable, 0),
                "the cloud has normals, which the output has no room for");
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
  rejects([] { VoxelDownsampleOp op(VoxelDownsampleConfig{.voxel_size = 0.0F}); },
          "voxel_size must be finite and positive");
  rejects([] { VoxelDownsampleOp op(VoxelDownsampleConfig{.voxel_size = INFINITY}); },
          "voxel_size must be finite and positive");
  rejects([] { VoxelDownsampleOp op(VoxelDownsampleConfig{.cloud = {.max_points = 0}}); },
          "max_points must be 1..");
  rejects(
      [] { VoxelDownsampleOp op(VoxelDownsampleConfig{.cloud = {.layout = OutputLayout::kKeep}}); },
      "layout must be kDense or kPadded");
  rejects(
      [] {
        CropBoxConfig config;
        config.cloud.layout = static_cast<OutputLayout>(7);
        CropBoxOp op(config);
      },
      "layout must be kKeep, kDense or kPadded");
  rejects(
      [] {
        CropBoxConfig config;
        config.box.min[1] = 2.0F;
        config.box.max[1] = 1.0F;
        CropBoxOp op(config);
      },
      "min <= max");
  rejects(
      [] {
        CropBoxConfig config;
        config.box.min_distance = 3.0F;
        config.box.max_distance = 1.0F;
        CropBoxOp op(config);
      },
      "0 <= min_distance <= max_distance");
  rejects(
      [] {
        CropBoxConfig config;
        config.box.use_transform = true;
        config.box.transform.matrix[3] = NAN;
        CropBoxOp op(config);
      },
      "transform must be finite");
  rejects(
      [] {
        GroundRemovalConfig config;
        config.ground.up[2] = 0.0F;
        GroundRemovalOp op(config);
      },
      "up must be a finite, nonzero vector");
  rejects(
      [] {
        GroundRemovalConfig config;
        config.ground.max_iterations = 0;
        GroundRemovalOp op(config);
      },
      "max_iterations must be 1..");
  rejects(
      [] {
        GroundRemovalConfig config;
        config.ground.max_tilt_deg = 91.0F;
        GroundRemovalOp op(config);
      },
      "max_tilt_deg must be 0..90");
  rejects(
      [] {
        GroundRemovalConfig config;
        config.ground.distance_threshold = -1.0F;
        GroundRemovalOp op(config);
      },
      "distance_threshold must be finite");
  rejects(
      [] {
        OutlierRemovalConfig config;
        config.outliers.k = kMaxNeighbors + 1U;
        OutlierRemovalOp op(config);
      },
      "k must be 1..");
  rejects(
      [] {
        OutlierRemovalConfig config;
        config.outliers.search_radius = 0.0F;
        OutlierRemovalOp op(config);
      },
      "search_radius must be finite and positive");
  rejects(
      [] {
        OutlierRemovalConfig config;
        config.outliers.mode = OutlierMode::kRadius;
        config.outliers.radius = -1.0F;
        OutlierRemovalOp op(config);
      },
      "radius must be finite and positive");
}

// A platform whose copies from pageable host memory read it only when the
// stream gets to them, as one whose GPU reads pageable memory in place does:
// while `late_pageable_reads` is on, cudaMemcpyAsync from pageable memory to
// the GPU reads the host memory in a host function queued on the stream,
// 20 ms later, and records an event after the copy. This program's
// definition takes the place of the CUDA runtime's for the module's library.
struct LatePageableReads {
  std::mutex mutex;
  bool on{};
  std::vector<cudaEvent_t> copied;
  std::vector<void*> bounce;  // Pinned, freed with the reads.
};

LatePageableReads& late_pageable_reads() {
  static LatePageableReads reads;
  return reads;
}

struct LateRead {
  void* to;
  const void* from;
  std::size_t bytes;
};

void late_read(void* data) {
  const auto* read = static_cast<const LateRead*>(data);
  std::this_thread::sleep_for(std::chrono::milliseconds{20});
  std::memcpy(read->to, read->from, read->bytes);
  delete read;
}

}  // namespace
}  // namespace holoscan::pcl_filters

extern "C" cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t count,
                                       cudaMemcpyKind kind, cudaStream_t stream) {
  using Real = cudaError_t (*)(void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t);
  static const auto real = reinterpret_cast<Real>(dlsym(RTLD_NEXT, "cudaMemcpyAsync"));
  auto& reads = holoscan::pcl_filters::late_pageable_reads();
  const std::lock_guard lock(reads.mutex);
  if (!reads.on || count == 0 || (kind != cudaMemcpyHostToDevice && kind != cudaMemcpyDefault)) {
    return real(dst, src, count, kind, stream);
  }
  cudaPointerAttributes from{};
  cudaPointerAttributes to{};
  const bool known = cudaPointerGetAttributes(&from, src) == cudaSuccess &&
                     cudaPointerGetAttributes(&to, dst) == cudaSuccess;
  if (!known) {
    (void)cudaGetLastError();  // Leaves no error behind for the caller.
  }
  if (!known || from.type != cudaMemoryTypeUnregistered ||
      (to.type != cudaMemoryTypeDevice && to.type != cudaMemoryTypeManaged)) {
    return real(dst, src, count, kind, stream);
  }
  void* bounce = nullptr;
  if (const cudaError_t status = cudaMallocHost(&bounce, count); status != cudaSuccess) {
    return status;
  }
  reads.bounce.push_back(bounce);
  if (const cudaError_t status = cudaLaunchHostFunc(
          stream, holoscan::pcl_filters::late_read,
          new holoscan::pcl_filters::LateRead{.to = bounce, .from = src, .bytes = count});
      status != cudaSuccess) {
    return status;
  }
  if (const cudaError_t status = real(dst, bounce, count, cudaMemcpyHostToDevice, stream);
      status != cudaSuccess) {
    return status;
  }
  cudaEvent_t copied = nullptr;
  if (const cudaError_t status = cudaEventCreateWithFlags(&copied, cudaEventDisableTiming);
      status != cudaSuccess) {
    return status;
  }
  reads.copied.push_back(copied);
  return cudaEventRecord(copied, stream);
}

namespace holoscan::pcl_filters {
namespace {

// What a graph run of host clouds sends and gets back.
struct HostRun {
  std::vector<std::vector<float>> frames;
  std::vector<std::vector<float>> outputs;
  std::atomic<std::uint32_t> received{0};
  // Copies from the host that were still running when a cloud came out.
  std::atomic<std::uint32_t> unfinished_reads{0};
};

// Sends each of the run's clouds, points only, in a runtime pool of plain
// host memory, once the previous one came back.
class HostCloudSourceOp final : public holoscan::Operator<> {
 public:
  explicit HostCloudSourceOp(std::shared_ptr<HostRun> run) : run_(std::move(run)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    std::uint64_t points = 1;
    for (const std::vector<float>& frame : run_->frames) {
      points = std::max<std::uint64_t>(points, frame.size() / 3);
    }
    hm::declare_point_cloud_output(
        spec, point_cloud, "point_cloud",
        {.max_points = points, .memory_kind = holoscan::MemoryKind::kHost});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{2}});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (emitted_ >= run_->frames.size() ||
        run_->received.load(std::memory_order_acquire) < emitted_) {
      return {};
    }
    const std::vector<float>& xyz = run_->frames[emitted_];
    const hm::PointCloudShape shape =
        hm::unorganized_cloud(static_cast<std::int64_t>(xyz.size() / 3));
    auto loan = hm::allocate_point_cloud(point_cloud, shape, {}, context.cuda_stream());
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    if (auto filled = loan->write_host_and_commit(
            hm::PointCloudFields::points,
            [&](const holoscan::TensorOutputWriteGuard& writer)
                -> holoscan::expected<void, holoscan::Error> {
              std::memcpy(writer.data(), xyz.data(), xyz.size() * sizeof(float));
              return {};
            });
        !filled) {
      return filled;
    }
    if (auto sent = hm::emit_point_cloud(
            point_cloud, std::move(*loan), {.frame_id = kFrameId}, shape, -1,
            {.capture_time = context.activation_time(), .frame_id = emitted_ + 1U});
        !sent) {
      return sent;
    }
    ++emitted_;
    return {};
  }

  holoscan::Output<CloudMessage> point_cloud;

 private:
  std::shared_ptr<HostRun> run_;
  std::uint32_t emitted_{};
};

// Counts the copies from the host still running as each cloud arrives, after
// the filter's compute() returned, then copies its points back to the host.
class PointsSinkOp final : public holoscan::Operator<> {
 public:
  explicit PointsSinkOp(std::shared_ptr<HostRun> run) : run_(std::move(run)) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.input(point_cloud, "point_cloud"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{point_cloud});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    {
      auto& reads = late_pageable_reads();
      const std::lock_guard lock(reads.mutex);
      for (cudaEvent_t copied : reads.copied) {
        run_->unfinished_reads += cudaEventQuery(copied) == cudaErrorNotReady ? 1U : 0U;
      }
    }
    auto cloud = point_cloud.receive();
    if (!cloud) {
      return holoscan::make_unexpected(std::move(cloud).error());
    }
    const holoscan::Tensor& points = *cloud->data.points;
    std::vector<float> host(static_cast<std::size_t>(points.size()));
    if (auto copied =
            points.copy_to_host(host.data(), host.size() * sizeof(float), context.cuda_stream());
        !copied) {
      return copied;
    }
    run_->outputs.push_back(std::move(host));
    const std::uint32_t received = run_->received.fetch_add(1U, std::memory_order_release) + 1U;
    if (received == run_->frames.size()) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<CloudMessage> point_cloud;

 private:
  std::shared_ptr<HostRun> run_;
};

// Clouds in plain host memory, on a platform where the copy reads it late:
// the producer may reuse that memory once the crop box's compute() returns,
// so the copy must be done by then. The clouds grow, so the operator's
// staging and scratch buffers are replaced on the way. This pins the SDK
// 5.0.0's runtime pools of host memory, which do not take a deallocation
// stream, so the operator itself waits; a pool that did could defer the
// reuse instead, and this test would then need to check that.
void check_pageable_input_copied_before_release() {
  auto run = std::make_shared<HostRun>();
  for (const std::uint32_t points : {20'000U, 200'000U, 400'000U, 400'000U}) {
    test::Random random(points);
    std::vector<float> xyz(3 * std::size_t{points});
    for (float& value : xyz) {
      value = random.uniform(-2.0F, 2.0F);
    }
    run->frames.push_back(std::move(xyz));
  }
  holoscan::Graph graph{"pageable_input"};
  auto source = graph.op<HostCloudSourceOp>("source", run);
  // The box keeps every valid point, so the output is the input.
  CropBoxConfig keep_all;
  keep_all.cloud = {.max_points = 400'000U, .fields = {}};
  auto crop = graph.op<CropBoxOp>("crop", keep_all);
  auto sink = graph.op<PointsSinkOp>("sink", run);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->point_cloud, crop->point_cloud, lossless);
  graph.add_flow(crop->filtered_point_cloud, sink->point_cloud, lossless);
  graph.partition("pageable_input").include_all();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{.operator_path = "crop",
                                            .output_port = "filtered_point_cloud",
                                            .device = holoscan::DeviceId{0}});
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  VERIFY_MSG(plan.ok(), holoscan::format_diagnostics(plan.diagnostics(), plan.graph()));
  if (!plan.ok()) return;
  auto& reads = late_pageable_reads();
  {
    const std::lock_guard lock(reads.mutex);
    reads.on = true;
  }
  const int status = holoscan::run(plan, holoscan::RunSignalPolicy::kCallerManaged);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
  {
    const std::lock_guard lock(reads.mutex);
    reads.on = false;
    VERIFY_MSG(reads.copied.size() == run->frames.size(),
               std::to_string(reads.copied.size()) + " copies from pageable memory, expected " +
                   std::to_string(run->frames.size()));
    for (cudaEvent_t copied : reads.copied) {
      cudaEventDestroy(copied);
    }
    for (void* bounce : reads.bounce) {
      cudaFreeHost(bounce);
    }
    reads.copied.clear();
    reads.bounce.clear();
  }
  VERIFY(status == 0);
  VERIFY_MSG(run->unfinished_reads == 0,
             std::to_string(run->unfinished_reads.load()) +
                 " copies from pageable input were still running after compute() returned");
  VERIFY(run->outputs.size() == run->frames.size());
  for (std::size_t frame = 0; frame < run->outputs.size() && frame < run->frames.size(); ++frame) {
    const std::vector<float>& got = run->outputs[frame];
    const std::vector<float>& want = run->frames[frame];
    VERIFY_MSG(got.size() == want.size() &&
                   std::memcmp(got.data(), want.data(), want.size() * sizeof(float)) == 0,
               "frame " + std::to_string(frame) + ": the output is not the input");
  }
}

// Takes the plane.
class PlaneSinkOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(plane, "plane"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{plane});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& /*context*/) override {
    auto got = plane.receive();
    return got ? holoscan::expected<void, holoscan::Error>{}
               : holoscan::make_unexpected(std::move(got).error());
  }

  holoscan::Input<holoscan::Tensor> plane;
};

// The README's live graph: every edge kDropOldest with depth 1, the ground
// filter forking to the voxel filter and the plane's consumer, split by the
// automatic partitioning. It compiles and runs a cloud through.
void check_live_graph() {
  auto run = std::make_shared<HostRun>();
  run->frames.push_back(scene_cloud(0, false, Fields::kPoints).xyz);
  holoscan::Graph graph{"live"};
  auto camera = graph.op<HostCloudSourceOp>("camera", run);
  GroundRemovalConfig ground = ground_config();
  ground.cloud.fields = {};
  auto ground_filter = graph.op<GroundRemovalOp>("ground", ground);
  auto voxels = graph.op<VoxelDownsampleOp>(
      "voxels", VoxelDownsampleConfig{.voxel_size = 0.05F,
                                      .cloud = {.layout = OutputLayout::kDense, .fields = {}}});
  auto planner = graph.op<PlaneSinkOp>("planner");
  auto mapper = graph.op<PointsSinkOp>("mapper", run);
  const holoscan::ConnectionOptions live{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                         .queue_depth = 1U};
  graph.add_flow(camera->point_cloud, ground_filter->point_cloud, live);
  graph.add_flow(ground_filter->filtered_point_cloud, voxels->point_cloud, live);
  graph.add_flow(ground_filter->plane, planner->plane, live);
  graph.add_flow(voxels->filtered_point_cloud, mapper->point_cloud, live);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  holoscan::CompileOptions options;
  for (const auto& [op, port] :
       {std::pair{"ground", "filtered_point_cloud"}, std::pair{"ground", "plane"},
        std::pair{"voxels", "filtered_point_cloud"}}) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
  }
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  VERIFY_MSG(plan.ok(), holoscan::format_diagnostics(plan.diagnostics(), plan.graph()));
  if (!plan.ok()) return;
  VERIFY(holoscan::run(plan, holoscan::RunSignalPolicy::kCallerManaged) == 0);
  VERIFY(run->outputs.size() == 1);
}

// What the process writes to stdout and stderr, where the SDK logs, while
// `body` runs.
std::string capture_output(const std::function<void()>& body) {
  std::cout.flush();
  std::cerr.flush();
  std::fflush(nullptr);
  FILE* file = std::tmpfile();
  VERIFY(file != nullptr);
  if (file == nullptr) return {};
  const int saved_out = dup(STDOUT_FILENO);
  const int saved_err = dup(STDERR_FILENO);
  dup2(fileno(file), STDOUT_FILENO);
  dup2(fileno(file), STDERR_FILENO);
  struct Restore {
    int out;
    int err;
    ~Restore() {
      std::cout.flush();
      std::cerr.flush();
      std::fflush(nullptr);
      dup2(out, STDOUT_FILENO);
      dup2(err, STDERR_FILENO);
      close(out);
      close(err);
    }
  };
  {
    const Restore restore{saved_out, saved_err};
    body();
  }
  std::string output;
  std::rewind(file);
  char buffer[4096];
  for (std::size_t n; (n = std::fread(buffer, 1, sizeof(buffer), file)) > 0;) {
    output.append(buffer, n);
  }
  std::fclose(file);
  return output;
}

std::size_t count_of(const std::string& text, std::string_view needle) {
  std::size_t count = 0;
  for (std::size_t at = text.find(needle); at != std::string::npos;
       at = text.find(needle, at + needle.size())) {
    ++count;
  }
  return count;
}

// Host clouds are copied to the GPU every frame, which is logged once, at the
// first, naming the operator, the fields and the bytes; device clouds log
// nothing.
void check_host_copy_logged_once() {
  const holoscan::LogLevel level = holoscan::log_level();
  holoscan::set_log_level(holoscan::LogLevel::INFO);
  const std::vector<TestCloud> clouds{scene_cloud(0, true, Fields::kAll),
                                      scene_cloud(1, true, Fields::kAll)};
  for (const Memory memory : {Memory::kDevice, Memory::kPinned}) {
    Outputs out;
    const std::string output =
        capture_output([&] { out = run_op<CropBoxOp>(crop_config(), clouds, memory); });
    VERIFY_MSG(out.ok, out.failure);
    const std::size_t logged = count_of(output, "copied to the GPU");
    const std::size_t expected = memory == Memory::kDevice ? 0U : 1U;
    VERIFY_MSG(logged == expected, std::string(memory_name(memory)) + ": the copy was logged " +
                                       std::to_string(logged) + " times:\n" + output);
    if (expected == 0 || logged != 1) continue;
    const std::size_t bytes = std::size_t{kWidth} * kHeight * (12U + 3U + 12U + 4U);
    for (const std::string& part :
         {std::string("CropBoxOp"), std::string("points, colors, normals and intensity"),
          std::to_string(bytes) + " bytes", std::string("device memory")}) {
      VERIFY_MSG(output.find(part) != std::string::npos, "the copy log names '" + part + "'");
    }
  }
  holoscan::set_log_level(level);
}

// CloudFilterOp's emit error, as the operators return it.
class EmitErrorProbe final : public CloudFilterOp {
 public:
  EmitErrorProbe() : CloudFilterOp("ProbeOp", CloudSettings{}) {}
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& /*context*/) override {
    return {};
  }
  using CloudFilterOp::emit_failure;
};

// A failed emit names the operator and the port and keeps its code; one a
// stop refused goes back unchanged, as the SDK ends the run stopped only then.
void check_emit_errors() {
  const EmitErrorProbe probe;
  const holoscan::Error stopped{holoscan::ErrorCode::kBackpressured,
                                "RUNTIME_PUBLICATION_STOPPED: the run is stopping"};
  const holoscan::Error got = probe.emit_failure("filtered_point_cloud", stopped).value();
  VERIFY_MSG(got.code == stopped.code && got.message == stopped.message,
             "a stop's error comes back unchanged, not as '" + got.message + "'");
  const holoscan::Error full =
      probe.emit_failure("plane", holoscan::Error{holoscan::ErrorCode::kBackpressured, ""}).value();
  VERIFY(full.code == holoscan::ErrorCode::kBackpressured);
  VERIFY_MSG(full.message.starts_with("ProbeOp: emitting plane failed"), full.message);
}

}  // namespace
}  // namespace holoscan::pcl_filters

int main() {
  using holoscan::pcl_filters::test::TestCase;
  namespace pf = holoscan::pcl_filters;
  return holoscan::pcl_filters::test::run({
      TestCase{"CropBoxOp", pf::check_crop_op},
      TestCase{"OutlierRemovalOp", pf::check_outlier_op},
      TestCase{"GroundRemovalOp", pf::check_ground_op},
      TestCase{"VoxelDownsampleOp", pf::check_voxel_op},
      TestCase{"empty and all-invalid clouds", pf::check_empty_and_invalid},
      TestCase{"rejected inputs", pf::check_rejected_inputs},
      TestCase{"rejected settings", pf::check_rejected_configs},
      TestCase{"pageable input copied before compute() returns, growing clouds",
               pf::check_pageable_input_copied_before_release},
      TestCase{"host-to-device copy logged once", pf::check_host_copy_logged_once},
      TestCase{"emit errors", pf::check_emit_errors},
      TestCase{"the README's live graph", pf::check_live_graph},
  });
}
