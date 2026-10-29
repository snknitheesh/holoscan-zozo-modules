// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The four operators through the SDK's operator test harness, on clouds of the
// synthetic scene made at test time: organized and dense input in host and
// device memory, with and without colors; both output layouts, checked
// against the filters run directly and the CPU references; the plane output;
// the capture time and frame id carried to every output; empty and
// all-invalid clouds; and the rejection of unusable inputs and settings.

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
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/pcl_filters/crop_box_op.hpp>
#include <holoscan/pcl_filters/ground_removal_op.hpp>
#include <holoscan/pcl_filters/outlier_removal_op.hpp>
#include <holoscan/pcl_filters/synthetic_scene.hpp>
#include <holoscan/pcl_filters/voxel_downsample_op.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "reference.hpp"
#include "test_support.hpp"

namespace holoscan::pcl_filters {
namespace {

using std::chrono_literals::operator""ms;

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr int kWidth = 96;
constexpr int kHeight = 72;
// Outputs captured per port. The capture's queue holds this many, and the
// runtime sizes the output's pool for them, each of max_points.
constexpr std::size_t kCaptures = 4;

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

struct TestCloud {
  std::vector<float> xyz;
  std::vector<std::uint8_t> rgb;
  std::vector<std::int64_t> shape;  // [H, W, 3] or [N, 3].
};

TestCloud scene_cloud(std::uint32_t frame, bool organized) {
  TestCloud cloud;
  std::vector<SceneLabel> labels;
  synthetic_scene_cloud(synthetic_scene(kWidth, kHeight, frame), cloud.xyz, cloud.rgb, labels);
  cloud.shape = organized ? std::vector<std::int64_t>{kHeight, kWidth, 3}
                          : std::vector<std::int64_t>{kHeight * kWidth, 3};
  return cloud;
}

holoscan::Tensor points_tensor(const TestCloud& cloud, bool device) {
  return make(cloud.shape, kFloat32, cloud.xyz.data(), cloud.xyz.size() * 4, device);
}

holoscan::Tensor colors_tensor(const TestCloud& cloud, bool device) {
  return make(cloud.shape, kUInt8, cloud.rgb.data(), cloud.rgb.size(), device);
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
std::vector<std::uint8_t> bytes(const Snapshot& snapshot) {
  std::vector<std::uint8_t> values(snapshot.bytes().size());
  std::memcpy(values.data(), snapshot.bytes().data(), values.size());
  return values;
}

// The cloud a keep mask leaves, in a layout.
TestCloud expected(const TestCloud& in, const std::vector<std::uint8_t>& keep,
                   OutputLayout layout) {
  TestCloud out;
  const std::size_t n = keep.size();
  if (layout == OutputLayout::kKeep) {
    out.shape = in.shape;
    out.xyz.assign(3 * n, std::numeric_limits<float>::quiet_NaN());
    out.rgb.assign(3 * n, 0U);
  }
  std::int64_t kept = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (keep[i] == 0U) {
      continue;
    }
    const std::size_t at = layout == OutputLayout::kKeep ? i : static_cast<std::size_t>(kept);
    if (layout == OutputLayout::kDense) {
      out.xyz.resize(3 * (at + 1));
      out.rgb.resize(3 * (at + 1));
    }
    std::copy_n(in.xyz.begin() + 3 * i, 3, out.xyz.begin() + 3 * at);
    std::copy_n(in.rgb.begin() + 3 * i, 3, out.rgb.begin() + 3 * at);
    ++kept;
  }
  if (layout == OutputLayout::kDense) {
    out.shape = {kept, 3};
  }
  return out;
}

// Bitwise equality, any NaN equal to any NaN.
bool same_points(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!(std::isnan(a[i]) && std::isnan(b[i])) && std::memcmp(&a[i], &b[i], 4) != 0) {
      return false;
    }
  }
  return true;
}

// Pushes `frames` clouds (with colors when `color`), runs the operator and
// returns the captured outputs; `plane` adds the plane output.
struct Outputs {
  bool ok{};
  std::string failure;
  std::vector<holoscan::testing::Snapshot<holoscan::Tensor>> points;
  std::vector<holoscan::testing::Snapshot<holoscan::Tensor>> colors;
  std::vector<holoscan::testing::Snapshot<holoscan::Tensor>> planes;
};

template <typename Op, typename Config>
Outputs run_op(const Config& config, const std::vector<TestCloud>& clouds, bool device,
               bool plane = false) {
  auto h = holoscan::testing::create_operator_test<Op>(config);
  std::vector<holoscan::testing::TestSample<holoscan::Tensor>> samples;
  for (std::uint32_t frame = 0; frame < clouds.size(); ++frame) {
    if (config.color) {
      h.push_frame(holoscan::testing::frame_at(
          capture_time(frame),
          holoscan::testing::input("points", points_tensor(clouds[frame], device)),
          holoscan::testing::input("colors", colors_tensor(clouds[frame], device))));
    } else {
      holoscan::testing::TestSample<holoscan::Tensor> sample =
          holoscan::testing::sample_at(points_tensor(clouds[frame], device), capture_time(frame));
      sample.metadata.frame_id = 7U + frame;
      samples.push_back(std::move(sample));
    }
  }
  if (!samples.empty()) {
    h.push_samples("points", std::move(samples));
  }
  if (device) {
    h.input_source_device("points", holoscan::DeviceId{0});
    if (config.color) {
      h.input_source_device("colors", holoscan::DeviceId{0});
    }
  }
  h.output_device("filtered_points", holoscan::DeviceId{0});
  h.template capture_snapshot<holoscan::Tensor>("filtered_points", kCaptures);
  if (config.color) {
    h.output_device("filtered_colors", holoscan::DeviceId{0});
    h.template capture_snapshot<holoscan::Tensor>("filtered_colors", kCaptures);
  }
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
  outputs.points = result.template snapshots<holoscan::Tensor>("filtered_points");
  if (config.color) {
    outputs.colors = result.template snapshots<holoscan::Tensor>("filtered_colors");
  }
  if (plane) {
    outputs.planes = result.template snapshots<holoscan::Tensor>("plane");
  }
  return outputs;
}

// Checks every output frame against the cloud `keep` leaves, and its
// capture time (and frame id, sent without colors).
template <typename Config>
void check_outputs(const std::string& name, const Outputs& out, const Config& config,
                   const std::vector<TestCloud>& clouds,
                   const std::vector<std::vector<std::uint8_t>>& keeps) {
  VERIFY_MSG(out.ok, name + ": " + out.failure);
  if (!out.ok) return;
  VERIFY_MSG(out.points.size() == clouds.size(), name + ": one output per cloud");
  if (config.color) {
    VERIFY_MSG(out.colors.size() == clouds.size(), name + ": one color output per cloud");
  }
  for (std::size_t frame = 0; frame < out.points.size() && frame < clouds.size(); ++frame) {
    const TestCloud want = expected(clouds[frame], keeps[frame], config.layout);
    const std::string at = name + " frame " + std::to_string(frame);
    VERIFY_MSG(out.points[frame].shape() == want.shape, at + ": shape");
    VERIFY_MSG(same_points(floats(out.points[frame]), want.xyz), at + ": points");
    const holoscan::SampleMetadata& metadata = out.points[frame].sample_metadata();
    VERIFY_MSG(metadata.capture_timestamp_ns == capture_time(frame).count(), at + ": capture time");
    if (!config.color) {
      VERIFY_MSG(metadata.frame_id == 7U + frame, at + ": frame id");
    } else if (frame < out.colors.size()) {
      VERIFY_MSG(out.colors[frame].shape() == want.shape, at + ": color shape");
      VERIFY_MSG(bytes(out.colors[frame]) == want.rgb, at + ": colors");
      VERIFY_MSG(
          out.colors[frame].sample_metadata().capture_timestamp_ns == capture_time(frame).count(),
          at + ": color capture time");
    }
  }
}

// Every layout, memory and color combination of an operator whose mask
// `reference` gives.
template <typename Op, typename Config, typename Reference>
void check_masking_op(const std::string& name, Config config, Reference&& reference) {
  for (const bool organized : {true, false}) {
    for (const bool device : {true, false}) {
      for (const bool color : {false, true}) {
        for (const OutputLayout layout : {OutputLayout::kKeep, OutputLayout::kDense}) {
          config.color = color;
          config.layout = layout;
          const std::vector<TestCloud> clouds{scene_cloud(0, organized), scene_cloud(1, organized)};
          std::vector<std::vector<std::uint8_t>> keeps;
          for (const TestCloud& cloud : clouds) {
            keeps.push_back(reference(cloud));
          }
          const std::string what = name + (organized ? " organized" : " dense") +
                                   (device ? " device" : " host") + (color ? " color" : "") +
                                   (layout == OutputLayout::kKeep ? " keep" : " packed");
          check_outputs(what, run_op<Op>(config, clouds, device), config, clouds, keeps);
        }
      }
    }
  }
}

// The filters run directly on a host cloud.
template <typename Run>
std::vector<std::uint8_t> device_mask(const TestCloud& cloud, Run&& run) {
  const auto n = static_cast<std::uint32_t>(cloud.xyz.size() / 3);
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
  // cloud's capture time.
  const std::vector<TestCloud> clouds{scene_cloud(0, true), scene_cloud(1, true)};
  const Outputs out = run_op<GroundRemovalOp>(config, clouds, true, true);
  VERIFY_MSG(out.ok, out.failure);
  VERIFY(out.planes.size() == 2);
  for (std::size_t frame = 0; frame < out.planes.size(); ++frame) {
    float want[4];
    segment(clouds[frame], want);
    VERIFY(out.planes[frame].shape() == std::vector<std::int64_t>{4});
    const std::vector<float> got = floats(out.planes[frame]);
    VERIFY_MSG(got.size() == 4 && std::memcmp(got.data(), want, 16) == 0,
               "the plane output is the filter's plane");
    float truth[4];
    synthetic_ground_plane(synthetic_scene(kWidth, kHeight, static_cast<std::uint32_t>(frame)),
                           truth);
    VERIFY(got.size() == 4 && std::abs(got[3] - truth[3]) < 0.01F &&
           got[0] * truth[0] + got[1] * truth[1] + got[2] * truth[2] > 0.9995F);
    VERIFY(out.planes[frame].sample_metadata().capture_timestamp_ns ==
           capture_time(static_cast<std::uint32_t>(frame)).count());
  }
}

void check_voxel_op() {
  for (const bool organized : {true, false}) {
    for (const bool device : {true, false}) {
      for (const bool color : {false, true}) {
        for (const VoxelMode mode : {VoxelMode::kCentroid, VoxelMode::kFirstPoint}) {
          VoxelDownsampleConfig config{.voxel_size = 0.04F, .mode = mode, .color = color};
          const std::vector<TestCloud> clouds{scene_cloud(0, organized), scene_cloud(1, organized)};
          const Outputs out = run_op<VoxelDownsampleOp>(config, clouds, device);
          const std::string name = std::string("voxels") + (organized ? " organized" : " dense") +
                                   (device ? " device" : " host") + (color ? " color" : "") +
                                   (mode == VoxelMode::kCentroid ? " centroid" : " first");
          VERIFY_MSG(out.ok, name + ": " + out.failure);
          if (!out.ok) continue;
          VERIFY(out.points.size() == 2);
          for (std::size_t frame = 0; frame < out.points.size(); ++frame) {
            const test::VoxelResult want =
                test::voxel_reference(clouds[frame].xyz, clouds[frame].rgb, 0.04F, mode);
            VERIFY_MSG(
                out.points[frame].shape() ==
                    (std::vector<std::int64_t>{static_cast<std::int64_t>(want.xyz.size() / 3), 3}),
                name + ": [voxels, 3]");
            VERIFY_MSG(same_points(floats(out.points[frame]), want.xyz), name + ": points");
            if (color && frame < out.colors.size()) {
              VERIFY_MSG(bytes(out.colors[frame]) == want.rgb, name + ": colors");
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
  empty.shape = {0, 3};
  TestCloud invalid = scene_cloud(0, true);
  std::fill(invalid.xyz.begin(), invalid.xyz.end(), std::numeric_limits<float>::quiet_NaN());
  // One run each, as the harness stages one tensor layout per port.
  for (const TestCloud& cloud : {empty, invalid}) {
    const std::string kind = cloud.xyz.empty() ? " empty" : " all-invalid";
    const std::vector<TestCloud> clouds{cloud};
    const std::vector<std::vector<std::uint8_t>> keeps{
        std::vector<std::uint8_t>(cloud.xyz.size() / 3, 0U)};
    for (const OutputLayout layout : {OutputLayout::kKeep, OutputLayout::kDense}) {
      CropBoxConfig crop = crop_config();
      crop.color = true;
      crop.layout = layout;
      check_outputs("crop" + kind, run_op<CropBoxOp>(crop, clouds, true), crop, clouds, keeps);
      OutlierRemovalConfig outliers = outlier_config();
      outliers.layout = layout;
      check_outputs("outliers" + kind, run_op<OutlierRemovalOp>(outliers, clouds, false), outliers,
                    clouds, keeps);
      GroundRemovalConfig ground = ground_config();
      ground.layout = layout;
      const Outputs out = run_op<GroundRemovalOp>(ground, clouds, true, true);
      check_outputs("ground" + kind, out, ground, clouds, keeps);
      for (const auto& plane : out.planes) {
        const std::vector<float> values = floats(plane);
        VERIFY_MSG(values.size() == 4 && std::isnan(values[0]), "no plane" + kind);
      }
    }
    const Outputs voxels =
        run_op<VoxelDownsampleOp>(VoxelDownsampleConfig{.color = true}, clouds, true);
    VERIFY_MSG(voxels.ok, "voxels" + kind + ": " + voxels.failure);
    VERIFY(voxels.points.size() == 1 && voxels.colors.size() == 1);
    for (const auto& snapshot : voxels.points) {
      VERIFY(snapshot.shape() == (std::vector<std::int64_t>{0, 3}));
    }
  }
}

// Runs a crop box on one unusable input and checks the failure names `what`.
void check_rejects(const CropBoxConfig& config, holoscan::Tensor points,
                   std::optional<holoscan::Tensor> colors, const std::string& what) {
  auto h = holoscan::testing::create_operator_test<CropBoxOp>(config);
  if (colors) {
    h.push_frame(holoscan::testing::frame_at(100ms, holoscan::testing::input("points", points),
                                             holoscan::testing::input("colors", *colors)));
  } else {
    h.push("points", points);
  }
  h.output_device("filtered_points", holoscan::DeviceId{0});
  if (config.color) {
    h.output_device("filtered_colors", holoscan::DeviceId{0});
  }
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), what + " is rejected");
  VERIFY_MSG(result.describe_failure().find(what) != std::string::npos,
             "failure names '" + what + "': " + result.describe_failure());
}

void check_rejected_inputs() {
  const std::vector<float> xyz(12, 1.0F);
  const std::vector<double> xyz64(12, 1.0);
  const std::vector<std::uint8_t> rgb(12, 9U);
  const std::string shape = "points must be float32 [N, 3] or [H, W, 3]";
  CropBoxConfig config = crop_config();
  check_rejects(config, make({4, 3}, {kDLFloat, 64, 1}, xyz64.data(), 96, false), {}, shape);
  check_rejects(config, make({3, 4}, kFloat32, xyz.data(), 48, false), {}, shape);
  check_rejects(config, make({12}, kFloat32, xyz.data(), 48, false), {}, shape);
  check_rejects(config, make({1, 2, 2, 3}, kFloat32, xyz.data(), 48, false), {}, shape);
  CropBoxConfig small = config;
  small.max_points = 3;
  check_rejects(small, make({4, 3}, kFloat32, xyz.data(), 48, false), {},
                "exceed the configured maximum of 3");
  CropBoxConfig with_color = config;
  with_color.color = true;
  check_rejects(with_color, make({4, 3}, kFloat32, xyz.data(), 48, false),
                make({2, 2, 3}, kUInt8, rgb.data(), 12, false),
                "colors must be uint8 RGB of the points' shape");
  check_rejects(with_color, make({4, 3}, kFloat32, xyz.data(), 48, false),
                make({4, 3}, kFloat32, xyz.data(), 48, false),
                "colors must be uint8 RGB of the points' shape");
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
  rejects([] { VoxelDownsampleOp op(VoxelDownsampleConfig{.max_points = 0}); },
          "max_points must be 1..");
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
  });
}
