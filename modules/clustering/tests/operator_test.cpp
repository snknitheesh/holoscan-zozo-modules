// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// EuclideanClusterOp through the SDK's operator test harness, on clouds made
// at test time: organized and unorganized input in host and device memory,
// with and without colors; both layouts and both connectivities, checked
// against the clusterer run directly; oriented boxes; the capture time and
// frame id carried to every output; an empty cloud; and the rejection of
// unusable inputs and settings.

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

#include <holoscan/clustering/euclidean_cluster_op.hpp>
#include <holoscan/clustering/synthetic_scene.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "reference.hpp"
#include "test_support.hpp"

namespace holoscan::clustering {
namespace {

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

// A frame of the synthetic camera, with a color per pixel.
TestCloud scene_cloud(std::uint32_t frame, bool organized) {
  TestCloud cloud;
  std::vector<std::int32_t> groups;
  synthetic_scene_cloud(synthetic_scene(kWidth, kHeight, frame), cloud.xyz, groups);
  cloud.rgb.resize(cloud.xyz.size());
  for (std::size_t i = 0; i < cloud.rgb.size(); ++i) {
    cloud.rgb[i] = static_cast<std::uint8_t>(1U + (i * 37U) % 250U);
  }
  cloud.shape = organized ? std::vector<std::int64_t>{kHeight, kWidth, 3}
                          : std::vector<std::int64_t>{kHeight * kWidth, 3};
  return cloud;
}

std::chrono::nanoseconds capture_time(std::uint32_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

template <typename T, typename Snapshot>
std::vector<T> values(const Snapshot& snapshot) {
  std::vector<T> out(snapshot.bytes().size() / sizeof(T));
  std::memcpy(out.data(), snapshot.bytes().data(), out.size() * sizeof(T));
  return out;
}

// What the clusterer gives directly, in the operator's layout.
struct Expected {
  std::vector<std::int64_t> label_shape;
  std::vector<std::int32_t> labels;
  std::vector<float> summaries;
  std::vector<std::int64_t> points_shape;
  std::vector<float> points;
  std::vector<std::uint8_t> colors;
};

Expected expected(const TestCloud& cloud, const EuclideanClusterConfig& config) {
  const auto n = static_cast<std::uint32_t>(cloud.xyz.size() / 3);
  const std::uint32_t width = cloud.shape.size() == 3 ? static_cast<std::uint32_t>(kWidth) : 0U;
  const std::vector<std::int32_t> labels =
      config.clustering.connectivity == Connectivity::kImage
          ? test::reference_labels(
                test::reference_image_components(cloud.xyz, kWidth, config.clustering.tolerance,
                                                 config.clustering.image_window),
                config.clustering)
          : test::reference_labels(
                test::reference_components(cloud.xyz, config.clustering.tolerance),
                config.clustering);
  Expected e;
  std::int32_t k = 0;
  for (const std::int32_t label : labels) {
    k = std::max(k, label + 1);
  }
  if (config.layout == OutputLayout::kKeep) {
    e.label_shape.assign(cloud.shape.begin(), cloud.shape.end() - 1);
    e.labels = labels;
    e.points_shape = cloud.shape;
    e.points.resize(cloud.xyz.size());
    e.colors.resize(cloud.rgb.size());
    for (std::size_t i = 0; i < n; ++i) {
      for (int c = 0; c < 3; ++c) {
        e.points[3 * i + c] =
            labels[i] >= 0 ? cloud.xyz[3 * i + c] : std::numeric_limits<float>::quiet_NaN();
        e.colors[3 * i + c] = labels[i] >= 0 ? cloud.rgb[3 * i + c] : std::uint8_t{0};
      }
    }
  } else {
    for (std::int32_t cluster = 0; cluster < k; ++cluster) {
      for (std::size_t i = 0; i < n; ++i) {
        if (labels[i] == cluster) {
          e.labels.push_back(cluster);
          e.points.insert(e.points.end(), cloud.xyz.begin() + 3 * i, cloud.xyz.begin() + 3 * i + 3);
          e.colors.insert(e.colors.end(), cloud.rgb.begin() + 3 * i, cloud.rgb.begin() + 3 * i + 3);
        }
      }
    }
    e.label_shape = {static_cast<std::int64_t>(e.labels.size())};
    e.points_shape = {static_cast<std::int64_t>(e.labels.size()), 3};
  }
  // Summaries from the clusterer itself, already checked against the CPU.
  EuclideanClusterer clusterer;
  float* xyz = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&xyz), cloud.xyz.size() * 4 + 4);
  cudaMemcpy(xyz, cloud.xyz.data(), cloud.xyz.size() * 4, cudaMemcpyHostToDevice);
  std::uint32_t clusters = 0;
  if (clusterer.cluster(xyz, n, width, config.clustering, clusters, nullptr) != cudaSuccess) {
    throw std::runtime_error("cluster failed");
  }
  float* summaries = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&summaries), (clusters + 1U) * sizeof(ClusterSummary));
  clusterer.write_summaries(xyz, config.oriented_boxes, summaries, nullptr);
  e.summaries.resize(std::size_t{clusters} * kSummaryFields);
  cudaMemcpy(e.summaries.data(), summaries, e.summaries.size() * 4, cudaMemcpyDeviceToHost);
  cudaFree(xyz);
  cudaFree(summaries);
  return e;
}

bool same_floats(const std::vector<float>& a, const std::vector<float>& b) {
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

using Snapshots = std::vector<holoscan::testing::Snapshot<holoscan::Tensor>>;

struct Outputs {
  bool ok{};
  std::string failure;
  Snapshots labels;
  Snapshots clusters;
  Snapshots points;
  Snapshots colors;
};

// Pushes the clouds (with colors when configured), runs the operator and
// returns the captured outputs.
Outputs run_op(const EuclideanClusterConfig& config, const std::vector<TestCloud>& clouds,
               bool device) {
  auto h = holoscan::testing::create_operator_test<EuclideanClusterOp>(config);
  std::vector<holoscan::testing::TestSample<holoscan::Tensor>> samples;
  for (std::uint32_t frame = 0; frame < clouds.size(); ++frame) {
    const TestCloud& cloud = clouds[frame];
    holoscan::Tensor points =
        make(cloud.shape, kFloat32, cloud.xyz.data(), cloud.xyz.size() * 4, device);
    if (config.color) {
      h.push_frame(holoscan::testing::frame_at(
          capture_time(frame), holoscan::testing::input("points", std::move(points)),
          holoscan::testing::input(
              "colors", make(cloud.shape, kUInt8, cloud.rgb.data(), cloud.rgb.size(), device))));
    } else {
      holoscan::testing::TestSample<holoscan::Tensor> sample =
          holoscan::testing::sample_at(std::move(points), capture_time(frame));
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
  std::vector<std::string> ports{"labels", "clusters"};
  if (config.emit_points) {
    ports.emplace_back("clustered_points");
  }
  if (config.color) {
    ports.emplace_back("clustered_colors");
  }
  for (const std::string& port : ports) {
    h.output_device(port, holoscan::DeviceId{0});
    h.template capture_snapshot<holoscan::Tensor>(port, kCaptures);
  }
  const auto result = h.run();
  Outputs out;
  out.ok = result.ok();
  if (!out.ok) {
    out.failure = result.describe_failure();
    return out;
  }
  out.labels = result.template snapshots<holoscan::Tensor>("labels");
  out.clusters = result.template snapshots<holoscan::Tensor>("clusters");
  if (config.emit_points) {
    out.points = result.template snapshots<holoscan::Tensor>("clustered_points");
  }
  if (config.color) {
    out.colors = result.template snapshots<holoscan::Tensor>("clustered_colors");
  }
  return out;
}

void check_frames(const std::string& name, const EuclideanClusterConfig& config,
                  const std::vector<TestCloud>& clouds, bool device) {
  const Outputs out = run_op(config, clouds, device);
  VERIFY_MSG(out.ok, name + ": " + out.failure);
  if (!out.ok) return;
  VERIFY_MSG(out.labels.size() == clouds.size() && out.clusters.size() == clouds.size(),
             name + ": one output per cloud");
  if (config.emit_points) VERIFY_MSG(out.points.size() == clouds.size(), name + ": points");
  if (config.color) VERIFY_MSG(out.colors.size() == clouds.size(), name + ": colors");
  for (std::size_t frame = 0; frame < out.labels.size() && frame < clouds.size(); ++frame) {
    const Expected want = expected(clouds[frame], config);
    const std::string at = name + " frame " + std::to_string(frame);
    VERIFY_MSG(out.labels[frame].shape() == want.label_shape, at + ": label shape");
    VERIFY_MSG(values<std::int32_t>(out.labels[frame]) == want.labels, at + ": labels");
    const std::vector<std::int64_t> summary_shape{
        static_cast<std::int64_t>(want.summaries.size() / kSummaryFields), kSummaryFields};
    VERIFY_MSG(out.clusters[frame].shape() == summary_shape, at + ": summary shape");
    VERIFY_MSG(same_floats(values<float>(out.clusters[frame]), want.summaries), at + ": summaries");
    if (config.emit_points && frame < out.points.size()) {
      VERIFY_MSG(out.points[frame].shape() == want.points_shape, at + ": points shape");
      VERIFY_MSG(same_floats(values<float>(out.points[frame]), want.points), at + ": points");
    }
    if (config.color && frame < out.colors.size()) {
      VERIFY_MSG(out.colors[frame].shape() == want.points_shape, at + ": colors shape");
      VERIFY_MSG(values<std::uint8_t>(out.colors[frame]) == want.colors, at + ": colors");
    }
    // Every output carries the cloud's capture time, and its frame id.
    for (const Snapshots* port : {&out.labels, &out.clusters, &out.points, &out.colors}) {
      if (frame < port->size()) {
        const holoscan::SampleMetadata& metadata = (*port)[frame].sample_metadata();
        VERIFY_MSG(metadata.capture_timestamp_ns == capture_time(frame).count(),
                   at + ": capture time");
        if (!config.color) {
          VERIFY_MSG(metadata.frame_id == 7U + frame, at + ": frame id");
        }
      }
    }
  }
}

void check_layouts() {
  for (const bool organized : {true, false}) {
    for (const bool device : {true, false}) {
      for (const OutputLayout layout : {OutputLayout::kKeep, OutputLayout::kDense}) {
        for (const int outputs : {0, 1, 2}) {
          EuclideanClusterConfig config;
          config.clustering = {.tolerance = 0.03F, .min_cluster_size = 5};
          config.layout = layout;
          config.emit_points = outputs >= 1;
          config.color = outputs == 2;
          config.oriented_boxes = outputs == 1;
          config.max_points = kWidth * kHeight;
          const std::string name = std::string(organized ? "organized" : "unorganized") +
                                   (device ? " device" : " host") +
                                   (layout == OutputLayout::kKeep ? " keep" : " dense") +
                                   (outputs == 0   ? ""
                                    : outputs == 1 ? " points"
                                                   : " points colors");
          check_frames(name, config, {scene_cloud(0, organized), scene_cloud(9, organized)},
                       device);
        }
      }
    }
  }
}

void check_image_connectivity() {
  for (const OutputLayout layout : {OutputLayout::kKeep, OutputLayout::kDense}) {
    EuclideanClusterConfig config;
    config.clustering = {.tolerance = 0.03F,
                         .min_cluster_size = 5,
                         .connectivity = Connectivity::kImage,
                         .image_window = 2};
    config.layout = layout;
    config.emit_points = true;
    config.color = true;
    config.max_points = kWidth * kHeight;
    check_frames(layout == OutputLayout::kKeep ? "image keep" : "image dense", config,
                 {scene_cloud(2, true), scene_cloud(3, true)}, true);
  }
  // An unorganized cloud cannot use it.
  EuclideanClusterConfig config;
  config.clustering.connectivity = Connectivity::kImage;
  config.max_points = kWidth * kHeight;
  const Outputs out = run_op(config, {scene_cloud(0, false)}, true);
  VERIFY_MSG(!out.ok && out.failure.find("organized") != std::string::npos,
             "image connectivity refuses [N, 3]: " + out.failure);
}

void check_empty_and_limits() {
  EuclideanClusterConfig config;
  config.emit_points = true;
  config.layout = OutputLayout::kDense;
  config.max_points = kWidth * kHeight;
  TestCloud empty;
  empty.shape = {0, 3};
  const Outputs out = run_op(config, {empty}, true);
  VERIFY_MSG(out.ok, "an empty cloud: " + out.failure);
  if (out.ok) {
    VERIFY(out.labels.size() == 1 && out.labels[0].shape() == std::vector<std::int64_t>{0});
    VERIFY(out.clusters.size() == 1 &&
           out.clusters[0].shape() == (std::vector<std::int64_t>{0, kSummaryFields}));
    VERIFY(out.points.size() == 1 && out.points[0].shape() == (std::vector<std::int64_t>{0, 3}));
  }
  // A cloud above max_points fails the frame.
  config.max_points = 100;
  const Outputs big = run_op(config, {scene_cloud(0, false)}, false);
  VERIFY_MSG(!big.ok && big.failure.find("exceed") != std::string::npos,
             "a cloud above max_points: " + big.failure);
}

void check_config() {
  const auto rejects = [](const EuclideanClusterConfig& config) {
    return test::throws<std::invalid_argument>([&] { validate_config(config); }) &&
           test::throws<std::invalid_argument>([&] { EuclideanClusterOp op(config); });
  };
  VERIFY(!rejects(EuclideanClusterConfig{}));
  EuclideanClusterConfig c;
  c.clustering.tolerance = 0.0F;
  VERIFY(rejects(c));
  c = {};
  c.clustering.tolerance = std::numeric_limits<float>::infinity();
  VERIFY(rejects(c));
  c = {};
  c.clustering.min_cluster_size = 0;
  VERIFY(rejects(c));
  c = {};
  c.clustering.min_cluster_size = 10;
  c.clustering.max_cluster_size = 9;
  VERIFY(rejects(c));
  c = {};
  c.clustering.max_clusters = 0;
  VERIFY(rejects(c));
  c = {};
  c.clustering.max_clusters = kMaxClustersLimit + 1;
  VERIFY(rejects(c));
  c = {};
  c.clustering.connectivity = Connectivity::kImage;
  c.clustering.image_window = kMaxImageWindow + 1;
  VERIFY(rejects(c));
  c = {};
  c.clustering.connectivity = static_cast<Connectivity>(7);
  VERIFY(rejects(c));
  c = {};
  c.color = true;
  VERIFY(rejects(c));
  c = {};
  c.max_points = 0;
  VERIFY(rejects(c));
  c = {};
  c.max_points = kMaxPointsLimit + 1;
  VERIFY(rejects(c));
  c = {};
  c.layout = static_cast<OutputLayout>(5);
  VERIFY(rejects(c));
}

}  // namespace
}  // namespace holoscan::clustering

int main() {
  namespace c = holoscan::clustering;
  return c::test::run({
      {"layouts, memory, outputs and colors", c::check_layouts},
      {"image connectivity", c::check_image_connectivity},
      {"empty cloud and max_points", c::check_empty_and_limits},
      {"configuration", c::check_config},
  });
}
