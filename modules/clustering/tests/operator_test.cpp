// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// EuclideanClusterOp through the SDK's operator test harness, on PointCloud
// messages made at test time: organized and unorganized clouds in host and
// device memory, with and without colors, normals and intensities, carried or
// not; both layouts and both connectivities, checked against the CPU
// reference and the clusterer run directly; oriented boxes; the capture time,
// frame id and header carried to every output; an empty cloud; and the
// rejection of unusable inputs and settings. Through a graph on one worker:
// clouds from a producer's host, pinned host and device pools, the pinned
// ones held until the operator's copy of them is done.

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
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/clustering/euclidean_cluster_op.hpp>
#include <holoscan/clustering/synthetic_scene.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "reference.hpp"
#include "tensor_support.hpp"
#include "test_support.hpp"

namespace holoscan::clustering {
namespace {

namespace hm = holoscan::messages;

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr int kWidth = 96;
constexpr int kHeight = 72;
constexpr const char* kFrame = "sensor_frame";
// Outputs captured per port. The capture's queue holds this many, and the
// runtime sizes the output's pool for them, each of max_points.
constexpr std::size_t kCaptures = 4;

using test::make;

struct TestCloud {
  std::vector<float> xyz;
  std::vector<std::uint8_t> rgb;
  std::vector<float> normals;
  std::vector<float> intensity;
  std::vector<std::int64_t> shape;  // [H, W, 3] or [N, 3].
};

// A frame of the synthetic camera, with a color, a normal and an intensity
// per pixel.
TestCloud scene_cloud(std::uint32_t frame, bool organized) {
  TestCloud cloud;
  std::vector<std::int32_t> groups;
  synthetic_scene_cloud(synthetic_scene(kWidth, kHeight, frame), cloud.xyz, groups);
  const std::size_t n = cloud.xyz.size() / 3;
  cloud.rgb.resize(3 * n);
  cloud.normals.resize(3 * n);
  cloud.intensity.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t c = 0; c < 3; ++c) {
      cloud.rgb[3 * i + c] = static_cast<std::uint8_t>(1U + ((3 * i + c) * 37U) % 250U);
      cloud.normals[3 * i + c] = static_cast<float>((i + c) % 7) / 7.0F - 0.5F;
    }
    cloud.intensity[i] = static_cast<float>(i % 1000) * 0.25F;
  }
  cloud.shape = organized ? std::vector<std::int64_t>{kHeight, kWidth, 3}
                          : std::vector<std::int64_t>{kHeight * kWidth, 3};
  return cloud;
}

// The cloud as a message, in host or device memory, carrying the optional
// fields `present` names (the others the empty [0] of an absent field).
hm::schema::PointCloudT cloud_message(const TestCloud& cloud, bool device,
                                      const hm::PointCloudContents& present) {
  const std::vector<std::int64_t> leading(cloud.shape.begin(), cloud.shape.end() - 1);
  const auto field = [&](bool has, const std::vector<std::int64_t>& shape, DLDataType dtype,
                         const void* data, std::size_t bytes) {
    return std::make_shared<holoscan::Tensor>(has ? make(shape, dtype, data, bytes, device)
                                                  : make({0}, dtype, nullptr, 0, device));
  };
  hm::schema::PointCloudT message;
  message.header = std::make_shared<holoscan::schema::HeaderT>();
  message.header->frame_id = kFrame;
  message.organized = cloud.shape.size() == 3;
  message.valid_count = -1;
  message.points = field(true, cloud.shape, kFloat32, cloud.xyz.data(), cloud.xyz.size() * 4);
  message.colors = field(present.colors, cloud.shape, kUInt8, cloud.rgb.data(), cloud.rgb.size());
  message.normals =
      field(present.normals, cloud.shape, kFloat32, cloud.normals.data(), cloud.normals.size() * 4);
  message.intensity = field(present.intensity, leading, kFloat32, cloud.intensity.data(),
                            cloud.intensity.size() * 4);
  return message;
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

using CloudSnapshot = holoscan::testing::Snapshot<hm::schema::PointCloudT>;

// The bytes of one field of a captured cloud.
template <typename T>
std::vector<T> field(const CloudSnapshot& cloud,
                     holoscan::TensorFieldSelector<hm::schema::PointCloudT> selector) {
  const auto& tensor = cloud.tensor(selector);
  std::vector<T> out(tensor.bytes().size() / sizeof(T));
  std::memcpy(out.data(), tensor.bytes().data(), out.size() * sizeof(T));
  return out;
}

// The CPU reference's labels for `xyz` (of an organized cloud of kWidth
// columns, for kImage).
std::vector<std::int32_t> reference(const std::vector<float>& xyz, const ClusterParams& params) {
  return params.connectivity == Connectivity::kImage
             ? test::reference_labels(test::reference_image_components(
                                          xyz, kWidth, params.tolerance, params.image_window),
                                      params)
             : test::reference_labels(test::reference_components(xyz, params.tolerance), params);
}

// What the clusterer gives directly, in the operator's layout.
struct Expected {
  std::vector<std::int64_t> label_shape;
  std::vector<std::int32_t> labels;
  std::vector<float> summaries;
  std::vector<std::int64_t> points_shape;
  std::vector<std::int64_t> intensity_shape;
  std::vector<float> points;
  std::vector<std::uint8_t> colors;
  std::vector<float> normals;
  std::vector<float> intensity;
};

Expected expected(const TestCloud& cloud, const EuclideanClusterConfig& config) {
  const auto n = static_cast<std::uint32_t>(cloud.xyz.size() / 3);
  const std::uint32_t width = cloud.shape.size() == 3 ? static_cast<std::uint32_t>(kWidth) : 0U;
  const std::vector<std::int32_t> labels = reference(cloud.xyz, config.clustering);
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  Expected e;
  std::int32_t k = 0;
  for (const std::int32_t label : labels) {
    k = std::max(k, label + 1);
  }
  if (config.layout == OutputLayout::kKeep) {
    e.label_shape.assign(cloud.shape.begin(), cloud.shape.end() - 1);
    e.labels = labels;
    e.points_shape = cloud.shape;
    e.intensity_shape = e.label_shape;
    for (std::size_t i = 0; i < n; ++i) {
      const bool kept = labels[i] >= 0;
      for (int c = 0; c < 3; ++c) {
        e.points.push_back(kept ? cloud.xyz[3 * i + c] : kNaN);
        e.colors.push_back(kept ? cloud.rgb[3 * i + c] : std::uint8_t{0});
        e.normals.push_back(kept ? cloud.normals[3 * i + c] : kNaN);
      }
      e.intensity.push_back(kept ? cloud.intensity[i] : kNaN);
    }
  } else {
    for (std::int32_t cluster = 0; cluster < k; ++cluster) {
      for (std::size_t i = 0; i < n; ++i) {
        if (labels[i] == cluster) {
          e.labels.push_back(cluster);
          e.points.insert(e.points.end(), cloud.xyz.begin() + 3 * i, cloud.xyz.begin() + 3 * i + 3);
          e.colors.insert(e.colors.end(), cloud.rgb.begin() + 3 * i, cloud.rgb.begin() + 3 * i + 3);
          e.normals.insert(e.normals.end(), cloud.normals.begin() + 3 * i,
                           cloud.normals.begin() + 3 * i + 3);
          e.intensity.push_back(cloud.intensity[i]);
        }
      }
    }
    e.label_shape = {static_cast<std::int64_t>(e.labels.size())};
    e.points_shape = {static_cast<std::int64_t>(e.labels.size()), 3};
    e.intensity_shape = e.label_shape;
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
  std::vector<CloudSnapshot> clouds;
};

// Pushes the clouds, with the optional fields `present` names, runs the
// operator and returns the captured outputs.
Outputs run_op(const EuclideanClusterConfig& config, const std::vector<TestCloud>& clouds,
               bool device, const hm::PointCloudContents& present = {}) {
  auto h = holoscan::testing::create_operator_test<EuclideanClusterOp>(config);
  std::vector<holoscan::testing::TestSample<hm::schema::PointCloudT>> samples;
  for (std::uint32_t frame = 0; frame < clouds.size(); ++frame) {
    holoscan::testing::TestSample<hm::schema::PointCloudT> sample = holoscan::testing::sample_at(
        cloud_message(clouds[frame], device, present), capture_time(frame));
    sample.metadata.frame_id = 7U + frame;
    samples.push_back(std::move(sample));
  }
  h.push_samples("point_cloud", std::move(samples));
  if (device) {
    h.input_source_device("point_cloud", holoscan::DeviceId{0});
  }
  for (const char* port : {"labels", "clusters"}) {
    h.output_device(port, holoscan::DeviceId{0});
    h.template capture_snapshot<holoscan::Tensor>(port, kCaptures);
  }
  if (config.emit_points) {
    h.output_device("clustered_point_cloud", holoscan::DeviceId{0});
    h.template capture_snapshot<hm::schema::PointCloudT>("clustered_point_cloud", kCaptures);
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
    out.clouds = result.template snapshots<hm::schema::PointCloudT>("clustered_point_cloud");
  }
  return out;
}

void check_frames(const std::string& name, const EuclideanClusterConfig& config,
                  const std::vector<TestCloud>& clouds, bool device,
                  const hm::PointCloudContents& present = {}) {
  const Outputs out = run_op(config, clouds, device, present);
  VERIFY_MSG(out.ok, name + ": " + out.failure);
  if (!out.ok) return;
  VERIFY_MSG(out.labels.size() == clouds.size() && out.clusters.size() == clouds.size(),
             name + ": one output per cloud");
  if (config.emit_points) VERIFY_MSG(out.clouds.size() == clouds.size(), name + ": clouds");
  // The fields the clustered cloud carries: those configured that the input has.
  const bool colors = config.fields.colors && present.colors;
  const bool normals = config.fields.normals && present.normals;
  const bool intensity = config.fields.intensity && present.intensity;
  const bool dense = config.layout == OutputLayout::kDense;
  for (std::size_t frame = 0; frame < out.labels.size() && frame < clouds.size(); ++frame) {
    const Expected want = expected(clouds[frame], config);
    const std::string at = name + " frame " + std::to_string(frame);
    VERIFY_MSG(out.labels[frame].shape() == want.label_shape, at + ": label shape");
    VERIFY_MSG(values<std::int32_t>(out.labels[frame]) == want.labels, at + ": labels");
    const std::vector<std::int64_t> summary_shape{
        static_cast<std::int64_t>(want.summaries.size() / kSummaryFields), kSummaryFields};
    VERIFY_MSG(out.clusters[frame].shape() == summary_shape, at + ": summary shape");
    VERIFY_MSG(same_floats(values<float>(out.clusters[frame]), want.summaries), at + ": summaries");
    if (config.emit_points && frame < out.clouds.size()) {
      const CloudSnapshot& cloud = out.clouds[frame];
      const hm::schema::PointCloudT metadata = cloud.metadata();
      const std::vector<std::int64_t> absent{0};
      VERIFY_MSG(metadata.organized == (!dense && clouds[frame].shape.size() == 3),
                 at + ": organized");
      VERIFY_MSG(metadata.valid_count ==
                     (dense ? static_cast<std::int64_t>(want.labels.size()) : std::int64_t{-1}),
                 at + ": valid_count");
      VERIFY_MSG(metadata.header && metadata.header->frame_id == kFrame, at + ": header frame");
      VERIFY_MSG(cloud.tensor(hm::PointCloudFields::points).shape() == want.points_shape,
                 at + ": points shape");
      VERIFY_MSG(same_floats(field<float>(cloud, hm::PointCloudFields::points), want.points),
                 at + ": points");
      VERIFY_MSG(cloud.tensor(hm::PointCloudFields::colors).shape() ==
                     (colors ? want.points_shape : absent),
                 at + ": colors shape");
      if (colors) {
        VERIFY_MSG(field<std::uint8_t>(cloud, hm::PointCloudFields::colors) == want.colors,
                   at + ": colors");
      }
      VERIFY_MSG(cloud.tensor(hm::PointCloudFields::normals).shape() ==
                     (normals ? want.points_shape : absent),
                 at + ": normals shape");
      if (normals) {
        VERIFY_MSG(same_floats(field<float>(cloud, hm::PointCloudFields::normals), want.normals),
                   at + ": normals");
      }
      VERIFY_MSG(cloud.tensor(hm::PointCloudFields::intensity).shape() ==
                     (intensity ? want.intensity_shape : absent),
                 at + ": intensity shape");
      if (intensity) {
        VERIFY_MSG(
            same_floats(field<float>(cloud, hm::PointCloudFields::intensity), want.intensity),
            at + ": intensity");
      }
    }
    // Every output carries the cloud's capture time and its frame id.
    std::vector<const holoscan::SampleMetadata*> metadata{&out.labels[frame].sample_metadata(),
                                                          &out.clusters[frame].sample_metadata()};
    if (frame < out.clouds.size()) {
      metadata.push_back(&out.clouds[frame].sample_metadata());
    }
    for (const holoscan::SampleMetadata* m : metadata) {
      VERIFY_MSG(m->capture_timestamp_ns == capture_time(frame).count(), at + ": capture time");
      VERIFY_MSG(m->frame_id == 7U + frame, at + ": frame id");
    }
  }
}

void check_layouts() {
  const hm::PointCloudContents all{.colors = true, .normals = true, .intensity = true};
  for (const bool organized : {true, false}) {
    for (const bool device : {true, false}) {
      for (const OutputLayout layout : {OutputLayout::kKeep, OutputLayout::kDense}) {
        for (const int outputs : {0, 1, 2}) {
          EuclideanClusterConfig config;
          config.clustering = {.tolerance = 0.03F, .min_cluster_size = 5};
          config.layout = layout;
          config.emit_points = outputs >= 1;
          if (outputs == 2) {
            config.fields = all;
          }
          config.oriented_boxes = outputs == 1;
          config.max_points = kWidth * kHeight;
          const std::string name = std::string(organized ? "organized" : "unorganized") +
                                   (device ? " device" : " host") +
                                   (layout == OutputLayout::kKeep ? " keep" : " dense") +
                                   (outputs == 0   ? ""
                                    : outputs == 1 ? " points"
                                                   : " points and fields");
          check_frames(name, config, {scene_cloud(0, organized), scene_cloud(9, organized)}, device,
                       all);
        }
      }
    }
  }
}

// The clustered cloud carries a field only when it is configured and the
// input has it.
void check_fields() {
  const std::vector<std::pair<hm::PointCloudContents, hm::PointCloudContents>> cases{
      {{.colors = true}, {}},
      {{}, {.colors = true, .normals = true, .intensity = true}},
      {{.colors = true, .intensity = true}, {.colors = true, .normals = true}},
      {{.normals = true}, {.normals = true}},
      {{.intensity = true}, {.intensity = true}},
  };
  for (const OutputLayout layout : {OutputLayout::kKeep, OutputLayout::kDense}) {
    for (const auto& [configured, present] : cases) {
      EuclideanClusterConfig config;
      config.clustering = {.tolerance = 0.03F, .min_cluster_size = 5};
      config.layout = layout;
      config.emit_points = true;
      config.fields = configured;
      config.max_points = kWidth * kHeight;
      const auto names = [](const hm::PointCloudContents& c) {
        return std::string(c.colors ? "c" : "-") + (c.normals ? "n" : "-") +
               (c.intensity ? "i" : "-");
      };
      check_frames(std::string(layout == OutputLayout::kKeep ? "keep" : "dense") +
                       " fields configured " + names(configured) + " present " + names(present),
                   config, {scene_cloud(4, true)}, true, present);
    }
  }
}

void check_image_connectivity() {
  const hm::PointCloudContents all{.colors = true, .normals = true, .intensity = true};
  for (const OutputLayout layout : {OutputLayout::kKeep, OutputLayout::kDense}) {
    EuclideanClusterConfig config;
    config.clustering = {.tolerance = 0.03F,
                         .min_cluster_size = 5,
                         .connectivity = Connectivity::kImage,
                         .image_window = 2};
    config.layout = layout;
    config.emit_points = true;
    config.fields = all;
    config.max_points = kWidth * kHeight;
    check_frames(layout == OutputLayout::kKeep ? "image keep" : "image dense", config,
                 {scene_cloud(2, true), scene_cloud(3, true)}, true, all);
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
    VERIFY(out.clouds.size() == 1 && out.clouds[0].tensor(hm::PointCloudFields::points).shape() ==
                                         (std::vector<std::int64_t>{0, 3}));
  }
  // A cloud whose objects are all below the minimum size: no cluster, and
  // the outputs say so.
  config.clustering.min_cluster_size = kWidth * kHeight;
  const Outputs none = run_op(config, {scene_cloud(0, true)}, true);
  VERIFY_MSG(none.ok, "no cluster: " + none.failure);
  if (none.ok) {
    VERIFY(none.clusters.size() == 1 &&
           none.clusters[0].shape() == (std::vector<std::int64_t>{0, kSummaryFields}));
    VERIFY(none.clouds.size() == 1 && none.clouds[0].tensor(hm::PointCloudFields::points).shape() ==
                                          (std::vector<std::int64_t>{0, 3}));
  }
  // A cloud above max_points fails the frame.
  config.clustering.min_cluster_size = 1;
  config.max_points = 100;
  const Outputs big = run_op(config, {scene_cloud(0, false)}, false);
  VERIFY_MSG(!big.ok && big.failure.find("exceed") != std::string::npos,
             "a cloud above max_points: " + big.failure);
}

// A cloud whose colors do not match its points, as no producer may send it,
// fails the frame with check_point_cloud()'s error.
void check_malformed() {
  hm::schema::PointCloudT message = cloud_message(scene_cloud(0, true), true, {});
  const std::vector<std::uint8_t> rgb(30, 1);
  message.colors = std::make_shared<holoscan::Tensor>(make({10, 3}, kUInt8, rgb.data(), 30, true));
  auto h = holoscan::testing::create_operator_test<EuclideanClusterOp>(
      EuclideanClusterConfig{.max_points = kWidth * kHeight});
  h.push_samples("point_cloud",
                 {holoscan::testing::sample_at(std::move(message), capture_time(0))});
  h.input_source_device("point_cloud", holoscan::DeviceId{0});
  for (const char* port : {"labels", "clusters"}) {
    h.output_device(port, holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::Tensor>(port, kCaptures);
  }
  const auto result = h.run();
  VERIFY_MSG(
      !result.ok() && result.describe_failure().find("EuclideanClusterOp") != std::string::npos,
      "a cloud with colors of another shape is refused: " + result.describe_failure());
}

// ---- Through a graph: clouds from a producer's pools ----------------------------------------

// Sends the organized clouds of the synthetic scene from a runtime pool in
// `memory`, one per tick, then idles. A full queue or pool is backpressure:
// the same cloud goes again on the next tick.
class PoolCloudsSourceOp final : public holoscan::Operator<> {
 public:
  PoolCloudsSourceOp(std::vector<TestCloud> clouds, holoscan::MemoryKind memory)
      : clouds_(std::move(clouds)), memory_(memory) {}
  void setup(holoscan::OperatorSpec& spec) override {
    hm::declare_point_cloud_output(
        spec, point_cloud, "point_cloud",
        {.max_points = kWidth * kHeight, .colors = true, .memory_kind = memory_});
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds(5)});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (next_ >= clouds_.size()) {
      return {};
    }
    const TestCloud& cloud = clouds_[next_];
    const hm::PointCloudShape shape = hm::organized_cloud(kHeight, kWidth);
    auto loan =
        hm::allocate_point_cloud(point_cloud, shape, {.colors = true}, context.cuda_stream());
    if (!loan) {
      return loan.error().code == holoscan::ErrorCode::kResourceExhausted
                 ? holoscan::expected<void, holoscan::Error>{}
                 : holoscan::make_unexpected(std::move(loan).error());
    }
    const auto write = [&](holoscan::TensorFieldSelector<hm::schema::PointCloudT> field,
                           const void* data, std::size_t bytes) {
      if (memory_ == holoscan::MemoryKind::kCudaDevice) {
        return loan->write_and_commit(
            field, context.cuda_stream(),
            [&](const holoscan::TensorOutputWriteGuard& guard)
                -> holoscan::expected<void, holoscan::Error> {
              if (cudaMemcpyAsync(guard.data_as<void>(), data, bytes, cudaMemcpyHostToDevice,
                                  context.cuda_stream().get()) != cudaSuccess) {
                return holoscan::make_unexpected(
                    holoscan::Error{holoscan::ErrorCode::kFailure, "cudaMemcpyAsync failed"});
              }
              return {};
            });
      }
      return loan->write_host_and_commit(field, [&](const holoscan::TensorOutputWriteGuard& guard) {
        std::memcpy(guard.data_as<void>(), data, bytes);
      });
    };
    if (auto written = write(hm::PointCloudFields::points, cloud.xyz.data(), cloud.xyz.size() * 4);
        !written) {
      return written;
    }
    if (auto written = write(hm::PointCloudFields::colors, cloud.rgb.data(), cloud.rgb.size());
        !written) {
      return written;
    }
    auto sent = hm::emit_point_cloud(
        point_cloud, std::move(*loan), {.frame_id = kFrame}, shape, -1,
        holoscan::EmitOptions{.capture_time = context.activation_time(), .frame_id = next_ + 1U});
    if (!sent && sent.error().code == holoscan::ErrorCode::kBackpressured) {
      return {};
    }
    if (!sent) {
      return sent;
    }
    ++next_;
    return {};
  }

  holoscan::Output<hm::schema::PointCloudT> point_cloud;

 private:
  std::vector<TestCloud> clouds_;
  holoscan::MemoryKind memory_;
  std::uint32_t next_{};
};

struct Received {
  std::vector<std::vector<std::int32_t>> labels;
  std::vector<std::vector<std::uint8_t>> colors;
};

// Keeps each frame's labels and clustered colors, and completes the run
// after `expected` frames.
class KeepOp final : public holoscan::Operator<> {
 public:
  KeepOp(std::shared_ptr<Received> received, std::size_t expected)
      : received_(std::move(received)), expected_(expected) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(labels, "labels").queue_depth(2U);
    spec.input(clusters, "clusters").queue_depth(2U);
    spec.input(cloud, "cloud").queue_depth(2U);
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAligned{
        .ports = holoscan::PortList{std::vector<holoscan::PortRef>{labels, clusters, cloud}},
        .window = std::chrono::nanoseconds{0}});
    result.input(labels, holoscan::Exact{});
    result.input(clusters, holoscan::Exact{});
    result.input(cloud, holoscan::Exact{});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto label_sample = labels.receive();
    auto cluster_sample = clusters.receive();
    auto cloud_sample = cloud.receive();
    if (!label_sample || !cluster_sample || !cloud_sample) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "KeepOp: an input did not arrive"});
    }
    std::vector<std::int32_t> ids(static_cast<std::size_t>(label_sample->data.size()));
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(cloud_sample->data.colors->size()));
    if (!ids.empty()) {
      if (auto copied =
              label_sample->data.copy_to_host(ids.data(), ids.size() * 4U, context.cuda_stream());
          !copied) {
        return copied;
      }
    }
    if (!rgb.empty()) {
      if (auto copied = cloud_sample->data.colors->copy_to_host(rgb.data(), rgb.size(),
                                                                context.cuda_stream());
          !copied) {
        return copied;
      }
    }
    received_->labels.push_back(std::move(ids));
    received_->colors.push_back(std::move(rgb));
    if (received_->labels.size() == expected_) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> labels;
  holoscan::Input<holoscan::Tensor> clusters;
  holoscan::Input<hm::schema::PointCloudT> cloud;

 private:
  std::shared_ptr<Received> received_;
  std::size_t expected_{};
};

void check_pools() {
  for (const holoscan::MemoryKind memory :
       {holoscan::MemoryKind::kHost, holoscan::MemoryKind::kPinnedHost,
        holoscan::MemoryKind::kCudaDevice}) {
    const std::string name = memory == holoscan::MemoryKind::kHost         ? "host pool"
                             : memory == holoscan::MemoryKind::kPinnedHost ? "pinned host pool"
                                                                           : "device pool";
    const std::vector<TestCloud> clouds{scene_cloud(0, true), scene_cloud(5, true),
                                        scene_cloud(9, true)};
    EuclideanClusterConfig config;
    config.clustering = {.tolerance = 0.03F, .min_cluster_size = 5};
    config.emit_points = true;
    config.fields = {.colors = true};
    config.max_points = kWidth * kHeight;
    holoscan::Graph graph{"clustering_pools"};
    auto source = graph.op<PoolCloudsSourceOp>("source", clouds, memory);
    auto cluster = graph.op<EuclideanClusterOp>("cluster", config);
    auto received = std::make_shared<Received>();
    auto keep = graph.op<KeepOp>("keep", received, clouds.size());
    const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
    graph.add_flow(source->point_cloud, cluster->point_cloud, lossless);
    graph.add_flow(cluster->labels, keep->labels, lossless);
    graph.add_flow(cluster->clusters, keep->clusters, lossless);
    graph.add_flow(cluster->clustered_point_cloud, keep->cloud, lossless);
    // Lossless: one worker, so that the source never runs ahead of the chain.
    graph.partition("clustering_pools").include_all();
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
    holoscan::CompileOptions options;
    for (const char* port : {"labels", "clusters", "clustered_point_cloud"}) {
      options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
          .operator_path = "cluster", .output_port = port, .device = holoscan::DeviceId{0}});
    }
    if (memory == holoscan::MemoryKind::kCudaDevice) {
      options.deployment.bind_tensor_output_device(
          holoscan::TensorOutputDevicePlacement{.operator_path = "source",
                                                .output_port = "point_cloud",
                                                .device = holoscan::DeviceId{0}});
    }
    const auto plan = holoscan::compile(graph, std::move(options));
    VERIFY_MSG(plan.ok(),
               name + ": " + holoscan::format_diagnostics(plan.diagnostics(), plan.graph()));
    if (!plan.ok()) continue;
    VERIFY_MSG(plan.scheduler_plan().worker_count == 1U, name + ": one worker");
    auto session = holoscan::run_async(plan);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (session.is_running()) {
      session.request_stop();
    }
    session.wait();
    VERIFY_MSG(session.termination() == holoscan::RunTermination::kRequestedCompleted,
               name + ": the run completes: " +
                   holoscan::format_diagnostics(session.diagnostics(), plan.graph()));
    VERIFY_MSG(received->labels.size() == clouds.size(), name + ": every cloud");
    for (std::size_t frame = 0; frame < received->labels.size() && frame < clouds.size(); ++frame) {
      const Expected want = expected(clouds[frame], config);
      VERIFY_MSG(received->labels[frame] == want.labels,
                 name + " frame " + std::to_string(frame) + ": labels");
      VERIFY_MSG(received->colors[frame] == want.colors,
                 name + " frame " + std::to_string(frame) + ": colors");
    }
  }
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
  for (const hm::PointCloudContents fields :
       {hm::PointCloudContents{.colors = true}, hm::PointCloudContents{.normals = true},
        hm::PointCloudContents{.intensity = true}}) {
    c = {};
    c.fields = fields;
    VERIFY(rejects(c));
  }
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
      {"layouts, memory, outputs and fields", c::check_layouts},
      {"the fields carried", c::check_fields},
      {"image connectivity", c::check_image_connectivity},
      {"empty cloud, no cluster and max_points", c::check_empty_and_limits},
      {"a malformed cloud", c::check_malformed},
      {"clouds from host, pinned host and device pools", c::check_pools},
      {"configuration", c::check_config},
  });
}
