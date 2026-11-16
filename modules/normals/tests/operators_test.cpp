// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The two operators through the SDK's operator test harness, on PointCloud
// messages of the synthetic scene made at test time: organized and
// unorganized clouds in host and device memory, with and without colors and
// intensities, both output layouts, checked bit for bit against
// NormalEstimator and PlaneSegmenter run directly; the header, capture time
// and frame id carried to every output; empty and all-invalid clouds; and the
// rejection of unusable clouds and settings. The harness stages host input in
// pinned host memory; through a graph, clouds from a producer's pageable and
// pinned host pools; and the holding of pinned host input on its own
// (detail::hold_until_done()).

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
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

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/messages/point_cloud.hpp>
#include <holoscan/normals/normal_estimation_op.hpp>
#include <holoscan/normals/plane_segmentation_op.hpp>
#include <holoscan/normals/synthetic_scene.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "host_input.hpp"
#include "test_support.hpp"

namespace holoscan::normals {
namespace {

namespace hm = holoscan::messages;
using std::chrono_literals::operator""ms;
using CloudSnapshot = holoscan::testing::Snapshot<hm::schema::PointCloudT>;

constexpr int kWidth = 96;
constexpr int kHeight = 72;
// Captures per port at most: the harness gives each capture a pool slot per
// tensor, and a cloud has four.
constexpr std::size_t kCaptures = 4;
constexpr const char* kFrame = "scene_optical";

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
  std::vector<float> intensity;
  std::vector<float> normals;
  bool with_normals{};              // Whether the message carries `normals`.
  std::vector<std::int64_t> shape;  // [H, W, 3] or [N, 3].

  [[nodiscard]] std::size_t count() const { return xyz.size() / 3; }
  [[nodiscard]] bool organized() const { return shape.size() == 3; }
  [[nodiscard]] std::vector<std::int64_t> point_shape() const {
    return {shape.begin(), shape.end() - 1};
  }
};

TestCloud scene_cloud(std::uint32_t frame, bool organized) {
  TestCloud cloud;
  std::vector<std::uint8_t> surfaces;
  std::vector<float> truth;
  synthetic_scene_cloud(synthetic_scene(kWidth, kHeight, frame), cloud.xyz, cloud.rgb, surfaces,
                        truth);
  cloud.intensity.resize(cloud.count());
  for (std::size_t i = 0; i < cloud.intensity.size(); ++i) {
    cloud.intensity[i] = 0.5F + static_cast<float>(i % 97U);
  }
  cloud.shape = organized ? std::vector<std::int64_t>{kHeight, kWidth, 3}
                          : std::vector<std::int64_t>{kHeight * kWidth, 3};
  return cloud;
}

// The message of `cloud`, with the fields asked for; the others absent.
hm::schema::PointCloudT message(const TestCloud& cloud, bool colors, bool intensity, bool device) {
  const std::size_t n = cloud.count();
  hm::schema::PointCloudT result;
  result.header = std::make_shared<holoscan::schema::HeaderT>();
  result.header->frame_id = kFrame;
  result.header->device_sequence = 42U;
  result.organized = cloud.organized();
  const std::vector<std::int64_t> absent{0};
  result.points = std::make_shared<holoscan::Tensor>(
      make(cloud.shape, kFloat32, cloud.xyz.data(), n * 12U, device));
  result.colors = std::make_shared<holoscan::Tensor>(
      colors ? make(cloud.shape, kUInt8, cloud.rgb.data(), n * 3U, device)
             : make(absent, kUInt8, nullptr, 0, device));
  result.normals = std::make_shared<holoscan::Tensor>(
      cloud.with_normals ? make(cloud.shape, kFloat32, cloud.normals.data(), n * 12U, device)
                         : make(absent, kFloat32, nullptr, 0, device));
  result.intensity = std::make_shared<holoscan::Tensor>(
      intensity ? make(cloud.point_shape(), kFloat32, cloud.intensity.data(), n * 4U, device)
                : make(absent, kFloat32, nullptr, 0, device));
  return result;
}

std::chrono::nanoseconds capture_time(std::uint32_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

template <typename Bytes>
std::vector<float> floats(const Bytes& bytes) {
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
  return values;
}

template <typename Bytes>
std::vector<std::uint8_t> bytes_of(const Bytes& bytes) {
  std::vector<std::uint8_t> values(bytes.size());
  std::memcpy(values.data(), bytes.data(), values.size());
  return values;
}

// Bitwise equality, any NaN equal to any NaN.
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

// Device scratch for running the kernels directly.
template <typename T>
struct Scratch {
  T* data{};
  explicit Scratch(std::size_t count) {
    if (cudaMalloc(&data, std::max<std::size_t>(count, 1) * sizeof(T)) != cudaSuccess) {
      throw std::runtime_error("cudaMalloc failed");
    }
  }
  ~Scratch() { cudaFree(data); }
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  std::vector<T> host(std::size_t count) const {
    std::vector<T> out(count);
    cudaDeviceSynchronize();
    if (count != 0) {
      cudaMemcpy(out.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost);
    }
    return out;
  }
};

struct DirectNormals {
  std::vector<float> normals;
  std::vector<float> curvature;
};

DirectNormals direct_normals(const TestCloud& cloud, NormalParams params) {
  const std::size_t n = cloud.count();
  params.search = resolve_search(params.search, cloud.organized());
  Scratch<float> xyz(3 * n);
  Scratch<float> normals(3 * n);
  Scratch<float> curvature(n);
  cudaMemcpy(xyz.data, cloud.xyz.data(), 12 * n, cudaMemcpyHostToDevice);
  NormalEstimator estimator;
  if (estimator.estimate(xyz.data, static_cast<std::uint32_t>(n),
                         cloud.organized() ? kWidth : static_cast<int>(n),
                         cloud.organized() ? kHeight : 1, params, normals.data, curvature.data,
                         nullptr) != cudaSuccess) {
    throw std::runtime_error("estimate failed");
  }
  return DirectNormals{normals.host(3 * n), curvature.host(n)};
}

// What NormalEstimationOp should emit for a cloud, in a layout.
struct NormalOutputs {
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> point_shape;
  std::vector<float> xyz;
  std::vector<std::uint8_t> rgb;
  std::vector<float> intensity;
  std::vector<float> normals;
  std::vector<float> curvature;
  std::int64_t kept{};
};

NormalOutputs expected_normals(const TestCloud& cloud, const NormalEstimationConfig& config) {
  const DirectNormals direct = direct_normals(cloud, config.normals);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  NormalOutputs out;
  for (std::size_t i = 0; i < cloud.count(); ++i) {
    const bool has = std::isfinite(direct.normals[3 * i]);
    out.kept += has ? 1 : 0;
    if (config.layout == OutputLayout::kDense && !has) {
      continue;
    }
    for (int c = 0; c < 3; ++c) {
      out.xyz.push_back(has ? cloud.xyz[3 * i + c] : nan);
      out.rgb.push_back(has ? cloud.rgb[3 * i + c] : 0U);
      out.normals.push_back(direct.normals[3 * i + c]);
    }
    out.intensity.push_back(has ? cloud.intensity[i] : nan);
    out.curvature.push_back(direct.curvature[i]);
  }
  if (config.layout == OutputLayout::kKeep) {
    out.shape = cloud.shape;
    out.point_shape = cloud.point_shape();
  } else {
    out.shape = {out.kept, 3};
    out.point_shape = {out.kept};
  }
  return out;
}

void check_normal_op() {
  for (const bool organized : {true, false}) {
    for (const bool device : {true, false}) {
      for (const int fields : {0, 1, 2}) {  // None, colors, colors and intensities.
        for (const OutputLayout layout : {OutputLayout::kKeep, OutputLayout::kDense}) {
          const bool colors = fields >= 1;
          const bool intensity = fields == 2;
          NormalEstimationConfig config;
          config.colors = colors;
          config.intensity = intensity;
          config.layout = layout;
          config.max_points = kWidth * kHeight;
          config.normals.radius = 0.15F;
          const std::vector<TestCloud> clouds{scene_cloud(0, organized), scene_cloud(1, organized)};
          auto h = holoscan::testing::create_operator_test<NormalEstimationOp>(config);
          std::vector<holoscan::testing::TestSample<hm::schema::PointCloudT>> samples;
          for (std::uint32_t f = 0; f < clouds.size(); ++f) {
            auto sample = holoscan::testing::sample_at(
                message(clouds[f], colors, intensity, device), capture_time(f));
            sample.metadata.frame_id = 7U + f;
            samples.push_back(std::move(sample));
          }
          h.push_samples("point_cloud", std::move(samples));
          if (device) {
            h.input_source_device("point_cloud", holoscan::DeviceId{0});
          }
          for (const char* output : {"point_cloud_with_normals", "curvature"}) {
            h.output_device(output, holoscan::DeviceId{0});
          }
          h.capture_snapshot<hm::schema::PointCloudT>("point_cloud_with_normals", kCaptures);
          h.capture_snapshot<holoscan::Tensor>("curvature", kCaptures);
          const std::string name = std::string("normals") + (organized ? " organized" : " dense") +
                                   (device ? " device" : " host") + (colors ? " colors" : "") +
                                   (intensity ? " intensity" : "") +
                                   (layout == OutputLayout::kKeep ? " keep" : " packed");
          const auto result = h.run();
          VERIFY_MSG(result.ok(), name + ": " + result.describe_failure());
          if (!result.ok()) {
            continue;
          }
          const auto out = result.snapshots<hm::schema::PointCloudT>("point_cloud_with_normals");
          const auto curvature = result.snapshots<holoscan::Tensor>("curvature");
          for (std::size_t f = 0; f < clouds.size(); ++f) {
            const std::string at = name + " frame " + std::to_string(f);
            if (out.size() <= f || curvature.size() <= f) {
              VERIFY_MSG(false, at + ": one output per cloud on every port");
              continue;
            }
            const NormalOutputs want = expected_normals(clouds[f], config);
            const CloudSnapshot& cloud = out[f];
            const hm::schema::PointCloudT metadata = cloud.metadata();
            VERIFY_MSG(metadata.organized == (organized && layout == OutputLayout::kKeep),
                       at + ": organized");
            VERIFY_MSG(metadata.valid_count == (layout == OutputLayout::kKeep ? -1 : want.kept),
                       at + ": valid_count");
            VERIFY_MSG(metadata.header && metadata.header->frame_id == kFrame &&
                           metadata.header->device_sequence == 42U,
                       at + ": the cloud's header");
            const auto& points = cloud.tensor(hm::PointCloudFields::points);
            VERIFY_MSG(points.shape() == want.shape, at + ": points shape");
            VERIFY_MSG(same_floats(floats(points.bytes()), want.xyz), at + ": points");
            const auto& normals = cloud.tensor(hm::PointCloudFields::normals);
            VERIFY_MSG(normals.shape() == want.shape, at + ": normals shape");
            VERIFY_MSG(same_floats(floats(normals.bytes()), want.normals), at + ": normals");
            const auto& rgb = cloud.tensor(hm::PointCloudFields::colors);
            VERIFY_MSG(rgb.shape() == (colors ? want.shape : std::vector<std::int64_t>{0}),
                       at + ": colors shape");
            if (colors) {
              VERIFY_MSG(bytes_of(rgb.bytes()) == want.rgb, at + ": colors");
            }
            const auto& values = cloud.tensor(hm::PointCloudFields::intensity);
            VERIFY_MSG(
                values.shape() == (intensity ? want.point_shape : std::vector<std::int64_t>{0}),
                at + ": intensity shape");
            if (intensity) {
              VERIFY_MSG(same_floats(floats(values.bytes()), want.intensity), at + ": intensity");
            }
            VERIFY_MSG(curvature[f].shape() == want.point_shape, at + ": curvature shape");
            VERIFY_MSG(same_floats(floats(curvature[f].bytes()), want.curvature),
                       at + ": curvature");
            for (const holoscan::SampleMetadata& sample :
                 {cloud.sample_metadata(), curvature[f].sample_metadata()}) {
              VERIFY_MSG(sample.capture_timestamp_ns == capture_time(f).count(),
                         at + ": capture time");
              VERIFY_MSG(sample.frame_id == 7U + f, at + ": frame id");
            }
          }
        }
      }
    }
  }
}

void check_plane_op() {
  for (const bool organized : {true, false}) {
    for (const bool device : {true, false}) {
      PlaneSegmentationConfig config;
      config.planes.min_inliers = 100;
      config.planes.max_planes = 6;
      config.max_points = kWidth * kHeight;
      std::vector<TestCloud> clouds{scene_cloud(0, organized), scene_cloud(1, organized)};
      auto h = holoscan::testing::create_operator_test<PlaneSegmentationOp>(config);
      std::vector<holoscan::testing::TestSample<hm::schema::PointCloudT>> samples;
      for (std::uint32_t f = 0; f < clouds.size(); ++f) {
        clouds[f].normals = direct_normals(clouds[f], NormalParams{.radius = 0.15F}).normals;
        clouds[f].with_normals = true;
        auto sample =
            holoscan::testing::sample_at(message(clouds[f], true, false, device), capture_time(f));
        sample.metadata.frame_id = 7U + f;
        samples.push_back(std::move(sample));
      }
      h.push_samples("point_cloud", std::move(samples));
      if (device) {
        h.input_source_device("point_cloud", holoscan::DeviceId{0});
      }
      const std::vector<std::string> outputs{"labels", "plane_coefficients", "plane_centroids",
                                             "plane_inliers"};
      for (const std::string& output : outputs) {
        h.output_device(output, holoscan::DeviceId{0});
        h.capture_snapshot<holoscan::Tensor>(output, kCaptures);
      }
      const std::string name = std::string("planes") + (organized ? " organized" : " dense") +
                               (device ? " device" : " host");
      const auto result = h.run();
      VERIFY_MSG(result.ok(), name + ": " + result.describe_failure());
      if (!result.ok()) {
        continue;
      }
      std::vector<std::vector<holoscan::testing::Snapshot<holoscan::Tensor>>> ports;
      for (const std::string& output : outputs) {
        ports.push_back(result.snapshots<holoscan::Tensor>(output));
      }
      for (std::size_t f = 0; f < clouds.size(); ++f) {
        const std::string at = name + " frame " + std::to_string(f);
        if (std::any_of(ports.begin(), ports.end(),
                        [f](const auto& port) { return port.size() <= f; })) {
          VERIFY_MSG(false, at + ": one output per cloud on every port");
          continue;
        }
        // The segmenter run directly.
        const std::size_t n = clouds[f].count();
        Scratch<float> xyz(3 * n);
        Scratch<float> point_normals(3 * n);
        Scratch<std::uint8_t> labels(n);
        Scratch<float> coefficients(4 * 6);
        Scratch<float> centroids(3 * 6);
        Scratch<std::uint32_t> inliers(6);
        cudaMemcpy(xyz.data, clouds[f].xyz.data(), 12 * n, cudaMemcpyHostToDevice);
        cudaMemcpy(point_normals.data, clouds[f].normals.data(), 12 * n, cudaMemcpyHostToDevice);
        PlaneSegmenter segmenter;
        VERIFY(segmenter.segment(
                   xyz.data, point_normals.data, static_cast<std::uint32_t>(n), config.planes,
                   PlaneOutputs{labels.data, coefficients.data, centroids.data, inliers.data},
                   nullptr) == cudaSuccess);
        VERIFY_MSG(ports[0][f].shape() == clouds[f].point_shape(), at + ": labels shape");
        VERIFY_MSG(bytes_of(ports[0][f].bytes()) == labels.host(n), at + ": labels");
        VERIFY_MSG(ports[1][f].shape() == (std::vector<std::int64_t>{6, 4}),
                   at + ": coefficients shape");
        VERIFY_MSG(same_floats(floats(ports[1][f].bytes()), coefficients.host(24)),
                   at + ": coefficients");
        VERIFY_MSG(same_floats(floats(ports[2][f].bytes()), centroids.host(18)),
                   at + ": centroids");
        std::vector<std::uint32_t> got(6);
        std::memcpy(got.data(), ports[3][f].bytes().data(), 24);
        VERIFY_MSG(got == inliers.host(6), at + ": inliers");
        VERIFY_MSG(got[0] > 1000U, at + ": a plane is found");
        for (const auto& port : ports) {
          VERIFY_MSG(port[f].sample_metadata().capture_timestamp_ns == capture_time(f).count(),
                     at + ": capture time");
          VERIFY_MSG(port[f].sample_metadata().frame_id == 7U + f, at + ": frame id");
        }
      }
    }
  }
}

// An empty cloud gives empty outputs; an all-invalid one no normals and no planes.
void check_empty_and_invalid() {
  TestCloud empty;
  empty.shape = {0, 3};
  TestCloud invalid = scene_cloud(0, true);
  std::fill(invalid.xyz.begin(), invalid.xyz.end(), std::numeric_limits<float>::quiet_NaN());
  for (TestCloud cloud : {empty, invalid}) {
    const std::string kind = cloud.xyz.empty() ? " empty" : " all-invalid";
    const std::size_t n = cloud.count();
    for (const OutputLayout layout : {OutputLayout::kKeep, OutputLayout::kDense}) {
      NormalEstimationConfig config;
      config.layout = layout;
      config.max_points = kWidth * kHeight;
      auto h = holoscan::testing::create_operator_test<NormalEstimationOp>(config);
      h.push("point_cloud", message(cloud, false, false, true));
      h.input_source_device("point_cloud", holoscan::DeviceId{0});
      h.output_device("point_cloud_with_normals", holoscan::DeviceId{0});
      h.output_device("curvature", holoscan::DeviceId{0});
      h.capture_snapshot<hm::schema::PointCloudT>("point_cloud_with_normals", kCaptures);
      h.capture_snapshot<holoscan::Tensor>("curvature", kCaptures);
      const auto result = h.run();
      VERIFY_MSG(result.ok(), "normals" + kind + ": " + result.describe_failure());
      if (!result.ok()) {
        continue;
      }
      const auto out = result.snapshots<hm::schema::PointCloudT>("point_cloud_with_normals");
      VERIFY(out.size() == 1);
      if (out.size() != 1) {
        continue;
      }
      const std::vector<float> normals =
          floats(out[0].tensor(hm::PointCloudFields::normals).bytes());
      VERIFY(normals.size() == (layout == OutputLayout::kKeep ? 3 * n : 0U));
      VERIFY(std::all_of(normals.begin(), normals.end(), [](float v) { return std::isnan(v); }));
      if (layout == OutputLayout::kDense) {
        VERIFY(out[0].tensor(hm::PointCloudFields::points).shape() ==
               (std::vector<std::int64_t>{0, 3}));
        VERIFY(out[0].metadata().valid_count == 0);
        VERIFY(result.snapshots<holoscan::Tensor>("curvature")[0].shape() ==
               (std::vector<std::int64_t>{0}));
      }
    }
    cloud.normals.assign(cloud.xyz.size(), std::numeric_limits<float>::quiet_NaN());
    cloud.with_normals = true;
    PlaneSegmentationConfig config;
    config.max_points = kWidth * kHeight;
    auto h = holoscan::testing::create_operator_test<PlaneSegmentationOp>(config);
    h.push("point_cloud", message(cloud, false, false, true));
    h.input_source_device("point_cloud", holoscan::DeviceId{0});
    for (const char* output :
         {"labels", "plane_coefficients", "plane_centroids", "plane_inliers"}) {
      h.output_device(output, holoscan::DeviceId{0});
      h.capture_snapshot<holoscan::Tensor>(output, kCaptures);
    }
    const auto result = h.run();
    VERIFY_MSG(result.ok(), "planes" + kind + ": " + result.describe_failure());
    if (!result.ok()) {
      continue;
    }
    const std::vector<std::uint8_t> labels =
        bytes_of(result.snapshots<holoscan::Tensor>("labels")[0].bytes());
    VERIFY(labels.size() == n);
    VERIFY(std::all_of(labels.begin(), labels.end(),
                       [](std::uint8_t v) { return v == kLabelInvalid; }));
    const std::vector<std::uint8_t> inliers =
        bytes_of(result.snapshots<holoscan::Tensor>("plane_inliers")[0].bytes());
    VERIFY(std::all_of(inliers.begin(), inliers.end(), [](std::uint8_t v) { return v == 0U; }));
  }
}

template <typename Op, typename Config>
void check_rejects(const Config& config, hm::schema::PointCloudT cloud,
                   const std::vector<std::string>& outputs, const std::string& what) {
  auto h = holoscan::testing::create_operator_test<Op>(config);
  h.push("point_cloud", std::move(cloud));
  for (const std::string& output : outputs) {
    h.output_device(output, holoscan::DeviceId{0});
  }
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), what + " is rejected");
  VERIFY_MSG(result.describe_failure().find(what) != std::string::npos,
             "failure names '" + what + "': " + result.describe_failure());
}

void check_rejected_inputs() {
  const std::vector<std::string> normal_outputs{"point_cloud_with_normals", "curvature"};
  const std::vector<std::string> plane_outputs{"labels", "plane_coefficients", "plane_centroids",
                                               "plane_inliers"};
  TestCloud four;
  four.xyz.assign(12, 1.0F);
  four.rgb.assign(12, 9U);
  four.intensity.assign(4, 1.0F);
  four.shape = {4, 3};
  NormalEstimationConfig config;
  // The layout, as the messages module checks it.
  hm::schema::PointCloudT wrong_colors = message(four, false, false, false);
  wrong_colors.colors =
      std::make_shared<holoscan::Tensor>(make({2, 2, 3}, kUInt8, four.rgb.data(), 12, false));
  check_rejects<NormalEstimationOp>(config, wrong_colors, normal_outputs, "point_cloud.colors");
  hm::schema::PointCloudT doubles = message(four, false, false, false);
  const std::vector<double> xyz64(12, 1.0);
  doubles.points =
      std::make_shared<holoscan::Tensor>(make({4, 3}, {kDLFloat, 64, 1}, xyz64.data(), 96, false));
  check_rejects<NormalEstimationOp>(config, doubles, normal_outputs, "point_cloud.points");
  // The operators' own limits.
  NormalEstimationConfig small = config;
  small.max_points = 3;
  check_rejects<NormalEstimationOp>(small, message(four, false, false, false), normal_outputs,
                                    "exceed the configured maximum of 3");
  NormalEstimationConfig window = config;
  window.normals.search = NeighborSearch::kWindow;
  check_rejects<NormalEstimationOp>(window, message(four, false, false, false), normal_outputs,
                                    "the window search needs an organized");
  // Fields the output has no room for.
  check_rejects<NormalEstimationOp>(config, message(four, true, false, false), normal_outputs,
                                    "set NormalEstimationConfig::colors");
  check_rejects<NormalEstimationOp>(config, message(four, false, true, false), normal_outputs,
                                    "set NormalEstimationConfig::intensity");
  // A cloud without normals has no planes to find.
  check_rejects<PlaneSegmentationOp>(PlaneSegmentationConfig{}, message(four, false, false, false),
                                     plane_outputs, "the cloud carries no normals");
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
  rejects([] { NormalEstimationOp op(NormalEstimationConfig{.max_points = 0}); }, "max_points");
  rejects(
      [] {
        NormalEstimationConfig config;
        config.normals.k = 1;
        NormalEstimationOp op(config);
      },
      "k must be 3..64");
  rejects(
      [] {
        NormalEstimationConfig config;
        config.normals.k = 4;
        NormalEstimationOp op(config);
      },
      "min_neighbors must be at most k");
  rejects(
      [] {
        NormalEstimationConfig config;
        config.normals.window_radius = 99;
        NormalEstimationOp op(config);
      },
      "window_radius must be 1..15");
  rejects([] { PlaneSegmentationOp op(PlaneSegmentationConfig{.max_points = 0}); }, "max_points");
  rejects(
      [] {
        PlaneSegmentationConfig config;
        config.planes.max_planes = 65;
        PlaneSegmentationOp op(config);
      },
      "max_planes must be 1..64");
  rejects(
      [] {
        PlaneSegmentationConfig config;
        config.planes.max_normal_angle_deg = -1.0F;
        PlaneSegmentationOp op(config);
      },
      "max_normal_angle_deg must be 0..90");
}

// Pinned host memory whose allocator the test plays: it counts the streams it
// is asked to wait for before reuse, and can refuse to wait.
class PinnedMemory final : public holoscan::TensorMemoryReference {
 public:
  PinnedMemory(std::size_t bytes, bool waits_for_streams) : waits_(waits_for_streams) {
    if (cudaMallocHost(&base_, bytes) != cudaSuccess) {
      throw std::runtime_error("cudaMallocHost failed");
    }
  }
  ~PinnedMemory() override { cudaFreeHost(base_); }
  PinnedMemory(const PinnedMemory&) = delete;
  PinnedMemory& operator=(const PinnedMemory&) = delete;

  bool set_deallocation_stream(cudaStream_t stream) override {
    if (waits_ && stream != nullptr) {
      streams_.fetch_add(1);
    }
    return waits_;
  }
  [[nodiscard]] bool tracks_deallocation_stream() const noexcept override { return waits_; }
  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override { return false; }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> synchronize_for_host() override {
    return {};
  }

  [[nodiscard]] void* base() const { return base_; }
  [[nodiscard]] int streams() const { return streams_.load(); }

 private:
  void* base_{};
  bool waits_{};
  std::atomic<int> streams_{0};
};

holoscan::Tensor wrap_pinned(const std::shared_ptr<PinnedMemory>& memory,
                             const std::vector<std::int64_t>& shape) {
  auto context = std::make_shared<holoscan::DLManagedTensorContext>();
  context->dl_shape = shape;
  context->dl_strides.assign(shape.size(), 1);
  DLTensor& tensor = context->tensor.dl_tensor;
  tensor.data = memory->base();
  tensor.device = DLDevice{kDLCUDAHost, 0};
  tensor.ndim = static_cast<std::int32_t>(shape.size());
  tensor.dtype = kUInt8;
  tensor.shape = context->dl_shape.data();
  tensor.strides = context->dl_strides.data();
  context->memory_reference = memory;
  return holoscan::Tensor(std::move(context));
}

// Pinned host input is held until the stream that copies it is done: by its
// allocator when that can wait for a stream, without blocking; otherwise by
// waiting.
void check_pinned_input_held() {
  cudaStream_t stream{};
  VERIFY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  const auto stall = [&] {
    return cudaLaunchHostFunc(
               stream, [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds(300)); },
               nullptr) == cudaSuccess;
  };
  {
    auto memory = std::make_shared<PinnedMemory>(64, true);
    VERIFY(stall());
    VERIFY(detail::hold_until_done(wrap_pinned(memory, {64}), stream, "test").has_value());
    VERIFY_MSG(memory->streams() == 1, "the allocator is asked to wait for the stream");
    VERIFY_MSG(cudaStreamQuery(stream) == cudaErrorNotReady, "the host does not wait");
    VERIFY(cudaStreamSynchronize(stream) == cudaSuccess);
  }
  {
    auto memory = std::make_shared<PinnedMemory>(64, false);
    VERIFY(stall());
    VERIFY(detail::hold_until_done(wrap_pinned(memory, {64}), stream, "test").has_value());
    VERIFY_MSG(cudaStreamQuery(stream) == cudaSuccess,
               "the stream is done when the allocator cannot wait for it");
  }
  VERIFY(cudaStreamDestroy(stream) == cudaSuccess);
}

// A producer of clouds from a runtime pool in host or pinned host memory,
// which the harness, staging its inputs in memory of its own, cannot be: the
// scene's organized clouds with colors, one per tick.
class PoolCloudSourceOp final : public holoscan::Operator<> {
 public:
  PoolCloudSourceOp(std::vector<TestCloud> clouds, holoscan::MemoryKind memory_kind)
      : clouds_(std::move(clouds)), memory_kind_(memory_kind) {}
  void setup(holoscan::OperatorSpec& spec) override {
    hm::declare_point_cloud_output(spec, point_cloud, "point_cloud",
                                   {.max_points = std::uint64_t{kWidth} * kHeight,
                                    .colors = true,
                                    .memory_kind = memory_kind_});
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
      // Every slot still held by the consumers: send the same cloud next tick.
      return loan.error().code == holoscan::ErrorCode::kResourceExhausted
                 ? holoscan::expected<void, holoscan::Error>{}
                 : holoscan::make_unexpected(std::move(loan).error());
    }
    const auto copy = [](const void* from, std::size_t bytes) {
      return [from, bytes](const holoscan::TensorOutputWriteGuard& writer) {
        std::memcpy(writer.data_as<void>(), from, bytes);
      };
    };
    if (auto written = loan->write_host_and_commit(hm::PointCloudFields::points,
                                                   copy(cloud.xyz.data(), cloud.xyz.size() * 4U));
        !written) {
      return written;
    }
    if (auto written = loan->write_host_and_commit(hm::PointCloudFields::colors,
                                                   copy(cloud.rgb.data(), cloud.rgb.size()));
        !written) {
      return written;
    }
    auto sent = hm::emit_point_cloud(
        point_cloud, std::move(*loan), {.frame_id = kFrame}, shape, -1,
        holoscan::EmitOptions{.capture_time = context.activation_time(), .frame_id = next_ + 1U});
    if (!sent) {
      return sent.error().code == holoscan::ErrorCode::kBackpressured
                 ? holoscan::expected<void, holoscan::Error>{}
                 : sent;
    }
    ++next_;
    return {};
  }

  holoscan::Output<hm::schema::PointCloudT> point_cloud;

 private:
  std::vector<TestCloud> clouds_;
  holoscan::MemoryKind memory_kind_{};
  std::uint32_t next_{};
};

struct SinkResult {
  std::vector<std::vector<float>> normals;
  std::vector<std::vector<std::uint8_t>> colors;
};

// Copies the normals and colors of each cloud it receives, and completes the
// run after `expected` of them.
class CloudSinkOp final : public holoscan::Operator<> {
 public:
  CloudSinkOp(std::shared_ptr<SinkResult> result, std::size_t expected)
      : result_(std::move(result)), expected_(expected) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(point_cloud, "point_cloud");
    spec.input(curvature, "curvature");
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    using std::chrono_literals::operator""ns;
    holoscan::Contract result;
    result.trigger(holoscan::OnAligned{
        .ports = holoscan::PortList{std::vector<holoscan::PortRef>{point_cloud, curvature}},
        .window = 0ns});
    result.input(point_cloud, holoscan::Exact{});
    result.input(curvature, holoscan::Exact{});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto cloud = point_cloud.receive();
    auto values = curvature.receive();
    if (!cloud) {
      return holoscan::make_unexpected(std::move(cloud).error());
    }
    if (!values) {
      return holoscan::make_unexpected(std::move(values).error());
    }
    const std::size_t n = static_cast<std::size_t>(kWidth) * kHeight;
    std::vector<float> normals(3 * n);
    std::vector<std::uint8_t> colors(3 * n);
    if (auto copied =
            cloud->data.normals->copy_to_host(normals.data(), 12 * n, context.cuda_stream());
        !copied) {
      return copied;
    }
    if (auto copied = cloud->data.colors->copy_to_host(colors.data(), 3 * n, context.cuda_stream());
        !copied) {
      return copied;
    }
    result_->normals.push_back(std::move(normals));
    result_->colors.push_back(std::move(colors));
    if (result_->normals.size() == expected_) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<hm::schema::PointCloudT> point_cloud;
  holoscan::Input<holoscan::Tensor> curvature;

 private:
  std::shared_ptr<SinkResult> result_;
  std::size_t expected_{};
};

// Clouds from a producer's host or pinned host pool: copied to the GPU (the
// pinned copy held against the producer's reuse), with the normals and colors
// that device input gives.
void check_pool_clouds(holoscan::MemoryKind memory_kind) {
  const std::vector<TestCloud> clouds{scene_cloud(0, true), scene_cloud(1, true)};
  holoscan::Graph graph{"pool"};
  auto source = graph.op<PoolCloudSourceOp>("source", clouds, memory_kind);
  NormalEstimationConfig config{.colors = true, .max_points = kWidth * kHeight};
  config.normals.radius = 0.15F;
  auto normals = graph.op<NormalEstimationOp>("normals", config);
  auto result = std::make_shared<SinkResult>();
  auto sink = graph.op<CloudSinkOp>("sink", result, clouds.size());
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->point_cloud, normals->point_cloud, lossless);
  graph.add_flow(normals->point_cloud_with_normals, sink->point_cloud, lossless);
  graph.add_flow(normals->curvature, sink->curvature, lossless);
  graph.partition("pool").include_all();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  holoscan::CompileOptions options;
  for (const char* port : {"point_cloud_with_normals", "curvature"}) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "normals", .output_port = port, .device = holoscan::DeviceId{0}});
  }
  const auto plan = holoscan::compile(graph, std::move(options));
  VERIFY_MSG(plan.ok(), holoscan::format_diagnostics(plan.diagnostics(), plan.graph()));
  if (!plan.ok()) {
    return;
  }
  auto session = holoscan::run_async(plan);
  if (!session.wait_for(std::chrono::seconds(60))) {
    session.request_stop();
  }
  session.wait();
  VERIFY_MSG(
      session.termination() == holoscan::RunTermination::kRequestedCompleted,
      "run did not complete: " + holoscan::format_diagnostics(session.diagnostics(), plan.graph()));
  VERIFY(result->normals.size() == clouds.size());
  for (std::size_t f = 0; f < std::min(result->normals.size(), clouds.size()); ++f) {
    const NormalOutputs want = expected_normals(clouds[f], config);
    VERIFY_MSG(same_floats(result->normals[f], want.normals),
               "cloud " + std::to_string(f) + ": normals");
    VERIFY_MSG(result->colors[f] == want.rgb, "cloud " + std::to_string(f) + ": colors");
  }
}

}  // namespace
}  // namespace holoscan::normals

int main() {
  using holoscan::normals::test::TestCase;
  namespace hn = holoscan::normals;
  return holoscan::normals::test::run({
      TestCase{"NormalEstimationOp", hn::check_normal_op},
      TestCase{"PlaneSegmentationOp", hn::check_plane_op},
      TestCase{"empty and all-invalid clouds", hn::check_empty_and_invalid},
      TestCase{"rejected inputs", hn::check_rejected_inputs},
      TestCase{"rejected settings", hn::check_rejected_configs},
      TestCase{"pinned host input held", hn::check_pinned_input_held},
      TestCase{"clouds from a host pool",
               [] { hn::check_pool_clouds(holoscan::MemoryKind::kHost); }},
      TestCase{"clouds from a pinned host pool",
               [] { hn::check_pool_clouds(holoscan::MemoryKind::kPinnedHost); }},
  });
}
