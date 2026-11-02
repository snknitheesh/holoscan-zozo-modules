// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The two operators through the SDK's operator test harness, on clouds of the
// synthetic scene made at test time: organized and unorganized input in host
// and device memory, with and without colors, both output layouts, checked
// bit for bit against NormalEstimator and PlaneSegmenter run directly; the
// capture time and frame id carried to every output; empty and all-invalid
// clouds; and the rejection of unusable inputs and settings.

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
#include <holoscan/normals/normal_estimation_op.hpp>
#include <holoscan/normals/plane_segmentation_op.hpp>
#include <holoscan/normals/synthetic_scene.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "test_support.hpp"

namespace holoscan::normals {
namespace {

using std::chrono_literals::operator""ms;

constexpr int kWidth = 96;
constexpr int kHeight = 72;
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
  std::vector<std::uint8_t> surfaces;
  std::vector<float> truth;
  synthetic_scene_cloud(synthetic_scene(kWidth, kHeight, frame), cloud.xyz, cloud.rgb, surfaces,
                        truth);
  cloud.shape = organized ? std::vector<std::int64_t>{kHeight, kWidth, 3}
                          : std::vector<std::int64_t>{kHeight * kWidth, 3};
  return cloud;
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
  const std::size_t n = cloud.xyz.size() / 3;
  const bool organized = cloud.shape.size() == 3;
  params.search = resolve_search(params.search, organized);
  Scratch<float> xyz(3 * n);
  Scratch<float> normals(3 * n);
  Scratch<float> curvature(n);
  cudaMemcpy(xyz.data, cloud.xyz.data(), 12 * n, cudaMemcpyHostToDevice);
  NormalEstimator estimator;
  if (estimator.estimate(xyz.data, static_cast<std::uint32_t>(n),
                         organized ? kWidth : static_cast<int>(n), organized ? kHeight : 1, params,
                         normals.data, curvature.data, nullptr) != cudaSuccess) {
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
  std::vector<float> normals;
  std::vector<float> curvature;
};

NormalOutputs expected_normals(const TestCloud& cloud, const NormalEstimationConfig& config) {
  const DirectNormals direct = direct_normals(cloud, config.normals);
  const std::size_t n = cloud.xyz.size() / 3;
  NormalOutputs out;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::int64_t kept = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const bool has = std::isfinite(direct.normals[3 * i]);
    if (config.layout == OutputLayout::kDense && !has) {
      continue;
    }
    ++kept;
    for (int c = 0; c < 3; ++c) {
      out.xyz.push_back(has ? cloud.xyz[3 * i + c] : nan);
      out.rgb.push_back(has ? cloud.rgb[3 * i + c] : 0U);
      out.normals.push_back(direct.normals[3 * i + c]);
    }
    out.curvature.push_back(direct.curvature[i]);
  }
  if (config.layout == OutputLayout::kKeep) {
    out.shape = cloud.shape;
    out.point_shape.assign(cloud.shape.begin(), cloud.shape.end() - 1);
  } else {
    out.shape = {kept, 3};
    out.point_shape = {kept};
  }
  return out;
}

struct Captured {
  bool ok{};
  std::string failure;
  std::vector<std::vector<holoscan::testing::Snapshot<holoscan::Tensor>>> ports;
};

// Pushes the clouds to `inputs` (one tensor per input per frame), runs the
// operator and captures `outputs`.
template <typename Op, typename Config>
Captured run_op(const Config& config, const std::vector<std::string>& inputs,
                const std::vector<std::vector<holoscan::Tensor>>& frames,
                const std::vector<std::string>& outputs, bool device) {
  auto h = holoscan::testing::create_operator_test<Op>(config);
  if (inputs.size() == 1) {
    std::vector<holoscan::testing::TestSample<holoscan::Tensor>> samples;
    for (std::uint32_t f = 0; f < frames.size(); ++f) {
      auto sample = holoscan::testing::sample_at(frames[f][0], capture_time(f));
      sample.metadata.frame_id = 7U + f;
      samples.push_back(std::move(sample));
    }
    h.push_samples(inputs[0], std::move(samples));
  } else {
    for (std::uint32_t f = 0; f < frames.size(); ++f) {
      h.push_frame(holoscan::testing::frame_at(capture_time(f),
                                               holoscan::testing::input(inputs[0], frames[f][0]),
                                               holoscan::testing::input(inputs[1], frames[f][1])));
    }
  }
  if (device) {
    for (const std::string& input : inputs) {
      h.input_source_device(input, holoscan::DeviceId{0});
    }
  }
  for (const std::string& output : outputs) {
    h.output_device(output, holoscan::DeviceId{0});
    h.template capture_snapshot<holoscan::Tensor>(output, kCaptures);
  }
  const auto result = h.run();
  Captured captured;
  captured.ok = result.ok();
  if (!captured.ok) {
    captured.failure = result.describe_failure();
    return captured;
  }
  for (const std::string& output : outputs) {
    captured.ports.push_back(result.template snapshots<holoscan::Tensor>(output));
  }
  return captured;
}

void check_normal_op() {
  for (const bool organized : {true, false}) {
    for (const bool device : {true, false}) {
      for (const bool color : {false, true}) {
        for (const OutputLayout layout : {OutputLayout::kKeep, OutputLayout::kDense}) {
          NormalEstimationConfig config;
          config.color = color;
          config.layout = layout;
          config.max_points = kWidth * kHeight;
          config.normals.radius = 0.15F;
          const std::vector<TestCloud> clouds{scene_cloud(0, organized), scene_cloud(1, organized)};
          std::vector<std::vector<holoscan::Tensor>> frames;
          for (const TestCloud& cloud : clouds) {
            std::vector<holoscan::Tensor> frame{
                make(cloud.shape, kFloat32, cloud.xyz.data(), cloud.xyz.size() * 4, device)};
            if (color) {
              frame.push_back(
                  make(cloud.shape, kUInt8, cloud.rgb.data(), cloud.rgb.size(), device));
            }
            frames.push_back(std::move(frame));
          }
          std::vector<std::string> inputs{"points"};
          std::vector<std::string> outputs{"filtered_points"};
          if (color) {
            inputs.emplace_back("colors");
            outputs.emplace_back("filtered_colors");
          }
          outputs.insert(outputs.end(), {"normals", "curvature"});
          const std::string name = std::string("normals") + (organized ? " organized" : " dense") +
                                   (device ? " device" : " host") + (color ? " color" : "") +
                                   (layout == OutputLayout::kKeep ? " keep" : " packed");
          const Captured out = run_op<NormalEstimationOp>(config, inputs, frames, outputs, device);
          VERIFY_MSG(out.ok, name + ": " + out.failure);
          if (!out.ok) {
            continue;
          }
          const std::size_t normals_port = color ? 2 : 1;
          for (std::size_t f = 0; f < clouds.size(); ++f) {
            const std::string at = name + " frame " + std::to_string(f);
            if (out.ports[0].size() <= f || out.ports[normals_port].size() <= f ||
                out.ports[normals_port + 1].size() <= f) {
              VERIFY_MSG(false, at + ": one output per cloud on every port");
              continue;
            }
            const NormalOutputs want = expected_normals(clouds[f], config);
            VERIFY_MSG(out.ports[0][f].shape() == want.shape, at + ": points shape");
            VERIFY_MSG(same_floats(floats(out.ports[0][f]), want.xyz), at + ": points");
            VERIFY_MSG(out.ports[normals_port][f].shape() == want.shape, at + ": normals shape");
            VERIFY_MSG(same_floats(floats(out.ports[normals_port][f]), want.normals),
                       at + ": normals");
            VERIFY_MSG(out.ports[normals_port + 1][f].shape() == want.point_shape,
                       at + ": curvature shape");
            VERIFY_MSG(same_floats(floats(out.ports[normals_port + 1][f]), want.curvature),
                       at + ": curvature");
            if (color) {
              VERIFY_MSG(bytes(out.ports[1][f]) == want.rgb, at + ": colors");
            }
            for (const auto& port : out.ports) {
              const holoscan::SampleMetadata& metadata = port[f].sample_metadata();
              VERIFY_MSG(metadata.capture_timestamp_ns == capture_time(f).count(),
                         at + ": capture time");
              if (!color) {
                VERIFY_MSG(metadata.frame_id == 7U + f, at + ": frame id");
              }
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
      std::vector<DirectNormals> normals;
      std::vector<std::vector<holoscan::Tensor>> frames;
      for (const TestCloud& cloud : clouds) {
        normals.push_back(direct_normals(cloud, NormalParams{.radius = 0.15F}));
        frames.push_back(
            {make(cloud.shape, kFloat32, cloud.xyz.data(), cloud.xyz.size() * 4, device),
             make(cloud.shape, kFloat32, normals.back().normals.data(),
                  normals.back().normals.size() * 4, device)});
      }
      const std::string name = std::string("planes") + (organized ? " organized" : " dense") +
                               (device ? " device" : " host");
      const Captured out = run_op<PlaneSegmentationOp>(
          config, {"points", "normals"}, frames,
          {"labels", "plane_coefficients", "plane_centroids", "plane_inliers"}, device);
      VERIFY_MSG(out.ok, name + ": " + out.failure);
      if (!out.ok) {
        continue;
      }
      for (std::size_t f = 0; f < clouds.size(); ++f) {
        const std::string at = name + " frame " + std::to_string(f);
        if (std::any_of(out.ports.begin(), out.ports.end(),
                        [f](const auto& port) { return port.size() <= f; })) {
          VERIFY_MSG(false, at + ": one output per cloud on every port");
          continue;
        }
        // The segmenter run directly.
        const std::size_t n = clouds[f].xyz.size() / 3;
        Scratch<float> xyz(3 * n);
        Scratch<float> point_normals(3 * n);
        Scratch<std::uint8_t> labels(n);
        Scratch<float> coefficients(4 * 6);
        Scratch<float> centroids(3 * 6);
        Scratch<std::uint32_t> inliers(6);
        cudaMemcpy(xyz.data, clouds[f].xyz.data(), 12 * n, cudaMemcpyHostToDevice);
        cudaMemcpy(point_normals.data, normals[f].normals.data(), 12 * n, cudaMemcpyHostToDevice);
        PlaneSegmenter segmenter;
        VERIFY(segmenter.segment(
                   xyz.data, point_normals.data, static_cast<std::uint32_t>(n), config.planes,
                   PlaneOutputs{labels.data, coefficients.data, centroids.data, inliers.data},
                   nullptr) == cudaSuccess);
        std::vector<std::int64_t> label_shape(clouds[f].shape.begin(), clouds[f].shape.end() - 1);
        VERIFY_MSG(out.ports[0][f].shape() == label_shape, at + ": labels shape");
        VERIFY_MSG(bytes(out.ports[0][f]) == labels.host(n), at + ": labels");
        VERIFY_MSG(out.ports[1][f].shape() == (std::vector<std::int64_t>{6, 4}),
                   at + ": coefficients shape");
        VERIFY_MSG(same_floats(floats(out.ports[1][f]), coefficients.host(24)),
                   at + ": coefficients");
        VERIFY_MSG(same_floats(floats(out.ports[2][f]), centroids.host(18)), at + ": centroids");
        std::vector<std::uint32_t> got(6);
        std::memcpy(got.data(), out.ports[3][f].bytes().data(), 24);
        VERIFY_MSG(got == inliers.host(6), at + ": inliers");
        VERIFY_MSG(got[0] > 1000U, at + ": a plane is found");
        for (const auto& port : out.ports) {
          VERIFY_MSG(port[f].sample_metadata().capture_timestamp_ns == capture_time(f).count(),
                     at + ": capture time");
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
  for (const TestCloud& cloud : {empty, invalid}) {
    const std::string kind = cloud.xyz.empty() ? " empty" : " all-invalid";
    const std::size_t n = cloud.xyz.size() / 3;
    for (const OutputLayout layout : {OutputLayout::kKeep, OutputLayout::kDense}) {
      NormalEstimationConfig config;
      config.layout = layout;
      config.max_points = kWidth * kHeight;
      const Captured out = run_op<NormalEstimationOp>(
          config, {"points"}, {{make(cloud.shape, kFloat32, cloud.xyz.data(), n * 12, true)}},
          {"filtered_points", "normals", "curvature"}, true);
      VERIFY_MSG(out.ok, "normals" + kind + ": " + out.failure);
      if (!out.ok) {
        continue;
      }
      VERIFY(out.ports[1].size() == 1);
      const std::vector<float> normals = floats(out.ports[1][0]);
      VERIFY(std::all_of(normals.begin(), normals.end(), [](float v) { return std::isnan(v); }));
      if (layout == OutputLayout::kDense) {
        VERIFY(out.ports[0][0].shape() == (std::vector<std::int64_t>{0, 3}));
        VERIFY(out.ports[2][0].shape() == (std::vector<std::int64_t>{0}));
      }
    }
    std::vector<float> nan_normals(cloud.xyz.size(), std::numeric_limits<float>::quiet_NaN());
    PlaneSegmentationConfig config;
    config.max_points = kWidth * kHeight;
    const Captured out = run_op<PlaneSegmentationOp>(
        config, {"points", "normals"},
        {{make(cloud.shape, kFloat32, cloud.xyz.data(), n * 12, true),
          make(cloud.shape, kFloat32, nan_normals.data(), n * 12, true)}},
        {"labels", "plane_coefficients", "plane_centroids", "plane_inliers"}, true);
    VERIFY_MSG(out.ok, "planes" + kind + ": " + out.failure);
    if (!out.ok) {
      continue;
    }
    const std::vector<std::uint8_t> labels = bytes(out.ports[0][0]);
    VERIFY(labels.size() == n);
    VERIFY(std::all_of(labels.begin(), labels.end(),
                       [](std::uint8_t v) { return v == kLabelInvalid; }));
    const std::vector<std::uint8_t> inliers = bytes(out.ports[3][0]);
    VERIFY(std::all_of(inliers.begin(), inliers.end(), [](std::uint8_t v) { return v == 0U; }));
  }
}

template <typename Op, typename Config>
void check_rejects(const Config& config, const std::vector<std::string>& inputs,
                   std::vector<holoscan::Tensor> tensors, const std::vector<std::string>& outputs,
                   const std::string& what) {
  auto h = holoscan::testing::create_operator_test<Op>(config);
  if (inputs.size() == 1) {
    h.push(inputs[0], tensors[0]);
  } else {
    h.push_frame(holoscan::testing::frame_at(100ms, holoscan::testing::input(inputs[0], tensors[0]),
                                             holoscan::testing::input(inputs[1], tensors[1])));
  }
  for (const std::string& output : outputs) {
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
  const std::vector<std::uint8_t> rgb(12, 9U);
  const std::string shape = "points must be float32 [N, 3] or [H, W, 3]";
  const std::vector<std::string> normal_outputs{"filtered_points", "normals", "curvature"};
  NormalEstimationConfig config;
  check_rejects<NormalEstimationOp>(config, {"points"},
                                    {make({4, 3}, {kDLFloat, 64, 1}, xyz64.data(), 96, false)},
                                    normal_outputs, shape);
  check_rejects<NormalEstimationOp>(
      config, {"points"}, {make({3, 4}, kFloat32, xyz.data(), 48, false)}, normal_outputs, shape);
  check_rejects<NormalEstimationOp>(config, {"points"},
                                    {make({1, 2, 2, 3}, kFloat32, xyz.data(), 48, false)},
                                    normal_outputs, shape);
  NormalEstimationConfig small = config;
  small.max_points = 3;
  check_rejects<NormalEstimationOp>(small, {"points"},
                                    {make({4, 3}, kFloat32, xyz.data(), 48, false)}, normal_outputs,
                                    "exceed the configured maximum of 3");
  NormalEstimationConfig window = config;
  window.normals.search = NeighborSearch::kWindow;
  check_rejects<NormalEstimationOp>(window, {"points"},
                                    {make({4, 3}, kFloat32, xyz.data(), 48, false)}, normal_outputs,
                                    "the window search needs an organized");
  NormalEstimationConfig with_color = config;
  with_color.color = true;
  check_rejects<NormalEstimationOp>(with_color, {"points", "colors"},
                                    {make({4, 3}, kFloat32, xyz.data(), 48, false),
                                     make({2, 2, 3}, kUInt8, rgb.data(), 12, false)},
                                    {"filtered_points", "filtered_colors", "normals", "curvature"},
                                    "colors must be uint8 of the points' shape");
  const std::vector<std::string> plane_outputs{"labels", "plane_coefficients", "plane_centroids",
                                               "plane_inliers"};
  check_rejects<PlaneSegmentationOp>(PlaneSegmentationConfig{}, {"points", "normals"},
                                     {make({4, 3}, kFloat32, xyz.data(), 48, false),
                                      make({2, 2, 3}, kFloat32, xyz.data(), 48, false)},
                                     plane_outputs, "normals must be float32 of the points' shape");
  check_rejects<PlaneSegmentationOp>(
      PlaneSegmentationConfig{}, {"points", "normals"},
      {make({4, 3}, kFloat32, xyz.data(), 48, false), make({4, 3}, kUInt8, rgb.data(), 12, false)},
      plane_outputs, "normals must be float32 of the points' shape");
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
  });
}
