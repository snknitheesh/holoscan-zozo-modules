// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// IcpOp on its own, through the SDK's operator test harness: pairs of clouds
// from host and device memory, with and without an initial guess, their
// transforms checked against the truth and the aligned cloud against the
// transform; a stored target (kMap) used for later sources; odometry along a
// camera path; a failed registration still reported; the capture time carried
// to every output; and unusable inputs and configurations rejected.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/icp/icp_op.hpp>
#include <holoscan/icp/synthetic_scene.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "cloud_support.hpp"
#include "test_support.hpp"

namespace holoscan::icp {
namespace {

using std::chrono_literals::operator""ms;
using test::degrees;
using test::radians;

constexpr int kWidth = 160;
constexpr int kHeight = 120;
constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kFloat64{kDLFloat, 64U, 1U};
constexpr DepthNoise kNoise{.sigma = 0.0005F, .sigma_per_m2 = 0.0005F, .seed = 1U};

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

// The organized cloud seen from `pose` as a [H, W, 3] tensor.
holoscan::Tensor cloud_tensor(const Matrix4& pose, bool device, std::uint32_t seed,
                              int width = kWidth, int height = kHeight) {
  DepthNoise noise = kNoise;
  noise.seed = seed;
  const std::vector<float> xyz =
      test::render(synthetic_scene(SceneKind::kRoom), pose, width, height, noise).to_host();
  return make({height, width, 3}, kFloat32, xyz.data(), xyz.size() * sizeof(float), device);
}

holoscan::Tensor matrix_tensor(const Matrix4& m, bool f32, bool device) {
  if (f32) {
    std::array<float, 16> values{};
    for (std::size_t i = 0; i < 16; ++i) values[i] = static_cast<float>(m[i]);
    return make({4, 4}, kFloat32, values.data(), sizeof(values), device);
  }
  return make({4, 4}, kFloat64, m.data(), sizeof(Matrix4), device);
}

IcpConfig base_config(IcpMode mode) {
  IcpConfig config;
  config.mode = mode;
  config.options.max_correspondence_distance = 0.2F;
  config.options.min_correspondence_distance = 0.03F;
  config.options.voxel_size = 0.02F;
  config.options.normal_radius = 0.06F;
  config.options.max_iterations = 100;
  config.max_points = kWidth * kHeight;
  return config;
}

std::chrono::nanoseconds capture_time(std::uint32_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

// The values of a captured float64 tensor.
template <typename Snapshot>
std::vector<double> doubles(const Snapshot& snapshot) {
  std::vector<double> values(snapshot.bytes().size() / sizeof(double));
  std::memcpy(values.data(), snapshot.bytes().data(), values.size() * sizeof(double));
  return values;
}

template <typename Snapshot>
Matrix4 matrix(const Snapshot& snapshot) {
  Matrix4 m{};
  const std::vector<double> values = doubles(snapshot);
  if (snapshot.shape() == std::vector<std::int64_t>{4, 4} && values.size() == 16U) {
    std::copy(values.begin(), values.end(), m.begin());
  }
  return m;
}

template <typename Snapshot>
void check_time(const Snapshot& snapshot, std::uint32_t frame, const std::string& what) {
  VERIFY_MSG(snapshot.sample_metadata().capture_timestamp_ns == capture_time(frame).count(),
             what + " of frame " + std::to_string(frame) + " carries the source's capture time");
}

// The source camera of frame `frame`, relative to its target camera.
Matrix4 pair_truth(std::uint32_t frame) {
  const double s = static_cast<double>(frame);
  return transform_from({radians(1.0 + s), radians(-2.0 + s), radians(0.5)},
                        {0.03 - 0.02 * s, 0.01, 0.04});
}

Matrix4 pair_target_pose(std::uint32_t frame) {
  return transform_from({0.0, radians(5.0 * frame), 0.0}, {0.1 * frame, 0.0, 0.3});
}

void check_pairs(bool device, bool guess, bool aligned_output, bool f32_guess = false) {
  IcpConfig config = base_config(IcpMode::kPairs);
  config.initial_guess_input = guess;
  config.aligned_output = aligned_output;
  auto h = holoscan::testing::create_operator_test<IcpOp>(config);
  constexpr std::uint32_t kFrames = 3;
  std::vector<holoscan::Tensor> sources;
  for (std::uint32_t frame = 0; frame < kFrames; ++frame) {
    const Matrix4 a = pair_target_pose(frame);
    const Matrix4 truth = pair_truth(frame);
    holoscan::Tensor source = cloud_tensor(compose(a, truth), device, 2U * frame + 1U);
    holoscan::Tensor target = cloud_tensor(a, device, 2U * frame);
    sources.push_back(source);
    if (guess) {
      // A guess 2 degrees and 5 cm off.
      const Matrix4 rough =
          compose(transform_from({0.0, radians(2.0), 0.0}, {0.05, 0.0, 0.0}), truth);
      h.push_frame(holoscan::testing::frame_at(
          capture_time(frame), holoscan::testing::input("source", source),
          holoscan::testing::input("target", target),
          holoscan::testing::input("initial_guess", matrix_tensor(rough, f32_guess, device))));
    } else {
      h.push_frame(holoscan::testing::frame_at(capture_time(frame),
                                               holoscan::testing::input("source", source),
                                               holoscan::testing::input("target", target)));
    }
  }
  if (device) {
    h.input_source_device("source", holoscan::DeviceId{0});
    h.input_source_device("target", holoscan::DeviceId{0});
    if (guess) h.input_source_device("initial_guess", holoscan::DeviceId{0});
  }
  h.capture_snapshot<holoscan::Tensor>("transform");
  h.capture_snapshot<holoscan::Tensor>("stats");
  if (aligned_output) {
    h.output_device("aligned", holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::Tensor>("aligned");
  }
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto transforms = result.snapshots<holoscan::Tensor>("transform");
  const auto stats = result.snapshots<holoscan::Tensor>("stats");
  VERIFY(transforms.size() == kFrames && stats.size() == kFrames);
  for (std::uint32_t frame = 0; frame < kFrames && frame < transforms.size(); ++frame) {
    const Matrix4 m = matrix(transforms[frame]);
    const IcpResult r = result_from(doubles(stats[frame]), m);
    const TransformError error = transform_error(m, pair_truth(frame));
    const std::string what = "frame " + std::to_string(frame);
    VERIFY_MSG(r.converged && r.status == IcpStatus::kConverged, what + " converged");
    VERIFY_MSG(degrees(error.rotation_rad) < 0.2 && error.translation < 0.005,
               what + ": error " + std::to_string(degrees(error.rotation_rad)) + " deg, " +
                   std::to_string(error.translation * 1000.0) + " mm");
    VERIFY_MSG(r.fitness > 0.5 && r.iterations > 0 && r.source_points > 0U &&
                   r.target_points > 0U && r.correspondences > 0U,
               what + " statistics");
    check_time(transforms[frame], frame, "transform");
    check_time(stats[frame], frame, "stats");
  }
  if (!aligned_output) return;
  const auto aligned = result.snapshots<holoscan::Tensor>("aligned");
  VERIFY(aligned.size() == kFrames);
  for (std::uint32_t frame = 0; frame < kFrames && frame < aligned.size(); ++frame) {
    VERIFY(aligned[frame].shape() == (std::vector<std::int64_t>{kHeight, kWidth, 3}));
    std::vector<float> got(static_cast<std::size_t>(kWidth) * kHeight * 3U);
    std::vector<float> in(got.size());
    if (aligned[frame].bytes().size() != got.size() * sizeof(float)) continue;
    std::memcpy(got.data(), aligned[frame].bytes().data(), got.size() * sizeof(float));
    VERIFY(sources[frame].copy_to_host(in.data(), in.size() * sizeof(float)).has_value());
    const Matrix4 m = matrix(transforms[frame]);
    int wrong = 0;
    for (std::size_t i = 0; i < in.size(); i += 3) {
      if (!std::isfinite(in[i])) {
        wrong += std::isfinite(got[i]) ? 1 : 0;
        continue;
      }
      for (int row = 0; row < 3; ++row) {
        const double expected = m[4 * row] * in[i] + m[4 * row + 1] * in[i + 1] +
                                m[4 * row + 2] * in[i + 2] + m[4 * row + 3];
        wrong += std::fabs(got[i + row] - expected) > 1e-5 ? 1 : 0;
      }
    }
    VERIFY_MSG(wrong == 0, std::to_string(wrong) + " aligned coordinates differ");
    check_time(aligned[frame], frame, "aligned");
  }
}

// kMap: one target, then sources from around it, each aligned to the stored
// target.
void check_map() {
  const IcpConfig config = base_config(IcpMode::kMap);
  auto h = holoscan::testing::create_operator_test<IcpOp>(config);
  const Matrix4 map_pose = pair_target_pose(0);
  h.push_frame(holoscan::testing::frame_at(
      capture_time(0), holoscan::testing::input("target", cloud_tensor(map_pose, true, 0U))));
  constexpr std::uint32_t kFrames = 3;
  for (std::uint32_t frame = 1; frame <= kFrames; ++frame) {
    h.push_frame(holoscan::testing::frame_at(
        capture_time(frame),
        holoscan::testing::input(
            "source", cloud_tensor(compose(map_pose, pair_truth(frame - 1U)), true, 10U + frame))));
  }
  h.input_source_device("source", holoscan::DeviceId{0});
  h.input_source_device("target", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::Tensor>("transform");
  h.capture_snapshot<holoscan::Tensor>("stats");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto transforms = result.snapshots<holoscan::Tensor>("transform");
  const auto stats = result.snapshots<holoscan::Tensor>("stats");
  VERIFY_MSG(transforms.size() == kFrames && stats.size() == kFrames,
             std::to_string(transforms.size()) + " transforms from " + std::to_string(kFrames) +
                 " sources");
  for (std::uint32_t i = 0; i < transforms.size() && i < stats.size(); ++i) {
    const Matrix4 m = matrix(transforms[i]);
    const IcpResult r = result_from(doubles(stats[i]), m);
    const TransformError error = transform_error(m, pair_truth(i));
    VERIFY_MSG(r.converged && degrees(error.rotation_rad) < 0.2 && error.translation < 0.005,
               "source " + std::to_string(i) + " against the stored target: " +
                   std::to_string(error.translation * 1000.0) + " mm");
    check_time(transforms[i], i + 1U, "transform");
  }
}

// Odometry along the room path, from host memory, at 320x240: each pose
// against the truth, relative to the first frame.
void check_odometry() {
  constexpr int kOdometryWidth = 320;
  constexpr int kOdometryHeight = 240;
  IcpConfig config = base_config(IcpMode::kOdometry);
  config.options.max_correspondence_distance = 0.1F;
  config.max_points = kOdometryWidth * kOdometryHeight;
  auto h = holoscan::testing::create_operator_test<IcpOp>(config);
  constexpr std::uint32_t kFrames = 12;
  constexpr std::uint32_t kPath = 150;  // The path's length in frames; the test runs a part.
  std::vector<holoscan::testing::TestSample<holoscan::Tensor>> samples;
  for (std::uint32_t frame = 0; frame < kFrames; ++frame) {
    samples.push_back(
        holoscan::testing::sample_at(cloud_tensor(synthetic_trajectory(frame, kPath), false, frame,
                                                  kOdometryWidth, kOdometryHeight),
                                     capture_time(frame)));
  }
  h.push_samples("source", std::move(samples));
  h.capture_snapshot<holoscan::Tensor>("pose");
  h.capture_snapshot<holoscan::Tensor>("transform");
  h.capture_snapshot<holoscan::Tensor>("stats");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto poses = result.snapshots<holoscan::Tensor>("pose");
  const auto transforms = result.snapshots<holoscan::Tensor>("transform");
  const auto stats = result.snapshots<holoscan::Tensor>("stats");
  VERIFY(poses.size() == kFrames && transforms.size() == kFrames && stats.size() == kFrames);
  for (std::uint32_t frame = 0; frame < poses.size() && frame < stats.size(); ++frame) {
    const Matrix4 pose = matrix(poses[frame]);
    const IcpResult r = result_from(doubles(stats[frame]), matrix(transforms[frame]));
    const std::string what = "frame " + std::to_string(frame);
    check_time(poses[frame], frame, "pose");
    if (frame == 0U) {
      VERIFY_MSG(pose == identity_transform() && r.iterations == 0 && !r.converged,
                 "the first frame is the origin, with nothing to register to");
      continue;
    }
    VERIFY_MSG(r.converged, what + " converged");
    const TransformError error = transform_error(pose, synthetic_trajectory(frame, kPath));
    VERIFY_MSG(error.translation < 0.01 && degrees(error.rotation_rad) < 0.2,
               what + ": pose error " + std::to_string(error.translation * 1000.0) + " mm");
  }
}

// A source with no overlap still gives a transform and statistics, which say
// the registration failed.
void check_failure_reported() {
  IcpConfig config = base_config(IcpMode::kPairs);
  config.options.max_correspondence_distance = 0.05F;
  config.options.min_correspondence_distance = 0.0F;
  auto h = holoscan::testing::create_operator_test<IcpOp>(config);
  // Every source point lies 20 m farther along the camera axis than any
  // target point.
  std::vector<float> xyz =
      test::render(synthetic_scene(SceneKind::kRoom), identity_transform(), kWidth, kHeight, kNoise)
          .to_host();
  const holoscan::Tensor target =
      make({kHeight, kWidth, 3}, kFloat32, xyz.data(), xyz.size() * sizeof(float), false);
  for (std::size_t i = 2; i < xyz.size(); i += 3) xyz[i] += 20.0F;
  const holoscan::Tensor source =
      make({kHeight * kWidth, 3}, kFloat32, xyz.data(), xyz.size() * sizeof(float), false);
  h.push_frame(holoscan::testing::frame_at(capture_time(0),
                                           holoscan::testing::input("source", source),
                                           holoscan::testing::input("target", target)));
  h.capture_snapshot<holoscan::Tensor>("transform");
  h.capture_snapshot<holoscan::Tensor>("stats");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto transforms = result.snapshots<holoscan::Tensor>("transform");
  const auto stats = result.snapshots<holoscan::Tensor>("stats");
  VERIFY(transforms.size() == 1U && stats.size() == 1U);
  if (stats.size() != 1U || transforms.size() != 1U) return;
  const IcpResult r = result_from(doubles(stats[0]), matrix(transforms[0]));
  VERIFY(!r.converged && r.status == IcpStatus::kTooFewCorrespondences);
  VERIFY(r.fitness == 0.0 && r.correspondences == 0U && r.iterations == 0);
  VERIFY(matrix(transforms[0]) == identity_transform());
}

// Runs the operator on one unusable input and checks the failure names `what`.
void check_rejects(const IcpConfig& config, holoscan::Tensor source,
                   std::optional<holoscan::Tensor> guess, const std::string& what) {
  auto h = holoscan::testing::create_operator_test<IcpOp>(config);
  const std::vector<float> xyz(12, 1.0F);
  const holoscan::Tensor target = make({4, 3}, kFloat32, xyz.data(), 48, false);
  if (guess) {
    h.push_frame(holoscan::testing::frame_at(100ms, holoscan::testing::input("source", source),
                                             holoscan::testing::input("target", target),
                                             holoscan::testing::input("initial_guess", *guess)));
  } else {
    h.push_frame(holoscan::testing::frame_at(100ms, holoscan::testing::input("source", source),
                                             holoscan::testing::input("target", target)));
  }
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), what + " is rejected");
  VERIFY_MSG(result.describe_failure().find(what) != std::string::npos,
             "failure names '" + what + "': " + result.describe_failure());
}

void check_rejected_inputs() {
  const IcpConfig config = base_config(IcpMode::kPairs);
  const std::vector<float> xyz(12, 1.0F);
  const std::vector<double> xyz64(12, 1.0);
  const char* kShape = "source must be float32 [N, 3] or [H, W, 3]";
  check_rejects(config, make({4, 3}, kFloat64, xyz64.data(), 96, false), {}, kShape);
  check_rejects(config, make({3, 4}, kFloat32, xyz.data(), 48, false), {}, kShape);
  check_rejects(config, make({12}, kFloat32, xyz.data(), 48, false), {}, kShape);
  IcpConfig small = config;
  small.max_points = 3;
  check_rejects(small, make({4, 3}, kFloat32, xyz.data(), 48, false), {},
                "more than the configured maximum of 3");
  IcpConfig guessed = config;
  guessed.initial_guess_input = true;
  const holoscan::Tensor points = make({4, 3}, kFloat32, xyz.data(), 48, false);
  const std::vector<double> nine(9, 0.0);
  check_rejects(guessed, points, make({3, 3}, kFloat64, nine.data(), 72, false),
                "initial_guess must be float32 or float64 [4, 4]");
  Matrix4 scaled = identity_transform();
  scaled[0] = 2.0;
  check_rejects(guessed, points, matrix_tensor(scaled, false, false),
                "initial_guess is not a rigid transform");
}

void check_rejected_config() {
  const auto rejects = [](auto change) {
    IcpConfig config = base_config(IcpMode::kPairs);
    change(config);
    return test::throws<std::invalid_argument>([&] { IcpOp op(config); });
  };
  VERIFY(!rejects([](IcpConfig&) {}));
  VERIFY(rejects([](IcpConfig& c) { c.max_points = 0; }));
  VERIFY(rejects([](IcpConfig& c) { c.max_points = kMaxPointsLimit + 1U; }));
  VERIFY(rejects([](IcpConfig& c) { c.map_max_age = std::chrono::nanoseconds::zero(); }));
  VERIFY(rejects([](IcpConfig& c) { c.options.max_correspondence_distance = -1.0F; }));
  VERIFY(rejects([](IcpConfig& c) { c.mode = static_cast<IcpMode>(7); }));
  VERIFY(rejects([](IcpConfig& c) {
    c.transform_output = false;
    c.stats_output = false;
  }));
  // A pose is an output only in kOdometry.
  VERIFY(rejects([](IcpConfig& c) {
    c.transform_output = false;
    c.stats_output = false;
    c.pose_output = true;
  }));
  // The statistics decode only when complete and their status is known.
  std::vector<double> stats(kStatsSize, 0.0);
  VERIFY(!test::throws<std::invalid_argument>(
      [&] { (void)result_from(stats, identity_transform()); }));
  stats[kStatStatus] = 9.0;
  VERIFY(
      test::throws<std::invalid_argument>([&] { (void)result_from(stats, identity_transform()); }));
  VERIFY(test::throws<std::invalid_argument>(
      [&] { (void)result_from(std::vector<double>(3, 0.0), identity_transform()); }));
}

}  // namespace
}  // namespace holoscan::icp

int main() {
  namespace icp = holoscan::icp;
  return icp::test::run({
      {"pairs, device memory", [] { icp::check_pairs(true, false, false); }},
      {"pairs, host memory, aligned output", [] { icp::check_pairs(false, false, true); }},
      {"pairs with float32 initial guesses, device memory, aligned output",
       [] { icp::check_pairs(true, true, true, true); }},
      {"pairs with float64 initial guesses, host memory",
       [] { icp::check_pairs(false, true, false); }},
      {"stored target", icp::check_map},
      {"odometry from host memory", icp::check_odometry},
      {"a failed registration is reported", icp::check_failure_reported},
      {"unusable inputs are rejected", icp::check_rejected_inputs},
      {"unusable configurations are rejected", icp::check_rejected_config},
  });
}
