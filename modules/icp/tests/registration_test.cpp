// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// IcpRegistration and IcpOdometry on synthetic scenes with known poses: rigid
// transforms from small to large recovered by both methods on three scenes,
// with depth noise and the partial overlap of two viewpoints; robust kernels
// against outliers; failures reported as such; bit-identical repeated runs; a
// CPU reference on a small cloud; and the drift of odometry along a camera
// path.

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/icp/registration.hpp>
#include <holoscan/icp/synthetic_scene.hpp>

#include "cloud_support.hpp"
#include "cpu_reference.hpp"
#include "test_support.hpp"

namespace holoscan::icp {
namespace {

using test::degrees;
using test::DeviceCloud;
using test::radians;
using test::render;

constexpr int kWidth = 320;
constexpr int kHeight = 240;
// A time-of-flight camera's noise: 0.5 mm plus 0.5 mm per square meter of
// depth, so 2.5 mm at 2 m and 8.5 mm at 4 m.
constexpr DepthNoise kNoise{.sigma = 0.0005F, .sigma_per_m2 = 0.0005F, .seed = 1U};

const char* name(SceneKind kind) {
  switch (kind) {
    case SceneKind::kRoom:
      return "room";
    case SceneKind::kObjects:
      return "objects";
    case SceneKind::kCorner:
      return "corner";
  }
  return "?";
}

const char* name(IcpMethod method) {
  return method == IcpMethod::kPointToPlane ? "point-to-plane" : "point-to-point";
}

// Where the target camera stands in each scene.
Matrix4 target_pose(SceneKind kind) {
  switch (kind) {
    case SceneKind::kRoom:
      return transform_from({0.0, radians(5.0), 0.0}, {0.2, 0.0, 0.4});
    case SceneKind::kObjects:
      return transform_from({radians(-12.0), 0.0, 0.0}, {0.0, -0.3, 0.0});
    case SceneKind::kCorner:
      return transform_from({0.0, radians(-10.0), 0.0}, {0.0, 0.0, 0.2});
  }
  return identity_transform();
}

struct Motion {
  const char* label;
  // The source camera's pose relative to the target's: the true transform.
  std::array<double, 3> axis_angle;
  std::array<double, 3> translation;
  // Where the registration starts, relative to the truth; none starts at
  // the identity.
  bool guess;
  float max_correspondence_distance;
};

// Small and medium motions start at the identity; the large one starts from
// a rough guess, 4 degrees and 8 cm from the truth.
const Motion kMotions[] = {
    {"small", {radians(1.0), radians(1.5), radians(-0.5)}, {0.03, -0.01, 0.02}, false, 0.10F},
    {"medium", {radians(-3.0), radians(6.0), radians(2.0)}, {-0.12, 0.05, 0.10}, false, 0.30F},
    {"large", {radians(10.0), radians(-25.0), radians(8.0)}, {0.40, -0.10, -0.35}, true, 0.15F},
};

// Coarse to fine down to 3 cm. Point-to-point converges linearly, so it
// gets more iterations.
IcpOptions options_for(IcpMethod method, float max_distance) {
  IcpOptions o;
  o.method = method;
  o.max_correspondence_distance = max_distance;
  o.min_correspondence_distance = std::fmin(0.03F, max_distance);
  o.max_iterations = method == IcpMethod::kPointToPlane ? 100 : 600;
  o.voxel_size = 0.02F;
  o.normal_radius = 0.06F;
  return o;
}

// The largest errors accepted: point-to-point lands farther from the truth
// on subsampled clouds, whose voxel centroids differ from view to view.
double max_rotation_deg(IcpMethod method) { return method == IcpMethod::kPointToPlane ? 0.1 : 0.3; }
double max_translation(IcpMethod method) {
  return method == IcpMethod::kPointToPlane ? 0.004 : 0.008;
}

IcpResult register_clouds(IcpRegistration& icp, const DeviceCloud& source,
                          const DeviceCloud& target, const Matrix4& initial) {
  IcpResult result;
  test::cuda_check(icp.set_target(target.data(), target.count(), nullptr), "set_target");
  test::cuda_check(icp.align(source.data(), source.count(), initial, nullptr, result), "align");
  return result;
}

void check_known_transforms() {
  for (const SceneKind kind : {SceneKind::kRoom, SceneKind::kObjects, SceneKind::kCorner}) {
    const SyntheticScene scene = synthetic_scene(kind);
    const Matrix4 a = target_pose(kind);
    const DeviceCloud target = render(scene, a, kWidth, kHeight, kNoise);
    for (const Motion& motion : kMotions) {
      const Matrix4 truth = transform_from(motion.axis_angle, motion.translation);
      DepthNoise noise = kNoise;
      noise.seed = 2U;
      const DeviceCloud source = render(scene, compose(a, truth), kWidth, kHeight, noise);
      const Matrix4 initial =
          motion.guess ? compose(transform_from({radians(2.0), radians(-3.0), radians(1.5)},
                                                {0.05, 0.04, -0.045}),
                                 truth)
                       : identity_transform();
      for (const IcpMethod method : {IcpMethod::kPointToPlane, IcpMethod::kPointToPoint}) {
        IcpRegistration icp(options_for(method, motion.max_correspondence_distance));
        const IcpResult result = register_clouds(icp, source, target, initial);
        const TransformError error = transform_error(result.transform, truth);
        std::cout << "  " << name(kind) << ' ' << motion.label << ' ' << name(method)
                  << ": rotation_error_deg=" << degrees(error.rotation_rad)
                  << " translation_error_mm=" << error.translation * 1000.0
                  << " iterations=" << result.iterations << " fitness=" << result.fitness
                  << " rmse_mm=" << result.rmse * 1000.0 << " status=" << to_string(result.status)
                  << '\n';
        const std::string what = std::string(name(kind)) + ' ' + motion.label + ' ' + name(method);
        VERIFY_MSG(result.converged, what + " did not converge");
        VERIFY_MSG(degrees(error.rotation_rad) < max_rotation_deg(method),
                   what + " rotation error " + std::to_string(degrees(error.rotation_rad)));
        VERIFY_MSG(error.translation < max_translation(method),
                   what + " translation error " + std::to_string(error.translation));
        VERIFY_MSG(result.fitness > 0.4, what + " fitness " + std::to_string(result.fitness));
        VERIFY_MSG(result.rmse < 0.02, what + " rmse " + std::to_string(result.rmse));
      }
    }
  }
}

// Outliers: every 10th source point is moved 0.5 m to 1 m off its surface.
void check_robust_kernels() {
  const SyntheticScene scene = synthetic_scene(SceneKind::kRoom);
  const Matrix4 a = target_pose(SceneKind::kRoom);
  const Matrix4 truth = transform_from({radians(1.0), radians(-2.0), 0.0}, {0.04, 0.0, -0.03});
  const DeviceCloud target = render(scene, a, kWidth, kHeight, kNoise);
  std::vector<float> host = render(scene, compose(a, truth), kWidth, kHeight).to_host();
  for (std::size_t i = 0; i < host.size() / 3U; i += 10U) {
    host[3U * i + 2U] -= 0.5F + 0.5F * static_cast<float>(i % 7U) / 7.0F;
  }
  const DeviceCloud source(host);
  for (const RobustKernel kernel : {RobustKernel::kHuber, RobustKernel::kTukey}) {
    IcpOptions o = options_for(IcpMethod::kPointToPlane, 0.10F);
    o.robust_kernel = kernel;
    o.robust_scale = 0.05F;
    IcpRegistration icp(o);
    const IcpResult result = register_clouds(icp, source, target, identity_transform());
    const TransformError error = transform_error(result.transform, truth);
    const std::string what = kernel == RobustKernel::kHuber ? "Huber" : "Tukey";
    std::cout << "  " << what << ": rotation_error_deg=" << degrees(error.rotation_rad)
              << " translation_error_mm=" << error.translation * 1000.0
              << " fitness=" << result.fitness << '\n';
    VERIFY_MSG(result.converged, what + " did not converge");
    VERIFY_MSG(degrees(error.rotation_rad) < 0.1 && error.translation < 0.004,
               what + " missed the truth");
  }
}

void check_failures() {
  const SyntheticScene scene = synthetic_scene(SceneKind::kObjects);
  const DeviceCloud target = render(scene, identity_transform(), kWidth, kHeight, kNoise);

  // No overlap at all: the source starts 10 m away.
  {
    IcpRegistration icp(options_for(IcpMethod::kPointToPlane, 0.10F));
    const DeviceCloud source = render(scene, identity_transform(), kWidth, kHeight, kNoise);
    const Matrix4 far = transform_from({0.0, 0.0, 0.0}, {10.0, 0.0, 0.0});
    const IcpResult result = register_clouds(icp, source, target, far);
    VERIFY(!result.converged);
    VERIFY(result.status == IcpStatus::kTooFewCorrespondences);
    VERIFY(result.iterations == 0);
    VERIFY(result.fitness == 0.0 && result.correspondences == 0U);
    VERIFY(result.transform == far);
  }
  // Too few iterations for a medium motion.
  {
    IcpOptions o = options_for(IcpMethod::kPointToPoint, 0.30F);
    o.max_iterations = 2;
    IcpRegistration icp(o);
    const Matrix4 truth = transform_from({0.0, radians(6.0), 0.0}, {0.15, 0.0, 0.1});
    const DeviceCloud source = render(scene, truth, kWidth, kHeight, kNoise);
    const IcpResult result = register_clouds(icp, source, target, identity_transform());
    VERIFY(!result.converged);
    VERIFY(result.status == IcpStatus::kMaxIterations);
    VERIFY(result.iterations == 2);
    VERIFY(result.fitness > 0.0);
  }
  // No target yet.
  {
    IcpRegistration icp(options_for(IcpMethod::kPointToPlane, 0.10F));
    const DeviceCloud source = render(scene, identity_transform(), kWidth, kHeight, kNoise);
    IcpResult result;
    test::cuda_check(
        icp.align(source.data(), source.count(), identity_transform(), nullptr, result), "align");
    VERIFY(!result.converged && result.status == IcpStatus::kTooFewCorrespondences);
    VERIFY(result.source_points > 0U && result.target_points == 0U);
  }
  // An empty or all-NaN source.
  {
    IcpRegistration icp(options_for(IcpMethod::kPointToPlane, 0.10F));
    const std::vector<float> nans(300U, std::nanf(""));
    const DeviceCloud source(nans);
    const IcpResult result = register_clouds(icp, source, target, identity_transform());
    VERIFY(!result.converged && result.source_points == 0U && result.fitness == 0.0);
  }
  // A bad initial guess is refused.
  {
    IcpRegistration icp(options_for(IcpMethod::kPointToPlane, 0.10F));
    const DeviceCloud source = render(scene, identity_transform(), 32, 24);
    Matrix4 bad = identity_transform();
    bad[15] = 2.0;
    IcpResult result;
    VERIFY(icp.align(source.data(), source.count(), bad, nullptr, result) == cudaErrorInvalidValue);
  }
}

void check_determinism() {
  const SyntheticScene scene = synthetic_scene(SceneKind::kRoom);
  const Matrix4 a = target_pose(SceneKind::kRoom);
  const Matrix4 truth =
      transform_from({radians(-3.0), radians(6.0), radians(2.0)}, {-0.12, 0.05, 0.10});
  const DeviceCloud target = render(scene, a, kWidth, kHeight, kNoise);
  const DeviceCloud source = render(scene, compose(a, truth), kWidth, kHeight, kNoise);
  for (const IcpMethod method : {IcpMethod::kPointToPlane, IcpMethod::kPointToPoint}) {
    for (const float voxel : {0.0F, 0.02F}) {
      IcpOptions o = options_for(method, 0.30F);
      o.voxel_size = voxel;
      IcpRegistration first(o);
      const IcpResult a1 = register_clouds(first, source, target, identity_transform());
      // The same instance again, and a new one on another stream.
      const IcpResult a2 = register_clouds(first, source, target, identity_transform());
      cudaStream_t stream = nullptr;
      test::cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
      IcpRegistration second(o);
      IcpResult b;
      test::cuda_check(second.set_target(target.data(), target.count(), stream), "set_target");
      test::cuda_check(second.align(source.data(), source.count(), identity_transform(), stream, b),
                       "align");
      cudaStreamDestroy(stream);
      const auto same = [](const IcpResult& x, const IcpResult& y) {
        return std::memcmp(x.transform.data(), y.transform.data(), sizeof(Matrix4)) == 0 &&
               std::memcmp(&x.fitness, &y.fitness, sizeof(double)) == 0 &&
               std::memcmp(&x.rmse, &y.rmse, sizeof(double)) == 0 && x.iterations == y.iterations &&
               x.correspondences == y.correspondences && x.source_points == y.source_points &&
               x.target_points == y.target_points;
      };
      const std::string what =
          std::string(name(method)) + (voxel > 0.0F ? " voxel" : " full resolution");
      VERIFY_MSG(same(a1, a2), what + ": a second run on the same instance differs");
      VERIFY_MSG(same(a1, b), what + ": a run on another instance and stream differs");
    }
  }
}

// A small cloud, every point used: the GPU after a fixed number of iterations
// against the CPU reference after the same.
void check_cpu_reference() {
  const SyntheticScene scene = synthetic_scene(SceneKind::kCorner);
  const Matrix4 a = target_pose(SceneKind::kCorner);
  const Matrix4 truth =
      transform_from({radians(2.0), radians(-3.0), radians(1.0)}, {0.05, -0.02, 0.04});
  const DeviceCloud target = render(scene, a, 64, 48, kNoise);
  const DeviceCloud source = render(scene, compose(a, truth), 64, 48, kNoise);
  const std::vector<float> host_target = target.to_host();
  const std::vector<float> host_source = source.to_host();
  for (const IcpMethod method : {IcpMethod::kPointToPlane, IcpMethod::kPointToPoint}) {
    IcpOptions o;
    o.method = method;
    o.max_correspondence_distance = 0.25F;
    o.normal_radius = 0.3F;
    o.max_iterations = 8;
    o.translation_epsilon = 0.0;
    o.rotation_epsilon = 0.0;
    o.relative_rmse_epsilon = 0.0;
    IcpRegistration icp(o);
    const IcpResult gpu = register_clouds(icp, source, target, identity_transform());
    const test::ReferenceResult cpu =
        test::reference_icp(host_source, host_target, o, identity_transform(), gpu.iterations);
    const TransformError difference = transform_error(gpu.transform, cpu.transform);
    const TransformError error = transform_error(cpu.transform, truth);
    std::cout << "  " << name(method) << ": iterations=" << gpu.iterations
              << " gpu_vs_cpu_rotation_deg=" << degrees(difference.rotation_rad)
              << " gpu_vs_cpu_translation_mm=" << difference.translation * 1000.0
              << " correspondences gpu=" << gpu.correspondences << " cpu=" << cpu.correspondences
              << " rmse_mm gpu=" << gpu.rmse * 1000.0 << " cpu=" << cpu.rmse * 1000.0
              << " cpu_error_mm=" << error.translation * 1000.0 << '\n';
    const std::string what = name(method);
    VERIFY_MSG(gpu.iterations == 8, what + ": the GPU stopped early");
    VERIFY_MSG(degrees(difference.rotation_rad) < 0.01 && difference.translation < 1e-4,
               what + ": the GPU and the CPU reference disagree");
    VERIFY_MSG(gpu.correspondences + 2U >= cpu.correspondences &&
                   cpu.correspondences + 2U >= gpu.correspondences,
               what + ": the correspondence counts disagree");
    VERIFY_MSG(std::fabs(gpu.rmse - cpu.rmse) < 1e-5, what + ": the RMSEs disagree");
  }
}

void check_odometry() {
  constexpr std::uint32_t kFrames = 150;
  const SyntheticScene scene = synthetic_scene(SceneKind::kRoom);
  for (const IcpMethod method : {IcpMethod::kPointToPlane, IcpMethod::kPointToPoint}) {
    IcpOptions o = options_for(method, 0.10F);
    IcpOdometry odometry(o);
    double path = 0.0;
    double worst_step = 0.0;
    double sum_squared = 0.0;
    std::uint32_t unconverged = 0;
    Matrix4 last_truth = identity_transform();
    OdometryStep step;
    for (std::uint32_t frame = 0; frame <= kFrames; ++frame) {
      const Matrix4 truth = synthetic_trajectory(frame, kFrames);
      DepthNoise noise = kNoise;
      noise.seed = frame;
      const DeviceCloud cloud = render(scene, truth, kWidth, kHeight, noise);
      test::cuda_check(odometry.track(cloud.data(), cloud.count(), nullptr, step), "track");
      if (frame == 0U) {
        VERIFY(step.first && step.pose == identity_transform());
        continue;
      }
      path += translation_norm(compose(rigid_inverse(last_truth), truth));
      last_truth = truth;
      unconverged += step.result.converged ? 0U : 1U;
      const TransformError error = transform_error(step.pose, truth);
      worst_step = std::fmax(worst_step, error.translation);
      sum_squared += error.translation * error.translation;
    }
    const TransformError final_error = transform_error(step.pose, last_truth);
    const double ate = std::sqrt(sum_squared / kFrames);
    std::cout << "  " << name(method) << ": frames=" << kFrames << " path_m=" << path
              << " final_translation_error_mm=" << final_error.translation * 1000.0
              << " final_rotation_error_deg=" << degrees(final_error.rotation_rad)
              << " drift_percent=" << 100.0 * final_error.translation / path
              << " ate_rmse_mm=" << ate * 1000.0 << " max_error_mm=" << worst_step * 1000.0
              << " unconverged=" << unconverged << '\n';
    // Frame to frame, errors add up. Point-to-point drifts more: matching
    // samples to samples, it tends to hold back the motion.
    const bool plane = method == IcpMethod::kPointToPlane;
    const double max_drift = plane ? 0.015 : 0.06;
    const double max_rotation_drift = plane ? 1.0 : 4.0;
    const std::string what = name(method);
    VERIFY_MSG(unconverged == 0U, what + ": frames did not converge");
    VERIFY_MSG(final_error.translation < max_drift * path,
               what + ": drift above " + std::to_string(100.0 * max_drift) + "% of the path");
    VERIFY_MSG(degrees(final_error.rotation_rad) < max_rotation_drift,
               what + ": rotation drift above " + std::to_string(max_rotation_drift) + " degrees");
  }
}

void check_options() {
  const auto rejects = [](auto change) {
    IcpOptions o;
    change(o);
    return test::throws<std::invalid_argument>([&] { validate_options(o); });
  };
  VERIFY(!rejects([](IcpOptions&) {}));
  VERIFY(rejects([](IcpOptions& o) { o.max_correspondence_distance = 0.0F; }));
  VERIFY(rejects([](IcpOptions& o) { o.max_correspondence_distance = std::nanf(""); }));
  VERIFY(rejects([](IcpOptions& o) { o.max_iterations = 0; }));
  VERIFY(rejects([](IcpOptions& o) { o.voxel_size = -0.01F; }));
  VERIFY(rejects([](IcpOptions& o) { o.normal_radius = -1.0F; }));
  VERIFY(rejects([](IcpOptions& o) { o.translation_epsilon = -1.0; }));
  VERIFY(rejects([](IcpOptions& o) {
    o.robust_kernel = RobustKernel::kTukey;
    o.robust_scale = 0.0F;
  }));
  VERIFY(rejects([](IcpOptions& o) { o.min_correspondences = 5U; }));
  VERIFY(test::throws<std::invalid_argument>([] {
    IcpOptions o;
    o.max_iterations = 0;
    IcpRegistration icp(o);
  }));
}

}  // namespace
}  // namespace holoscan::icp

int main() {
  using holoscan::icp::test::TestCase;
  std::cout << std::fixed << std::setprecision(4);
  return holoscan::icp::test::run({
      {"known transforms, both methods, three scenes", holoscan::icp::check_known_transforms},
      {"robust kernels against outliers", holoscan::icp::check_robust_kernels},
      {"failures reported", holoscan::icp::check_failures},
      {"bit-identical repeated runs", holoscan::icp::check_determinism},
      {"CPU reference", holoscan::icp::check_cpu_reference},
      {"odometry drift along a camera path", holoscan::icp::check_odometry},
      {"options validated", holoscan::icp::check_options},
  });
}
