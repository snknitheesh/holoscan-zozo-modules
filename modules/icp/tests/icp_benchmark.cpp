// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Times depth odometry per frame at 320x240, 640x480 and 1280x720: clouds of
// the synthetic room along its camera path, registered frame to frame with
// point-to-plane ICP at the demo's settings, or with another voxel size (and
// the normal radius and finest correspondence distance scaled with it). The time is that of
// IcpOdometry::track(): subsampling, normals, grids and every iteration,
// until the result is on the host. Also prints the drift. Fails only if a
// frame fails to register.
//
// Usage: icp_benchmark [frames per resolution, default 150] [voxel size in m, default 0.02]

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <holoscan/icp/registration.hpp>
#include <holoscan/icp/synthetic_scene.hpp>

#include "cloud_support.hpp"

namespace {

using holoscan::icp::compose;
using holoscan::icp::IcpOdometry;
using holoscan::icp::IcpOptions;
using holoscan::icp::Matrix4;
using holoscan::icp::OdometryStep;
using holoscan::icp::rigid_inverse;
using holoscan::icp::SceneKind;
using holoscan::icp::synthetic_scene;
using holoscan::icp::synthetic_trajectory;
using holoscan::icp::transform_error;
using holoscan::icp::translation_norm;

struct Resolution {
  int width;
  int height;
};

double percentile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1U));
  return values[index];
}

}  // namespace

int main(int argc, char** argv) try {
  const std::uint32_t frames = argc > 1 ? static_cast<std::uint32_t>(std::stoul(argv[1])) : 150U;
  const float voxel = argc > 2 ? std::stof(argv[2]) : 0.02F;
  if (frames < 2U || !(voxel > 0.0F && voxel <= 0.05F)) {
    std::cerr << "icp_benchmark: needs at least 2 frames and a voxel of up to 5 cm\n";
    return 2;
  }
  cudaDeviceProp properties{};
  holoscan::icp::test::cuda_check(cudaGetDeviceProperties(&properties, 0), "device");
  std::cout << "icp_benchmark: GPU " << properties.name << ", " << frames
            << " frames per resolution, point-to-plane, voxel " << voxel * 100.0F
            << " cm, correspondences 10 cm to " << std::max(3.0F, 150.0F * voxel)
            << " cm, normals within 3 voxels\n"
            << std::fixed;
  // The demo's settings.
  IcpOptions options;
  options.max_correspondence_distance = 0.10F;
  options.voxel_size = voxel;
  options.normal_radius = 3.0F * voxel;
  options.min_correspondence_distance = std::max(0.03F, 1.5F * voxel);
  options.max_iterations = 100;
  const holoscan::icp::SyntheticScene scene = synthetic_scene(SceneKind::kRoom);
  const holoscan::icp::DepthNoise noise{.sigma = 0.0005F, .sigma_per_m2 = 0.0005F, .seed = 0U};
  bool failed = false;
  for (const Resolution r : {Resolution{320, 240}, Resolution{640, 480}, Resolution{1280, 720}}) {
    IcpOdometry odometry(options);
    std::vector<double> times;
    double path = 0.0;
    std::uint64_t iterations = 0;
    std::uint64_t points = 0;
    std::uint32_t unregistered = 0;
    OdometryStep step;
    for (std::uint32_t frame = 0; frame <= frames; ++frame) {
      // Rendered first, so the timing covers only the registration.
      holoscan::icp::DepthNoise n = noise;
      n.seed = frame;
      const holoscan::icp::test::DeviceCloud cloud = holoscan::icp::test::render(
          scene, synthetic_trajectory(frame, frames), r.width, r.height, n);
      const auto start = std::chrono::steady_clock::now();
      holoscan::icp::test::cuda_check(odometry.track(cloud.data(), cloud.count(), nullptr, step),
                                      "track");
      const auto end = std::chrono::steady_clock::now();
      if (frame == 0U) continue;  // The first frame only becomes the target.
      times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
      path += translation_norm(compose(rigid_inverse(synthetic_trajectory(frame - 1U, frames)),
                                       synthetic_trajectory(frame, frames)));
      iterations += static_cast<std::uint64_t>(step.result.iterations);
      points += step.result.source_points;
      unregistered += step.result.converged ? 0U : 1U;
    }
    const auto error = transform_error(step.pose, synthetic_trajectory(frames, frames));
    double mean = 0.0;
    for (const double t : times) mean += t;
    mean /= static_cast<double>(times.size());
    std::cout << std::setprecision(2) << "icp_benchmark: " << r.width << 'x' << r.height
              << " ms_per_frame mean=" << mean << " median=" << percentile(times, 0.5)
              << " p95=" << percentile(times, 0.95) << " max=" << percentile(times, 1.0)
              << std::setprecision(1) << " iterations_per_frame="
              << static_cast<double>(iterations) / static_cast<double>(times.size())
              << " points_per_frame="
              << static_cast<double>(points) / static_cast<double>(times.size())
              << std::setprecision(2) << " drift_percent=" << 100.0 * error.translation / path
              << " unconverged=" << unregistered << '\n';
    failed = failed || unregistered > frames / 20U;
  }
  std::cout << "icp_benchmark: " << (failed ? "FAIL" : "PASS") << '\n';
  return failed ? 1 : 0;
} catch (const std::exception& error) {
  std::cerr << "icp_benchmark: " << error.what() << '\n';
  return 1;
}
