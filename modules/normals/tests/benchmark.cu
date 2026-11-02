// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Times normal estimation (each search) and plane segmentation on clouds of
// the synthetic scene, as the operators run them in the kKeep layout. Prints
// a Markdown table of the mean and the 95th percentile over the iterations,
// in milliseconds of GPU time on one stream.
//
//   normals_benchmark [--iterations N] [--size WxH]...

#include <cuda_runtime_api.h>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/normals/normals.hpp>
#include <holoscan/normals/planes.hpp>
#include <holoscan/normals/synthetic_scene.hpp>

namespace {

namespace hn = holoscan::normals;

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

struct Timing {
  double mean{};
  double p95{};
};

Timing time_it(const std::function<void()>& step, int iterations, cudaStream_t stream) {
  for (int i = 0; i < 3; ++i) {
    step();
  }
  check(cudaStreamSynchronize(stream), "warm-up");
  cudaEvent_t start{};
  cudaEvent_t stop{};
  check(cudaEventCreate(&start), "cudaEventCreate");
  check(cudaEventCreate(&stop), "cudaEventCreate");
  std::vector<double> times;
  for (int i = 0; i < iterations; ++i) {
    check(cudaEventRecord(start, stream), "cudaEventRecord");
    step();
    check(cudaEventRecord(stop, stream), "cudaEventRecord");
    check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float ms = 0.0F;
    check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    times.push_back(ms);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::sort(times.begin(), times.end());
  Timing timing;
  for (const double t : times) {
    timing.mean += t / static_cast<double>(times.size());
  }
  timing.p95 = times[std::min(times.size() - 1, static_cast<std::size_t>(0.95 * times.size()))];
  return timing;
}

void run(int width, int height, int iterations) {
  const hn::SyntheticScene scene = hn::synthetic_scene(width, height, 0);
  const auto n = static_cast<std::uint32_t>(width) * static_cast<std::uint32_t>(height);
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
  float* xyz = nullptr;
  float* normals = nullptr;
  float* curvature = nullptr;
  std::uint8_t* labels = nullptr;
  float* planes = nullptr;
  std::uint32_t* inliers = nullptr;
  check(cudaMalloc(&xyz, 12U * n), "cudaMalloc");
  check(cudaMalloc(&normals, 12U * n), "cudaMalloc");
  check(cudaMalloc(&curvature, 4U * n), "cudaMalloc");
  check(cudaMalloc(&labels, n), "cudaMalloc");
  check(cudaMalloc(&planes, 7U * hn::kMaxPlanes * sizeof(float)), "cudaMalloc");
  check(cudaMalloc(&inliers, hn::kMaxPlanes * sizeof(std::uint32_t)), "cudaMalloc");
  check(hn::render_synthetic_scene(scene, nullptr, nullptr, xyz, stream), "render");

  hn::NormalEstimator estimator;
  hn::PlaneSegmenter segmenter;
  const auto normals_step = [&](hn::NormalParams params) {
    return [&, params] {
      check(estimator.estimate(xyz, n, width, height, params, normals, curvature, stream),
            "estimate");
    };
  };
  hn::NormalParams window3;
  window3.search = hn::NeighborSearch::kWindow;
  hn::NormalParams window5 = window3;
  window5.window_radius = 5;
  hn::NormalParams knn;
  knn.search = hn::NeighborSearch::kKnn;
  knn.radius = 0.1F;
  hn::NormalParams radius;
  radius.search = hn::NeighborSearch::kRadius;
  radius.radius = 0.04F;
  const hn::PlaneParams plane_params;
  const hn::PlaneOutputs outputs{labels, planes, planes + 4 * hn::kMaxPlanes, inliers};
  const auto planes_step = [&] {
    check(segmenter.segment(xyz, normals, n, plane_params, outputs, stream), "segment");
  };

  std::vector<std::pair<std::string, std::function<void()>>> steps{
      {"NormalEstimationOp (window 7x7)", normals_step(window3)},
      {"NormalEstimationOp (window 11x11)", normals_step(window5)},
      {"NormalEstimationOp (k-nearest 24, 10 cm)", normals_step(knn)},
      {"NormalEstimationOp (radius 4 cm)", normals_step(radius)},
  };
  std::printf("| %dx%d | mean ms | p95 ms |\n| --- | --- | --- |\n", width, height);
  for (const auto& [name, step] : steps) {
    const Timing t = time_it(step, iterations, stream);
    std::printf("| %s | %.3f | %.3f |\n", name.c_str(), t.mean, t.p95);
  }
  // The planes of the window normals.
  normals_step(window3)();
  const Timing t = time_it(planes_step, iterations, stream);
  std::printf("| PlaneSegmentationOp (8 planes, 256 hypotheses) | %.3f | %.3f |\n", t.mean, t.p95);
  const Timing both = time_it(
      [&] {
        normals_step(window3)();
        planes_step();
      },
      iterations, stream);
  std::printf("| Both (window 7x7, then planes) | %.3f | %.3f |\n\n", both.mean, both.p95);
  std::fflush(stdout);

  cudaFree(xyz);
  cudaFree(normals);
  cudaFree(curvature);
  cudaFree(labels);
  cudaFree(planes);
  cudaFree(inliers);
  cudaStreamDestroy(stream);
}

int parse(std::string_view text) {
  int value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc() || end != text.data() + text.size() || value <= 0) {
    throw std::invalid_argument("not a positive number: " + std::string(text));
  }
  return value;
}

}  // namespace

int main(int argc, char** argv) try {
  int iterations = 100;
  std::vector<std::pair<int, int>> sizes;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string_view option{argv[i]};
    const std::string_view value{argv[i + 1]};
    if (option == "--iterations") {
      iterations = parse(value);
    } else if (option == "--size") {
      const std::size_t x = value.find('x');
      if (x == std::string_view::npos) {
        throw std::invalid_argument("--size takes WxH");
      }
      sizes.emplace_back(parse(value.substr(0, x)), parse(value.substr(x + 1)));
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  if (sizes.empty()) {
    sizes = {{640, 480}, {1280, 720}};
  }
  for (const auto& [width, height] : sizes) {
    run(width, height, iterations);
  }
  return 0;
} catch (const std::exception& error) {
  std::cerr << "normals_benchmark: " << error.what() << '\n';
  return 1;
}
