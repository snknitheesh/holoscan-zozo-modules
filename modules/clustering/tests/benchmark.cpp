// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Times the clusterer on organized clouds of the synthetic depth camera, and
// on a wall that fills the image (one cluster of every point), as
// EuclideanClusterOp runs it: cluster, then the labels and the summaries
// (kKeep), or the clustered count and the grouped points too (kDense).
// Prints a Markdown table of the mean and the 95th percentile over the
// iterations, in milliseconds of GPU time on one stream, waits included.
//
//   clustering_benchmark [--iterations N] [--size WxH]...

#include <cuda_runtime_api.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/clustering/clustering.hpp>
#include <holoscan/clustering/synthetic_scene.hpp>

namespace {

namespace hc = holoscan::clustering;

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

struct Timing {
  double mean{};
  double p95{};
};

// Runs `step` `iterations` times after a warm-up of at least 20 runs and
// 200 ms, which lets a Jetson's clocks settle, timing each on `stream` with
// the host waits it makes.
Timing time_it(const std::function<void()>& step, int iterations, cudaStream_t stream) {
  const auto warm = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
  for (int i = 0; i < 20 || std::chrono::steady_clock::now() < warm; ++i) {
    step();
    check(cudaStreamSynchronize(stream), "warm-up");
  }
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
  const hc::SyntheticScene scene = hc::synthetic_scene(width, height, 0);
  const auto n = static_cast<std::uint32_t>(width) * static_cast<std::uint32_t>(height);
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
  float* xyz = nullptr;
  float* out_xyz = nullptr;
  std::int32_t* labels = nullptr;
  float* summaries = nullptr;
  check(cudaMalloc(reinterpret_cast<void**>(&xyz), n * 12U), "cudaMalloc");
  check(cudaMalloc(reinterpret_cast<void**>(&out_xyz), n * 12U), "cudaMalloc");
  check(cudaMalloc(reinterpret_cast<void**>(&labels), n * 4U), "cudaMalloc");
  check(cudaMalloc(reinterpret_cast<void**>(&summaries), 1024U * sizeof(hc::ClusterSummary)),
        "cudaMalloc");
  check(hc::render_synthetic_scene(scene, nullptr, nullptr, xyz, stream), "render");
  check(cudaStreamSynchronize(stream), "render");
  // The worst case: a tilted wall filling the image at 1 to 1.5 m, one cluster of every point.
  float* wall = nullptr;
  check(cudaMalloc(reinterpret_cast<void**>(&wall), n * 12U), "cudaMalloc");
  {
    std::vector<float> host(3U * n);
    for (int v = 0; v < height; ++v) {
      for (int u = 0; u < width; ++u) {
        const float z = 1.0F + 0.5F * static_cast<float>(u) / static_cast<float>(width);
        float* p = host.data() + 3 * (static_cast<std::size_t>(v) * width + u);
        p[0] = (static_cast<float>(u) - scene.cx) * z / scene.fx;
        p[1] = (static_cast<float>(v) - scene.cy) * z / scene.fy;
        p[2] = z;
      }
    }
    check(cudaMemcpy(wall, host.data(), host.size() * 4U, cudaMemcpyHostToDevice), "cudaMemcpy");
  }

  hc::EuclideanClusterer clusterer;
  std::uint32_t found = 0;
  const auto keep = [&](const hc::ClusterParams& params, bool oriented, const float* cloud) {
    return [&, params, oriented, cloud] {
      check(clusterer.cluster(cloud, n, static_cast<std::uint32_t>(width), params, found, stream),
            "cluster");
      check(clusterer.write_labels(labels, stream), "write_labels");
      check(clusterer.write_summaries(cloud, oriented, summaries, stream), "write_summaries");
    };
  };
  const auto dense = [&](const hc::ClusterParams& params) {
    return [&, params] {
      check(clusterer.cluster(xyz, n, static_cast<std::uint32_t>(width), params, found, stream),
            "cluster");
      std::uint32_t clustered = 0;
      check(clusterer.clustered_count(clustered, stream), "clustered_count");
      check(clusterer.write_grouped(xyz, nullptr, out_xyz, nullptr, labels, stream),
            "write_grouped");
      check(clusterer.write_summaries(xyz, false, summaries, stream), "write_summaries");
    };
  };
  const auto params = [](float tolerance, hc::Connectivity connectivity, int window = 2) {
    return hc::ClusterParams{.tolerance = tolerance,
                             .min_cluster_size = 300,
                             .connectivity = connectivity,
                             .image_window = window};
  };
  using C = hc::Connectivity;
  struct Case {
    std::string name;
    std::function<void()> step;
  };
  const std::vector<Case> cases{
      {"Euclidean, 2.5 cm", keep(params(0.025F, C::kEuclidean), false, xyz)},
      {"Euclidean, 1 cm", keep(params(0.01F, C::kEuclidean), false, xyz)},
      {"Euclidean, 5 cm", keep(params(0.05F, C::kEuclidean), false, xyz)},
      {"Euclidean, 2.5 cm, oriented boxes", keep(params(0.025F, C::kEuclidean), true, xyz)},
      {"Euclidean, 2.5 cm, dense", dense(params(0.025F, C::kEuclidean))},
      {"Image window 2, 2.5 cm", keep(params(0.025F, C::kImage), false, xyz)},
      {"Image window 1, 2.5 cm", keep(params(0.025F, C::kImage, 1), false, xyz)},
      {"Image window 2, 2.5 cm, oriented boxes", keep(params(0.025F, C::kImage), true, xyz)},
      {"Wall, Euclidean, 2.5 cm, oriented boxes", keep(params(0.025F, C::kEuclidean), true, wall)},
      {"Wall, image window 2, 2.5 cm, oriented boxes", keep(params(0.025F, C::kImage), true, wall)},
  };
  std::cout << "\n"
            << width << "x" << height << " (" << n << " points), " << iterations
            << " iterations\n\n| Case | Clusters | Mean ms | p95 ms |\n| --- | --- | --- | --- |\n";
  for (const Case& c : cases) {
    const Timing timing = time_it(c.step, iterations, stream);
    char line[200];
    std::snprintf(line, sizeof(line), "| %s | %u | %.3f | %.3f |\n", c.name.c_str(), found,
                  timing.mean, timing.p95);
    std::cout << line;
  }
  cudaFree(xyz);
  cudaFree(wall);
  cudaFree(out_xyz);
  cudaFree(labels);
  cudaFree(summaries);
  cudaStreamDestroy(stream);
}

int parse_int(std::string_view text) {
  int value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc() || end != text.data() + text.size() || value < 1) {
    throw std::invalid_argument("not a positive number: " + std::string(text));
  }
  return value;
}

}  // namespace

int main(int argc, char** argv) try {
  int iterations = 100;
  std::vector<std::pair<int, int>> sizes;
  for (int i = 1; i < argc; i += 2) {
    const std::string_view option{argv[i]};
    if (i + 1 >= argc) {
      throw std::invalid_argument("missing value for " + std::string(option));
    }
    const std::string_view value{argv[i + 1]};
    if (option == "--iterations") {
      iterations = parse_int(value);
    } else if (option == "--size") {
      const std::size_t x = value.find('x');
      if (x == std::string_view::npos) {
        throw std::invalid_argument("--size takes WxH");
      }
      sizes.emplace_back(parse_int(value.substr(0, x)), parse_int(value.substr(x + 1)));
    } else {
      throw std::invalid_argument("unknown option " + std::string(option));
    }
  }
  if (sizes.empty()) {
    sizes = {{640, 480}, {1280, 720}};
  }
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
  std::cout << "clustering_benchmark on " << properties.name << '\n';
  for (const auto& [width, height] : sizes) {
    run(width, height, iterations);
  }
  return 0;
} catch (const std::exception& error) {
  std::cerr << "clustering_benchmark: " << error.what() << '\n';
  return 1;
}
