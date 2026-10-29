// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Times each filter on organized clouds of the synthetic scene, as its
// operator runs it: the filter, then the organized output (apply_mask) or the
// dense one (count, wait for it, pack). Prints a Markdown table of the mean
// and the 95th percentile over the iterations, in milliseconds of GPU time
// on one stream, waits included.
//
//   pcl_filters_benchmark [--iterations N] [--size WxH]...

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

#include <holoscan/pcl_filters/filters.hpp>
#include <holoscan/pcl_filters/synthetic_scene.hpp>

namespace {

namespace pf = holoscan::pcl_filters;

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

struct Timing {
  double mean{};
  double p95{};
};

// Runs `step` `iterations` times after a warm-up, timing each on `stream`
// with the host waits it makes.
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
  const pf::SyntheticScene scene = pf::synthetic_scene(width, height, 0);
  const auto n = static_cast<std::uint32_t>(width) * static_cast<std::uint32_t>(height);
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
  float* xyz = nullptr;
  std::uint8_t* rgb = nullptr;
  float* out_xyz = nullptr;
  std::uint8_t* out_rgb = nullptr;
  std::uint8_t* keep = nullptr;
  float* plane = nullptr;
  std::uint32_t* count = nullptr;
  check(cudaMalloc(&xyz, n * 12U), "cudaMalloc");
  check(cudaMalloc(&rgb, n * 3U), "cudaMalloc");
  check(cudaMalloc(&out_xyz, n * 12U), "cudaMalloc");
  check(cudaMalloc(&out_rgb, n * 3U), "cudaMalloc");
  check(cudaMalloc(&keep, n), "cudaMalloc");
  check(cudaMalloc(&plane, 16), "cudaMalloc");
  check(cudaMallocHost(&count, 4), "cudaMallocHost");
  check(pf::render_synthetic_scene(scene, nullptr, rgb, xyz, stream), "render");

  pf::Compactor compactor;
  const auto organized = [&] {
    check(pf::apply_mask(xyz, rgb, keep, n, out_xyz, out_rgb, stream), "apply_mask");
  };
  const auto dense = [&] {
    check(compactor.count(keep, n, stream), "count");
    check(cudaMemcpyAsync(count, compactor.device_count(), 4, cudaMemcpyDeviceToHost, stream),
          "cudaMemcpyAsync");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    check(compactor.write(xyz, rgb, out_xyz, out_rgb, stream), "write");
  };

  pf::CropBoxParams box;
  box.min[2] = 1.0F;
  box.max[2] = 5.0F;
  box.max_distance = 6.0F;
  pf::VoxelGrid grid;
  pf::GroundSegmenter segmenter;
  float truth[4];
  pf::synthetic_ground_plane(scene, truth);
  pf::GroundParams ground;
  std::copy(truth, truth + 3, ground.up);
  ground.distance_threshold = 0.02F;
  ground.max_iterations = 256;
  pf::OutlierFilter outliers;
  pf::OutlierParams statistical{.k = 8, .std_multiplier = 2.0F, .search_radius = 0.1F};
  pf::OutlierParams radius{.mode = pf::OutlierMode::kRadius, .radius = 0.05F, .min_neighbors = 3};

  struct Case {
    std::string name;
    std::function<void()> step;
  };
  const std::vector<Case> cases{
      {"CropBoxOp (organized)",
       [&] {
         check(pf::launch_crop_box(xyz, n, box, keep, stream), "crop");
         organized();
       }},
      {"CropBoxOp (dense)",
       [&] {
         check(pf::launch_crop_box(xyz, n, box, keep, stream), "crop");
         dense();
       }},
      {"VoxelDownsampleOp (2 cm)",
       [&] {
         check(grid.group(xyz, n, 0.02F, stream), "group");
         check(cudaMemcpyAsync(count, grid.device_count(), 4, cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync");
         check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
         check(grid.write(xyz, rgb, pf::VoxelMode::kCentroid, out_xyz, out_rgb, stream), "write");
       }},
      {"VoxelDownsampleOp (5 cm)",
       [&] {
         check(grid.group(xyz, n, 0.05F, stream), "group");
         check(cudaMemcpyAsync(count, grid.device_count(), 4, cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync");
         check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
         check(grid.write(xyz, rgb, pf::VoxelMode::kCentroid, out_xyz, out_rgb, stream), "write");
       }},
      {"GroundRemovalOp (RANSAC, 256 iterations)",
       [&] {
         check(segmenter.segment(xyz, n, ground, plane, keep, nullptr, stream), "ground");
         organized();
       }},
      {"OutlierRemovalOp (statistical, k 8, 10 cm)",
       [&] {
         check(outliers.filter(xyz, n, statistical, keep, stream), "statistical");
         organized();
       }},
      {"OutlierRemovalOp (radius, 5 cm, 3)",
       [&] {
         check(outliers.filter(xyz, n, radius, keep, stream), "radius");
         organized();
       }},
  };
  std::cout << "\n"
            << width << "x" << height << " (" << n << " points), " << iterations
            << " iterations\n\n| Filter | Mean ms | p95 ms |\n| --- | --- | --- |\n";
  for (const Case& c : cases) {
    const Timing timing = time_it(c.step, iterations, stream);
    char line[160];
    std::snprintf(line, sizeof(line), "| %s | %.3f | %.3f |\n", c.name.c_str(), timing.mean,
                  timing.p95);
    std::cout << line;
  }
  cudaFree(xyz);
  cudaFree(rgb);
  cudaFree(out_xyz);
  cudaFree(out_rgb);
  cudaFree(keep);
  cudaFree(plane);
  cudaFreeHost(count);
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
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string_view option{argv[i]};
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
  std::cout << "pcl_filters_benchmark on " << properties.name << '\n';
  for (const auto& [width, height] : sizes) {
    run(width, height, iterations);
  }
  return 0;
} catch (const std::exception& error) {
  std::cerr << "pcl_filters_benchmark: " << error.what() << '\n';
  return 1;
}
