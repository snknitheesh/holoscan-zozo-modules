// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Times a map update (HeightMapper::update() and write_layers() of the
// elevation and variance, as HeightMapOp runs them) and the traversability on
// depth-camera clouds of the synthetic course, for each input size and map
// size. Prints a Markdown table of the mean and the 95th percentile over the
// iterations, in milliseconds of GPU time on one stream.
//
//   height_map_benchmark [--iterations N] [--size WxH]... [--map N]...

#include <cuda_runtime_api.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/height_map/height_mapper.hpp>
#include <holoscan/height_map/synthetic_terrain.hpp>
#include <holoscan/height_map/traversability.hpp>
#include <holoscan/height_map/views.hpp>

namespace {

namespace hm = holoscan::height_map;

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

struct Timing {
  double mean{};
  double p95{};
};

Timing time_it(const std::function<void(int)>& step, int iterations, cudaStream_t stream) {
  for (int i = 0; i < 3; ++i) {
    step(i);
  }
  check(cudaStreamSynchronize(stream), "warm-up");
  cudaEvent_t start{};
  cudaEvent_t stop{};
  check(cudaEventCreate(&start), "cudaEventCreate");
  check(cudaEventCreate(&stop), "cudaEventCreate");
  std::vector<double> times;
  for (int i = 0; i < iterations; ++i) {
    check(cudaEventRecord(start, stream), "cudaEventRecord");
    step(3 + i);
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

void run(int width, int height, int cells, int iterations) {
  const hm::Terrain terrain = hm::course_terrain();
  hm::TerrainCamera camera;
  camera.width = width;
  camera.height = height;
  camera.fx = camera.fy = 0.7F * static_cast<float>(width);
  camera.cx = 0.5F * static_cast<float>(width - 1);
  camera.cy = 0.5F * static_cast<float>(height - 1);
  camera.pose = hm::camera_pose(0.5, 0.0, 1.0, 0.0, 0.6);
  camera.noise_base = 0.002F;
  const auto n = static_cast<std::uint32_t>(width) * static_cast<std::uint32_t>(height);
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
  float* depth = nullptr;
  float* xyz = nullptr;
  check(cudaMalloc(&depth, 4U * n), "cudaMalloc");
  check(cudaMalloc(&xyz, 12U * n), "cudaMalloc");
  check(hm::render_terrain_depth(terrain, camera, depth, xyz, stream), "render");

  // A 10 m square map either way.
  hm::HeightMapParams params;
  params.geometry = hm::MapGeometry{.resolution = 10.0F / static_cast<float>(cells),
                                    .rows = cells,
                                    .cols = cells,
                                    .origin_x = -5.0,
                                    .origin_y = -5.0};
  params.mode = hm::MapMode::kFollow;
  const std::size_t map_cells = params.geometry.cells();
  float* layers = nullptr;
  std::uint8_t* rgb = nullptr;
  std::uint8_t* unknown = nullptr;
  float* range = nullptr;
  check(cudaMalloc(&layers, 6U * 4U * map_cells), "cudaMalloc");
  check(cudaMalloc(&rgb, 3U * map_cells), "cudaMalloc");
  check(cudaMalloc(&unknown, map_cells), "cudaMalloc");
  check(cudaMalloc(&range, 2U * sizeof(float)), "cudaMalloc");
  const hm::MapLayers map_layers{.elevation = layers, .variance = layers + map_cells};
  const hm::TraversabilityLayers cost_layers{.cost = layers + 2 * map_cells,
                                             .unknown = unknown,
                                             .slope = layers + 3 * map_cells,
                                             .step_height = layers + 4 * map_cells,
                                             .roughness = layers + 5 * map_cells};

  std::vector<std::pair<std::string, std::function<void(int)>>> steps;
  for (const hm::ElevationMode mode : {hm::ElevationMode::kFused, hm::ElevationMode::kMax}) {
    params.elevation = mode;
    auto mapper = std::make_shared<hm::HeightMapper>(params);
    steps.emplace_back(
        std::string("Map update (") + (mode == hm::ElevationMode::kFused ? "fused" : "max") +
            ", moving 1 cm a frame)",
        [=](int i) {
          // The sensor moves, so the window shifts every few frames.
          hm::Transform pose = camera.pose;
          pose.m[3] += 0.01 * i;
          check(mapper->update(xyz, n, pose, 33'000'000LL * (i + 1), stream), "update");
          check(mapper->write_layers(map_layers, stream), "write_layers");
        });
  }
  const hm::TraversabilityParams traversability;
  steps.emplace_back("Traversability (5x5 plane fit)", [=](int) {
    check(hm::compute_traversability(layers, cells, cells, params.geometry.resolution,
                                     traversability, cost_layers, stream),
          "traversability");
  });
  steps.emplace_back("Elevation and cost images", [=](int) {
    check(hm::colorize_elevation(layers, cells, cells, 0.0F, 0.0F, range, rgb, stream), "colorize");
    check(hm::colorize_cost(layers + 2 * map_cells, cells, cells, rgb, stream), "colorize");
  });
  std::printf("| %dx%d input, %dx%d map | mean ms | p95 ms |\n| --- | --- | --- |\n", width, height,
              cells, cells);
  for (const auto& [name, step] : steps) {
    const Timing t = time_it(step, iterations, stream);
    std::printf("| %s | %.3f | %.3f |\n", name.c_str(), t.mean, t.p95);
  }
  // How much of the map the view fills, for the record.
  std::vector<float> elevation(map_cells);
  check(cudaMemcpyAsync(elevation.data(), layers, 4U * map_cells, cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync");
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  const auto known =
      std::count_if(elevation.begin(), elevation.end(), [](float v) { return std::isfinite(v); });
  std::printf("\n%u points, %zu of %zu cells known\n\n", n, static_cast<std::size_t>(known),
              map_cells);
  std::fflush(stdout);

  cudaFree(depth);
  cudaFree(xyz);
  cudaFree(layers);
  cudaFree(rgb);
  cudaFree(unknown);
  cudaFree(range);
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
  std::vector<int> maps;
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
    } else if (option == "--map") {
      maps.push_back(parse(value));
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  if (sizes.empty()) {
    sizes = {{640, 480}, {1280, 720}};
  }
  if (maps.empty()) {
    maps = {200, 500};
  }
  for (const int cells : maps) {
    for (const auto& [width, height] : sizes) {
      run(width, height, cells, iterations);
    }
  }
  return 0;
} catch (const std::exception& error) {
  std::cerr << "height_map_benchmark: " << error.what() << '\n';
  return 1;
}
