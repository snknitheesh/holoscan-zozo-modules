// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Times integration, ray casting and meshing on the GPU for 640x480 and
// 1280x720 depth images and 256^3 and 512^3 volumes of the tabletop scene,
// and reports the GPU memory each volume and its mesh scratch take. A volume
// that does not fit in the free GPU memory is skipped.
//
//   tsdf_benchmark [--iterations N] [--max-dims 256|512]

#include <cuda_runtime_api.h>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/tsdf/marching_cubes.hpp>
#include <holoscan/tsdf/synthetic_scene.hpp>
#include <holoscan/tsdf/tsdf_volume.hpp>

#include "scene_support.hpp"

namespace {

using holoscan::tsdf::test::cuda_check;
using holoscan::tsdf::test::Device;
namespace ht = holoscan::tsdf;

constexpr std::uint32_t kMaxVertices = 1U << 22;
constexpr std::uint32_t kMaxTriangles = 1U << 23;

double mib(std::uint64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

// The median time of `iterations` runs of `work` on `stream`, in ms.
template <typename Work>
double time_ms(int iterations, cudaStream_t stream, Work&& work) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cuda_check(cudaEventCreate(&start), "cudaEventCreate");
  cuda_check(cudaEventCreate(&stop), "cudaEventCreate");
  work();  // Warm up.
  std::vector<float> times;
  for (int i = 0; i < iterations; ++i) {
    cuda_check(cudaEventRecord(start, stream), "cudaEventRecord");
    work();
    cuda_check(cudaEventRecord(stop, stream), "cudaEventRecord");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float ms = 0.0F;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    times.push_back(ms);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

int run(int iterations, int max_dims) {
  cudaDeviceProp properties{};
  cuda_check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
  std::printf("GPU: %s\n", properties.name);
  cudaStream_t stream = nullptr;
  cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
  for (const int dims : {256, 512}) {
    if (dims > max_dims) {
      continue;
    }
    const ht::TsdfVolumeConfig config = ht::scene_volume(ht::SceneKind::kTabletop, dims, false);
    const std::uint64_t scratch = ht::MeshExtractor::scratch_bytes(
        static_cast<std::uint64_t>(dims) * dims * dims, kMaxVertices, kMaxTriangles);
    const std::uint64_t mesh_buffers = 28ULL * kMaxVertices + 12ULL * kMaxTriangles;
    std::size_t free = 0;
    std::size_t total = 0;
    cuda_check(cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
    const std::uint64_t need = ht::volume_bytes(config) + scratch + mesh_buffers;
    if (need + (std::uint64_t{512} << 20) > free) {
      std::printf("%d^3: skipped, needs %.0f MiB and %.0f MiB are free\n", dims, mib(need),
                  mib(free));
      continue;
    }
    ht::TsdfVolume volume(config);
    std::printf(
        "%d^3 (%.2f mm voxels): volume %.0f MiB, mesh scratch %.0f MiB and outputs %.0f MiB"
        " for %u vertices / %u triangles\n",
        dims, 1000.0 * config.voxel_size, mib(volume.bytes()), mib(scratch), mib(mesh_buffers),
        kMaxVertices, kMaxTriangles);
    for (const auto [width, height] : {std::pair{640, 480}, std::pair{1280, 720}}) {
      cuda_check(volume.reset(stream), "reset");
      ht::test::Frame frame(width, height);
      // A surface to mesh and ray cast: a quarter of the orbit.
      constexpr std::uint32_t kOrbit = 40;
      for (std::uint32_t i = 0; i < kOrbit; i += 4) {
        ht::test::render(frame, ht::SceneKind::kTabletop, i, kOrbit, ht::test::kNoise);
        cuda_check(volume.integrate(ht::test::depth_image(frame), nullptr, frame.pose,
                                    ht::test::integrate_params(frame), stream),
                   "integrate");
      }
      const double integrate = time_ms(iterations, stream, [&] {
        cuda_check(volume.integrate(ht::test::depth_image(frame), nullptr, frame.pose,
                                    ht::test::integrate_params(frame), stream),
                   "integrate");
      });
      const std::size_t pixels = static_cast<std::size_t>(width) * height;
      Device<float> depth(pixels);
      Device<float> normals(pixels * 3);
      const ht::RaycastParams params{.intrinsics = frame.intrinsics,
                                     .width = width,
                                     .height = height,
                                     .depth_min = 0.1F,
                                     .depth_max = 3.0F};
      const double raycast = time_ms(iterations, stream, [&] {
        cuda_check(volume.raycast(frame.pose, params,
                                  ht::RaycastImages{depth.get(), normals.get(), nullptr}, stream),
                   "raycast");
      });
      ht::MeshExtractor extractor(kMaxVertices, kMaxTriangles);
      Device<float> vertices(std::size_t{kMaxVertices} * 3);
      Device<float> vertex_normals(std::size_t{kMaxVertices} * 3);
      Device<std::uint32_t> triangles(std::size_t{kMaxTriangles} * 3);
      Device<std::uint32_t> counts(ht::kMeshCountsSize);
      const ht::MeshBuffers out{vertices.get(), vertex_normals.get(), nullptr, triangles.get(),
                                counts.get()};
      const double mesh = time_ms(iterations, stream, [&] {
        cuda_check(extractor.extract(volume, out, stream), "extract");
      });
      const std::vector<std::uint32_t> c = counts.host();
      std::printf(
          "  %dx%d: integrate %.3f ms, ray cast %.3f ms, mesh %.3f ms (%u vertices, %u "
          "triangles)\n",
          width, height, integrate, raycast, mesh, c[0], c[1]);
    }
  }
  cudaStreamDestroy(stream);
  std::printf("tsdf_benchmark: done\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) try {
  int iterations = 20;
  int max_dims = 512;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string_view option{argv[i]};
    const std::string_view value{argv[i + 1]};
    int number = 0;
    if (std::from_chars(value.data(), value.data() + value.size(), number).ec != std::errc() ||
        number < 1) {
      std::fprintf(stderr, "error: %s needs a positive number\n", argv[i]);
      return 2;
    }
    if (option == "--iterations") {
      iterations = number;
    } else if (option == "--max-dims") {
      max_dims = number;
    } else {
      std::fprintf(stderr, "error: unknown option %s\n", argv[i]);
      return 2;
    }
  }
  return run(iterations, max_dims);
} catch (const std::exception& error) {
  std::fprintf(stderr, "error: %s\n", error.what());
  return 1;
}
