// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Times the Stitcher at the demo's sizes: four 1280x720 fisheye cameras into a
// 1024x1024 bird's-eye view and a 4096x1024 panorama. The tables' build on the
// host clock (once per calibration); each frame with CUDA events over 200
// stitches queued back to back, the best of three rounds. Prints a Markdown
// table. Not a test: run it by hand (see DEVELOPER.md).

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include <holoscan/stitching/stitcher.hpp>
#include <holoscan/stitching/synthetic_world.hpp>

#include "gpu_support.hpp"

namespace st = holoscan::stitching;

namespace {

constexpr int kStitches = 200;
constexpr int kRounds = 3;

double time_stitches(st::Stitcher& stitcher, const st::test::RigFrame& frame, void* output,
                     std::size_t pitch, cudaStream_t stream) {
  cudaEvent_t start;
  cudaEvent_t stop;
  st::test::cuda_check(cudaEventCreate(&start), "cudaEventCreate");
  st::test::cuda_check(cudaEventCreate(&stop), "cudaEventCreate");
  double best = 1e30;
  for (int round = 0; round < kRounds; ++round) {
    st::test::cuda_check(cudaEventRecord(start, stream), "cudaEventRecord");
    for (int i = 0; i < kStitches; ++i) {
      st::test::cuda_check(stitcher.stitch(frame.images, output, pitch, stream), "stitch");
    }
    st::test::cuda_check(cudaEventRecord(stop, stream), "cudaEventRecord");
    st::test::cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float ms = 0.0F;
    st::test::cuda_check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    best = std::min(best, static_cast<double>(ms) / kStitches);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return best;
}

}  // namespace

int main() {
  if (!st::test::have_gpu()) {
    std::printf("no CUDA device\n");
    return 77;
  }
  cudaDeviceProp properties{};
  cudaGetDeviceProperties(&properties, 0);
  const st::SyntheticWorld world;
  const st::CameraRig rig = st::surround_rig(1280, 720);
  cudaStream_t stream = nullptr;
  st::test::cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                       "cudaStreamCreate");

  struct Output {
    std::string name;
    st::OutputGeometry geometry;
  };
  const std::vector<Output> outputs{
      {"bird's-eye 1024x1024", st::ground_view(1024, 1024, 5.5)},
      {"cylindrical 4096x1024", st::wall_panorama(world, st::Projection::kCylindrical, 4096, 1024)},
      {"spherical 4096x1024", st::wall_panorama(world, st::Projection::kSpherical, 4096, 1024)},
  };
  struct Mode {
    std::string name;
    st::BlendSettings settings;
  };
  st::BlendSettings hard;
  hard.mode = st::BlendMode::kHardSeam;
  st::BlendSettings exposure;
  exposure.exposure_compensation = true;
  st::BlendSettings rgb;
  rgb.output_format = st::PixelFormat::kRgb8;
  const std::vector<Mode> modes{{"feather", st::BlendSettings{}},
                                {"hard seam", hard},
                                {"feather + exposure", exposure},
                                {"feather, RGB8 out", rgb}};

  std::printf("GPU: %s\n\n", properties.name);
  std::printf("| Output | Input | Blend | Tables (ms, once) | Table memory (MB) | Frame (ms) |\n");
  std::printf("| --- | --- | --- | ---: | ---: | ---: |\n");
  for (const Output& out : outputs) {
    for (const st::PixelFormat input : {st::PixelFormat::kRgba8, st::PixelFormat::kRgb8}) {
      const auto frame = st::test::render_frame(world, rig, st::RigPose{}, input, {}, 1);
      for (const Mode& mode : modes) {
        if (input == st::PixelFormat::kRgb8 && mode.name != "feather") continue;
        const auto begin = std::chrono::steady_clock::now();
        st::Stitcher stitcher(rig, out.geometry, mode.settings);
        const double build_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                .count();
        const std::size_t pitch = static_cast<std::size_t>(out.geometry.width) *
                                  st::channels(mode.settings.output_format);
        st::test::DeviceBuffer output(pitch * out.geometry.height);
        // Warm up, then time.
        st::test::cuda_check(stitcher.stitch(frame.images, output.data(), pitch, stream), "stitch");
        st::test::cuda_check(cudaStreamSynchronize(stream), "stitch");
        const double frame_ms = time_stitches(stitcher, frame, output.data(), pitch, stream);
        std::printf("| %s | 4 x %s 1280x720 | %s | %.1f | %.1f | %.3f |\n", out.name.c_str(),
                    std::string(st::format_name(input)).c_str(), mode.name.c_str(), build_ms,
                    static_cast<double>(stitcher.table_bytes()) / 1e6, frame_ms);
      }
    }
  }
  cudaStreamDestroy(stream);
  return 0;
}
