// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Time per frame of the GPU analyzer at 1920x1080 and 3840x2160 RGB8, from
// device and from host memory: the kernels, the copy back, the stream wait and
// the evaluation, as ImageQualityOp spends it. Fails only if a 1080p frame from
// device memory takes longer than a 30 fps frame period on average.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include <holoscan/image_quality/analyzer.hpp>
#include <holoscan/image_quality/synthetic.hpp>

namespace {

namespace iq = holoscan::image_quality;

struct Timing {
  double mean_ms{};
  double p50_ms{};
  double p95_ms{};
};

Timing time_frames(iq::ImageQualityAnalyzer& analyzer, const iq::ImageView& image, bool device,
                   cudaStream_t stream, int frames) {
  std::vector<double> ms;
  for (int i = 0; i < frames + 10; ++i) {
    const auto start = std::chrono::steady_clock::now();
    const iq::QualityReport report = analyzer.analyze(image, device, stream);
    const double elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (report.tiles.empty()) throw std::runtime_error("no report");
    if (i >= 10) ms.push_back(elapsed);  // The first frames allocate.
  }
  std::sort(ms.begin(), ms.end());
  Timing t;
  for (const double v : ms) t.mean_ms += v / static_cast<double>(ms.size());
  t.p50_ms = ms[ms.size() / 2];
  t.p95_ms = ms[ms.size() * 95 / 100];
  return t;
}

}  // namespace

int main() try {
  cudaDeviceProp properties{};
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    throw std::runtime_error("no CUDA device");
  }
  cudaStream_t stream{};
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
    throw std::runtime_error("cudaStreamCreate failed");
  }
  std::printf("image_quality_benchmark: %s, %d multiprocessors\n", properties.name,
              properties.multiProcessorCount);
  bool passed = true;
  const int sizes[][2] = {{1920, 1080}, {3840, 2160}};
  for (const auto& size : sizes) {
    const iq::HostImage image = iq::render_scene(size[0], size[1], 0);
    void* device_pixels = nullptr;
    if (cudaMalloc(&device_pixels, image.pixels.size()) != cudaSuccess ||
        cudaMemcpy(device_pixels, image.pixels.data(), image.pixels.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("device copy failed");
    }
    iq::ImageView on_device = image.view();
    on_device.data = static_cast<const std::uint8_t*>(device_pixels);
    for (const bool from_device : {true, false}) {
      iq::ImageQualityAnalyzer analyzer({}, {});
      const Timing t =
          time_frames(analyzer, from_device ? on_device : image.view(), from_device, stream, 100);
      std::printf(
          "image_quality_benchmark: %dx%d RGB8 from %s memory: mean %.3f ms, p50 %.3f ms, "
          "p95 %.3f ms per frame\n",
          size[0], size[1], from_device ? "device" : "host", t.mean_ms, t.p50_ms, t.p95_ms);
      if (from_device && size[0] == 1920 && t.mean_ms > 1000.0 / 30.0) {
        passed = false;
      }
    }
    cudaFree(device_pixels);
  }
  cudaStreamDestroy(stream);
  std::printf("image_quality_benchmark: %s\n", passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
} catch (const std::exception& error) {
  std::fprintf(stderr, "image_quality_benchmark: %s\n", error.what());
  return 1;
}
