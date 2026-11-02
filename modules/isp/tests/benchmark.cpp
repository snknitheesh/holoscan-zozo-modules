// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Times IspPipeline's CUDA backend on synthetic 12-bit frames at 1920x1080 and
// 3840x2160, for each demosaic method: the plain pipeline (black level, manual
// white balance, colour matrix, sRGB, RGB8 out), and the full one (lens
// shading, gray-world white balance, sharpening, statistics). Mean GPU time
// per frame, from CUDA events, after a warm-up.
//
//   isp_benchmark [--iterations N] [--cpu true]
//
// --cpu also times the CPU backend once per case.

#include <cuda_runtime_api.h>

#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/isp/isp_params.hpp>
#include <holoscan/isp/isp_pipeline.hpp>
#include <holoscan/isp/synthetic_raw.hpp>
#include <holoscan/isp/synthetic_raw_cuda.hpp>

namespace {

using holoscan::isp::Backend;
using holoscan::isp::DemosaicMethod;
using holoscan::isp::IspParams;
using holoscan::isp::IspPipeline;

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

struct Case {
  const char* name;
  bool full;
};

}  // namespace

int main(int argc, char** argv) try {
  int iterations = 100;
  bool cpu = false;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string_view option{argv[i]};
    const std::string_view value{argv[i + 1]};
    if (option == "--iterations") {
      const auto [end, error] =
          std::from_chars(value.data(), value.data() + value.size(), iterations);
      if (error != std::errc() || iterations < 1) throw std::invalid_argument("--iterations N");
    } else if (option == "--cpu") {
      cpu = value == "true";
    } else {
      throw std::invalid_argument("usage: isp_benchmark [--iterations N] [--cpu true]");
    }
  }
  cudaDeviceProp properties{};
  int device = 0;
  check(cudaGetDevice(&device), "cudaGetDevice");
  check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
  std::printf("%s, %d iterations, mean milliseconds per frame\n", properties.name, iterations);
  std::printf("%-10s %-9s %-6s %9s%s\n", "size", "demosaic", "stages", "cuda ms",
              cpu ? "    cpu ms" : "");

  cudaStream_t stream = nullptr;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  check(cudaEventCreate(&begin), "cudaEventCreate");
  check(cudaEventCreate(&end), "cudaEventCreate");
  for (const auto& [width, height] : {std::pair{1920, 1080}, std::pair{3840, 2160}}) {
    holoscan::isp::SyntheticRawConfig config;
    config.width = width;
    config.height = height;
    config.vignetting = 0.3F;
    const std::size_t pitch = holoscan::isp::raw_row_bytes(config.format, width);
    void* raw = nullptr;
    void* out = nullptr;
    void* histogram = nullptr;
    void* stats = nullptr;
    check(cudaMalloc(&raw, pitch * height), "cudaMalloc");
    check(cudaMalloc(&out, static_cast<std::size_t>(width) * height * 3U), "cudaMalloc");
    check(cudaMalloc(&histogram,
                     4U * holoscan::isp::kHistogramChannels * holoscan::isp::kHistogramBins),
          "cudaMalloc");
    check(cudaMalloc(&stats, 4U * holoscan::isp::stats::kStatsSize), "cudaMalloc");
    check(holoscan::isp::render_synthetic_raw_cuda(holoscan::isp::make_synthetic_scene(config), raw,
                                                   pitch, stream),
          "render");
    std::vector<unsigned char> host_raw;
    std::vector<unsigned char> host_out;
    if (cpu) {
      host_raw.resize(pitch * height);
      host_out.resize(static_cast<std::size_t>(width) * height * 3U);
      check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      check(cudaMemcpy(host_raw.data(), raw, host_raw.size(), cudaMemcpyDeviceToHost),
            "cudaMemcpy");
    }
    for (const DemosaicMethod method : {DemosaicMethod::kBilinear, DemosaicMethod::kMalvar}) {
      for (const Case c : {Case{"plain", false}, Case{"full", true}}) {
        IspParams params = holoscan::isp::synthetic_isp_params(config);
        params.demosaic = method;
        if (!c.full) {
          params.lens_shading.reset();
        } else {
          params.white_balance = holoscan::isp::WhiteBalanceMode::kGrayWorld;
          params.sharpen = 0.5F;
        }
        IspPipeline pipeline(Backend::kCuda, params);
        const holoscan::isp::RawImageView view =
            holoscan::isp::synthetic_raw_view(config, raw, pitch);
        const holoscan::isp::RgbImageView rgb{out, static_cast<std::size_t>(width) * 3U,
                                              holoscan::isp::OutputFormat::kRgb8};
        const holoscan::isp::StatsOutput statistics{
            c.full ? static_cast<std::uint32_t*>(histogram) : nullptr,
            c.full ? static_cast<float*>(stats) : nullptr};
        for (int i = 0; i < 5; ++i) pipeline.process(view, rgb, statistics, {}, stream);
        check(cudaEventRecord(begin, stream), "cudaEventRecord");
        for (int i = 0; i < iterations; ++i) pipeline.process(view, rgb, statistics, {}, stream);
        check(cudaEventRecord(end, stream), "cudaEventRecord");
        check(cudaEventSynchronize(end), "cudaEventSynchronize");
        float ms = 0.0F;
        check(cudaEventElapsedTime(&ms, begin, end), "cudaEventElapsedTime");
        const std::string size = std::to_string(width) + "x" + std::to_string(height);
        std::printf("%-10s %-9s %-6s %9.3f", size.c_str(),
                    std::string(holoscan::isp::demosaic_name(method)).c_str(), c.name,
                    ms / static_cast<float>(iterations));
        if (cpu) {
          IspPipeline reference(Backend::kCpu, params);
          std::vector<std::uint32_t> host_histogram(holoscan::isp::kHistogramChannels *
                                                    holoscan::isp::kHistogramBins);
          std::vector<float> host_stats(holoscan::isp::stats::kStatsSize);
          const auto start = std::chrono::steady_clock::now();
          reference.process(
              holoscan::isp::synthetic_raw_view(config, host_raw.data(), pitch),
              {host_out.data(), static_cast<std::size_t>(width) * 3U,
               holoscan::isp::OutputFormat::kRgb8},
              {c.full ? host_histogram.data() : nullptr, c.full ? host_stats.data() : nullptr});
          const std::chrono::duration<double, std::milli> elapsed =
              std::chrono::steady_clock::now() - start;
          std::printf(" %9.1f", elapsed.count());
        }
        std::printf("\n");
      }
    }
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    cudaFree(raw);
    cudaFree(out);
    cudaFree(histogram);
    cudaFree(stats);
  }
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  cudaStreamDestroy(stream);
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "isp_benchmark: %s\n", error.what());
  return 1;
}
