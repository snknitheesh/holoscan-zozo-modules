// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Times each conversion at 1920x1080 and 3840x2160 on every backend here, and
// prints a Markdown table: the mean time a call takes from submission until
// its result is in device memory: the best of three rounds of 200 calls, after
// 50 to warm up, as the GPU's and the VIC's clocks ramp up under load. The
// cuda backend's calls are queued back to back and timed with CUDA events;
// the vic backend's return when the VIC is done, so they are timed on the
// host. Not a test: run it on an idle machine, as DEVELOPER.md describes.
//
//   color_convert_benchmark [iterations]

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/color_convert/converter.hpp>
#include <holoscan/color_convert/reference.hpp>

namespace cc = holoscan::color_convert;

namespace {

void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

struct Case {
  cc::PixelFormat from;
  cc::PixelFormat to;
  // Output size as a fraction of the input's, or a fixed size.
  int out_width;  // 0: the input's.
  int out_height;
  cc::ResizeMode mode;
  bool letterbox;
  const char* label;
};

std::string size_name(int w, int h) { return std::to_string(w) + "x" + std::to_string(h); }

}  // namespace

int main(int argc, char** argv) {
  const int iterations = argc > 1 ? std::atoi(argv[1]) : 200;
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cerr << "no CUDA device\n";
    return 1;
  }
  cudaDeviceProp properties{};
  cuda_check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
  const cc::PlatformFacts facts = cc::probe_platform();
  std::cout << "GPU: " << properties.name << "; VIC: " << (facts.has_vic ? "yes" : "no") << "; "
            << iterations << " calls each\n\n";

  using cc::PixelFormat;
  const std::vector<Case> cases = {
      {PixelFormat::kNv12, PixelFormat::kRgb8, 0, 0, cc::ResizeMode::kNearest, false, "convert"},
      {PixelFormat::kNv12, PixelFormat::kRgba8, 0, 0, cc::ResizeMode::kNearest, false, "convert"},
      {PixelFormat::kNv12, PixelFormat::kBgr8, 0, 0, cc::ResizeMode::kNearest, false, "convert"},
      {PixelFormat::kNv12, PixelFormat::kMono8, 0, 0, cc::ResizeMode::kNearest, false, "convert"},
      {PixelFormat::kNv16, PixelFormat::kRgb8, 0, 0, cc::ResizeMode::kNearest, false, "convert"},
      {PixelFormat::kI420, PixelFormat::kRgb8, 0, 0, cc::ResizeMode::kNearest, false, "convert"},
      {PixelFormat::kYuyv, PixelFormat::kRgb8, 0, 0, cc::ResizeMode::kNearest, false, "convert"},
      {PixelFormat::kYuyv, PixelFormat::kRgba8, 0, 0, cc::ResizeMode::kNearest, false, "convert"},
      {PixelFormat::kUyvy, PixelFormat::kRgba8, 0, 0, cc::ResizeMode::kNearest, false, "convert"},
      {PixelFormat::kYuyv, PixelFormat::kNv12, 0, 0, cc::ResizeMode::kNearest, false, "convert"},
      {PixelFormat::kRgb8, PixelFormat::kNv12, 0, 0, cc::ResizeMode::kNearest, false, "convert"},
      {PixelFormat::kRgba8, PixelFormat::kNv12, 0, 0, cc::ResizeMode::kNearest, false, "convert"},
      {PixelFormat::kRgb8, PixelFormat::kBgr8, 0, 0, cc::ResizeMode::kNearest, false, "convert"},
      {PixelFormat::kRgb16, PixelFormat::kRgb8, 0, 0, cc::ResizeMode::kNearest, false, "convert"},
      {PixelFormat::kNv12, PixelFormat::kRgba8, 1280, 720, cc::ResizeMode::kBilinear, false,
       "resize bilinear"},
      {PixelFormat::kNv12, PixelFormat::kRgb8, 640, 640, cc::ResizeMode::kBilinear, true,
       "resize bilinear, letterbox"},
      {PixelFormat::kYuyv, PixelFormat::kRgba8, 640, 360, cc::ResizeMode::kNearest, false,
       "resize nearest"},
      {PixelFormat::kNv12, PixelFormat::kRgb8, 1280, 720, cc::ResizeMode::kArea, false,
       "resize area"},
      {PixelFormat::kNv12, PixelFormat::kNv12, 1280, 720, cc::ResizeMode::kBilinear, false,
       "resize bilinear"},
  };
  std::vector<cc::Backend> backends{cc::Backend::kCuda};
  if (facts.vpi_compiled && facts.has_vic) {
    backends.push_back(cc::Backend::kVic);
  }

  cudaStream_t stream = nullptr;
  cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cuda_check(cudaEventCreate(&start), "event");
  cuda_check(cudaEventCreate(&stop), "event");

  std::cout << "| Input | Output | Operation | Backend | ms |\n|---|---|---|---|---|\n";
  for (const auto [width, height] : {std::pair{1920, 1080}, std::pair{3840, 2160}}) {
    for (const Case& c : cases) {
      const int out_w = c.out_width == 0 ? width : c.out_width;
      const int out_h = c.out_height == 0 ? height : c.out_height;
      std::vector<std::uint8_t> host(cc::packed_size(PixelFormat::kRgb8, width, height));
      cc::draw_test_pattern(cc::packed_view(host.data(), PixelFormat::kRgb8, width, height,
                                            cc::MemoryLocation::kHost),
                            0);
      std::vector<std::uint8_t> input(cc::packed_size(c.from, width, height));
      cc::reference_convert(
          cc::packed_view(host.data(), PixelFormat::kRgb8, width, height,
                          cc::MemoryLocation::kHost),
          cc::packed_view(input.data(), c.from, width, height, cc::MemoryLocation::kHost), {});
      void* src = nullptr;
      void* dst = nullptr;
      cuda_check(cudaMalloc(&src, input.size()), "cudaMalloc");
      cuda_check(cudaMalloc(&dst, cc::packed_size(c.to, out_w, out_h)), "cudaMalloc");
      cuda_check(cudaMemcpy(src, input.data(), input.size(), cudaMemcpyHostToDevice), "upload");
      const cc::ImageView in =
          cc::packed_view(src, c.from, width, height, cc::MemoryLocation::kDevice);
      const cc::ImageView out =
          cc::packed_view(dst, c.to, out_w, out_h, cc::MemoryLocation::kDevice);
      const cc::ConversionParams params{
          .matrix = cc::ColorMatrix::kBt709, .mode = c.mode, .letterbox = c.letterbox};
      for (const cc::Backend backend : backends) {
        std::optional<double> ms;
        if (backend == cc::Backend::kVic &&
            cc::vic_limitation(c.from, width, height, c.to, out_w, out_h, params)) {
          // Not on the VIC.
        } else {
          cc::Converter converter(backend);
          for (int i = 0; i < 50; ++i) converter.convert(in, out, params, stream);
          cuda_check(cudaStreamSynchronize(stream), "sync");
          for (int round = 0; round < 3; ++round) {
            double round_ms = 0.0;
            if (backend == cc::Backend::kCuda) {
              cuda_check(cudaEventRecord(start, stream), "record");
              for (int i = 0; i < iterations; ++i) converter.convert(in, out, params, stream);
              cuda_check(cudaEventRecord(stop, stream), "record");
              cuda_check(cudaEventSynchronize(stop), "sync");
              float elapsed = 0.0F;
              cuda_check(cudaEventElapsedTime(&elapsed, start, stop), "elapsed");
              round_ms = static_cast<double>(elapsed) / iterations;
            } else {
              const auto begin = std::chrono::steady_clock::now();
              for (int i = 0; i < iterations; ++i) converter.convert(in, out, params, stream);
              cuda_check(cudaStreamSynchronize(stream), "sync");
              round_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - begin)
                             .count() /
                         iterations;
            }
            ms = ms ? std::min(*ms, round_ms) : round_ms;
          }
        }
        std::cout << "| " << cc::format_name(c.from) << ' ' << size_name(width, height) << " | "
                  << cc::format_name(c.to) << ' ' << size_name(out_w, out_h) << " | " << c.label
                  << " | " << cc::backend_name(backend) << " | ";
        if (ms) {
          std::cout << std::fixed << std::setprecision(3) << *ms;
        } else {
          std::cout << "n/a";
        }
        std::cout << " |\n";
      }
      cudaFree(src);
      cudaFree(dst);
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  return 0;
}
