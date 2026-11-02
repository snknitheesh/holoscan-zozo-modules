// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Times every denoising method and backend, exposure fusion and the tone map,
// at 1920x1080 and 3840x2160: the mean time per frame between CUDA events
// around the call (for the vpi backend this includes its host waits), after a
// warm-up. CI runs it with few iterations, to keep it working.
//
//   denoise_benchmark [--iterations N] [--sizes 1080,2160]

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/denoise/denoiser.hpp>
#include <holoscan/denoise/exposure_fusion.hpp>
#include <holoscan/denoise/synthetic_scene.hpp>

#include "image_support.hpp"

namespace dn = holoscan::denoise;
using dn::test::cuda_check;
using dn::test::DeviceImage;

namespace {

// Mean milliseconds per call of `body` on `stream`, after `warmup` untimed calls.
double time_ms(const std::function<void()>& body, cudaStream_t stream, int iterations, int warmup) {
  for (int i = 0; i < warmup; ++i) body();
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cuda_check(cudaEventCreate(&start), "cudaEventCreate");
  cuda_check(cudaEventCreate(&stop), "cudaEventCreate");
  cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  cuda_check(cudaEventRecord(start, stream), "cudaEventRecord");
  for (int i = 0; i < iterations; ++i) body();
  cuda_check(cudaEventRecord(stop, stream), "cudaEventRecord");
  cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize");
  float ms = 0.0F;
  cuda_check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms / static_cast<double>(iterations);
}

struct Method {
  std::string name;
  dn::Backend backend;
  dn::DenoiseParams params;
  dn::VpiEngine engine{dn::VpiEngine::kAuto};
};

}  // namespace

int main(int argc, char** argv) try {
  int iterations = 50;
  std::vector<int> heights{1080, 2160};
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string_view option = argv[i];
    const std::string_view value = argv[i + 1];
    if (option == "--iterations") {
      std::from_chars(value.data(), value.data() + value.size(), iterations);
    } else if (option == "--sizes") {
      heights.clear();
      for (std::string_view rest = value; !rest.empty();) {
        const std::size_t comma = rest.find(',');
        int h = 0;
        std::from_chars(rest.data(), rest.data() + std::min(comma, rest.size()), h);
        heights.push_back(h);
        rest = comma == std::string_view::npos ? std::string_view{} : rest.substr(comma + 1);
      }
    } else {
      throw std::invalid_argument("unknown option " + std::string(option));
    }
  }
  iterations = std::max(iterations, 1);
  const int warmup = std::min(iterations, 5);

  std::vector<Method> methods;
  const auto params = [](dn::SpatialFilter spatial, bool temporal) {
    dn::DenoiseParams p;
    p.spatial = spatial;
    p.temporal = temporal;
    return p;
  };
  methods.push_back(
      {"bilateral", dn::Backend::kCuda, params(dn::SpatialFilter::kBilateral, false)});
  methods.push_back({"nlm", dn::Backend::kCuda, params(dn::SpatialFilter::kNlm, false)});
  methods.push_back({"temporal", dn::Backend::kCuda, params(dn::SpatialFilter::kNone, true)});
  methods.push_back(
      {"temporal+bilateral", dn::Backend::kCuda, params(dn::SpatialFilter::kBilateral, true)});
  methods.push_back({"temporal+nlm", dn::Backend::kCuda, params(dn::SpatialFilter::kNlm, true)});
  const auto backends = dn::available_backends();
  if (std::find(backends.begin(), backends.end(), dn::Backend::kVpi) != backends.end()) {
    methods.push_back({"vpi tnr (cuda)", dn::Backend::kVpi, params(dn::SpatialFilter::kNone, true),
                       dn::VpiEngine::kCuda});
    if (dn::probe_platform().vic_tnr) {
      methods.push_back({"vpi tnr (vic)", dn::Backend::kVpi, params(dn::SpatialFilter::kNone, true),
                         dn::VpiEngine::kVic});
    }
  }

  cudaStream_t stream = nullptr;
  cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
  std::printf("%-22s %-6s", "method", "format");
  for (const int h : heights) std::printf(" %9dx%d", h * 16 / 9, h);
  std::printf("   (ms per frame)\n");

  for (const Method& method : methods) {
    for (const dn::PixelFormat format : {dn::PixelFormat::kRgb8, dn::PixelFormat::kNv12}) {
      // NV12 is VPI's own format; the cuda kernels' cost follows the bytes.
      if (format == dn::PixelFormat::kNv12 && method.backend != dn::Backend::kVpi &&
          method.name != "temporal+bilateral") {
        continue;
      }
      std::printf("%-22s %-6s", method.name.c_str(), std::string(dn::format_name(format)).c_str());
      for (const int height : heights) {
        const int width = height * 16 / 9;
        const dn::LowLightScene scene{.width = width, .height = height};
        DeviceImage noisy(dn::PixelFormat::kRgb8, width, height);
        cuda_check(dn::render_noisy(scene, dn::NoiseModel{}, 0, noisy.view(), stream),
                   "render_noisy");
        DeviceImage in(format, width, height);
        if (format == dn::PixelFormat::kNv12) {
          in.upload(dn::test::rgb_to_nv12(noisy.download(), width, height));
        } else {
          cuda_check(cudaMemcpy(in.view().planes[0].data, noisy.view().planes[0].data,
                                noisy.bytes(), cudaMemcpyDeviceToDevice),
                     "cudaMemcpy");
        }
        DeviceImage out(format, width, height);
        dn::Denoiser denoiser(method.backend, method.params,
                              dn::VpiOptions{.engine = method.engine});
        const double ms = time_ms([&] { denoiser.process(in.view(), out.view(), stream); }, stream,
                                  iterations, warmup);
        std::printf(" %12.3f", ms);
        std::fflush(stdout);
      }
      std::printf("\n");
    }
  }

  std::printf("%-22s %-6s", "fusion, 3 brackets", "RGB8");
  for (const int height : heights) {
    const int width = height * 16 / 9;
    const dn::HdrScene scene{.width = width, .height = height};
    std::vector<std::unique_ptr<DeviceImage>> images;
    std::vector<dn::ImageView> views;
    for (const float exposure : dn::kBracketExposures) {
      images.push_back(std::make_unique<DeviceImage>(dn::PixelFormat::kRgb8, width, height));
      cuda_check(dn::render_bracket(scene, 0, exposure, images.back()->view(), stream),
                 "render_bracket");
      views.push_back(images.back()->view());
    }
    DeviceImage out(dn::PixelFormat::kRgb8, width, height);
    dn::ExposureFusion fusion(dn::FusionParams{});
    std::printf(" %12.3f", time_ms([&] { fusion.fuse(views, out.view(), stream); }, stream,
                                   iterations, warmup));
    std::fflush(stdout);
  }
  std::printf("\n%-22s %-6s", "tone map", "RGB32F");
  for (const int height : heights) {
    const int width = height * 16 / 9;
    const auto radiance = dn::render_radiance(dn::HdrScene{.width = width, .height = height}, 0);
    void* device = nullptr;
    cuda_check(cudaMalloc(&device, radiance.size() * sizeof(float)), "cudaMalloc");
    cuda_check(cudaMemcpy(device, radiance.data(), radiance.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy");
    const dn::HdrImageView view{.format = dn::HdrFormat::kRgb32f,
                                .width = width,
                                .height = height,
                                .data = device,
                                .pitch = static_cast<std::size_t>(width) * 3 * sizeof(float)};
    DeviceImage out(dn::PixelFormat::kRgb8, width, height);
    dn::ToneMapper mapper(dn::ToneMapParams{});
    std::printf(" %12.3f",
                time_ms([&] { mapper.map(view, out.view(), stream); }, stream, iterations, warmup));
    cudaFree(device);
  }
  std::printf("\n");
  cudaStreamDestroy(stream);
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "denoise_benchmark: %s\n", error.what());
  return 1;
}
