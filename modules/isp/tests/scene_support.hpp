// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// For the GPU tests: synthetic raw frames on the host, running an IspPipeline
// on them in host or device memory, the ground truth as the ISP should code
// it, and the measures it is compared with (PSNR, CIEDE2000).

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/isp/isp_params.hpp>
#include <holoscan/isp/isp_pipeline.hpp>
#include <holoscan/isp/synthetic_raw.hpp>

namespace holoscan::isp::test {

// A raw frame in host memory, rows packed.
struct RawFrame {
  SyntheticRawConfig config;
  std::vector<unsigned char> bytes;
  std::size_t pitch{};
  [[nodiscard]] RawImageView view() const {
    return synthetic_raw_view(config, bytes.data(), pitch);
  }
};

inline RawFrame render(const SyntheticRawConfig& config) {
  RawFrame frame{config, {}, raw_row_bytes(config.format, config.width)};
  // uint16 rows stay aligned.
  frame.pitch = (frame.pitch + 1U) & ~std::size_t{1};
  frame.bytes.assign(frame.pitch * config.height, 0);
  render_synthetic_raw(make_synthetic_scene(config), frame.bytes.data(), frame.pitch);
  return frame;
}

// What a pipeline produced for a frame, in host memory.
struct Result {
  OutputFormat format{};
  int width{};
  int height{};
  std::vector<unsigned char> image;  // Rows packed.
  std::vector<std::uint32_t> histogram;
  std::vector<float> stats;
};

inline void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

// Runs `pipeline` on `raw` (host bytes, `pitch` per row), staging through
// device memory for the CUDA backend.
inline Result run(IspPipeline& pipeline, const RawImageView& raw, std::size_t raw_bytes,
                  OutputFormat format, bool with_stats = true, FrameExposure exposure = {}) {
  Result result{format, raw.width, raw.height, {}, {}, {}};
  const std::size_t pitch =
      static_cast<std::size_t>(raw.width) * output_channels(format) * output_channel_bytes(format);
  result.image.assign(pitch * raw.height, 0);
  if (with_stats) {
    result.histogram.assign(static_cast<std::size_t>(kHistogramChannels) * kHistogramBins, 0);
    result.stats.assign(stats::kStatsSize, 0.0F);
  }
  if (pipeline.location() == MemoryLocation::kHost) {
    pipeline.process(raw, RgbImageView{result.image.data(), pitch, format},
                     StatsOutput{with_stats ? result.histogram.data() : nullptr,
                                 with_stats ? result.stats.data() : nullptr},
                     exposure);
    return result;
  }
  void* d_raw = nullptr;
  void* d_out = nullptr;
  std::uint32_t* d_hist = nullptr;
  float* d_stats = nullptr;
  cudaStream_t stream = nullptr;
  try {
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    cuda_check(cudaMalloc(&d_raw, raw_bytes), "cudaMalloc");
    cuda_check(cudaMalloc(&d_out, result.image.size()), "cudaMalloc");
    // On the pipeline's stream: a synchronous copy from pageable memory may
    // return before its DMA is done, and a non-blocking stream would not wait for it.
    cuda_check(cudaMemcpyAsync(d_raw, raw.data, raw_bytes, cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync");
    if (with_stats) {
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_hist), result.histogram.size() * 4U),
                 "cudaMalloc");
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_stats), result.stats.size() * 4U),
                 "cudaMalloc");
    }
    RawImageView device = raw;
    device.data = d_raw;
    pipeline.process(device, RgbImageView{d_out, pitch, format}, StatsOutput{d_hist, d_stats},
                     exposure, stream);
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    cuda_check(cudaMemcpy(result.image.data(), d_out, result.image.size(), cudaMemcpyDeviceToHost),
               "cudaMemcpy");  // After the stream's work: synchronized above.
    if (with_stats) {
      cuda_check(cudaMemcpy(result.histogram.data(), d_hist, result.histogram.size() * 4U,
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy");
      cuda_check(cudaMemcpy(result.stats.data(), d_stats, result.stats.size() * 4U,
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy");
    }
  } catch (...) {
    cudaFree(d_raw);
    cudaFree(d_out);
    cudaFree(d_hist);
    cudaFree(d_stats);
    if (stream != nullptr) cudaStreamDestroy(stream);
    throw;
  }
  cudaFree(d_raw);
  cudaFree(d_out);
  cudaFree(d_hist);
  cudaFree(d_stats);
  cudaStreamDestroy(stream);
  return result;
}

inline Result run(IspPipeline& pipeline, const RawFrame& frame, OutputFormat format,
                  bool with_stats = true, FrameExposure exposure = {}) {
  return run(pipeline, frame.view(), frame.bytes.size(), format, with_stats, exposure);
}

// Channel k of pixel (x, y) of an output, 0..1.
inline float value(const Result& r, int x, int y, int k) {
  const int channels = output_channels(r.format);
  const std::size_t i = (static_cast<std::size_t>(y) * r.width + x) * channels + k;
  switch (r.format) {
    case OutputFormat::kRgb16: {
      std::uint16_t v;
      std::memcpy(&v, r.image.data() + 2U * i, 2U);
      return static_cast<float>(v) / 65535.0F;
    }
    case OutputFormat::kRgb32f: {
      float v;
      std::memcpy(&v, r.image.data() + 4U * i, 4U);
      return v;
    }
    default:
      return static_cast<float>(r.image[i]) / 255.0F;
  }
}

// The truth an ideal ISP codes at (x, y): synthetic_truth() through `lut`, 0..1.
inline void truth(const SyntheticRawScene& scene, const std::vector<float>& lut, int x, int y,
                  float out[3]) {
  float linear[3];
  synthetic_truth(scene, x, y, linear);
  for (int k = 0; k < 3; ++k) {
    const float f = linear[k] * static_cast<float>(lut.size() - 1);
    const auto i = static_cast<std::size_t>(f);
    out[k] = i >= lut.size() - 1 ? lut.back()
                                 : lut[i] + (f - static_cast<float>(i)) * (lut[i + 1] - lut[i]);
  }
}

// PSNR in dB of an output against the truth, both on the 8-bit scale.
inline double psnr(const Result& r, const SyntheticRawConfig& config,
                   const std::vector<float>& lut) {
  const SyntheticRawScene scene = make_synthetic_scene(config);
  double squared = 0.0;
  for (int y = 0; y < r.height; ++y) {
    for (int x = 0; x < r.width; ++x) {
      float t[3];
      truth(scene, lut, x, y, t);
      for (int k = 0; k < 3; ++k) {
        const double d =
            255.0 * (static_cast<double>(value(r, x, y, k)) - std::round(255.0 * t[k]) / 255.0);
        squared += d * d;
      }
    }
  }
  const double mse = squared / (3.0 * r.width * r.height);
  return mse == 0.0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
}

// CIE L*a*b* (D65) of sRGB-coded RGB, 0..1.
inline std::array<double, 3> lab_of_srgb(const double rgb[3]) {
  double linear[3];
  for (int k = 0; k < 3; ++k) {
    const double v = rgb[k];
    linear[k] = v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
  }
  const double xyz[3] = {
      (0.4124 * linear[0] + 0.3576 * linear[1] + 0.1805 * linear[2]) / 0.95047,
      0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2],
      (0.0193 * linear[0] + 0.1192 * linear[1] + 0.9505 * linear[2]) / 1.08883,
  };
  double f[3];
  for (int k = 0; k < 3; ++k) {
    f[k] = xyz[k] > 216.0 / 24389.0 ? std::cbrt(xyz[k]) : (24389.0 / 27.0 * xyz[k] + 16.0) / 116.0;
  }
  return {116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])};
}

// CIEDE2000 colour difference (Sharma, Wu and Dalal's formulation).
inline double delta_e2000(const std::array<double, 3>& lab1, const std::array<double, 3>& lab2) {
  constexpr double kPi = 3.14159265358979323846;
  const auto deg = [](double rad) { return rad * 180.0 / kPi; };
  const auto rad = [](double d) { return d * kPi / 180.0; };
  const double c1 = std::hypot(lab1[1], lab1[2]);
  const double c2 = std::hypot(lab2[1], lab2[2]);
  const double c_mean = (c1 + c2) / 2.0;
  const double c7 = std::pow(c_mean, 7.0);
  const double g = 0.5 * (1.0 - std::sqrt(c7 / (c7 + std::pow(25.0, 7.0))));
  const double a1 = (1.0 + g) * lab1[1];
  const double a2 = (1.0 + g) * lab2[1];
  const double cp1 = std::hypot(a1, lab1[2]);
  const double cp2 = std::hypot(a2, lab2[2]);
  const auto hue = [&](double b, double a) {
    if (a == 0.0 && b == 0.0) return 0.0;
    const double h = deg(std::atan2(b, a));
    return h < 0.0 ? h + 360.0 : h;
  };
  const double hp1 = hue(lab1[2], a1);
  const double hp2 = hue(lab2[2], a2);
  const double dl = lab2[0] - lab1[0];
  const double dc = cp2 - cp1;
  double dh = 0.0;
  if (cp1 * cp2 != 0.0) {
    dh = hp2 - hp1;
    if (dh > 180.0) dh -= 360.0;
    if (dh < -180.0) dh += 360.0;
  }
  const double dhh = 2.0 * std::sqrt(cp1 * cp2) * std::sin(rad(dh) / 2.0);
  const double l_mean = (lab1[0] + lab2[0]) / 2.0;
  const double cp_mean = (cp1 + cp2) / 2.0;
  double h_mean = hp1 + hp2;
  if (cp1 * cp2 != 0.0) {
    h_mean = std::abs(hp1 - hp2) <= 180.0 ? h_mean / 2.0
             : h_mean < 360.0             ? (h_mean + 360.0) / 2.0
                                          : (h_mean - 360.0) / 2.0;
  }
  const double t = 1.0 - 0.17 * std::cos(rad(h_mean - 30.0)) + 0.24 * std::cos(rad(2.0 * h_mean)) +
                   0.32 * std::cos(rad(3.0 * h_mean + 6.0)) -
                   0.20 * std::cos(rad(4.0 * h_mean - 63.0));
  const double d_theta = 30.0 * std::exp(-std::pow((h_mean - 275.0) / 25.0, 2.0));
  const double cp7 = std::pow(cp_mean, 7.0);
  const double rc = 2.0 * std::sqrt(cp7 / (cp7 + std::pow(25.0, 7.0)));
  const double l50 = (l_mean - 50.0) * (l_mean - 50.0);
  const double sl = 1.0 + 0.015 * l50 / std::sqrt(20.0 + l50);
  const double sc = 1.0 + 0.045 * cp_mean;
  const double sh = 1.0 + 0.015 * cp_mean * t;
  const double rt = -std::sin(rad(2.0 * d_theta)) * rc;
  return std::sqrt((dl / sl) * (dl / sl) + (dc / sc) * (dc / sc) + (dhh / sh) * (dhh / sh) +
                   rt * (dc / sc) * (dhh / sh));
}

// CIEDE2000 between the output and the truth over the inner half of each of
// the colour checker's 24 patches, means compared; sRGB tone curve only.
struct PatchErrors {
  double mean{};
  double worst{};
};
inline PatchErrors checker_delta_e(const Result& r, const SyntheticRawConfig& config,
                                   const std::vector<float>& lut) {
  const SyntheticRawScene scene = make_synthetic_scene(config);
  const ChartLayout c = chart_layout(scene);
  PatchErrors errors;
  for (int patch = 0; patch < 24; ++patch) {
    const int x0 = c.left + scene.shift + (patch % 6) * c.cell_width + c.patch_width / 4;
    const int y0 = c.top + (patch / 6) * c.cell_height + c.patch_height / 4;
    double got[3] = {};
    double want[3] = {};
    int n = 0;
    for (int y = y0; y < y0 + c.patch_height / 2; ++y) {
      for (int x = x0; x < x0 + c.patch_width / 2; ++x) {
        float t[3];
        truth(scene, lut, x, y, t);
        for (int k = 0; k < 3; ++k) {
          got[k] += value(r, x, y, k);
          want[k] += std::round(255.0 * t[k]) / 255.0;
        }
        ++n;
      }
    }
    for (int k = 0; k < 3; ++k) {
      got[k] /= n;
      want[k] /= n;
    }
    const double de = delta_e2000(lab_of_srgb(want), lab_of_srgb(got));
    errors.mean += de / 24.0;
    errors.worst = std::max(errors.worst, de);
  }
  return errors;
}

}  // namespace holoscan::isp::test
