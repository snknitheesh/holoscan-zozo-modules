// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The VIC against the CUDA kernel, on a Jetson: every pair of the formats
// the VIC handles, for every matrix and range, same-size and resized
// (nearest and bilinear, crop, letterbox), within a tolerance. The VIC
// interpolates chroma where the kernel replicates it, and rounds its own
// way, so the two differ by a few codes; a wrong matrix, range, channel order
// or geometry differs by far more. Also auto's choice. Skipped (77) where
// there is no VIC, unless HOLOSCAN_CI_PLATFORM is jetson, where it fails.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/color_convert/converter.hpp>
#include <holoscan/color_convert/reference.hpp>

#include "test_support.hpp"
#include "textbook.hpp"

namespace holoscan::color_convert {
namespace {

using test::Image;

void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) { cuda_check(cudaMalloc(&data_, bytes), "cudaMalloc"); }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  [[nodiscard]] std::uint8_t* get() const { return static_cast<std::uint8_t*>(data_); }

 private:
  void* data_{};
};

using F = PixelFormat;
constexpr F kVicFormats[] = {F::kNv12, F::kYuyv, F::kUyvy, F::kRgba8, F::kBgra8, F::kMono8};

// How far two images of one format are apart, per stored sample.
struct Difference {
  double mean{};
  int p99{};
  int max{};
};

Difference compare(const Image& a, const Image& b) {
  std::vector<int> histogram(256, 0);
  double sum = 0.0;
  for (std::size_t i = 0; i < a.bytes.size(); ++i) {
    const int d = std::abs(static_cast<int>(a.bytes[i]) - static_cast<int>(b.bytes[i]));
    ++histogram[d];
    sum += d;
  }
  Difference result{.mean = sum / static_cast<double>(a.bytes.size())};
  std::size_t seen = 0;
  bool p99_found = false;
  for (int d = 0; d < 256; ++d) {
    seen += histogram[d];
    if (!p99_found && seen >= a.bytes.size() * 99 / 100) {
      result.p99 = d;
      p99_found = true;
    }
    if (histogram[d] != 0) {
      result.max = d;
    }
  }
  return result;
}

// A smooth RGB picture: gradients and slow waves, no hard edges, so that the
// VIC's chroma interpolation and the kernel's replication agree closely.
Image smooth_rgb(int width, int height) {
  Image img(PixelFormat::kRgb8, width, height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const double u = static_cast<double>(x) / width;
      const double v = static_cast<double>(y) / height;
      test::write(img, x, y,
                  test::Px{{std::floor(20.0 + 215.0 * u),
                            std::floor(127.5 + 100.0 * std::sin(6.0 * u + 4.0 * v)),
                            std::floor(30.0 + 200.0 * v)}});
    }
  }
  return img;
}

Image run(Converter& converter, const Image& src, PixelFormat format, int width, int height,
          const ConversionParams& params, cudaStream_t stream) {
  DeviceBuffer in(src.bytes.size());
  Image out(format, width, height);
  DeviceBuffer dst(out.bytes.size());
  cuda_check(cudaMemcpy(in.get(), src.bytes.data(), src.bytes.size(), cudaMemcpyHostToDevice),
             "upload");
  cuda_check(cudaMemset(dst.get(), 0xA5, out.bytes.size()), "memset");
  // The legacy stream's copy and memset before the conversion's own stream.
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  converter.convert(
      packed_view(in.get(), src.format, src.width, src.height, MemoryLocation::kDevice),
      packed_view(dst.get(), format, width, height, MemoryLocation::kDevice), params, stream);
  cuda_check(cudaStreamSynchronize(stream), "sync");
  cuda_check(cudaMemcpy(out.bytes.data(), dst.get(), out.bytes.size(), cudaMemcpyDeviceToHost),
             "download");
  return out;
}

struct Geometry {
  const char* name;
  int width;
  int height;
  std::optional<Rect> crop;
  bool letterbox;
};

void check_parity() {
  Converter vic(Backend::kVic);
  Converter cuda(Backend::kCuda);
  cudaStream_t stream = nullptr;
  cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
  const Image rgb = smooth_rgb(1280, 720);
  const Geometry geometries[] = {
      {"same size", 1280, 720, std::nullopt, false},
      {"down 640x360", 640, 360, std::nullopt, false},
      {"up 1920x1080", 1920, 1080, std::nullopt, false},
      {"crop", 512, 288, Rect{100, 60, 1024, 576}, false},
      {"letterbox 640x640", 640, 640, std::nullopt, true},
  };
  int compared = 0;
  for (const ColorMatrix matrix :
       {ColorMatrix::kBt601, ColorMatrix::kBt709, ColorMatrix::kBt2020}) {
    for (const ColorRange range : {ColorRange::kLimited, ColorRange::kFull}) {
      for (const PixelFormat from : kVicFormats) {
        Image src(from, rgb.width, rgb.height);
        Image rgb_copy = rgb;
        reference_convert(rgb_copy.view(), src.view(),
                          ConversionParams{.matrix = matrix, .range = range});
        for (const PixelFormat to : kVicFormats) {
          // The matrix and range matter only between colour models.
          const bool matrix_matters = format_info(from).model != format_info(to).model;
          if (!matrix_matters && (matrix != ColorMatrix::kBt601 || range != ColorRange::kLimited)) {
            continue;
          }
          if (vic_limitation(from, rgb.width, rgb.height, to, rgb.width, rgb.height,
                             ConversionParams{.matrix = matrix, .range = range})) {
            continue;  // Not the VIC's (BT.2020 between colour models).
          }
          for (const Geometry& g : geometries) {
            for (const ResizeMode mode : {ResizeMode::kNearest, ResizeMode::kBilinear}) {
              const bool resized = g.width != rgb.width || g.height != rgb.height || g.crop;
              if (!resized && mode == ResizeMode::kNearest) continue;
              const ConversionParams params{.matrix = matrix,
                                            .range = range,
                                            .mode = mode,
                                            .crop = g.crop,
                                            .letterbox = g.letterbox,
                                            .fill_rgb = {16, 32, 64}};
              const Image a = run(vic, src, to, g.width, g.height, params, stream);
              const Image b = run(cuda, src, to, g.width, g.height, params, stream);
              const Difference d = compare(a, b);
              ++compared;
              // The VIC interpolates chroma and rounds its own way; resizing adds
              // its own sampling. Measured on Orin over every case here: mean up
              // to 0.9, 99th percentile up to 3, largest 6 codes.
              const double mean_limit = resized ? 1.5 : 1.0;
              const int p99_limit = resized ? 4 : 3;
              constexpr int kMaxLimit = 10;
              std::ostringstream what;
              what << format_name(from) << " -> " << format_name(to) << " " << matrix_name(matrix)
                   << " " << range_name(range) << " " << g.name << " " << resize_mode_name(mode)
                   << ": mean " << std::fixed << std::setprecision(3) << d.mean << " p99 " << d.p99
                   << " max " << d.max;
              if (std::getenv("COLOR_CONVERT_PARITY_VERBOSE") != nullptr) {
                std::cout << what.str() << '\n';
              }
              VERIFY_MSG(d.mean <= mean_limit && d.p99 <= p99_limit && d.max <= kMaxLimit,
                         what.str());
            }
          }
        }
      }
    }
  }
  cudaStreamDestroy(stream);
  std::cout << compared << " conversions compared\n";
}

// auto takes the VIC for what it can do, and CUDA for the rest.
void check_auto() {
  Converter converter(Backend::kAuto);
  DeviceBuffer buffer(1920 * 1080 * 4);
  const ImageView nv12 =
      packed_view(buffer.get(), PixelFormat::kNv12, 1920, 1080, MemoryLocation::kDevice);
  const ImageView rgba =
      packed_view(buffer.get(), PixelFormat::kRgba8, 1920, 1080, MemoryLocation::kDevice);
  const ImageView rgb =
      packed_view(buffer.get(), PixelFormat::kRgb8, 640, 360, MemoryLocation::kDevice);
  VERIFY(converter.choose(nv12, rgba, ConversionParams{}).first == Backend::kVic);
  VERIFY(converter.choose(nv12, rgb, ConversionParams{}).first == Backend::kCuda);
  VERIFY(converter.choose(nv12, rgba, ConversionParams{.mode = ResizeMode::kArea}).first ==
         Backend::kCuda);
}

}  // namespace
}  // namespace holoscan::color_convert

int main() {
  namespace cc = holoscan::color_convert;
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cout << "no CUDA device; skipping\n";
    return 77;
  }
  const cc::PlatformFacts facts = cc::probe_platform();
  if (!facts.vpi_compiled || !facts.has_vic) {
    const char* platform = std::getenv("HOLOSCAN_CI_PLATFORM");
    if (platform != nullptr && std::string(platform) == "jetson") {
      std::cerr << "HOLOSCAN_CI_PLATFORM is jetson, but "
                << (facts.vpi_compiled ? "VPI reports no VIC" : "this build has no VPI") << '\n';
      return 1;
    }
    std::cout << "no VIC here; skipping\n";
    return 77;
  }
  return cc::test::run({
      {"VIC against CUDA", cc::check_parity},
      {"auto's choice on a Jetson", cc::check_auto},
  });
}
