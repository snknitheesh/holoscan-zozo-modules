// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The CUDA kernel against the CPU reference, byte for byte: every format
// pair for every matrix and range, every resize mode with crops, letterboxes
// and fused conversions, odd sizes, pitched rows, the synthetic camera's
// pattern, and determinism. The SDK-free tests check the reference against
// the colour science, so together these check the kernel against it.

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
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

// Device memory freed when it goes.
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

struct Stream {
  Stream() { cuda_check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream"); }
  ~Stream() { cudaStreamDestroy(s); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  cudaStream_t s{};
};

// `src` converted on the GPU into a `format` image of `width` x `height`.
Image gpu_convert(Converter& converter, const Image& src, PixelFormat format, int width, int height,
                  const ConversionParams& params, cudaStream_t stream) {
  DeviceBuffer in(src.bytes.size());
  Image out(format, width, height);
  DeviceBuffer dst(out.bytes.size());
  cuda_check(cudaMemcpy(in.get(), src.bytes.data(), src.bytes.size(), cudaMemcpyHostToDevice),
             "upload");
  // Poison the destination, so that a pixel the kernel misses shows.
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

Image cpu_convert(const Image& src, PixelFormat format, int width, int height,
                  const ConversionParams& params) {
  Image in = src;
  Image out(format, width, height);
  reference_convert(in.view(), out.view(), params);
  return out;
}

std::size_t differing_bytes(const Image& a, const Image& b) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < a.bytes.size() && i < b.bytes.size(); ++i) {
    count += a.bytes[i] != b.bytes[i];
  }
  return count + (a.bytes.size() > b.bytes.size() ? a.bytes.size() - b.bytes.size()
                                                  : b.bytes.size() - a.bytes.size());
}

std::string describe(PixelFormat from, PixelFormat to, int w, int h, const ConversionParams& p) {
  return std::string(format_name(from)) + " -> " + std::string(format_name(to)) + " " +
         std::to_string(w) + "x" + std::to_string(h) + " " + std::string(matrix_name(p.matrix)) +
         " " + std::string(range_name(p.range)) + " " + std::string(resize_mode_name(p.mode)) +
         (p.crop ? " crop" : "") + (p.letterbox ? " letterbox" : "");
}

void expect_same(Converter& converter, const Image& src, PixelFormat format, int width, int height,
                 const ConversionParams& params, cudaStream_t stream) {
  const Image gpu = gpu_convert(converter, src, format, width, height, params, stream);
  const Image cpu = cpu_convert(src, format, width, height, params);
  const std::size_t wrong = differing_bytes(gpu, cpu);
  VERIFY_MSG(wrong == 0, describe(src.format, format, width, height, params) + ": " +
                             std::to_string(wrong) + " bytes differ from the reference");
}

// A random image in `format`: every byte random, so that chroma differs
// inside blocks and nothing about the content helps the kernel.
Image noise(PixelFormat format, int width, int height, std::uint32_t seed) {
  Image img(format, width, height);
  std::uint32_t state = seed * 2654435761U + 1U;
  for (std::uint8_t& byte : img.bytes) {
    state = state * 1664525U + 1013904223U;
    byte = static_cast<std::uint8_t>(state >> 24);
  }
  return img;
}

void check_every_pair() {
  Converter converter(Backend::kCuda);
  Stream stream;
  for (const ColorMatrix matrix :
       {ColorMatrix::kBt601, ColorMatrix::kBt709, ColorMatrix::kBt2020}) {
    for (const ColorRange range : {ColorRange::kLimited, ColorRange::kFull}) {
      for (const PixelFormat from : kAllFormats) {
        const Image src = noise(from, 66, 38, 17U);
        for (const PixelFormat to : kAllFormats) {
          expect_same(
              converter, src, to, 66, 38,
              ConversionParams{.matrix = matrix, .range = range, .mode = ResizeMode::kNearest},
              stream.s);
        }
      }
    }
  }
}

void check_resize_modes() {
  Converter converter(Backend::kCuda);
  Stream stream;
  struct Case {
    int width;
    int height;
    std::optional<Rect> crop;
    bool letterbox;
  };
  const Case cases[] = {
      {64, 36, std::nullopt, false}, {200, 120, std::nullopt, false},
      {30, 17, std::nullopt, false}, {48, 48, Rect{7, 3, 50, 29}, false},
      {64, 64, std::nullopt, true},  {41, 64, Rect{10, 6, 61, 40}, true},
  };
  for (const ResizeMode mode : {ResizeMode::kNearest, ResizeMode::kBilinear, ResizeMode::kArea}) {
    for (const PixelFormat from : {PixelFormat::kNv12, PixelFormat::kYuyv, PixelFormat::kRgb8,
                                   PixelFormat::kBgra8, PixelFormat::kMono16}) {
      const Image src = noise(from, 98, 54, 23U);
      for (const PixelFormat to : {PixelFormat::kRgb8, PixelFormat::kNv12, PixelFormat::kI420,
                                   PixelFormat::kUyvy, PixelFormat::kMono8, PixelFormat::kRgb16}) {
        for (const Case& c : cases) {
          if (needs_even_width(to) && c.width % 2 != 0) continue;
          expect_same(converter, src, to, c.width, c.height,
                      ConversionParams{.matrix = ColorMatrix::kBt709,
                                       .mode = mode,
                                       .crop = c.crop,
                                       .letterbox = c.letterbox,
                                       .fill_rgb = {200, 30, 90}},
                      stream.s);
        }
      }
    }
  }
}

void check_odd_sizes() {
  Converter converter(Backend::kCuda);
  Stream stream;
  for (const PixelFormat from : kAllFormats) {
    for (const PixelFormat to : kAllFormats) {
      for (const auto [w, h] : {std::pair{1, 1}, std::pair{3, 5}, std::pair{17, 9}}) {
        const int sw = needs_even_width(from) ? w + 1 : w;
        const int dw = needs_even_width(to) ? w + 1 : w;
        const Image src = noise(from, sw, h, 5U);
        expect_same(converter, src, to, dw, h, ConversionParams{.mode = ResizeMode::kBilinear},
                    stream.s);
      }
    }
  }
}

// Rows with padding after them, in the source and the destination.
void check_pitched() {
  Converter converter(Backend::kCuda);
  Stream stream;
  const Image src = noise(PixelFormat::kNv12, 50, 20, 9U);
  const Image cpu = cpu_convert(src, PixelFormat::kRgba8, 40, 16,
                                ConversionParams{.mode = ResizeMode::kBilinear});
  constexpr std::size_t kPitch = 64;
  DeviceBuffer in(kPitch * (20 + 10));
  DeviceBuffer out(256 * 16);
  cuda_check(cudaMemcpy2D(in.get(), kPitch, src.bytes.data(), 50, 50, 20, cudaMemcpyHostToDevice),
             "upload y");
  cuda_check(cudaMemcpy2D(in.get() + kPitch * 20, kPitch, src.bytes.data() + 50 * 20, 50, 50, 10,
                          cudaMemcpyHostToDevice),
             "upload uv");
  ImageView source{.format = PixelFormat::kNv12, .width = 50, .height = 20};
  source.planes[0] = {in.get(), kPitch};
  source.planes[1] = {in.get() + kPitch * 20, kPitch};
  ImageView destination{.format = PixelFormat::kRgba8, .width = 40, .height = 16};
  destination.planes[0] = {out.get(), 256};
  converter.convert(source, destination, ConversionParams{.mode = ResizeMode::kBilinear}, stream.s);
  cuda_check(cudaStreamSynchronize(stream.s), "sync");
  Image gpu(PixelFormat::kRgba8, 40, 16);
  cuda_check(cudaMemcpy2D(gpu.bytes.data(), 160, out.get(), 256, 160, 16, cudaMemcpyDeviceToHost),
             "download");
  VERIFY(differing_bytes(gpu, cpu) == 0);
}

// The synthetic camera's pattern, drawn on the GPU, equals the CPU's.
void check_test_pattern() {
  Stream stream;
  for (const std::uint32_t frame : {0U, 7U, 1000U}) {
    Image cpu(PixelFormat::kRgb8, 1920, 1080);
    draw_test_pattern(cpu.view(), frame);
    DeviceBuffer buffer(cpu.bytes.size());
    cuda_check(launch_test_pattern(packed_view(buffer.get(), PixelFormat::kRgb8, 1920, 1080,
                                               MemoryLocation::kDevice),
                                   frame, stream.s),
               "pattern");
    cuda_check(cudaStreamSynchronize(stream.s), "sync");
    Image gpu(PixelFormat::kRgb8, 1920, 1080);
    cuda_check(cudaMemcpy(gpu.bytes.data(), buffer.get(), gpu.bytes.size(), cudaMemcpyDeviceToHost),
               "download");
    VERIFY_MSG(differing_bytes(gpu, cpu) == 0, "frame " + std::to_string(frame));
  }
}

// Full-size camera frames: the pattern in NV12 and YUYV to RGB at 1080p and
// to a letterboxed 640x640, as the demo converts them.
void check_camera_sizes() {
  Converter converter(Backend::kCuda);
  Stream stream;
  Image rgb(PixelFormat::kRgb8, 1920, 1080);
  draw_test_pattern(rgb.view(), 3U);
  for (const PixelFormat camera : {PixelFormat::kNv12, PixelFormat::kYuyv}) {
    const ConversionParams params{.matrix = ColorMatrix::kBt709};
    const Image frame = cpu_convert(rgb, camera, 1920, 1080, params);
    ConversionParams same = params;
    same.mode = ResizeMode::kNearest;
    expect_same(converter, frame, PixelFormat::kRgba8, 1920, 1080, same, stream.s);
    ConversionParams box = params;
    box.letterbox = true;
    expect_same(converter, frame, PixelFormat::kRgb8, 640, 640, box, stream.s);
  }
}

// The same input gives the same bytes every time, on any stream.
void check_determinism() {
  Converter converter(Backend::kCuda);
  Stream first;
  Stream second;
  const Image src = noise(PixelFormat::kYuyv, 640, 360, 31U);
  const ConversionParams params{.mode = ResizeMode::kArea, .letterbox = true};
  const Image a = gpu_convert(converter, src, PixelFormat::kNv12, 300, 200, params, first.s);
  const Image b = gpu_convert(converter, src, PixelFormat::kNv12, 300, 200, params, second.s);
  const Image c = gpu_convert(converter, src, PixelFormat::kNv12, 300, 200, params, first.s);
  VERIFY(a.bytes == b.bytes);
  VERIFY(a.bytes == c.bytes);
}

// Host images and an unusable conversion are refused.
void check_rejects() {
  Converter converter(Backend::kCuda);
  Image host(PixelFormat::kRgb8, 8, 8);
  DeviceBuffer device(8 * 8 * 3);
  const ImageView dst =
      packed_view(device.get(), PixelFormat::kRgb8, 8, 8, MemoryLocation::kDevice);
  VERIFY(test::throws<std::invalid_argument>(
      [&] { converter.convert(host.view(), dst, ConversionParams{}, nullptr); }));
  ImageView odd = packed_view(device.get(), PixelFormat::kYuyv, 3, 8, MemoryLocation::kDevice);
  VERIFY(test::throws<std::invalid_argument>(
      [&] { converter.convert(dst, odd, ConversionParams{}, nullptr); }));
  VERIFY(test::throws<std::invalid_argument>(
      [&] { converter.convert(dst, dst, ConversionParams{.crop = Rect{4, 4, 8, 8}}, nullptr); }));
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
  return cc::test::run({
      {"every format pair equals the reference", cc::check_every_pair},
      {"resize modes, crops and letterboxes equal the reference", cc::check_resize_modes},
      {"odd sizes equal the reference", cc::check_odd_sizes},
      {"pitched rows", cc::check_pitched},
      {"the synthetic pattern on the GPU", cc::check_test_pattern},
      {"camera-size frames", cc::check_camera_sizes},
      {"determinism", cc::check_determinism},
      {"unusable conversions refused", cc::check_rejects},
  });
}
