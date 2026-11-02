// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Device images for the tests: packed buffers that upload and download host
// copies, and conversions of host images to NV12.

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/denoise/denoise_params.hpp>

namespace holoscan::denoise::test {

inline void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

// Bytes of a packed image of `format`.
inline std::size_t image_bytes(PixelFormat format, int width, int height) {
  const auto pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  return format == PixelFormat::kNv12 ? pixels + pixels / 2 : pixels * pixel_bytes(format);
}

// A packed image in device memory.
class DeviceImage {
 public:
  DeviceImage(PixelFormat format, int width, int height)
      : format_(format),
        width_(width),
        height_(height),
        bytes_(image_bytes(format, width, height)) {
    cuda_check(cudaMalloc(&data_, std::max<std::size_t>(bytes_, 1)), "cudaMalloc");
  }
  ~DeviceImage() { cudaFree(data_); }
  DeviceImage(const DeviceImage&) = delete;
  DeviceImage& operator=(const DeviceImage&) = delete;

  [[nodiscard]] ImageView view() const {
    ImageView v{.format = format_, .width = width_, .height = height_};
    const auto row = static_cast<std::size_t>(width_) * pixel_bytes(format_);
    v.planes[0] = Plane{.data = static_cast<std::uint8_t*>(data_), .pitch = row};
    if (format_ == PixelFormat::kNv12) {
      v.planes[1] = Plane{.data = static_cast<std::uint8_t*>(data_) + row * height_, .pitch = row};
    }
    return v;
  }
  void upload(const std::vector<std::uint8_t>& host) {
    if (host.size() != bytes_) throw std::invalid_argument("upload: wrong size");
    cuda_check(cudaMemcpy(data_, host.data(), bytes_, cudaMemcpyHostToDevice), "cudaMemcpy");
  }
  [[nodiscard]] std::vector<std::uint8_t> download() const {
    std::vector<std::uint8_t> host(bytes_);
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    cuda_check(cudaMemcpy(host.data(), data_, bytes_, cudaMemcpyDeviceToHost), "cudaMemcpy");
    return host;
  }
  [[nodiscard]] std::size_t bytes() const { return bytes_; }

 private:
  PixelFormat format_;
  int width_;
  int height_;
  std::size_t bytes_;
  void* data_{};
};

// Full-range BT.601 NV12 of a packed RGB8 image, even sizes; chroma the 2x2 mean.
inline std::vector<std::uint8_t> rgb_to_nv12(const std::vector<std::uint8_t>& rgb, int width,
                                             int height) {
  const auto pixels = static_cast<std::size_t>(width) * height;
  std::vector<std::uint8_t> out(pixels + pixels / 2);
  const auto byte = [](double v) {
    return static_cast<std::uint8_t>(std::clamp(std::floor(v + 0.5), 0.0, 255.0));
  };
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::uint8_t* p = rgb.data() + (static_cast<std::size_t>(y) * width + x) * 3;
      out[static_cast<std::size_t>(y) * width + x] =
          byte(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]);
    }
  }
  for (int y = 0; y < height / 2; ++y) {
    for (int x = 0; x < width / 2; ++x) {
      double r = 0;
      double g = 0;
      double b = 0;
      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          const std::uint8_t* p =
              rgb.data() + (static_cast<std::size_t>(2 * y + dy) * width + 2 * x + dx) * 3;
          r += p[0] / 4.0;
          g += p[1] / 4.0;
          b += p[2] / 4.0;
        }
      }
      std::uint8_t* uv = out.data() + pixels + static_cast<std::size_t>(y) * width + 2 * x;
      uv[0] = byte(128 - 0.168736 * r - 0.331264 * g + 0.5 * b);
      uv[1] = byte(128 + 0.5 * r - 0.418688 * g - 0.081312 * b);
    }
  }
  return out;
}

}  // namespace holoscan::denoise::test
