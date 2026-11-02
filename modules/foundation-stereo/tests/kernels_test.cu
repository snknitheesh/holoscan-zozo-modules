// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// FoundationStereoOp's kernels against the CPU references: the resize and
// normalization of every pixel format into the model, down and up, with
// padded rows; the resize and rescale of the model's disparity back to the
// image, with its invalidation; a constant disparity, which must come back
// multiplied by exactly the width ratio; and the depth, which must be
// depth_from_disparity() of each pixel, fx * B / d.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/foundation_stereo/kernels.hpp>
#include <holoscan/foundation_stereo/reference.hpp>
#include <holoscan/foundation_stereo/stereo_geometry.hpp>

#include "test_support.hpp"

namespace holoscan::foundation_stereo {
namespace {

// Device memory for the length of a test.
template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (cudaMalloc(&data_, count * sizeof(T)) != cudaSuccess) {
      throw std::runtime_error("cudaMalloc failed");
    }
  }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() const { return data_; }
  void upload(const std::vector<T>& values) {
    if (cudaMemcpy(data_, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      throw std::runtime_error("upload failed");
    }
  }
  std::vector<T> download() const {
    std::vector<T> values(count_);
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(values.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
      throw std::runtime_error("download failed");
    }
    return values;
  }

 private:
  T* data_{};
  std::size_t count_{};
};

const char* name_of(PixelFormat format) {
  switch (format) {
    case PixelFormat::kMono8:
      return "MONO8";
    case PixelFormat::kRgb8:
      return "RGB8";
    case PixelFormat::kBgr8:
      return "BGR8";
    case PixelFormat::kRgba8:
      return "RGBA8";
    default:
      return "BGRA8";
  }
}

void check_preprocess(PixelFormat format, int width, int height, int mw, int mh) {
  const std::string what = std::string(name_of(format)) + " " + std::to_string(width) + "x" +
                           std::to_string(height) + " to " + std::to_string(mw) + "x" +
                           std::to_string(mh);
  const int n = channels(format);
  // Rows padded by 7 bytes, with a pattern that differs per channel.
  const std::size_t pitch = static_cast<std::size_t>(width) * n + 7;
  std::vector<std::uint8_t> pixels(pitch * height, 0xEE);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < n; ++c) {
        pixels[y * pitch + x * n + c] =
            static_cast<std::uint8_t>((x * 7 + y * 13 + c * 61 + (x * y) % 17) & 0xFF);
      }
    }
  }
  DeviceBuffer<std::uint8_t> image(pixels.size());
  image.upload(pixels);
  DeviceBuffer<float> out(static_cast<std::size_t>(3) * mw * mh);
  VERIFY(launch_preprocess(ImageView{.data = image.get(),
                                     .pitch = pitch,
                                     .width = width,
                                     .height = height,
                                     .format = format},
                           mw, mh, out.get(), nullptr) == cudaSuccess);
  const auto gpu = out.download();
  const auto cpu = reference::preprocess(pixels.data(), pitch, width, height, format, mw, mh);
  double worst = 0;
  for (std::size_t i = 0; i < gpu.size(); ++i) {
    worst = std::max(worst, std::abs(static_cast<double>(gpu[i]) - cpu[i]));
  }
  // A normalized 0..255 value is about 0.0171 per level. The GPU's float
  // sample positions are up to ~1e-4 px off the reference's doubles at 1280
  // px, which across this pattern's steepest steps is up to 0.03 levels.
  VERIFY_MSG(worst < 1e-3, what + ": worst difference " + std::to_string(worst));
}

void check_postprocess(int mw, int mh, int width, int height, bool invalidate) {
  const std::string what = std::to_string(mw) + "x" + std::to_string(mh) + " to " +
                           std::to_string(width) + "x" + std::to_string(height) +
                           (invalidate ? "" : ", keeping out-of-view pixels");
  // A disparity field with negative, zero and large values.
  std::vector<float> model(static_cast<std::size_t>(mw) * mh);
  for (int y = 0; y < mh; ++y) {
    for (int x = 0; x < mw; ++x) {
      model[static_cast<std::size_t>(y) * mw + x] =
          static_cast<float>(20.0 * std::sin(0.11 * x + 0.07 * y) + 0.15 * x - 3.0);
    }
  }
  DeviceBuffer<float> input(model.size());
  input.upload(model);
  DeviceBuffer<float> disparity(static_cast<std::size_t>(width) * height);
  DeviceBuffer<float> depth(static_cast<std::size_t>(width) * height);
  const PostprocessParams params{.model_width = mw,
                                 .model_height = mh,
                                 .width = width,
                                 .height = height,
                                 .fx_baseline = 42.5F,
                                 .doffs = 0.75F,
                                 .invalidate_out_of_view = invalidate};
  VERIFY(launch_postprocess(input.get(), params, disparity.get(), depth.get(), nullptr) ==
         cudaSuccess);
  const auto gpu = disparity.download();
  const auto gpu_depth = depth.download();
  const auto raw = reference::rescale(model, mw, mh, width, height);
  std::size_t wrong = 0;
  std::size_t wrong_depth = 0;
  std::size_t zeros = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * width + x;
      const double scale_tolerance = 1e-4 * std::max(1.0, std::abs(raw[i]));
      if (reference::near_limit(raw[i], x, scale_tolerance)) continue;
      const double expected = reference::valid(raw[i], x, invalidate);
      wrong += std::abs(gpu[i] - expected) > scale_tolerance;
      zeros += gpu[i] == 0.0F;
      // Exactly the formula, on the disparity the operator gives.
      wrong_depth += gpu_depth[i] != depth_from_disparity(gpu[i], 42.5F, 0.75F);
    }
  }
  VERIFY_MSG(wrong == 0, what + ": " + std::to_string(wrong) + " disparities off");
  VERIFY_MSG(wrong_depth == 0, what + ": " + std::to_string(wrong_depth) + " depths off");
  VERIFY_MSG(zeros > 0 && zeros < gpu.size(), what + ": some, not all, pixels invalid");
}

// A constant model disparity comes back multiplied by width / model width,
// whatever the heights, and the depth is fx * B / d.
void check_constant_rescale() {
  const int mw = 160;
  const int mh = 96;
  const int width = 1280;
  const int height = 720;
  const std::vector<float> model(static_cast<std::size_t>(mw) * mh, 5.0F);
  DeviceBuffer<float> input(model.size());
  input.upload(model);
  DeviceBuffer<float> disparity(static_cast<std::size_t>(width) * height);
  DeviceBuffer<float> depth(static_cast<std::size_t>(width) * height);
  const float fx = 640.0F;
  const float baseline = 0.05F;
  VERIFY(launch_postprocess(input.get(),
                            PostprocessParams{.model_width = mw,
                                              .model_height = mh,
                                              .width = width,
                                              .height = height,
                                              .fx_baseline = fx * baseline},
                            disparity.get(), depth.get(), nullptr) == cudaSuccess);
  const auto d = disparity.download();
  const auto z = depth.download();
  // 5 model pixels are 40 image pixels; columns 0..39 match outside the right image.
  std::size_t wrong = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * width + x;
      const float expected = x < 40 ? 0.0F : 40.0F;
      const float expected_depth = x < 40 ? 0.0F : fx * baseline / 40.0F;
      wrong += d[i] != expected || std::abs(z[i] - expected_depth) > 1e-6F;
    }
  }
  VERIFY_MSG(wrong == 0, std::to_string(wrong) + " pixels differ from 40 px and 0.8 m");
}

void check_rejects() {
  float* out = nullptr;
  VERIFY(launch_preprocess(ImageView{.data = nullptr, .pitch = 3, .width = 1, .height = 1}, 8, 8,
                           out, nullptr) == cudaErrorInvalidValue);
  VERIFY(launch_postprocess(
             nullptr,
             PostprocessParams{.model_width = 8, .model_height = 8, .width = 8, .height = 8}, out,
             nullptr, nullptr) == cudaErrorInvalidValue);
  cudaGetLastError();
}

}  // namespace
}  // namespace holoscan::foundation_stereo

int main() {
  namespace fs = holoscan::foundation_stereo;
  using fs::PixelFormat;
  return fs::test::run({
      {"preprocess MONO8, down",
       [] { fs::check_preprocess(PixelFormat::kMono8, 1280, 720, 736, 480); }},
      {"preprocess RGB8, down",
       [] { fs::check_preprocess(PixelFormat::kRgb8, 640, 480, 160, 96); }},
      {"preprocess BGR8, up", [] { fs::check_preprocess(PixelFormat::kBgr8, 100, 60, 320, 224); }},
      {"preprocess RGBA8, odd sizes",
       [] { fs::check_preprocess(PixelFormat::kRgba8, 333, 251, 192, 128); }},
      {"preprocess BGRA8, same size",
       [] { fs::check_preprocess(PixelFormat::kBgra8, 160, 96, 160, 96); }},
      {"postprocess up", [] { fs::check_postprocess(160, 96, 1280, 720, true); }},
      {"postprocess down", [] { fs::check_postprocess(736, 480, 640, 360, true); }},
      {"postprocess without out-of-view invalidation",
       [] { fs::check_postprocess(160, 96, 333, 251, false); }},
      {"constant disparity rescaled, and its depth", fs::check_constant_rescale},
      {"rejects unusable arguments", fs::check_rejects},
  });
}
