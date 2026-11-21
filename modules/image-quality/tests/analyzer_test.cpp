// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The GPU analyzer: every accuracy case with host and device input; the
// statistics bit for bit equal to the CPU reference's over many images, sizes,
// formats and row pitches; and the same whatever number of blocks a tile is
// split into.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/image_quality/analyzer.hpp>
#include <holoscan/image_quality/quality.hpp>
#include <holoscan/image_quality/reference.hpp>
#include <holoscan/image_quality/synthetic.hpp>

#include "accuracy_cases.hpp"
#include "test_support.hpp"

namespace holoscan::image_quality::test {
namespace {

// A copy of a host image in device memory, rows `padding` bytes apart beyond their pixels.
class DeviceImage {
 public:
  DeviceImage(const HostImage& image, std::size_t padding) : view_(image.view()) {
    const std::size_t row = view_.pitch;
    view_.pitch = row + padding;
    if (cudaMalloc(&data_, view_.pitch * image.height) != cudaSuccess ||
        cudaMemset(data_, 0xEE, view_.pitch * image.height) != cudaSuccess ||
        cudaMemcpy2D(data_, view_.pitch, image.pixels.data(), row, row, image.height,
                     cudaMemcpyHostToDevice) != cudaSuccess ||
        // The analyzer's stream does not wait for the legacy default stream.
        cudaDeviceSynchronize() != cudaSuccess) {
      throw std::runtime_error("device image copy failed");
    }
    view_.data = static_cast<const std::uint8_t*>(data_);
  }
  ~DeviceImage() { cudaFree(data_); }
  DeviceImage(const DeviceImage&) = delete;
  DeviceImage& operator=(const DeviceImage&) = delete;
  [[nodiscard]] const ImageView& view() const noexcept { return view_; }

 private:
  void* data_{};
  ImageView view_;
};

class GpuSource final : public StatsSource {
 public:
  GpuSource(const StatsOptions& options, bool device, int parts = 0)
      : stats_(AnalyzerOptions{.stats = options, .parts_per_tile = parts}), device_(device) {
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
      throw std::runtime_error("cudaStreamCreate failed");
    }
  }
  ~GpuSource() override { cudaStreamDestroy(stream_); }
  GpuSource(const GpuSource&) = delete;
  GpuSource& operator=(const GpuSource&) = delete;

  FrameStats gather(const HostImage& image) override {
    if (!device_) return stats_.gather(image.view(), false, stream_);
    const DeviceImage copy(image, 3);
    return stats_.gather(copy.view(), true, stream_);
  }

 private:
  GpuStats stats_;
  bool device_;
  cudaStream_t stream_{};
};

std::unique_ptr<StatsSource> gpu_host(const StatsOptions& options) {
  return std::make_unique<GpuSource>(options, false);
}

std::unique_ptr<StatsSource> gpu_device(const StatsOptions& options) {
  return std::make_unique<GpuSource>(options, true);
}

std::string describe(const FrameStats& a, const FrameStats& b) {
  std::string what;
  if (a.luma_histogram != b.luma_histogram) what += " luma_histogram";
  if (a.gradient_histogram != b.gradient_histogram) what += " gradient_histogram";
  if (a.high_pass_histogram != b.high_pass_histogram) what += " high_pass_histogram";
  if (a.flat_threshold != b.flat_threshold) {
    what += " flat_threshold " + std::to_string(a.flat_threshold) + " vs " +
            std::to_string(b.flat_threshold);
  }
  if (a.has_previous != b.has_previous) what += " has_previous";
  for (std::size_t i = 0; i < a.tiles.size() && i < b.tiles.size(); ++i) {
    if (!(a.tiles[i] == b.tiles[i])) {
      what += " tile " + std::to_string(i);
      break;
    }
  }
  return what.empty() ? " geometry" : what;
}

// Sequences of frames, each gathered by the GPU (host and device input, pitched
// rows) and the CPU reference: every statistic equal.
void check_matches_reference() {
  struct Sequence {
    std::string name;
    std::vector<HostImage> frames;
    StatsOptions options;
  };
  std::vector<Sequence> sequences;
  for (const PixelFormat format : {PixelFormat::kMono8, PixelFormat::kRgb8, PixelFormat::kBgr8,
                                   PixelFormat::kRgba8, PixelFormat::kBgra8}) {
    HostImage noisy = render_scene(321, 203, 1, format);
    add_noise(noisy, 7.0, 5);
    HostImage blurred = render_scene(321, 203, 2, format);
    gaussian_blur(blurred, 2.0);
    sequences.push_back({"format " + std::to_string(static_cast<int>(format)),
                         {render_scene(321, 203, 0, format), std::move(noisy), std::move(blurred),
                          render_scene(321, 203, 2, format)},
                         {}});
  }
  const int sizes[][2] = {{1, 1}, {2, 3}, {3, 3}, {17, 5}, {64, 48}, {1280, 720}, {1921, 1081}};
  for (const auto& size : sizes) {
    HostImage dark = render_scene(size[0], size[1], 4);
    apply_gain(dark, 0.2);
    sequences.push_back({std::to_string(size[0]) + "x" + std::to_string(size[1]),
                         {render_scene(size[0], size[1], 3), std::move(dark)},
                         {}});
  }
  sequences.push_back({"odd grid",
                       {render_scene(400, 300, 0), render_scene(400, 300, 1)},
                       {.tiles_x = 7, .tiles_y = 5, .flat_fraction = 0.3}});
  sequences.push_back({"one tile", {render_scene(200, 100, 0)}, {.tiles_x = 1, .tiles_y = 1}});

  for (const Sequence& sequence : sequences) {
    for (const bool device : {false, true}) {
      ReferenceStats reference(sequence.options);
      GpuSource gpu(sequence.options, device);
      for (std::size_t i = 0; i < sequence.frames.size(); ++i) {
        const FrameStats expected = reference.gather(sequence.frames[i].view());
        const FrameStats actual = gpu.gather(sequence.frames[i]);
        VERIFY_MSG(actual == expected, sequence.name + (device ? " device" : " host") + " frame " +
                                           std::to_string(i) + ": differs in" +
                                           describe(actual, expected));
      }
    }
  }
}

// Splitting tiles into more or fewer blocks changes nothing.
void check_parts() {
  HostImage image = render_scene(777, 555, 6);
  add_noise(image, 5.0, 1);
  std::vector<FrameStats> results;
  for (const int parts : {0, 1, 3, 64}) {
    GpuSource gpu({}, true, parts);
    results.push_back(gpu.gather(image));
  }
  for (std::size_t i = 1; i < results.size(); ++i) {
    VERIFY_MSG(results[i] == results[0], "parts setting " + std::to_string(i) + " differs");
  }
  GpuStats stats({});
  VERIFY(stats.parts_per_tile(1920, 1080) >= 1);
  VERIFY(stats.parts_per_tile(16, 9) == 1);
}

void check_rejects() {
  VERIFY(throws<std::invalid_argument>([] { GpuStats({.parts_per_tile = -1}); }));
  VERIFY(throws<std::invalid_argument>([] { GpuStats({.stats = {.tiles_x = 0}}); }));
  GpuStats stats({});
  VERIFY(throws<std::invalid_argument>([&] { (void)stats.gather(ImageView{}, false, nullptr); }));
}

// The analyzer resets: the frame after a reset has no previous frame.
void check_reset() {
  ImageQualityAnalyzer analyzer({}, {});
  const HostImage image = render_scene(160, 120, 0);
  VERIFY(analyzer.analyze(image.view(), false, nullptr).flags == 0U);
  VERIFY(analyzer.analyze(image.view(), false, nullptr).flags == kFrozen);
  analyzer.reset();
  const QualityReport r = analyzer.analyze(image.view(), false, nullptr);
  VERIFY(r.flags == 0U && std::isnan(r.metric(Metric::kFrameDifference)));
}

}  // namespace
}  // namespace holoscan::image_quality::test

int main() {
  using namespace holoscan::image_quality::test;  // NOLINT(build/namespaces)
  std::vector<TestCase> cases = accuracy_cases("gpu_host_", gpu_host);
  for (TestCase& c : accuracy_cases("gpu_device_", gpu_device)) cases.push_back(std::move(c));
  cases.push_back({"matches_reference", check_matches_reference});
  cases.push_back({"parts", check_parts});
  cases.push_back({"rejects", check_rejects});
  cases.push_back({"reset", check_reset});
  return run(cases);
}
