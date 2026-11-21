// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The local recordings: every frame of each recording's color stream (8-bit PPM)
// and left infrared eye (8-bit PGM) through the GPU analyzer, from host and from
// device memory, must give the CPU reference's statistics bit for bit, and so
// the same report. The rate of each flag with the default thresholds is printed:
// it is a record of what real pixels give, not a pass criterion, except that the
// lit color frames of a live camera are never frozen (the infrared eyes without
// the projector are nearly black, and differ by less than frozen_difference). Reads
// $HOLOSCAN_TEST_DATA/stereo-camera/<recording>/; skipped (exit 77) when the
// variable is unset, failed when it names no recording with an index.csv.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/image_quality/analyzer.hpp>
#include <holoscan/image_quality/quality.hpp>
#include <holoscan/image_quality/reference.hpp>
#include <holoscan/image_quality/synthetic.hpp>

#include "test_support.hpp"

namespace holoscan::image_quality {
namespace {

namespace fs = std::filesystem;

constexpr int kSkip = 77;
constexpr int kFlagCount = 8;
constexpr std::size_t kFrozenBit = 5;
static_assert((1U << kFrozenBit) == kFrozen);

// An 8-bit binary PGM (P5) or PPM (P6), as the recordings hold the eyes and the color images.
HostImage read_pnm(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error(path.string() + ": cannot open");
  std::string magic;
  file >> magic;
  int values[3] = {};
  for (int& value : values) {
    file >> std::ws;
    while (file.peek() == '#') {
      std::string comment;
      std::getline(file, comment);
      file >> std::ws;
    }
    file >> value;
  }
  file.get();  // The single whitespace before the pixels.
  const bool gray = magic == "P5";
  if ((!gray && magic != "P6") || values[2] != 255 || values[0] < 1 || values[1] < 1) {
    throw std::runtime_error(path.string() + ": not an 8-bit binary PGM or PPM");
  }
  HostImage image{values[0], values[1], gray ? PixelFormat::kMono8 : PixelFormat::kRgb8, {}};
  image.pixels.resize(static_cast<std::size_t>(image.width) * image.height * (gray ? 1U : 3U));
  file.read(reinterpret_cast<char*>(image.pixels.data()),
            static_cast<std::streamsize>(image.pixels.size()));
  if (!file) throw std::runtime_error(path.string() + ": truncated");
  return image;
}

// The paths of column `column` of index.csv, in frame order.
std::vector<fs::path> stream_paths(const fs::path& recording, const std::string& column) {
  std::ifstream index(recording / "index.csv");
  std::string line;
  std::getline(index, line);
  std::vector<std::string> header;
  std::stringstream names(line);
  std::string field;
  while (std::getline(names, field, ',')) header.push_back(field);
  const auto at = std::find(header.begin(), header.end(), column);
  if (at == header.end()) throw std::runtime_error("index.csv has no column " + column);
  const auto position = static_cast<std::size_t>(at - header.begin());
  std::vector<fs::path> paths;
  while (std::getline(index, line)) {
    std::vector<std::string> fields;
    std::stringstream row(line);
    while (std::getline(row, field, ',')) fields.push_back(field);
    if (fields.size() > position) paths.push_back(recording / fields[position]);
  }
  return paths;
}

// A device copy of each frame, for the analyzer's device-memory path.
class DeviceImage {
 public:
  DeviceImage() = default;
  DeviceImage(const DeviceImage&) = delete;
  DeviceImage& operator=(const DeviceImage&) = delete;
  ~DeviceImage() { cudaFree(data_); }

  ImageView upload(const HostImage& image) {
    if (bytes_ < image.pixels.size()) {
      cudaFree(data_);
      data_ = nullptr;
      if (cudaMalloc(&data_, image.pixels.size()) != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed");
      }
      bytes_ = image.pixels.size();
    }
    // On the legacy stream, which the analyzer's non-blocking stream does not
    // wait for: done before it is read.
    if (cudaMemcpy(data_, image.pixels.data(), image.pixels.size(), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
      throw std::runtime_error("cudaMemcpy failed");
    }
    ImageView view = image.view();
    view.data = static_cast<const std::uint8_t*>(data_);
    return view;
  }

 private:
  void* data_{};
  std::size_t bytes_{};
};

// One stream of a recording: GPU == CPU on every frame, and the flags' rates.
void check_stream(const fs::path& recording, const std::string& label, const std::string& column,
                  bool never_frozen) {
  const std::vector<fs::path> paths = stream_paths(recording, column);
  VERIFY_MSG(paths.size() >= 2, label + ": too few frames in index.csv");
  cudaStream_t stream{};
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
    throw std::runtime_error("cudaStreamCreate failed");
  }
  const StatsOptions options;
  const QualityThresholds thresholds;
  GpuStats gpu(AnalyzerOptions{.stats = options});
  ReferenceStats reference(options);
  QualityEvaluator gpu_evaluator(thresholds);
  QualityEvaluator reference_evaluator(thresholds);
  DeviceImage device;
  std::array<int, kFlagCount> counts{};
  int frames = 0;
  int failed = 0;
  int mismatched = 0;
  double obstructed_share = 0.0;
  double overexposed_share = 0.0;
  double color_cast = 0.0;
  int width = 0;
  int height = 0;
  for (const fs::path& path : paths) {
    const HostImage image = read_pnm(path);
    width = image.width;
    height = image.height;
    // Every other frame from device memory, the rest from host memory.
    const bool on_device = frames % 2 == 1;
    const FrameStats from_gpu =
        gpu.gather(on_device ? device.upload(image) : image.view(), on_device, stream);
    const FrameStats from_cpu = reference.gather(image.view());
    const QualityReport gpu_report = gpu_evaluator.evaluate(from_gpu);
    const QualityReport report = reference_evaluator.evaluate(from_cpu);
    if (!(from_gpu == from_cpu) || gpu_report.flags != report.flags ||
        std::memcmp(gpu_report.metrics.data(), report.metrics.data(), sizeof(report.metrics)) !=
            0) {
      ++mismatched;
    }
    ++frames;
    failed += report.flags != 0U ? 1 : 0;
    for (int bit = 0; bit < kFlagCount; ++bit) {
      counts[static_cast<std::size_t>(bit)] += (report.flags >> bit) & 1U;
    }
    obstructed_share = std::max(
        obstructed_share, static_cast<double>(report.metric(Metric::kObstructedTileFraction)));
    overexposed_share = std::max(overexposed_share,
                                 static_cast<double>(report.metric(Metric::kOverexposedFraction)));
    color_cast = std::max(color_cast, static_cast<double>(report.metric(Metric::kColorCast)));
  }
  cudaStreamDestroy(stream);
  VERIFY_MSG(mismatched == 0, label + ": " + std::to_string(mismatched) + " of " +
                                  std::to_string(frames) + " frames differ between GPU and CPU");
  VERIFY_MSG(!never_frozen || counts[kFrozenBit] == 0,
             label + ": a live camera's frames flagged frozen");
  std::cout << "  " << label << " " << width << "x" << height << ": " << frames
            << " frames, GPU == CPU on " << frames - mismatched << ", flagged " << failed << ":";
  for (int bit = 0; bit < kFlagCount; ++bit) {
    if (counts[static_cast<std::size_t>(bit)] != 0) {
      std::cout << ' ' << flag_names(1U << static_cast<unsigned>(bit)) << ' '
                << counts[static_cast<std::size_t>(bit)];
    }
  }
  std::cout << std::fixed << std::setprecision(3) << "; most: obstructed tiles " << obstructed_share
            << ", overexposed pixels " << overexposed_share << ", color cast " << color_cast
            << '\n';
}

}  // namespace
}  // namespace holoscan::image_quality

int main() {
  namespace iq = holoscan::image_quality;
  namespace fs = std::filesystem;
  const char* root = std::getenv("HOLOSCAN_TEST_DATA");
  if (root == nullptr || *root == '\0') {
    std::cout << "HOLOSCAN_TEST_DATA is not set; skipping the real-data tests\n";
    return iq::kSkip;
  }
  const fs::path base = fs::path(root) / "stereo-camera";
  std::vector<fs::path> recordings;
  if (fs::is_directory(base)) {
    for (const auto& entry : fs::directory_iterator(base)) {
      if (fs::is_regular_file(entry.path() / "index.csv")) recordings.push_back(entry.path());
    }
  }
  std::sort(recordings.begin(), recordings.end());
  if (recordings.empty()) {
    // The variable asks for the real-data tests: missing files fail them.
    std::cerr << base << " holds no recordings with an index.csv\n";
    return 1;
  }
  std::vector<iq::test::TestCase> cases;
  for (std::size_t i = 0; i < recordings.size(); ++i) {
    const fs::path recording = recordings[i];
    const std::string name = "recording " + std::to_string(i + 1);
    cases.push_back(
        {name + " color", [=] { iq::check_stream(recording, name + " color", "rgb", true); }});
    cases.push_back({name + " infrared",
                     [=] { iq::check_stream(recording, name + " infrared", "left", false); }});
  }
  return iq::test::run(cases);
}
