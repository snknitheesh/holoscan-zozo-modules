// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The local recordings' RGB frames through VideoEncoder and VideoDecoder: every frame of each
// recording comes back once, with its own pts, at the quality each setting must reach on camera
// images (H.264 VBR at 8 Mbit/s, H.265 at a constant QP of 25, H.264 CBR at 2 Mbit/s). Reads
// $HOLOSCAN_TEST_DATA/stereo-camera/<recording>/index.csv and its rgb/ PPM frames; skipped (exit
// 77) when the variable is unset, failed when it names no recording. On the backend auto picks,
// or in CI the lane's backend (ci_backend.hpp).

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/codec/test_video.hpp>
#include <holoscan/codec/video_codec.hpp>

#include "ci_backend.hpp"
#include "test_support.hpp"

namespace holoscan::codec {
namespace {

namespace fs = std::filesystem;

struct Image {
  int width{};
  int height{};
  std::vector<std::uint8_t> rgb;
};

// An 8-bit binary PPM (P6), as the recordings hold their RGB frames.
Image read_ppm(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  std::string magic;
  int max_value = 0;
  Image image;
  file >> magic >> image.width >> image.height >> max_value;
  file.get();
  if (!file || magic != "P6" || max_value != 255 || image.width < 1 || image.height < 1) {
    throw std::runtime_error(path.string() + ": not an 8-bit binary PPM");
  }
  image.rgb.resize(static_cast<std::size_t>(image.width) * image.height * 3);
  file.read(reinterpret_cast<char*>(image.rgb.data()),
            static_cast<std::streamsize>(image.rgb.size()));
  if (!file) throw std::runtime_error(path.string() + ": truncated");
  return image;
}

// The RGB frames of one recording, in index.csv's order (its "rgb" column).
std::vector<fs::path> rgb_frames(const fs::path& recording) {
  std::ifstream index(recording / "index.csv");
  std::string line;
  if (!std::getline(index, line)) return {};
  std::vector<std::string> columns;
  std::stringstream header(line);
  for (std::string column; std::getline(header, column, ',');) columns.push_back(column);
  const auto rgb = std::find(columns.begin(), columns.end(), "rgb");
  if (rgb == columns.end()) return {};
  const auto column = static_cast<std::size_t>(rgb - columns.begin());
  std::vector<fs::path> frames;
  while (std::getline(index, line)) {
    std::vector<std::string> cells;
    std::stringstream row(line);
    for (std::string cell; std::getline(row, cell, ',');) cells.push_back(cell);
    if (cells.size() > column) frames.push_back(recording / cells[column]);
  }
  return frames;
}

struct Setting {
  const char* name;
  Codec codec;
  RateControl rate_control;
  std::uint32_t bitrate;
  int qp;
  // Mean PSNR over the recording; measured on the RTX 5090 at 1.5 to 3 dB more.
  double min_mean_psnr;
};

constexpr Setting kSettings[] = {
    {"h264 vbr 8 Mbit/s", Codec::kH264, RateControl::kVbr, 8'000'000, 23, 37.0},
    {"h265 cqp 25", Codec::kH265, RateControl::kConstQp, 8'000'000, 25, 35.5},
    {"h264 cbr 2 Mbit/s", Codec::kH264, RateControl::kCbr, 2'000'000, 23, 33.5},
};

void check_recording(const std::vector<Image>& images, const std::string& name, Backend backend) {
  const int width = images.front().width;
  const int height = images.front().height;
  const std::size_t bytes = images.front().rgb.size();
  cudaStream_t stream = nullptr;
  void* device = nullptr;
  void* out = nullptr;
  VERIFY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  VERIFY(cudaMalloc(&device, bytes) == cudaSuccess && cudaMalloc(&out, bytes) == cudaSuccess);
  for (const Setting& setting : kSettings) {
    EncoderSettings settings;
    settings.codec = setting.codec;
    settings.backend = backend;
    settings.rate_control = setting.rate_control;
    settings.bitrate = setting.bitrate;
    settings.qp = setting.qp;
    VideoEncoder encoder(settings);
    VideoDecoder decoder(setting.codec, backend);
    double sum = 0.0;
    double worst = 1e9;
    std::size_t decoded = 0;
    std::size_t stream_bytes = 0;
    const auto measure = [&](const DecodedFrame& frame) {
      const auto index = static_cast<std::size_t>(frame.pts);
      VERIFY_MSG(index == decoded,
                 "frame " + std::to_string(decoded) + " has the pts of " + std::to_string(index));
      convert_from_nv12(frame.image, PixelFormat::kRgb8, static_cast<std::uint8_t*>(out),
                        static_cast<std::size_t>(width) * 3, width, height, stream);
      std::vector<std::uint8_t> host(bytes);
      VERIFY(cudaMemcpyAsync(host.data(), out, bytes, cudaMemcpyDeviceToHost, stream) ==
             cudaSuccess);
      VERIFY(cudaStreamSynchronize(stream) == cudaSuccess);
      const double value =
          index < images.size() ? psnr(images[index].rgb, host, width, height, 3) : 0.0;
      sum += value;
      worst = std::min(worst, value);
      ++decoded;
    };
    for (std::size_t i = 0; i < images.size(); ++i) {
      VERIFY(cudaMemcpyAsync(device, images[i].rgb.data(), bytes, cudaMemcpyHostToDevice, stream) ==
             cudaSuccess);
      const EncodedFrame unit =
          encoder.encode(DeviceImage{.data = static_cast<const std::uint8_t*>(device),
                                     .pitch = static_cast<std::size_t>(width) * 3,
                                     .width = width,
                                     .height = height},
                         static_cast<std::int64_t>(i), stream);
      stream_bytes += unit.data.size();
      decoder.decode(unit.data, unit.pts, stream, measure);
    }
    decoder.flush(stream, measure);
    const double mean = decoded == 0 ? 0.0 : sum / static_cast<double>(decoded);
    std::cout << "  " << name << ' ' << setting.name << " on " << backend_name(encoder.backend())
              << ": " << decoded << '/' << images.size() << " frames, PSNR mean " << std::fixed
              << std::setprecision(1) << mean << " dB, worst " << worst << " dB, "
              << static_cast<double>(stream_bytes) * 8.0 * 30.0 /
                     static_cast<double>(images.size()) / 1e6
              << " Mbit/s at 30 frames/s\n";
    VERIFY_MSG(decoded == images.size(),
               name + " " + setting.name + ": " + std::to_string(decoded) + " frames decoded");
    VERIFY_MSG(mean >= setting.min_mean_psnr, name + " " + setting.name + ": mean PSNR " +
                                                  std::to_string(mean) + " below " +
                                                  std::to_string(setting.min_mean_psnr));
  }
  cudaFree(device);
  cudaFree(out);
  cudaStreamDestroy(stream);
}

}  // namespace
}  // namespace holoscan::codec

int main() {
  using namespace holoscan::codec;  // NOLINT(build/namespaces)
  const char* root = std::getenv("HOLOSCAN_TEST_DATA");
  if (root == nullptr || *root == '\0') {
    std::cout << "SKIPPED: HOLOSCAN_TEST_DATA is not set\n";
    return test::kSkipped;
  }
  const Backend backend = test::ci_backend().value_or(Backend::kAuto);
  std::vector<std::filesystem::path> recordings;
  const std::filesystem::path base = std::filesystem::path(root) / "stereo-camera";
  if (std::filesystem::is_directory(base)) {
    for (const auto& entry : std::filesystem::directory_iterator(base)) {
      if (entry.is_directory() && !rgb_frames(entry.path()).empty()) {
        recordings.push_back(entry.path());
      }
    }
  }
  std::sort(recordings.begin(), recordings.end());
  if (recordings.empty()) {
    std::cout << "FAIL: no recording with RGB frames under " << base << '\n';
    return 1;
  }
  std::vector<test::TestCase> cases;
  for (std::size_t r = 0; r < recordings.size(); ++r) {
    const std::filesystem::path recording = recordings[r];
    const std::string name = "recording " + std::to_string(r + 1);
    cases.push_back({name, [=] {
                       std::vector<Image> images;
                       for (const auto& file : rgb_frames(recording)) {
                         images.push_back(read_ppm(file));
                       }
                       VERIFY(!images.empty());
                       if (images.empty()) return;
                       std::cout << "  " << name << ": " << images.size() << " frames of "
                                 << images.front().width << 'x' << images.front().height << '\n';
                       check_recording(images, name, backend);
                     }});
  }
  return test::run(cases);
}
