// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The denoiser on the local recordings: for each clip in
// $HOLOSCAN_TEST_DATA/stereo-camera/<clip>/ with RGB frames (8-bit PPM, listed
// in index.csv), its first frames are denoised in order by each backend this
// build has. The recordings are of a still scene and have no clean truth, so the
// checks are what a still scene allows: the noise estimate is a plausible
// sensor's, the denoised frames are less noisy than the camera's (by the same
// estimate) and change less from frame to frame, their brightness is the
// camera's per channel, and the cuda backend gives the same bytes again.
// Skipped (exit 77) when HOLOSCAN_TEST_DATA is not set, as in CI, whose tests
// make their own data.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/denoise/denoiser.hpp>

#include "image_support.hpp"
#include "test_support.hpp"

namespace holoscan::denoise {
namespace {

namespace fs = std::filesystem;
using test::DeviceImage;

constexpr int kFrames = 16;
// The frames each check skips while the temporal filter fills its history.
constexpr int kSettle = 4;

struct RgbImage {
  int width{};
  int height{};
  std::vector<std::uint8_t> pixels;
};

// Skips whitespace and '#' comments in a PNM header.
void skip_space(std::istream& in) {
  while (in) {
    const int c = in.peek();
    if (c == '#') {
      std::string line;
      std::getline(in, line);
    } else if (std::isspace(c) != 0) {
      in.get();
    } else {
      return;
    }
  }
}

// A binary 8-bit PPM (P6).
RgbImage read_ppm(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::string magic;
  int max_value = 0;
  RgbImage image;
  in >> magic;
  skip_space(in);
  in >> image.width;
  skip_space(in);
  in >> image.height;
  skip_space(in);
  in >> max_value;
  in.get();
  if (!in || magic != "P6" || max_value != 255 || image.width <= 0 || image.height <= 0) {
    throw std::runtime_error(path.string() + ": not an 8-bit binary PPM");
  }
  image.pixels.resize(static_cast<std::size_t>(image.width) * image.height * 3U);
  in.read(reinterpret_cast<char*>(image.pixels.data()),
          static_cast<std::streamsize>(image.pixels.size()));
  if (!in) throw std::runtime_error(path.string() + ": truncated");
  return image;
}

// The first `count` RGB frames named by the clip's index.csv (column "rgb").
std::vector<RgbImage> read_clip(const fs::path& clip, int count) {
  std::ifstream index(clip / "index.csv");
  std::string line;
  std::getline(index, line);
  std::vector<std::string> columns;
  for (std::stringstream header(line); std::getline(header, line, ',');) columns.push_back(line);
  const auto rgb = std::find(columns.begin(), columns.end(), "rgb");
  if (rgb == columns.end()) throw std::runtime_error(clip.string() + ": index.csv has no rgb");
  const auto column = static_cast<std::size_t>(rgb - columns.begin());
  std::vector<RgbImage> frames;
  while (static_cast<int>(frames.size()) < count && std::getline(index, line)) {
    std::vector<std::string> cells;
    for (std::stringstream row(line); std::getline(row, line, ',');) cells.push_back(line);
    if (cells.size() <= column) throw std::runtime_error(clip.string() + ": short index row");
    frames.push_back(read_ppm(clip / cells[column]));
  }
  return frames;
}

// The noise standard deviation the denoiser estimates for `pixels`
// (Immerkaer's method on green): a pass at strength 0, which leaves them as
// they are.
float estimate_noise(const std::vector<std::uint8_t>& pixels, int width, int height) {
  DenoiseParams params;
  params.temporal = false;
  params.strength = 0.0F;
  Denoiser estimator(Backend::kCuda, params);
  DeviceImage in(PixelFormat::kRgb8, width, height);
  DeviceImage out(PixelFormat::kRgb8, width, height);
  in.upload(pixels);
  estimator.process(in.view(), out.view(), nullptr);
  return estimator.noise_sigma(nullptr);
}

// The mean absolute difference between two images.
double mean_change(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) sum += std::abs(static_cast<int>(a[i]) - b[i]);
  return sum / static_cast<double>(a.size());
}

std::array<double, 3> channel_means(const std::vector<std::uint8_t>& rgb) {
  std::array<double, 3> sums{};
  for (std::size_t i = 0; i < rgb.size(); ++i) sums[i % 3] += rgb[i];
  for (double& sum : sums) sum /= static_cast<double>(rgb.size() / 3);
  return sums;
}

// The frames denoised in order by a new denoiser of `backend`.
std::vector<std::vector<std::uint8_t>> denoise_clip(Backend backend,
                                                    const std::vector<RgbImage>& frames,
                                                    std::string& engine) {
  DenoiseParams params;
  if (backend == Backend::kVpi) params.spatial = SpatialFilter::kNone;
  Denoiser denoiser(backend, params);
  const int width = frames.front().width;
  const int height = frames.front().height;
  DeviceImage in(PixelFormat::kRgb8, width, height);
  DeviceImage out(PixelFormat::kRgb8, width, height);
  std::vector<std::vector<std::uint8_t>> result;
  for (const RgbImage& frame : frames) {
    in.upload(frame.pixels);
    denoiser.process(in.view(), out.view(), nullptr);
    result.push_back(out.download());
  }
  engine = std::string(denoiser.engine());
  return result;
}

void check_clip(const fs::path& clip) {
  const std::vector<RgbImage> frames = read_clip(clip, kFrames);
  const std::string name = clip.filename().string();
  VERIFY_MSG(frames.size() == kFrames, name + ": " + std::to_string(frames.size()) + " frames");
  if (frames.size() != kFrames) return;
  const int width = frames.front().width;
  const int height = frames.front().height;
  for (const RgbImage& frame : frames) {
    VERIFY_MSG(frame.width == width && frame.height == height, name + ": frames of one size");
  }
  const float noisy_sigma = estimate_noise(frames.back().pixels, width, height);
  double noisy_change = 0.0;
  for (int f = kSettle + 1; f < kFrames; ++f) {
    noisy_change += mean_change(frames[f].pixels, frames[f - 1].pixels);
  }
  noisy_change /= kFrames - kSettle - 1;
  std::printf("%s: %dx%d, camera noise %.2f, frame-to-frame change %.3f\n", name.c_str(), width,
              height, noisy_sigma, noisy_change);
  VERIFY_MSG(noisy_sigma > 0.5F && noisy_sigma < 20.0F,
             name + ": implausible noise estimate " + std::to_string(noisy_sigma));

  for (const Backend backend : available_backends()) {
    std::string engine;
    const auto denoised = denoise_clip(backend, frames, engine);
    const std::string what = name + " " + engine;
    const float sigma = estimate_noise(denoised.back(), width, height);
    double change = 0.0;
    double worst_shift = 0.0;
    for (int f = kSettle; f < kFrames; ++f) {
      if (f > kSettle) change += mean_change(denoised[f], denoised[f - 1]);
      const auto in = channel_means(frames[f].pixels);
      const auto out = channel_means(denoised[f]);
      for (int c = 0; c < 3; ++c) worst_shift = std::max(worst_shift, std::abs(out[c] - in[c]));
    }
    change /= kFrames - kSettle - 1;
    std::printf(
        "  %-9s noise %.2f (%.0f%% of the camera's), change %.3f (%.0f%%), "
        "brightness within %.3f\n",
        engine.c_str(), sigma, 100.0 * sigma / noisy_sigma, change, 100.0 * change / noisy_change,
        worst_shift);
    VERIFY_MSG(sigma < 0.8F * noisy_sigma, what + ": the denoised frame is not less noisy");
    VERIFY_MSG(change < 0.8 * noisy_change, what + ": the denoised frames change as much");
    VERIFY_MSG(worst_shift < 1.0, what + ": brightness moved by " + std::to_string(worst_shift));
    if (backend == Backend::kCuda) {
      std::string again;
      VERIFY_MSG(denoise_clip(backend, frames, again) == denoised, what + ": not deterministic");
    }
  }
}

}  // namespace
}  // namespace holoscan::denoise

int main() {
  namespace dn = holoscan::denoise;
  const char* data = std::getenv("HOLOSCAN_TEST_DATA");
  if (data == nullptr || *data == '\0') {
    std::printf("HOLOSCAN_TEST_DATA is not set: skipped\n");
    return 77;
  }
  const dn::fs::path root = dn::fs::path(data) / "stereo-camera";
  std::vector<dn::fs::path> clips;
  if (dn::fs::is_directory(root)) {
    for (const auto& entry : dn::fs::directory_iterator(root)) {
      if (entry.is_directory() && dn::fs::exists(entry.path() / "index.csv") &&
          dn::fs::is_directory(entry.path() / "rgb")) {
        clips.push_back(entry.path());
      }
    }
  }
  std::sort(clips.begin(), clips.end());
  std::vector<dn::test::TestCase> cases;
  cases.push_back({"recordings found", [&] {
                     VERIFY_MSG(!clips.empty(), root.string() + " has no clip with RGB frames");
                   }});
  for (const auto& clip : clips) {
    cases.push_back({"recording " + clip.filename().string(), [clip] { dn::check_clip(clip); }});
  }
  return dn::test::run(cases);
}
