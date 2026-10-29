// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Writes a small recording of the module's synthetic scene in the recording
// format the demo reads: calibration.json, index.csv, 8-bit PGM eyes and 16-bit
// big-endian PGM depth in millimeters, 0 where the right eye does not see the
// point. Its depth is the scene's truth, so the demo's camera-depth check has a
// known answer without any real data.
//
//   write_test_recording <folder> <frames> <width> <height>

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <holoscan/stereo_disparity/synthetic_scene.hpp>

namespace sd = holoscan::stereo_disparity;

namespace {

void write_pgm(const std::filesystem::path& path, int width, int height, int max_value,
               const std::vector<std::uint16_t>& pixels) {
  std::ofstream file(path, std::ios::binary);
  file << "P5\n" << width << ' ' << height << '\n' << max_value << '\n';
  for (const std::uint16_t value : pixels) {
    if (max_value > 255) file.put(static_cast<char>(value >> 8U));
    file.put(static_cast<char>(value & 0xffU));
  }
  if (!file) throw std::runtime_error("cannot write " + path.string());
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc != 5) {
    std::cerr << "usage: write_test_recording <folder> <frames> <width> <height>\n";
    return 2;
  }
  const std::filesystem::path folder = argv[1];
  const int frames = std::atoi(argv[2]);
  const int width = std::atoi(argv[3]);
  const int height = std::atoi(argv[4]);
  std::filesystem::remove_all(folder);
  for (const char* part : {"left", "right", "depth"}) {
    std::filesystem::create_directories(folder / part);
  }
  const sd::SyntheticScene first = sd::make_scene(0, width, height);
  std::ofstream calibration(folder / "calibration.json");
  calibration << std::setprecision(17) << "{\n  \"version\": 1,\n"
              << "  \"camera\": {\"name\": \"Synthetic\", \"serial\": \"0\"},\n"
              << "  \"rectified\": true,\n";
  for (const char* eye : {"left", "right", "depth"}) {
    calibration << "  \"" << eye << "\": {\"width\": " << width << ", \"height\": " << height
                << ", \"fx\": " << first.focal << ", \"fy\": " << first.focal
                << ", \"cx\": " << first.cx << ", \"cy\": " << first.cy
                << ", \"distortion\": {\"model\": \"none\", \"coefficients\": []}";
    if (std::string(eye) == "depth") {
      calibration << ", \"from_left\": {\"rotation\": [1, 0, 0, 0, 1, 0, 0, 0, 1], "
                     "\"translation\": [0, 0, 0]}, \"scale\": 0.001";
    }
    calibration << "},\n";
  }
  calibration << "  \"right_from_left\": {\"rotation\": [1, 0, 0, 0, 1, 0, 0, 0, 1], "
              << "\"translation\": [" << -first.baseline << ", 0, 0]}\n}\n";
  std::ofstream index(folder / "index.csv");
  index << "frame,timestamp_ns,device_sequence,left,right,rgb,depth\n";

  const std::size_t pixels = static_cast<std::size_t>(width) * height;
  std::uint8_t* gpu = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&gpu), 2 * pixels) != cudaSuccess)
    throw std::runtime_error("cudaMalloc");
  std::vector<std::uint8_t> pair(2 * pixels);
  for (int frame = 0; frame < frames; ++frame) {
    const sd::SyntheticScene scene =
        sd::make_scene(static_cast<std::uint32_t>(frame), width, height);
    if (sd::render_scene(scene, 1, gpu, gpu + pixels, nullptr) != cudaSuccess ||
        cudaMemcpy(pair.data(), gpu, pair.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error("render_scene");
    }
    std::ostringstream name;
    name << std::setw(6) << std::setfill('0') << frame << ".pgm";
    write_pgm(folder / "left" / name.str(), width, height, 255,
              std::vector<std::uint16_t>(pair.begin(), pair.begin() + pixels));
    write_pgm(folder / "right" / name.str(), width, height, 255,
              std::vector<std::uint16_t>(pair.begin() + pixels, pair.end()));
    std::vector<std::uint16_t> depth(pixels);
    for (int v = 0; v < height; ++v) {
      for (int u = 0; u < width; ++u) {
        const float d = sd::true_disparity(scene, static_cast<float>(u), static_cast<float>(v));
        depth[static_cast<std::size_t>(v) * width + u] =
            d > 0.0F
                ? static_cast<std::uint16_t>(std::lround(1000.0 * scene.focal * scene.baseline / d))
                : 0U;
      }
    }
    write_pgm(folder / "depth" / name.str(), width, height, 65535, depth);
    index << frame << ',' << 1000000000LL * (frame + 1) << ',' << frame + 1 << ",left/"
          << name.str() << ",right/" << name.str() << ",,depth/" << name.str() << '\n';
  }
  cudaFree(gpu);
  return 0;
} catch (const std::exception& error) {
  std::cerr << "write_test_recording: " << error.what() << '\n';
  return 1;
}
