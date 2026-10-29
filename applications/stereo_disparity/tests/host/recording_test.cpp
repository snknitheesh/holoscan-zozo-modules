// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The recording reader: calibration.json as the stereo camera recordings write
// it, the geometry it gives, what it refuses, and 8- and 16-bit PGM images.

#include "recording.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "test_support.hpp"

namespace stereo_disparity_app {
namespace {

using test::throws;

// A RealSense D585's calibration at 1280x720, as recorded.
constexpr char kCalibration[] = R"({
  "version": 1,
  "camera": {
    "name": "RealSense D585 Prototype",
    "serial": "541522320091",
    "settings": {"emitter_enabled": "true", "fps": "30"}
  },
  "rectified": true,
  "left": {"width": 1280, "height": 720, "fx": 336.98760986328125, "fy": 336.98760986328125,
           "cx": 674.053955078125, "cy": 355.81134033203125,
           "distortion": {"model": "brown_conrady", "coefficients": [0, 0, 0, 0, 0]}},
  "right": {"width": 1280, "height": 720, "fx": 336.98760986328125, "fy": 336.98760986328125,
            "cx": 676.553955078125, "cy": 355.81134033203125,
            "distortion": {"model": "brown_conrady", "coefficients": [0, 0, 0, 0, 0]}},
  "right_from_left": {"rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                      "translation": [-0.094843290746212, 0, 0]},
  "rgb": {"width": 1280, "height": 720, "fx": 331.4, "fy": 331.6, "cx": 702.4, "cy": 382.2,
          "from_left": {"rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        "translation": [-0.0476, 0.0004, -9.1e-05]}},
  "depth": {"width": 1280, "height": 720, "fx": 336.98760986328125, "fy": 336.98760986328125,
            "cx": 674.053955078125, "cy": 355.81134033203125,
            "from_left": {"rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1], "translation": [0, 0, 0]},
            "scale": 0.00045200000749900937}
})";

std::string replaced(const char* original, const std::string& from, const std::string& to) {
  std::string text(original);
  const std::size_t at = text.find(from);
  if (at == std::string::npos) throw std::logic_error("no '" + from + "' to replace");
  return text.replace(at, from.size(), to);
}

void check_calibration() {
  const RecordingCalibration c = parse_calibration(kCalibration);
  VERIFY(c.camera == "RealSense D585 Prototype 541522320091");
  VERIFY(c.width == 1280 && c.height == 720);
  VERIFY(c.fx == 336.98760986328125 && c.cx == 674.053955078125);
  VERIFY(c.right_cx == 676.553955078125);
  VERIFY(c.baseline == 0.094843290746212);
  VERIFY(c.depth_scale && *c.depth_scale == 0.00045200000749900937);
}

void check_refused_calibrations() {
  const auto refuses = [](const std::string& json, const std::string& what) {
    try {
      (void)parse_calibration(json);
      VERIFY_MSG(false, "refused: " + what);
    } catch (const std::runtime_error& error) {
      VERIFY_MSG(std::string(error.what()).find(what) != std::string::npos,
                 "the error says '" + what + "': " + error.what());
    }
  };
  refuses(replaced(kCalibration, "\"rectified\": true", "\"rectified\": false"), "not rectified");
  refuses(replaced(kCalibration, "-0.094843290746212", "0.094843290746212"),
          "to the left eye's right");
  refuses(replaced(kCalibration, "\"fx\": 336.98760986328125, \"fy\"", "\"fy\""), "left.fx");
  refuses(replaced(kCalibration, "\"right\": {\"width\": 1280", "\"right\": {\"width\": 640"),
          "differ in size");
  refuses(std::string(kCalibration).substr(0, 200), "JSON");
  refuses("[1, 2", "JSON");
  // Depth that is not aligned with the left eye is left out, not refused.
  const RecordingCalibration unaligned = parse_calibration(
      replaced(kCalibration, "\"translation\": [0, 0, 0]", "\"translation\": [0.01, 0, 0]"));
  VERIFY(!unaligned.depth_scale.has_value());
}

void check_pgm() {
  std::string eight = "P5\n# a comment\n3 2\n255\n";
  eight += std::string("\x01\x02\x03\x04\x05\xff", 6);
  const Pgm small = parse_pgm(eight, "eight");
  VERIFY(small.width == 3 && small.height == 2 && small.max_value == 255);
  VERIFY(small.pixels.size() == 6 && small.pixels[0] == 1 && small.pixels[5] == 255);
  std::string sixteen = "P5 2 1 65535\n";
  sixteen += std::string("\x12\x34\xff\x01", 4);
  const Pgm wide = parse_pgm(sixteen, "sixteen");
  VERIFY(wide.pixels.size() == 2 && wide.pixels[0] == 0x1234 && wide.pixels[1] == 0xff01);
  VERIFY(throws<std::runtime_error>([] { (void)parse_pgm("P2 1 1 255\n1", "ascii"); }));
  VERIFY(throws<std::runtime_error>([] { (void)parse_pgm("P5 4 4 255\n\x01", "short"); }));
  VERIFY(throws<std::runtime_error>([] { (void)parse_pgm("P5 0 4 255\n", "empty"); }));
}

void check_recording_folder() {
  const std::filesystem::path folder =
      std::filesystem::temp_directory_path() / "stereo_disparity_recording_test";
  std::filesystem::remove_all(folder);
  std::filesystem::create_directories(folder);
  std::ofstream(folder / "calibration.json") << kCalibration;
  std::ofstream(folder / "index.csv")
      << "frame,timestamp_ns,device_sequence,left,right,rgb,depth\n"
      << "0,1,31,left/000000.pgm,right/000000.pgm,rgb/000000.ppm,depth/000000.pgm\n"
      << "1,2,32,left/000001.pgm,right/000001.pgm,rgb/000001.ppm,\n";
  const Recording recording = read_recording(folder);
  VERIFY(recording.frames.size() == 2);
  VERIFY(recording.frames[0].left == folder / "left/000000.pgm");
  VERIFY(recording.frames[0].depth == folder / "depth/000000.pgm");
  VERIFY(recording.frames[1].frame == 1 && recording.frames[1].depth.empty());
  std::ofstream(folder / "index.csv") << "frame,left\n0,left/000000.pgm\n";
  VERIFY(throws<std::runtime_error>([&] { (void)read_recording(folder); }));
  std::filesystem::remove_all(folder);
  VERIFY(throws<std::runtime_error>([&] { (void)read_recording(folder); }));
}

}  // namespace
}  // namespace stereo_disparity_app

int main() {
  using namespace stereo_disparity_app;  // NOLINT(build/namespaces)
  return test::run({
      {"calibration", check_calibration},
      {"refused calibrations", check_refused_calibrations},
      {"PGM images", check_pgm},
      {"recording folder", check_recording_folder},
  });
}
