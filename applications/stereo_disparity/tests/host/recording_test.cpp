// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The recording reader: a recording's camera_info files and index as the
// local recordings hold them, whether its depth is aligned with the left
// image, what it refuses, and 8- and 16-bit PGM images.

#include "recording.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <holoscan/messages/camera_info_yaml.hpp>

#include "test_support.hpp"

namespace stereo_disparity_app {
namespace {

namespace hm = holoscan::messages;
using test::throws;

// A rectified 1280x720 pair with a 95 mm baseline, laid out as the local
// recordings' files are; generic intrinsics, with fractions that must survive
// the files.
hm::StereoCameraInfo rectified_pair() {
  hm::CameraInfo left;
  left.header.frame_id = "left";
  left.width = 1280;
  left.height = 720;
  left.state = hm::CameraState::kRectified;
  left.fx = 340.125;
  left.fy = 340.125;
  left.cx = 640.375;
  left.cy = 359.625;
  hm::CameraInfo right = left;
  right.header.frame_id = "right";
  hm::StereoCameraInfo pair;
  pair.header.frame_id = "left";
  pair.left = left;
  pair.right = right;
  pair.right_from_left.translation = {-0.095125, 0.0, 0.0};
  pair.baseline = 0.095125;
  return pair;
}

void check_same_pixels() {
  const hm::StereoCameraInfo pair = rectified_pair();
  hm::CameraInfo depth = pair.left;
  depth.header.frame_id = "depth";
  depth.state = hm::CameraState::kUndistorted;
  VERIFY(same_pixels(depth, pair.left));
  hm::CameraInfo other = depth;
  other.cx += 1.0;
  VERIFY(!same_pixels(other, pair.left));
  other = depth;
  other.width = 640;
  VERIFY(!same_pixels(other, pair.left));
  other = depth;
  other.state = hm::CameraState::kRaw;
  other.distortion_model = hm::DistortionModel::kPlumbBob;
  other.coefficients = {0.1, 0.0, 0.0, 0.0, 0.0};
  VERIFY(!same_pixels(other, pair.left));
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
  const hm::StereoCameraInfo pair = rectified_pair();
  hm::write_stereo_camera_info_yaml(pair, folder / "left.yaml", folder / "right.yaml");
  hm::CameraInfo depth = pair.left;
  depth.header.frame_id = "depth";
  hm::write_camera_info_yaml(depth, folder / "depth.yaml");
  std::ofstream(folder / "index.csv")
      << "frame,timestamp_ns,device_sequence,left,right,rgb,depth\n"
      << "0,1,31,left/000000.pgm,right/000000.pgm,rgb/000000.ppm,depth/000000.pgm\n"
      << "1,2,32,left/000001.pgm,right/000001.pgm,rgb/000001.ppm,\n";
  Recording recording = read_recording(folder);
  VERIFY(recording.left_camera_info == folder / "left.yaml");
  VERIFY(recording.right_camera_info == folder / "right.yaml");
  VERIFY(recording.camera_info.left.state == hm::CameraState::kRectified);
  VERIFY(recording.camera_info.left.fx == pair.left.fx &&
         recording.camera_info.baseline == pair.baseline);
  VERIFY(recording.depth_aligned);
  VERIFY(recording.frames.size() == 2);
  VERIFY(recording.frames[0].left == folder / "left/000000.pgm");
  VERIFY(recording.frames[0].depth == folder / "depth/000000.pgm");
  VERIFY(recording.frames[1].frame == 1 && recording.frames[1].depth.empty());
  // Depth of another camera is left out, not refused; so is no depth.yaml.
  depth.cx += 10.0;
  hm::write_camera_info_yaml(depth, folder / "depth.yaml");
  VERIFY(!read_recording(folder).depth_aligned);
  std::filesystem::remove(folder / "depth.yaml");
  VERIFY(!read_recording(folder).depth_aligned);
  // A bad index, a missing camera_info file, no folder.
  std::ofstream(folder / "index.csv") << "frame,left\n0,left/000000.pgm\n";
  VERIFY(throws<std::runtime_error>([&] { (void)read_recording(folder); }));
  std::filesystem::remove(folder / "right.yaml");
  try {
    (void)read_recording(folder);
    VERIFY_MSG(false, "a recording without right.yaml is refused");
  } catch (const std::runtime_error& error) {
    VERIFY_MSG(std::string(error.what()).find("right.yaml") != std::string::npos,
               std::string("the error names right.yaml: ") + error.what());
  }
  std::filesystem::remove_all(folder);
  VERIFY(throws<std::runtime_error>([&] { (void)read_recording(folder); }));
}

}  // namespace
}  // namespace stereo_disparity_app

int main() {
  using namespace stereo_disparity_app;  // NOLINT(build/namespaces)
  return test::run({
      {"depth aligned with the left image", check_same_pixels},
      {"PGM images", check_pgm},
      {"recording folder", check_recording_folder},
  });
}
