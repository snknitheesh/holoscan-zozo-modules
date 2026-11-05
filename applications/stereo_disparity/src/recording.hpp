// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A small reader for stereo camera recordings, as the demo plays them: a
// folder with the pair's ROS camera_info files left.yaml and right.yaml (and
// depth.yaml when it has depth images), index.csv, 8-bit PGM left and right
// images and 16-bit PGM depth images (the format of the local test
// recordings, `$HOLOSCAN_TEST_DATA/stereo-camera/<clip>/`). Only what the
// demo needs is read. No SDK, CUDA or GPU.

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/messages/camera_info.hpp>

namespace stereo_disparity_app {

struct RecordingFrame {
  std::uint64_t frame{};
  std::filesystem::path left;
  std::filesystem::path right;
  std::filesystem::path depth;  // Empty without a depth image.
};

struct Recording {
  std::filesystem::path folder;
  // The pair's camera_info files, and what they hold.
  std::filesystem::path left_camera_info;
  std::filesystem::path right_camera_info;
  holoscan::messages::StereoCameraInfo camera_info;
  // Whether the depth images are aligned with the left image: depth.yaml
  // describes the left camera (the same size and intrinsics, no distortion).
  bool depth_aligned{};
  std::vector<RecordingFrame> frames;
};

// Reads a recording's camera_info files and index; the images are read per
// frame.
// @throws std::runtime_error naming the file and what is wrong: a file that
//         cannot be read or parsed, or no frames.
[[nodiscard]] Recording read_recording(const std::filesystem::path& folder);

// Whether a depth camera describes the left camera's pixels: the same size
// and intrinsics, and no distortion.
[[nodiscard]] bool same_pixels(const holoscan::messages::CameraInfo& depth,
                               const holoscan::messages::CameraInfo& left);

// An 8- or 16-bit binary PGM (P5); 16-bit samples are big-endian, as the
// format says. Samples are returned as they are, widened to 16 bits.
struct Pgm {
  int width{};
  int height{};
  int max_value{};
  std::vector<std::uint16_t> pixels;
};
// @throws std::runtime_error naming the file if it is not a binary PGM.
[[nodiscard]] Pgm read_pgm(const std::filesystem::path& path);
// The same, from a PGM's bytes; `name` names it in errors.
[[nodiscard]] Pgm parse_pgm(const std::string& bytes, const std::string& name);

}  // namespace stereo_disparity_app
