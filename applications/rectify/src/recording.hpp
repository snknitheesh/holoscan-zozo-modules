// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A stereo recording's frames: a folder with an index.csv of one line per
// frame, and 8-bit PGM left and right images. The pair's calibration beside
// them, left.yaml and right.yaml, is ROS camera_info files, which
// StereoRectifyOp reads itself. No SDK.

#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace rectify_app {

// The camera_info files of a recording's left and right cameras, in `clip`.
[[nodiscard]] std::filesystem::path left_camera_info(const std::filesystem::path& clip);
[[nodiscard]] std::filesystem::path right_camera_info(const std::filesystem::path& clip);

struct RecordedFrame {
  std::uint64_t frame{};
  std::int64_t timestamp_ns{};
  std::uint64_t device_sequence{};
  std::filesystem::path left;
  std::filesystem::path right;
};

// The frames of the recording in `clip`, from its index.csv, in order.
// @throws std::invalid_argument naming the file and line of a problem.
[[nodiscard]] std::vector<RecordedFrame> read_index(const std::filesystem::path& clip);

struct GrayImage {
  int width{};
  int height{};
  std::vector<std::uint8_t> pixels;  // Packed rows.
};

// An 8-bit binary PGM (P5, maxval 255).
// @throws std::invalid_argument naming the file and the problem.
[[nodiscard]] GrayImage read_pgm(const std::filesystem::path& path);

}  // namespace rectify_app
