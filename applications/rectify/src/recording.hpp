// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A stereo recording's frames, as holoscan-stereo-camera records them: an
// index.csv with one line per frame, and 8-bit PGM left and right infrared
// images. The calibration.json beside them is read with holoscan-rectify's
// read_stereo_calibration(). No SDK.

#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace rectify_app {

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
