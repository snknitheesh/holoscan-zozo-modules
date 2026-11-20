// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A recording's color frames: a folder with an index.csv of one line per frame naming each
// frame's 8-bit RGB PPM image, and the color camera's ROS camera_info file, rgb.yaml, beside
// them. No SDK.

#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace fiducial_app {

// The camera_info file of a recording's color camera, in `clip`.
[[nodiscard]] std::filesystem::path rgb_camera_info(const std::filesystem::path& clip);

struct RecordedFrame {
  std::uint64_t frame{};
  std::uint64_t device_sequence{};
  std::filesystem::path rgb;
};

// The frames of the recording in `clip`, from its index.csv (columns frame, device_sequence
// and rgb, by name, among any others), in order.
// @throws std::invalid_argument naming the file and line of a problem.
[[nodiscard]] std::vector<RecordedFrame> read_index(const std::filesystem::path& clip);

struct RgbImage {
  int width{};
  int height{};
  std::vector<std::uint8_t> pixels;  // Packed RGB rows.
};

// An 8-bit binary PPM (P6, maxval 255).
// @throws std::invalid_argument naming the file and the problem.
[[nodiscard]] RgbImage read_ppm(const std::filesystem::path& path);

}  // namespace fiducial_app
