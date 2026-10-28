// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A small reader for stereo camera recordings, as the demo plays them: a
// folder with calibration.json, index.csv, 8-bit PGM left and right images and
// 16-bit PGM depth images (the format of the stereo-camera test recordings,
// `$HOLOSCAN_TEST_DATA/stereo-camera/<clip>/`). Only what the demo needs is
// read. No SDK, CUDA or GPU.

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace stereo_disparity_app {

// What the demo takes from calibration.json.
struct RecordingCalibration {
  std::string camera;  // The camera's name and serial, for the log.
  int width{};         // The eyes' size.
  int height{};
  double fx{};  // The left eye's, in pixels.
  double cx{};  // The left eye's principal point.
  double cy{};
  double right_cx{};  // The right eye's principal point x.
  double baseline{};  // Meters; the right eye sits this far to the left eye's right.
  // The depth images, when the recording has them aligned with the left eye
  // (the same size and intrinsics, no offset): meters per unit.
  std::optional<double> depth_scale;
};

struct RecordingFrame {
  std::uint64_t frame{};
  std::filesystem::path left;
  std::filesystem::path right;
  std::filesystem::path depth;  // Empty without a depth image.
};

struct Recording {
  std::filesystem::path folder;
  RecordingCalibration calibration;
  std::vector<RecordingFrame> frames;
};

// Reads a recording's calibration and index; the images are read per frame.
// @throws std::runtime_error naming the file and what is wrong: a file that
//         cannot be read or parsed, eyes that are not rectified, a right eye
//         that is not to the left eye's right, or no frames.
[[nodiscard]] Recording read_recording(const std::filesystem::path& folder);

// The calibration from calibration.json's text.
// @throws std::runtime_error as above.
[[nodiscard]] RecordingCalibration parse_calibration(const std::string& json);

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
