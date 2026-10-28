// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Checks detections against the expected detections listed per image file,
// and reads the files that hold expectations and class names. No Holoscan.
// (Synthetic frames are checked by holoscan::yolo26::match_truth().)

#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <holoscan/yolo26/synthetic_scene.hpp>

namespace yolo26_app {

using holoscan::yolo26::BoxDetection;
using holoscan::yolo26::iou;

// A detection an image file must yield: its class, and a box that a
// detection of that class must overlap by an IoU of at least the threshold.
struct Expectation {
  std::string image;
  BoxDetection box;
};

// Reads lines of `image,class_id,x1,y1,x2,y2`, where `image` is a file name
// without its directory. Blank lines and lines starting with # are skipped.
// @throws std::runtime_error naming the file and line of a malformed entry.
[[nodiscard]] std::vector<Expectation> read_expectations(const std::filesystem::path& path);

// How many of `expected` a distinct detection of the same class overlaps by
// an IoU of at least `min_iou`.
[[nodiscard]] std::uint32_t count_matched(std::span<const BoxDetection> detections,
                                          std::span<const Expectation> expected, float min_iou);

// One class name per line; a trailing empty line is dropped.
// @throws std::runtime_error if the file cannot be read or holds no name.
[[nodiscard]] std::vector<std::string> read_labels(const std::filesystem::path& path);

}  // namespace yolo26_app
