// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// camera_info files laid out as ROS's camera_calibration writes them, made
// here rather than by the module's writer, so that reading is tested against
// an independent source.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "holoscan/messages/camera_info.hpp"

namespace holoscan::messages::test {

using Projection = std::array<double, 12>;

// How the numbers are written.
enum class RosPrecision {
  // 17 significant digits: every double reads back as itself.
  kExact,
  // As ROS's camera_calibration writes them: K and P with 5 decimals, D with
  // printf's "%8f" (6 decimals), R with 8, and each matrix row on a line of
  // its own.
  kCalibrator,
};

inline std::string ros_number(double value) {
  std::ostringstream text;
  text.precision(17);
  text << value;
  return text.str();
}

inline std::string ros_fixed(double value, int decimals, int width = 0) {
  char text[64];
  std::snprintf(text, sizeof(text), "%*.*f", width, decimals, value);
  return text;
}

inline std::string ros_matrix(std::string_view key, std::size_t rows, std::size_t cols,
                              const double* values) {
  std::string text = std::string(key) + ":\n  rows: " + std::to_string(rows) +
                     "\n  cols: " + std::to_string(cols) + "\n  data: [";
  for (std::size_t i = 0; i < rows * cols; ++i) {
    text += (i == 0 ? "" : ", ") + ros_number(values[i]);
  }
  return text + "]\n";
}

// A matrix the way the calibrator formats it: `decimals` decimals, a row per
// line, continued lines indented under the list.
inline std::string ros_calibrator_matrix(std::string_view key, std::size_t rows, std::size_t cols,
                                         const double* values, int decimals) {
  std::string text = std::string(key) + ":\n  rows: " + std::to_string(rows) +
                     "\n  cols: " + std::to_string(cols) + "\n  data: [";
  for (std::size_t i = 0; i < rows * cols; ++i) {
    if (i != 0) text += (i % cols == 0) ? ",\n         " : ", ";
    text += ros_fixed(values[i], decimals);
  }
  return text + "]\n";
}

inline std::string ros_coefficients(const std::vector<double>& d, RosPrecision precision) {
  if (precision == RosPrecision::kExact) {
    return ros_matrix("distortion_coefficients", 1, d.size(), d.data());
  }
  std::string text =
      "distortion_coefficients:\n  rows: 1\n  cols: " + std::to_string(d.size()) + "\n  data: [";
  for (std::size_t i = 0; i < d.size(); ++i) text += (i == 0 ? "" : ", ") + ros_fixed(d[i], 6, 8);
  return text + "]\n";
}

// K and D from `camera`, with R and P as given.
inline std::string ros_camera_info(const CameraInfo& camera, const Matrix3& r, const Projection& p,
                                   RosPrecision precision = RosPrecision::kExact) {
  const Matrix3 k{camera.fx, camera.skew, camera.cx, 0, camera.fy, camera.cy, 0, 0, 1};
  const bool exact = precision == RosPrecision::kExact;
  return "image_width: " + std::to_string(camera.width) +
         "\nimage_height: " + std::to_string(camera.height) +
         "\ncamera_name: " + camera.header.frame_id + "\n" +
         (exact ? ros_matrix("camera_matrix", 3, 3, k.data())
                : ros_calibrator_matrix("camera_matrix", 3, 3, k.data(), 5)) +
         "distortion_model: " + std::string(distortion_model_name(camera.distortion_model)) + "\n" +
         ros_coefficients(camera.coefficients, precision) +
         (exact ? ros_matrix("rectification_matrix", 3, 3, r.data())
                : ros_calibrator_matrix("rectification_matrix", 3, 3, r.data(), 8)) +
         (exact ? ros_matrix("projection_matrix", 3, 4, p.data())
                : ros_calibrator_matrix("projection_matrix", 3, 4, p.data(), 5));
}

// The rectified camera's projection f, cx, cy, with Tx = -f * baseline.
inline Projection ros_projection(double f, double cx, double cy, double baseline) {
  return {f, 0, cx, -f * baseline, 0, f, cy, 0, 0, 0, 1, 0};
}

// A raw pair's two files, as a stereo calibration writes them: K, D and R of
// each camera, P of the rectified cameras with focal length f and principal
// point (cx, cy).
inline std::pair<std::string, std::string> ros_stereo_camera_info(
    const StereoCameraInfo& pair, double f, double cx, double cy,
    RosPrecision precision = RosPrecision::kExact) {
  const Vector3& t = pair.right_from_left.translation;
  const double baseline = std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
  return {ros_camera_info(pair.left, pair.left.rectification.value_or(kIdentity3),
                          ros_projection(f, cx, cy, 0.0), precision),
          ros_camera_info(pair.right, pair.right.rectification.value_or(kIdentity3),
                          ros_projection(f, cx, cy, baseline), precision)};
}

// `camera` as read back from a kCalibrator file: K and D rounded as written.
// (R, rounded too, reads back as the rotation nearest to it.)
inline CameraInfo ros_calibrator_rounded(CameraInfo camera) {
  const auto rounded = [](double value, int decimals) {
    return std::stod(ros_fixed(value, decimals));
  };
  for (double* k : {&camera.fx, &camera.fy, &camera.cx, &camera.cy, &camera.skew}) {
    *k = rounded(*k, 5);
  }
  for (double& d : camera.coefficients) d = rounded(d, 6);
  return camera;
}

}  // namespace holoscan::messages::test
