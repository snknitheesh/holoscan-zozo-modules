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
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "holoscan/messages/camera_info.hpp"

namespace holoscan::messages::test {

using Projection = std::array<double, 12>;

inline std::string ros_number(double value) {
  std::ostringstream text;
  text.precision(17);
  text << value;
  return text.str();
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

// K and D from `camera`, with R and P as given.
inline std::string ros_camera_info(const CameraInfo& camera, const Matrix3& r,
                                   const Projection& p) {
  const Matrix3 k{camera.fx, camera.skew, camera.cx, 0, camera.fy, camera.cy, 0, 0, 1};
  return "image_width: " + std::to_string(camera.width) +
         "\nimage_height: " + std::to_string(camera.height) +
         "\ncamera_name: " + camera.header.frame_id + "\n" +
         ros_matrix("camera_matrix", 3, 3, k.data()) +
         "distortion_model: " + std::string(distortion_model_name(camera.distortion_model)) + "\n" +
         ros_matrix("distortion_coefficients", 1, camera.coefficients.size(),
                    camera.coefficients.data()) +
         ros_matrix("rectification_matrix", 3, 3, r.data()) +
         ros_matrix("projection_matrix", 3, 4, p.data());
}

// The rectified camera's projection f, cx, cy, with Tx = -f * baseline.
inline Projection ros_projection(double f, double cx, double cy, double baseline) {
  return {f, 0, cx, -f * baseline, 0, f, cy, 0, 0, 0, 1, 0};
}

// A raw pair's two files, as a stereo calibration writes them: K, D and R of
// each camera, P of the rectified cameras with focal length f and principal
// point (cx, cy).
inline std::pair<std::string, std::string> ros_stereo_camera_info(const StereoCameraInfo& pair,
                                                                  double f, double cx, double cy) {
  const Vector3& t = pair.right_from_left.translation;
  const double baseline = std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
  return {ros_camera_info(pair.left, pair.left.rectification.value_or(kIdentity3),
                          ros_projection(f, cx, cy, 0.0)),
          ros_camera_info(pair.right, pair.right.rectification.value_or(kIdentity3),
                          ros_projection(f, cx, cy, baseline))};
}

}  // namespace holoscan::messages::test
