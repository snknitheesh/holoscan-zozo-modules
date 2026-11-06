// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "ros_files.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace messages_demo {
namespace {

namespace hm = holoscan::messages;

// The calibrator's decimals: K and P 5, D 6 ("%8f"), R 8.
constexpr int kIntrinsicsDecimals = 5;
constexpr int kDistortionDecimals = 6;
constexpr int kRotationDecimals = 8;

std::string fixed(double value, int decimals) {
  std::array<char, 64> text{};
  const int length = std::snprintf(text.data(), text.size(), "%.*f", decimals, value);
  if (length < 0 || static_cast<std::size_t>(length) >= text.size()) {
    throw std::runtime_error("cannot format " + std::to_string(value));
  }
  return std::string(text.data(), static_cast<std::size_t>(length));
}

double rounded(double value, int decimals) {
  const std::string text = fixed(value, decimals);
  double result = 0.0;
  std::from_chars(text.data(), text.data() + text.size(), result);
  return result;
}

// A matrix as the calibrator lays it out: one row per line.
void matrix(std::ostream& out, std::string_view key, std::size_t rows, std::size_t cols,
            const double* values, int decimals) {
  out << key << ":\n  rows: " << rows << "\n  cols: " << cols << "\n  data: [";
  for (std::size_t i = 0; i < rows * cols; ++i) {
    if (i != 0) out << (i % cols == 0 ? ",\n         " : ", ");
    out << fixed(values[i], decimals);
  }
  out << "]\n";
}

void write_camera(const hm::CameraInfo& camera, const std::array<double, 12>& projection,
                  const std::filesystem::path& path) {
  std::ofstream out(path);
  out << "image_width: " << camera.width << "\nimage_height: " << camera.height
      << "\ncamera_name: " << camera.header.frame_id << '\n';
  const hm::Matrix3 k{camera.fx, camera.skew, camera.cx, 0, camera.fy, camera.cy, 0, 0, 1};
  matrix(out, "camera_matrix", 3, 3, k.data(), kIntrinsicsDecimals);
  out << "distortion_model: " << hm::distortion_model_name(camera.distortion_model) << '\n';
  matrix(out, "distortion_coefficients", 1, camera.coefficients.size(), camera.coefficients.data(),
         kDistortionDecimals);
  const hm::Matrix3 r = camera.rectification.value_or(hm::kIdentity3);
  matrix(out, "rectification_matrix", 3, 3, r.data(), kRotationDecimals);
  matrix(out, "projection_matrix", 3, 4, projection.data(), kIntrinsicsDecimals);
  out.close();
  if (!out) throw std::runtime_error(path.string() + ": cannot write");
}

}  // namespace

void write_ros_stereo_files(const hm::StereoCameraInfo& pair, double f, double cx, double cy,
                            const std::filesystem::path& left, const std::filesystem::path& right) {
  const hm::Vector3& t = pair.right_from_left.translation;
  const double baseline = std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
  write_camera(pair.left, {f, 0, cx, 0, 0, f, cy, 0, 0, 0, 1, 0}, left);
  write_camera(pair.right, {f, 0, cx, -f * baseline, 0, f, cy, 0, 0, 0, 1, 0}, right);
}

hm::CameraInfo as_written(hm::CameraInfo camera) {
  for (double* k : {&camera.fx, &camera.fy, &camera.cx, &camera.cy, &camera.skew}) {
    *k = rounded(*k, kIntrinsicsDecimals);
  }
  for (double& d : camera.coefficients) d = rounded(d, kDistortionDecimals);
  return camera;
}

}  // namespace messages_demo
