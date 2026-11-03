// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "ros_files.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace messages_demo {
namespace {

namespace hm = holoscan::messages;

void matrix(std::ostream& out, std::string_view key, std::size_t rows, std::size_t cols,
            const double* values) {
  out << key << ":\n  rows: " << rows << "\n  cols: " << cols << "\n  data: [";
  for (std::size_t i = 0; i < rows * cols; ++i) out << (i == 0 ? "" : ", ") << values[i];
  out << "]\n";
}

void write_camera(const hm::CameraInfo& camera, const std::array<double, 12>& projection,
                  const std::filesystem::path& path) {
  std::ofstream out(path);
  out.precision(17);
  out << "image_width: " << camera.width << "\nimage_height: " << camera.height
      << "\ncamera_name: " << camera.header.frame_id << '\n';
  const hm::Matrix3 k{camera.fx, camera.skew, camera.cx, 0, camera.fy, camera.cy, 0, 0, 1};
  matrix(out, "camera_matrix", 3, 3, k.data());
  out << "distortion_model: " << hm::distortion_model_name(camera.distortion_model) << '\n';
  matrix(out, "distortion_coefficients", 1, camera.coefficients.size(), camera.coefficients.data());
  const hm::Matrix3 r = camera.rectification.value_or(hm::kIdentity3);
  matrix(out, "rectification_matrix", 3, 3, r.data());
  matrix(out, "projection_matrix", 3, 4, projection.data());
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

}  // namespace messages_demo
