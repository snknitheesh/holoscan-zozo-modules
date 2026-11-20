// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Writes synthetic depth frames as 16-bit PGM files (millimeters), for the
// tests that feed the application a folder of depth images, with the camera's
// camera_info file, camera_info.yaml, and three the operator must refuse:
// other_size.yaml, the camera at half the size, raw.yaml, the camera with
// lens distortion, and skewed.yaml, the camera with a skew.
//
// Usage: write_test_images DIR COUNT WIDTH HEIGHT

#include <charconv>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string_view>

#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/point_cloud/synthetic_depth.hpp>

#include "image_io.hpp"

namespace {

int number(const char* text) {
  const std::string_view view(text);
  int value = 0;
  std::from_chars(view.data(), view.data() + view.size(), value);
  return value;
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc != 5) {
    std::cerr << "usage: " << argv[0] << " DIR COUNT WIDTH HEIGHT\n";
    return 2;
  }
  const std::filesystem::path directory(argv[1]);
  const int count = number(argv[2]);
  const int width = number(argv[3]);
  const int height = number(argv[4]);
  std::filesystem::create_directories(directory);
  namespace hm = holoscan::messages;
  namespace pc = holoscan::point_cloud;
  const hm::CameraInfo camera =
      pc::synthetic_camera_info(pc::synthetic_frame(width, height, 0, false), "depth_camera");
  hm::write_camera_info_yaml(camera, directory / "camera_info.yaml");
  hm::write_camera_info_yaml(
      pc::synthetic_camera_info(pc::synthetic_frame(width / 2, height / 2, 0, false),
                                "depth_camera"),
      directory / "other_size.yaml");
  hm::CameraInfo raw = camera;
  raw.state = hm::CameraState::kRaw;
  raw.distortion_model = hm::DistortionModel::kPlumbBob;
  raw.coefficients = {-0.1, 0.02, 0.0, 0.0, 0.0};
  hm::write_camera_info_yaml(raw, directory / "raw.yaml");
  hm::CameraInfo skewed = camera;
  skewed.skew = 0.5;
  hm::write_camera_info_yaml(skewed, directory / "skewed.yaml");
  for (int frame = 0; frame < count; ++frame) {
    const auto f = holoscan::point_cloud::synthetic_frame(width, height,
                                                          static_cast<std::uint32_t>(frame), false);
    depth_to_point_cloud_app::DepthImage image;
    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<std::size_t>(width) * height);
    for (int v = 0; v < height; ++v) {
      for (int u = 0; u < width; ++u) {
        image.pixels[static_cast<std::size_t>(v) * width + u] =
            holoscan::point_cloud::synthetic_depth_mm(f, u, v);
      }
    }
    char name[32];
    std::snprintf(name, sizeof(name), "depth_%04d.pgm", frame);
    depth_to_point_cloud_app::write_pgm_depth(directory / name, image);
  }
  return 0;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
