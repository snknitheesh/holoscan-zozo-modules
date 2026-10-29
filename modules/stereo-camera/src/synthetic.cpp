// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/stereo_camera/synthetic.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include <holoscan/stereo_camera/recording.hpp>

namespace holoscan::stereo_camera {
namespace {

constexpr double kBaseline = 0.1;

// A texture value for a plane at (x, y), in 2x2 blocks: enough structure for a
// matcher, and the same wherever either eye sees it.
std::uint8_t texture(std::uint32_t plane, int x, int y) noexcept {
  // splitmix64 of the block's coordinates.
  std::uint64_t z = (static_cast<std::uint64_t>(plane) << 48U) ^
                    (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x >> 1)) << 24U) ^
                    static_cast<std::uint64_t>(static_cast<std::uint32_t>(y >> 1));
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
  z ^= z >> 31U;
  return static_cast<std::uint8_t>(16U + z % 224U);
}

}  // namespace

SyntheticScene::SyntheticScene(int width, int height, std::uint64_t index)
    : width_(width), height_(height) {
  if (width < 64 || height < 64 || width > 4096 || height > 4096) {
    throw std::invalid_argument("synthetic images must be 64..4096 pixels a side");
  }
  background_ = std::max(4, width / 40);
  foreground_ = 3 * background_;
  // The rectangle moves right two pixels a frame, and back after 32 frames.
  x0_ = width / 4 + 2 * static_cast<int>(index % 32U);
  x1_ = x0_ + width / 3;
  y0_ = height / 4;
  y1_ = y0_ + height / 2;
}

bool SyntheticScene::in_rectangle(int x, int y) const noexcept {
  return x >= x0_ && x < x1_ && y >= y0_ && y < y1_;
}

int SyntheticScene::disparity(int u, int v) const noexcept {
  return in_rectangle(u, v) ? foreground_ : background_;
}

bool SyntheticScene::visible_in_right(int u, int v) const noexcept {
  const int d = disparity(u, v);
  if (u - d < 0) {
    return false;
  }
  // The background is hidden where the right eye sees the rectangle instead.
  return d == foreground_ || !in_rectangle(u - d + foreground_, v);
}

std::uint8_t SyntheticScene::left(int u, int v) const noexcept {
  return in_rectangle(u, v) ? texture(1, u, v) : texture(0, u, v);
}

std::uint8_t SyntheticScene::right(int u, int v) const noexcept {
  // The right eye's pixel (u, v) sees the point the left eye sees at (u + d, v),
  // for the nearest plane that is there.
  if (in_rectangle(u + foreground_, v)) {
    return texture(1, u + foreground_, v);
  }
  return texture(0, u + background_, v);
}

StereoCalibration synthetic_calibration(int width, int height, bool rgb, bool depth) {
  StereoCalibration calibration;
  calibration.left = CameraModel{.width = width,
                                 .height = height,
                                 .fx = 0.8 * width,
                                 .fy = 0.8 * width,
                                 .cx = (width - 1) / 2.0,
                                 .cy = (height - 1) / 2.0};
  calibration.right = calibration.left;
  calibration.right_from_left.translation = {-kBaseline, 0.0, 0.0};
  calibration.rectified = true;
  if (rgb) {
    calibration.rgb = AuxiliaryCamera{.model = calibration.left};
  }
  if (depth) {
    calibration.depth = DepthCamera{.model = calibration.left, .scale = 0.001};
  }
  calibration.camera.name = "synthetic";
  return calibration;
}

std::int64_t synthetic_timestamp_ns(std::uint64_t index) noexcept {
  return 1'000'000'000LL + static_cast<std::int64_t>(index) * 33'333'333LL;
}

void render_synthetic_frame(int width, int height, std::uint64_t index, Image& left, Image& right,
                            Image* rgb, Image* depth) {
  const SyntheticScene scene(width, height, index);
  left.resize(width, height, PixelFormat::kMono8);
  right.resize(width, height, PixelFormat::kMono8);
  if (rgb != nullptr) rgb->resize(width, height, PixelFormat::kRgb8);
  if (depth != nullptr) depth->resize(width, height, PixelFormat::kMono16);
  const double depth_from_disparity = 0.8 * width * kBaseline;
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * width + u;
      const std::uint8_t gray = scene.left(u, v);
      left.pixels[i] = gray;
      right.pixels[i] = scene.right(u, v);
      if (rgb != nullptr) {
        rgb->pixels[3 * i] = gray;
        rgb->pixels[3 * i + 1] = static_cast<std::uint8_t>(gray / 2 + 64);
        rgb->pixels[3 * i + 2] = static_cast<std::uint8_t>(255 - gray);
      }
      if (depth != nullptr) {
        // Millimetres: fx * baseline / disparity metres, at most a few metres.
        const auto mm = static_cast<std::uint16_t>(
            std::lround(1000.0 * depth_from_disparity / scene.disparity(u, v)));
        std::memcpy(depth->pixels.data() + 2 * i, &mm, sizeof(mm));
      }
    }
  }
}

void write_synthetic_recording(const std::filesystem::path& directory, int width, int height,
                               std::uint32_t frames, bool rgb, bool depth) {
  RecordingWriter writer(directory, synthetic_calibration(width, height, rgb, depth), rgb, depth);
  Image left;
  Image right;
  Image rgb_image;
  Image depth_image;
  for (std::uint32_t index = 0; index < frames; ++index) {
    render_synthetic_frame(width, height, index, left, right, rgb ? &rgb_image : nullptr,
                           depth ? &depth_image : nullptr);
    StereoFrameView frame{.left = left.view(),
                          .right = right.view(),
                          .index = index,
                          .timestamp_ns = synthetic_timestamp_ns(index),
                          .device_sequence = index + 1U};
    if (rgb) frame.rgb = rgb_image.view();
    if (depth) frame.depth = depth_image.view();
    writer.write(frame);
  }
  writer.close();
}

}  // namespace holoscan::stereo_camera
