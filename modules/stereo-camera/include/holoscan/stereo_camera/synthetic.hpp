// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A synthetic rectified stereo camera with known disparity, for tests and the
// demo's CI mode: two fronto-parallel textured planes, a background and a
// rectangle in front of it that moves from frame to frame. Its RGB image and
// depth are registered to the left eye. Everything is computed on the host,
// exactly, so a test can compare pixel for pixel. No SDK.

#pragma once

#include <cstdint>
#include <filesystem>

#include <holoscan/stereo_camera/calibration.hpp>
#include <holoscan/stereo_camera/image.hpp>

namespace holoscan::stereo_camera {

// The scene of one frame of a `width` x `height` synthetic camera.
class SyntheticScene {
 public:
  // @throws std::invalid_argument unless 64 <= width, height <= 4096.
  SyntheticScene(int width, int height, std::uint64_t index);

  [[nodiscard]] int width() const noexcept { return width_; }
  [[nodiscard]] int height() const noexcept { return height_; }
  // The background's and the rectangle's disparity in pixels.
  [[nodiscard]] int background_disparity() const noexcept { return background_; }
  [[nodiscard]] int foreground_disparity() const noexcept { return foreground_; }

  // The disparity of the left eye's pixel (u, v): the right eye sees the same
  // point at (u - disparity, v).
  [[nodiscard]] int disparity(int u, int v) const noexcept;
  // Whether the right eye sees the point of left pixel (u, v), rather than it
  // being outside the right image or hidden behind the rectangle.
  [[nodiscard]] bool visible_in_right(int u, int v) const noexcept;
  // The gray value of left pixel (u, v) and right pixel (u, v).
  [[nodiscard]] std::uint8_t left(int u, int v) const noexcept;
  [[nodiscard]] std::uint8_t right(int u, int v) const noexcept;

 private:
  [[nodiscard]] bool in_rectangle(int x, int y) const noexcept;

  int width_;
  int height_;
  int background_;
  int foreground_;
  int x0_;
  int x1_;
  int y0_;
  int y1_;
};

// The calibration of a `width` x `height` synthetic camera: rectified, no
// distortion, fx = fy = 0.8 * width, the principal point at the center, a 0.1 m
// baseline; with `rgb` and `depth`, an RGB camera and a depth image identical to
// the left eye (millimetres), and the camera named "synthetic".
[[nodiscard]] StereoCalibration synthetic_calibration(int width, int height, bool rgb, bool depth);

// The timestamp of synthetic frame `index`: 1 s plus 33.333333 ms a frame.
[[nodiscard]] std::int64_t synthetic_timestamp_ns(std::uint64_t index) noexcept;

// The images of synthetic frame `index`, reusing the images' memory: kMono8 eyes;
// an RGB image derived from the left eye's gray (r = g', g = g' / 2 + 64,
// b = 255 - g'); and the depth fx * baseline / disparity in millimetres, rounded
// to the nearest. `rgb` and `depth` may be null to skip them.
void render_synthetic_frame(int width, int height, std::uint64_t index, Image& left, Image& right,
                            Image* rgb, Image* depth);

// Writes a recording of `frames` synthetic frames, numbered from 0, to
// `directory`, which must not exist or be empty.
// @throws as RecordingWriter does.
void write_synthetic_recording(const std::filesystem::path& directory, int width, int height,
                               std::uint32_t frames, bool rgb, bool depth);

}  // namespace holoscan::stereo_camera
