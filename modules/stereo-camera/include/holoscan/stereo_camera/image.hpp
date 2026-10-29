// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Images on the host, as the sources produce them: borrowed views, owning
// images, and one frame of a stereo camera. No SDK.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace holoscan::stereo_camera {

// The pixel layouts the module carries. Each maps to one
// holoscan::schema::ImageEncoding of the same name.
enum class PixelFormat : std::uint8_t {
  kMono8,   // 8-bit gray: infrared or luma.
  kMono16,  // 16-bit unsigned, native byte order: depth.
  kRgb8,    // 8-bit R, G, B.
  kBgr8,    // 8-bit B, G, R.
};

// "mono8", "mono16", "rgb8" or "bgr8".
[[nodiscard]] std::string_view pixel_format_name(PixelFormat format) noexcept;
// Bytes per pixel: 1, 2, 3 or 3.
[[nodiscard]] int bytes_per_pixel(PixelFormat format) noexcept;
// Channels per pixel: 1, 1, 3 or 3.
[[nodiscard]] int channels(PixelFormat format) noexcept;

// A borrowed image in host memory. `pitch` is the row stride in bytes, at least
// width * bytes_per_pixel(format); rows may be padded, pixels within a row not.
struct ImageView {
  const std::uint8_t* data{};
  int width{};
  int height{};
  std::size_t pitch{};
  PixelFormat format{PixelFormat::kMono8};

  // The packed size of the image: width * height * bytes_per_pixel(format).
  [[nodiscard]] std::size_t packed_bytes() const noexcept;
};

// An image that owns its pixels, packed.
struct Image {
  int width{};
  int height{};
  PixelFormat format{PixelFormat::kMono8};
  std::vector<std::uint8_t> pixels;

  // Sizes `pixels` for `width` x `height` in `format`, keeping their capacity.
  void resize(int width, int height, PixelFormat format);
  [[nodiscard]] ImageView view() const noexcept;
  [[nodiscard]] std::size_t pitch() const noexcept;
};

// Copies `view` into `image`, packing its rows.
void copy_image(const ImageView& view, Image& image);
// Whether two images have the same size, format and pixels, padding aside.
[[nodiscard]] bool same_pixels(const ImageView& a, const ImageView& b) noexcept;

// One frame of a stereo camera, as a source hands it out. The views borrow the
// source's memory and stay valid until its next grab() or close().
struct StereoFrameView {
  ImageView left;
  ImageView right;
  // Present when the source is configured for them.
  std::optional<ImageView> rgb;
  // uint16 depth, registered as the calibration's depth camera says.
  std::optional<ImageView> depth;
  // The source's own number for the frame: 0, 1, 2, ... in a recording.
  std::uint64_t index{};
  // The source's timestamp for the frame, in nanoseconds of its own clock (for a
  // RealSense, the time domain librealsense reports; in a recording, what was
  // recorded).
  std::int64_t timestamp_ns{};
  // The camera's own sample number, or 0 when it has none; a gap in it means the
  // camera dropped a frame.
  std::uint64_t device_sequence{};
  // How long ago, on the host, the frame was captured, when the source knows; the
  // operator subtracts it from the graph's time to stamp the frame's capture time.
  std::optional<std::int64_t> age_ns;
};

}  // namespace holoscan::stereo_camera
