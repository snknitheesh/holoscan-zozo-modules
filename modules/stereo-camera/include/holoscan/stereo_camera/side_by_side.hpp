// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Splitting a side-by-side stereo frame, as a ZED or another USB stereo camera
// sends it over V4L2, into its left and right eyes: the left half of each row
// is the left eye. No SDK.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

#include <holoscan/stereo_camera/image.hpp>

namespace holoscan::stereo_camera {

// The pixel format of the camera's side-by-side frame.
enum class SideBySideFormat : std::uint8_t {
  kYuyv,  // Packed 4:2:2, Y0 U Y1 V; each eye's width must be even.
  kGray,  // 8-bit gray.
};

// "yuyv" or "gray".
[[nodiscard]] std::string_view side_by_side_format_name(SideBySideFormat format) noexcept;
[[nodiscard]] std::optional<SideBySideFormat> parse_side_by_side_format(
    std::string_view name) noexcept;

// Splits a side-by-side frame of `full_width` x `height` pixels, `pitch` bytes a
// row, into `left` and `right`, each `full_width / 2` wide, in `output`: kMono8
// keeps the luma (or the gray) as it is; kRgb8 converts YUYV with BT.601
// limited-range coefficients, and replicates gray. Reuses the images' memory.
// @throws std::invalid_argument for an odd full width, an eye width that YUYV
//         cannot split (odd), a pitch shorter than a row, or kMono16 or kBgr8
//         output.
void split_side_by_side(const std::uint8_t* frame, int full_width, int height, std::size_t pitch,
                        SideBySideFormat input, PixelFormat output, Image& left, Image& right);

// One pixel of YUYV in BT.601 limited range, as split_side_by_side() converts it.
void yuv_to_rgb(int y, int u, int v, std::uint8_t* rgb) noexcept;

}  // namespace holoscan::stereo_camera
