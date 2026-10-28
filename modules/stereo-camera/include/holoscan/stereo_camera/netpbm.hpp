// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Binary Netpbm images, the frames of a recording: PGM (P5) with 8-bit or
// 16-bit gray, and PPM (P6) with 8-bit RGB. 16-bit samples are big-endian in
// the file, as the format requires. No SDK.

#pragma once

#include <filesystem>
#include <string_view>

#include <holoscan/stereo_camera/image.hpp>

namespace holoscan::stereo_camera {

// Decodes a P5 file with a maxval of 255 (kMono8) or 65535 (kMono16), or a P6
// file with a maxval of 255 (kRgb8), into `image`, reusing its memory. Comments
// in the header are allowed. Anything else, a header that does not parse, or
// pixel data that is too short is an error.
// @throws std::invalid_argument naming `what` (a file name) and the problem.
void decode_netpbm(std::string_view bytes, Image& image, std::string_view what = "image");

// Reads and decodes `path`.
// @throws std::invalid_argument naming the file and the problem.
void read_netpbm(const std::filesystem::path& path, Image& image);

// Writes `view` as P5 (kMono8, kMono16) or P6 (kRgb8; kBgr8 is written as RGB),
// replacing `path`.
// @throws std::runtime_error if the file cannot be written.
void write_netpbm(const std::filesystem::path& path, const ImageView& view);

// The file name extension for a format: ".pgm" for gray, ".ppm" for color.
[[nodiscard]] std::string_view netpbm_extension(PixelFormat format) noexcept;

}  // namespace holoscan::stereo_camera
