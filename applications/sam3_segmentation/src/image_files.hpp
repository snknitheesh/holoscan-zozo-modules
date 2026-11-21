// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Image files for the application: JPEG and binary PPM (P6) in, PPM out. JPEG pixels are decoded on
// the GPU with nvJPEG; the rest needs no library. No Holoscan.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <holoscan/sam3/pixel_math.hpp>

namespace sam3_app {

using holoscan::sam3::blue_first;
using holoscan::sam3::channels;
using holoscan::sam3::kMaxImageSide;
using holoscan::sam3::PixelFormat;

enum class ImageFileKind : std::uint8_t { kJpeg, kPpm };

struct ImageFileInfo {
  ImageFileKind kind{ImageFileKind::kJpeg};
  int width{};
  int height{};
};

// The kind and size of a JPEG (by its frame header) or a binary PPM with a
// maxval of 255, without decoding it.
// @throws std::runtime_error naming the file and what it does not support.
[[nodiscard]] ImageFileInfo read_image_info(const std::filesystem::path& path);

// The .jpg, .jpeg and .ppm files (any letter case) in `directory`, in name
// order; or `path` itself when it is such a file.
// @throws std::runtime_error if `path` cannot be listed or holds no image.
[[nodiscard]] std::vector<std::filesystem::path> list_image_files(
    const std::filesystem::path& path);

// A file's bytes.
// @throws std::runtime_error if it cannot be read.
[[nodiscard]] std::vector<std::uint8_t> read_file(const std::filesystem::path& path);

// The packed RGB8 pixels of a binary PPM.
// @throws std::runtime_error for an unsupported or truncated file.
[[nodiscard]] std::vector<std::uint8_t> read_ppm(const std::filesystem::path& path, int& width,
                                                 int& height);

// Writes packed RGB8 pixels as a binary PPM.
// @throws std::runtime_error if the file cannot be written.
void write_ppm(const std::filesystem::path& path, int width, int height,
               std::span<const std::uint8_t> rgb);

// Converts `width` x `height` pixels of `format`, rows `pitch` bytes apart, to
// packed RGB8.
[[nodiscard]] std::vector<std::uint8_t> to_rgb(const std::uint8_t* pixels, std::size_t pitch,
                                               int width, int height, PixelFormat format);

// Decodes JPEG files with nvJPEG into GPU memory. One decoder per thread.
class JpegDecoder {
 public:
  // @throws std::runtime_error if nvJPEG cannot be initialized.
  JpegDecoder();
  ~JpegDecoder();
  JpegDecoder(const JpegDecoder&) = delete;
  JpegDecoder& operator=(const JpegDecoder&) = delete;

  // Decodes `jpeg` as interleaved RGB8 (or BGR8 with `bgr`) into `pixels`,
  // rows `pitch` bytes apart, on `stream`. The image must be width x height.
  // Returns an empty string on success, or what went wrong.
  [[nodiscard]] std::string decode(std::span<const std::uint8_t> jpeg, int width, int height,
                                   bool bgr, std::uint8_t* pixels, std::size_t pitch,
                                   cudaStream_t stream);

 private:
  struct State;
  State* state_{};
};

}  // namespace sam3_app
