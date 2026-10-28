// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "holoscan/stereo_camera/netpbm.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace holoscan::stereo_camera {
namespace {

// Largest image side a Netpbm file may have here.
constexpr std::int64_t kMaxSide = 16384;

class HeaderReader {
 public:
  HeaderReader(std::string_view bytes, std::string_view what) : bytes_(bytes), what_(what) {}

  [[noreturn]] void fail(const std::string& problem) const {
    throw std::invalid_argument(std::string(what_) + ": " + problem);
  }

  // Skips white space and '#' comments up to the end of their line.
  void skip() {
    while (position_ < bytes_.size()) {
      const char c = bytes_[position_];
      if (c == '#') {
        while (position_ < bytes_.size() && bytes_[position_] != '\n') ++position_;
      } else if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f') {
        ++position_;
      } else {
        return;
      }
    }
  }

  std::int64_t number(const char* name) {
    skip();
    std::int64_t value = 0;
    std::size_t digits = 0;
    while (position_ < bytes_.size() && bytes_[position_] >= '0' && bytes_[position_] <= '9') {
      value = value * 10 + (bytes_[position_] - '0');
      ++position_;
      if (++digits > 9) fail(std::string("header ") + name + " is too large");
    }
    if (digits == 0) fail(std::string("header has no ") + name);
    return value;
  }

  // The single white-space byte between the header and the pixels.
  void end_of_header() {
    if (position_ >= bytes_.size() || !(bytes_[position_] == ' ' || bytes_[position_] == '\t' ||
                                        bytes_[position_] == '\n' || bytes_[position_] == '\r')) {
      fail("header does not end with white space");
    }
    ++position_;
  }

  std::size_t position() const noexcept { return position_; }
  void advance(std::size_t count) noexcept { position_ += count; }

 private:
  std::string_view bytes_;
  std::string_view what_;
  std::size_t position_{};
};

void write_file(const std::filesystem::path& path, const std::string& header,
                const std::uint8_t* data, std::size_t size) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(header.data(), static_cast<std::streamsize>(header.size()));
  file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
  file.close();
  if (!file) {
    throw std::runtime_error(path.string() + ": cannot write");
  }
}

}  // namespace

void decode_netpbm(std::string_view bytes, Image& image, std::string_view what) {
  HeaderReader reader(bytes, what);
  if (bytes.size() < 2 || bytes[0] != 'P' || (bytes[1] != '5' && bytes[1] != '6')) {
    reader.fail("not a binary PGM (P5) or PPM (P6) file");
  }
  const bool color = bytes[1] == '6';
  reader.advance(2);
  const std::int64_t width = reader.number("width");
  const std::int64_t height = reader.number("height");
  const std::int64_t maxval = reader.number("maxval");
  reader.end_of_header();
  if (width < 1 || height < 1 || width > kMaxSide || height > kMaxSide) {
    reader.fail("width and height must be 1.." + std::to_string(kMaxSide));
  }
  PixelFormat format = PixelFormat::kMono8;
  if (color) {
    if (maxval != 255) reader.fail("only 8-bit PPM (maxval 255) is supported");
    format = PixelFormat::kRgb8;
  } else if (maxval == 65535) {
    format = PixelFormat::kMono16;
  } else if (maxval != 255) {
    reader.fail("only 8-bit or 16-bit PGM (maxval 255 or 65535) is supported");
  }
  image.resize(static_cast<int>(width), static_cast<int>(height), format);
  const std::size_t size = image.pixels.size();
  if (bytes.size() - reader.position() < size) {
    reader.fail("pixel data is truncated: " + std::to_string(bytes.size() - reader.position()) +
                " of " + std::to_string(size) + " bytes");
  }
  const auto* pixels = reinterpret_cast<const std::uint8_t*>(bytes.data() + reader.position());
  if (format == PixelFormat::kMono16) {
    // Big-endian in the file, native in memory.
    auto* out = reinterpret_cast<std::uint16_t*>(image.pixels.data());
    for (std::size_t i = 0; i < size / 2; ++i) {
      out[i] = static_cast<std::uint16_t>((pixels[2 * i] << 8U) | pixels[2 * i + 1]);
    }
  } else {
    std::memcpy(image.pixels.data(), pixels, size);
  }
}

void read_netpbm(const std::filesystem::path& path, Image& image) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::invalid_argument(path.string() + ": cannot open");
  }
  const std::streamoff size = file.tellg();
  std::string bytes(size > 0 ? static_cast<std::size_t>(size) : 0U, '\0');
  file.seekg(0);
  if (size < 0 || !file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
    throw std::invalid_argument(path.string() + ": cannot read");
  }
  decode_netpbm(bytes, image, path.string());
}

void write_netpbm(const std::filesystem::path& path, const ImageView& view) {
  const bool color = channels(view.format) == 3;
  const std::string header = std::string(color ? "P6\n" : "P5\n") + std::to_string(view.width) +
                             " " + std::to_string(view.height) + "\n" +
                             (view.format == PixelFormat::kMono16 ? "65535\n" : "255\n");
  const std::size_t row = static_cast<std::size_t>(view.width) * bytes_per_pixel(view.format);
  std::vector<std::uint8_t> pixels(row * static_cast<std::size_t>(view.height));
  for (int y = 0; y < view.height; ++y) {
    const std::uint8_t* in = view.data + static_cast<std::size_t>(y) * view.pitch;
    std::uint8_t* out = pixels.data() + static_cast<std::size_t>(y) * row;
    switch (view.format) {
      case PixelFormat::kMono16:
        for (int x = 0; x < view.width; ++x) {
          std::uint16_t value = 0;
          std::memcpy(&value, in + 2 * x, 2);
          out[2 * x] = static_cast<std::uint8_t>(value >> 8U);
          out[2 * x + 1] = static_cast<std::uint8_t>(value & 0xFFU);
        }
        break;
      case PixelFormat::kBgr8:
        for (int x = 0; x < view.width; ++x) {
          out[3 * x] = in[3 * x + 2];
          out[3 * x + 1] = in[3 * x + 1];
          out[3 * x + 2] = in[3 * x];
        }
        break;
      default:
        std::memcpy(out, in, row);
        break;
    }
  }
  write_file(path, header, pixels.data(), pixels.size());
}

std::string_view netpbm_extension(PixelFormat format) noexcept {
  return channels(format) == 3 ? ".ppm" : ".pgm";
}

}  // namespace holoscan::stereo_camera
