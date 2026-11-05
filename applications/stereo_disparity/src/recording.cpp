// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "recording.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/messages/camera_info_yaml.hpp>

namespace stereo_disparity_app {
namespace {

std::string read_file(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error(path.string() + ": cannot open");
  std::ostringstream bytes;
  bytes << file.rdbuf();
  return bytes.str();
}

std::vector<std::string> split(const std::string& line, char separator) {
  std::vector<std::string> fields;
  std::size_t begin = 0;
  while (true) {
    const std::size_t end = line.find(separator, begin);
    fields.push_back(
        line.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
    if (end == std::string::npos) return fields;
    begin = end + 1;
  }
}

}  // namespace

bool same_pixels(const holoscan::messages::CameraInfo& depth,
                 const holoscan::messages::CameraInfo& left) {
  constexpr double kPixels = 1e-3;
  return depth.width == left.width && depth.height == left.height &&
         depth.state != holoscan::messages::CameraState::kRaw &&
         std::abs(depth.fx - left.fx) < kPixels && std::abs(depth.fy - left.fy) < kPixels &&
         std::abs(depth.cx - left.cx) < kPixels && std::abs(depth.cy - left.cy) < kPixels &&
         std::abs(depth.skew - left.skew) < kPixels;
}

Recording read_recording(const std::filesystem::path& folder) {
  Recording recording;
  recording.folder = folder;
  recording.left_camera_info = folder / "left.yaml";
  recording.right_camera_info = folder / "right.yaml";
  // The messages module's reader names the file and the line on a failure.
  try {
    recording.camera_info = holoscan::messages::read_stereo_camera_info_yaml(
        recording.left_camera_info, recording.right_camera_info);
    if (const std::filesystem::path depth = folder / "depth.yaml"; std::filesystem::exists(depth)) {
      recording.depth_aligned =
          same_pixels(holoscan::messages::read_camera_info_yaml(depth), recording.camera_info.left);
    }
  } catch (const std::invalid_argument& error) {
    throw std::runtime_error(error.what());
  }
  const std::filesystem::path index_file = folder / "index.csv";
  std::istringstream index(read_file(index_file));
  std::string line;
  if (!std::getline(index, line)) throw std::runtime_error(index_file.string() + ": empty");
  const std::vector<std::string> header = split(line, ',');
  const auto column = [&](const std::string& name) -> std::optional<std::size_t> {
    for (std::size_t i = 0; i < header.size(); ++i) {
      if (header[i] == name) return i;
    }
    return std::nullopt;
  };
  const auto frame = column("frame");
  const auto left = column("left");
  const auto right = column("right");
  const auto depth = column("depth");
  if (!frame || !left || !right) {
    throw std::runtime_error(index_file.string() + ": needs frame, left and right columns");
  }
  while (std::getline(index, line)) {
    if (line.empty()) continue;
    const std::vector<std::string> fields = split(line, ',');
    if (fields.size() < header.size()) {
      throw std::runtime_error(index_file.string() + ": short row '" + line + "'");
    }
    RecordingFrame entry;
    entry.frame = std::strtoull(fields[*frame].c_str(), nullptr, 10);
    entry.left = folder / fields[*left];
    entry.right = folder / fields[*right];
    if (depth && !fields[*depth].empty()) entry.depth = folder / fields[*depth];
    recording.frames.push_back(std::move(entry));
  }
  if (recording.frames.empty()) throw std::runtime_error(index_file.string() + ": no frames");
  return recording;
}

Pgm parse_pgm(const std::string& bytes, const std::string& name) {
  std::size_t at = 0;
  // The header: P5, width, height and maximum value, separated by white space
  // and comments; then one white space character and the samples.
  const auto token = [&]() -> std::string {
    while (at < bytes.size()) {
      if (bytes[at] == '#') {
        while (at < bytes.size() && bytes[at] != '\n') ++at;
      } else if (std::isspace(static_cast<unsigned char>(bytes[at]))) {
        ++at;
      } else {
        break;
      }
    }
    const std::size_t begin = at;
    while (at < bytes.size() && !std::isspace(static_cast<unsigned char>(bytes[at]))) ++at;
    return bytes.substr(begin, at - begin);
  };
  if (token() != "P5") throw std::runtime_error(name + ": not a binary PGM (P5)");
  Pgm pgm;
  try {
    pgm.width = std::stoi(token());
    pgm.height = std::stoi(token());
    pgm.max_value = std::stoi(token());
  } catch (const std::exception&) {
    throw std::runtime_error(name + ": unreadable PGM header");
  }
  if (pgm.width < 1 || pgm.height < 1 || pgm.width > 16384 || pgm.height > 16384 ||
      pgm.max_value < 1 || pgm.max_value > 65535) {
    throw std::runtime_error(name + ": PGM size or maximum value out of range");
  }
  ++at;  // The single white space character before the samples.
  const std::size_t sample_bytes = pgm.max_value > 255 ? 2 : 1;
  const std::size_t count = static_cast<std::size_t>(pgm.width) * pgm.height;
  if (bytes.size() < at + count * sample_bytes) {
    throw std::runtime_error(name + ": PGM shorter than its header says");
  }
  pgm.pixels.resize(count);
  const auto* data = reinterpret_cast<const unsigned char*>(bytes.data() + at);
  for (std::size_t i = 0; i < count; ++i) {
    pgm.pixels[i] = sample_bytes == 2
                        ? static_cast<std::uint16_t>(data[2 * i] << 8U | data[2 * i + 1])
                        : data[i];
  }
  return pgm;
}

Pgm read_pgm(const std::filesystem::path& path) {
  return parse_pgm(read_file(path), path.string());
}

}  // namespace stereo_disparity_app
