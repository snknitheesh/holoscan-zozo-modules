// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "recording.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rectify_app {
namespace {

// Largest image side a PGM may have here.
constexpr std::int64_t kMaxSide = 16384;

template <typename T>
T number(std::string_view text, const std::string& where) {
  T value{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc() || end != text.data() + text.size()) {
    throw std::invalid_argument(where + ": '" + std::string(text) + "' is not a number");
  }
  return value;
}

// A path from the index, which must stay inside the recording.
std::filesystem::path inside(const std::filesystem::path& clip, const std::string& relative,
                             const std::string& where) {
  const std::filesystem::path path(relative);
  if (relative.empty() || path.is_absolute()) {
    throw std::invalid_argument(where + ": paths must be relative to the recording");
  }
  for (const auto& part : path) {
    if (part == "..") throw std::invalid_argument(where + ": paths must stay inside the recording");
  }
  return clip / path;
}

}  // namespace

std::vector<RecordedFrame> read_index(const std::filesystem::path& clip) {
  const std::filesystem::path index = clip / "index.csv";
  std::ifstream file(index);
  if (!file) throw std::invalid_argument(index.string() + ": cannot open");
  std::string line;
  if (!std::getline(file, line)) throw std::invalid_argument(index.string() + ": empty");
  // The columns, by name: frame, timestamp_ns, device_sequence, left, right, ...
  std::vector<std::string> header;
  {
    std::stringstream columns(line);
    std::string column;
    while (std::getline(columns, column, ',')) {
      if (!column.empty() && column.back() == '\r') column.pop_back();
      header.push_back(column);
    }
  }
  const auto column_of = [&](const std::string& name) {
    for (std::size_t i = 0; i < header.size(); ++i) {
      if (header[i] == name) return i;
    }
    throw std::invalid_argument(index.string() + ": no '" + name + "' column");
  };
  const std::size_t frame_column = column_of("frame");
  const std::size_t time_column = column_of("timestamp_ns");
  const std::size_t sequence_column = column_of("device_sequence");
  const std::size_t left_column = column_of("left");
  const std::size_t right_column = column_of("right");
  const std::size_t last_column =
      std::max({frame_column, time_column, sequence_column, left_column, right_column});
  std::vector<RecordedFrame> frames;
  int line_number = 1;
  while (std::getline(file, line)) {
    ++line_number;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    const std::string where = index.string() + ":" + std::to_string(line_number);
    std::vector<std::string> fields;
    std::stringstream row(line);
    std::string field;
    while (std::getline(row, field, ',')) fields.push_back(field);
    if (fields.size() <= last_column) throw std::invalid_argument(where + ": too few fields");
    RecordedFrame frame;
    frame.frame = number<std::uint64_t>(fields[frame_column], where);
    frame.timestamp_ns = number<std::int64_t>(fields[time_column], where);
    frame.device_sequence = number<std::uint64_t>(fields[sequence_column], where);
    frame.left = inside(clip, fields[left_column], where);
    frame.right = inside(clip, fields[right_column], where);
    if (frame.frame != frames.size()) {
      throw std::invalid_argument(where + ": frames must count 0, 1, 2, ...");
    }
    frames.push_back(std::move(frame));
  }
  if (frames.empty()) throw std::invalid_argument(index.string() + ": no frames");
  return frames;
}

GrayImage read_pgm(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::invalid_argument(path.string() + ": cannot open");
  std::string magic;
  file >> magic;
  std::int64_t values[3] = {};
  for (std::int64_t& value : values) {
    file >> std::ws;
    while (file.peek() == '#') {
      std::string comment;
      std::getline(file, comment);
      file >> std::ws;
    }
    if (!(file >> value)) throw std::invalid_argument(path.string() + ": bad PGM header");
  }
  file.get();  // The single whitespace before the pixels.
  if (magic != "P5" || values[2] != 255) {
    throw std::invalid_argument(path.string() + ": not an 8-bit binary PGM (P5, maxval 255)");
  }
  if (values[0] < 1 || values[1] < 1 || values[0] > kMaxSide || values[1] > kMaxSide) {
    throw std::invalid_argument(path.string() + ": image size out of range");
  }
  GrayImage image{static_cast<int>(values[0]), static_cast<int>(values[1]), {}};
  image.pixels.resize(static_cast<std::size_t>(image.width) * image.height);
  file.read(reinterpret_cast<char*>(image.pixels.data()),
            static_cast<std::streamsize>(image.pixels.size()));
  if (!file) throw std::invalid_argument(path.string() + ": truncated");
  return image;
}

}  // namespace rectify_app
