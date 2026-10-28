// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Files for the SDK-free tests: a fresh folder under the test's working folder,
// and text files written into it.

#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace holoscan::stereo_camera::test {

// `root / name`, emptied.
inline std::filesystem::path fresh_folder(const std::filesystem::path& root,
                                          const std::string& name) {
  const std::filesystem::path folder = root / name;
  std::filesystem::remove_all(folder);
  std::filesystem::create_directories(folder);
  return folder;
}

inline void write_text(const std::filesystem::path& path, std::string_view text) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file << text;
  if (!file) {
    throw std::runtime_error("cannot write " + path.string());
  }
}

inline std::string read_text(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// The message of the std::invalid_argument `body` throws, or "" if it throws none.
template <typename Body>
std::string invalid_message(Body&& body) {
  try {
    std::forward<Body>(body)();
  } catch (const std::invalid_argument& error) {
    return error.what();
  }
  return "";
}

inline bool contains(std::string_view text, std::string_view part) {
  return text.find(part) != std::string_view::npos;
}

}  // namespace holoscan::stereo_camera::test
