// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// POSIX files for the recording: a file descriptor that closes itself, and
// whole reads and writes that retry short transfers and interruptions.

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace holoscan::recorder::detail {

class File {
 public:
  File() = default;
  explicit File(int fd) noexcept : fd_(fd) {}
  ~File() { close(); }
  File(File&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  File& operator=(File&& other) noexcept {
    if (this != &other) {
      close();
      fd_ = std::exchange(other.fd_, -1);
    }
    return *this;
  }
  File(const File&) = delete;
  File& operator=(const File&) = delete;

  // @throws RecordingError naming `path` if it cannot be opened.
  static File open_read(const std::filesystem::path& path);
  // Creates or truncates `path`. @throws RecordingError.
  static File create(const std::filesystem::path& path);

  [[nodiscard]] int fd() const noexcept { return fd_; }
  [[nodiscard]] explicit operator bool() const noexcept { return fd_ >= 0; }

  // All of `data`, at the end of the file. @throws RecordingError naming `what`.
  void write_all(std::span<const std::byte> data, const std::string& what) const;
  // Exactly `out.size()` bytes from `offset`. @throws RecordingError naming
  // `what`, also when the file ends first.
  void read_at(std::uint64_t offset, std::span<std::byte> out, const std::string& what) const;
  // Flushes the file to the disk. @throws RecordingError naming `what`.
  void sync(const std::string& what) const;
  void close() noexcept;

 private:
  int fd_{-1};
};

// The whole of a small file. @throws RecordingError.
[[nodiscard]] std::string read_text(const std::filesystem::path& path);

// Writes `text` to `path` through a temporary file in the same folder, renamed
// over it, so that a reader sees the old or the new file, never half of one.
// @throws RecordingError.
void replace_file(const std::filesystem::path& path, std::string_view text);

}  // namespace holoscan::recorder::detail
