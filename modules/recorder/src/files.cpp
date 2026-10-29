// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0

#include "files.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>
#include <string_view>
#include <system_error>

#include <holoscan/recorder/format.hpp>

namespace holoscan::recorder::detail {
namespace {

[[noreturn]] void fail(const std::string& what, const char* action, int error) {
  throw RecordingError(what + ": cannot " + action + ": " + std::strerror(error));
}

}  // namespace

File File::open_read(const std::filesystem::path& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    fail(path.string(), "open", errno);
  }
  return File(fd);
}

File File::create(const std::filesystem::path& path) {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) {
    fail(path.string(), "create", errno);
  }
  return File(fd);
}

void File::write_all(std::span<const std::byte> data, const std::string& what) const {
  while (!data.empty()) {
    const ssize_t written = ::write(fd_, data.data(), data.size());
    if (written < 0) {
      if (errno == EINTR) continue;
      fail(what, "write", errno);
    }
    data = data.subspan(static_cast<std::size_t>(written));
  }
}

void File::read_at(std::uint64_t offset, std::span<std::byte> out, const std::string& what) const {
  while (!out.empty()) {
    const ssize_t got = ::pread(fd_, out.data(), out.size(), static_cast<off_t>(offset));
    if (got < 0) {
      if (errno == EINTR) continue;
      fail(what, "read", errno);
    }
    if (got == 0) {
      throw RecordingError(what + ": the file ends before the data the index names");
    }
    out = out.subspan(static_cast<std::size_t>(got));
    offset += static_cast<std::uint64_t>(got);
  }
}

void File::sync(const std::string& what) const {
  if (::fsync(fd_) != 0 && errno != EINVAL) {
    fail(what, "flush", errno);
  }
}

void File::close() noexcept {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

std::string read_text(const std::filesystem::path& path) {
  const File file = File::open_read(path);
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error) {
    throw RecordingError(path.string() + ": " + error.message());
  }
  std::string text(static_cast<std::size_t>(size), '\0');
  file.read_at(0, std::as_writable_bytes(std::span(text)), path.string());
  return text;
}

void replace_file(const std::filesystem::path& path, std::string_view text) {
  std::filesystem::path temporary = path;
  temporary += ".tmp";
  {
    const File file = File::create(temporary);
    file.write_all(std::as_bytes(std::span(text)), temporary.string());
    file.sync(temporary.string());
  }
  if (::rename(temporary.c_str(), path.c_str()) != 0) {
    fail(path.string(), "replace", errno);
  }
}

}  // namespace holoscan::recorder::detail
