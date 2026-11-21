// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A minimal check-and-count harness: the SDK image has no unit-test framework,
// and these tests need none beyond named checks and a failing exit status.
// Also a temporary folder, for the camera_info files the tests write.

#pragma once

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace holoscan::rectify::test {

inline int& failures() {
  static int count = 0;
  return count;
}

inline void check(bool condition, const std::string& what, const char* file, int line) {
  if (!condition) {
    ++failures();
    std::cerr << file << ':' << line << ": check failed: " << what << '\n';
  }
}

struct TestCase {
  std::string name;
  std::function<void()> body;
};

// Runs every case, reporting each, and returns the process exit status.
inline int run(const std::vector<TestCase>& cases) {
  for (const TestCase& test : cases) {
    const int before = failures();
    try {
      test.body();
    } catch (const std::exception& error) {
      ++failures();
      std::cerr << test.name << ": unexpected exception: " << error.what() << '\n';
    }
    std::cout << (failures() == before ? "[ PASS ] " : "[ FAIL ] ") << test.name << '\n';
  }
  std::cout << (failures() == 0 ? "all passed" : std::to_string(failures()) + " check(s) failed")
            << '\n';
  return failures() == 0 ? 0 : 1;
}

template <typename Exception, typename Body>
bool throws(Body&& body) {
  try {
    std::forward<Body>(body)();
  } catch (const Exception&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

// A new, empty folder under the system's temporary folder, removed with
// everything in it when the object goes.
class TempDir {
 public:
  TempDir() {
    std::string pattern = (std::filesystem::temp_directory_path() / "rectify-test-XXXXXX").string();
    if (mkdtemp(pattern.data()) == nullptr) {
      throw std::runtime_error("cannot make a temporary folder from " + pattern);
    }
    path_ = pattern;
  }
  ~TempDir() {
    std::error_code ignored;  // Nothing to do about a folder that cannot be removed.
    std::filesystem::remove_all(path_, ignored);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
  [[nodiscard]] std::filesystem::path operator/(const char* name) const { return path_ / name; }

 private:
  std::filesystem::path path_;
};

}  // namespace holoscan::rectify::test

#define VERIFY(condition) \
  ::holoscan::rectify::test::check((condition), #condition, __FILE__, __LINE__)
#define VERIFY_MSG(condition, message) \
  ::holoscan::rectify::test::check((condition), (message), __FILE__, __LINE__)
