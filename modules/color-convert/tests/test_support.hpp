// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A minimal check-and-count harness: the SDK image has no unit-test framework,
// and these tests need none beyond named checks and a failing exit status.

#pragma once

#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace holoscan::color_convert::test {

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

// The exit status of a test that needs a CUDA device and has none: skipped
// (77) on a developer's machine, failed (1) inside a CI lane
// (HOLOSCAN_CI_PLATFORM set), whose runner has a GPU, so that a lane with a
// broken GPU does not pass silently.
inline int no_cuda_device() {
  if (std::getenv("HOLOSCAN_CI_PLATFORM") != nullptr) {
    std::cerr << "HOLOSCAN_CI_PLATFORM is set, but there is no CUDA device\n";
    return 1;
  }
  std::cout << "no CUDA device; skipping\n";
  return 77;
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

}  // namespace holoscan::color_convert::test

#define VERIFY(condition) \
  ::holoscan::color_convert::test::check((condition), #condition, __FILE__, __LINE__)
#define VERIFY_MSG(condition, message) \
  ::holoscan::color_convert::test::check((condition), (message), __FILE__, __LINE__)
