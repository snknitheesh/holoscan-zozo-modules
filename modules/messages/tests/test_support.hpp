// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A minimal check-and-count harness: the SDK image has no unit-test framework,
// and these tests need none beyond named checks and a failing exit status.

#pragma once

#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace holoscan::messages::test {

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

// Whether `body` throws std::invalid_argument with `text` in its message;
// prints what happened instead when not.
template <typename Body>
bool refuses(Body&& body, std::string_view text) {
  try {
    std::forward<Body>(body)();
  } catch (const std::invalid_argument& error) {
    if (std::string_view(error.what()).find(text) != std::string_view::npos) return true;
    std::cerr << "  refused with \"" << error.what() << "\", expected \"" << text << "\"\n";
    return false;
  } catch (const std::exception& error) {
    std::cerr << "  threw " << error.what() << ", expected std::invalid_argument\n";
    return false;
  }
  std::cerr << "  accepted, expected a refusal with \"" << text << "\"\n";
  return false;
}

}  // namespace holoscan::messages::test

#define VERIFY(condition) \
  ::holoscan::messages::test::check((condition), #condition, __FILE__, __LINE__)
#define VERIFY_MSG(condition, message) \
  ::holoscan::messages::test::check((condition), (message), __FILE__, __LINE__)
