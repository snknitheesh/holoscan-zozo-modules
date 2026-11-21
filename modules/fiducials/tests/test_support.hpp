// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// A minimal check-and-count harness: the SDK image has no unit-test framework,
// and these tests need none beyond named checks and a failing exit status.

#pragma once

#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/messages/camera_info.hpp>

namespace holoscan::fiducials::test {

// A camera of `width` x `height` with these intrinsics and lens, valid
// (holoscan::messages::validate()).
inline messages::CameraInfo camera(
    double fx, double fy, double cx, double cy, int width = 1280, int height = 720,
    messages::DistortionModel model = messages::DistortionModel::kNone,
    std::vector<double> coefficients = {}, double skew = 0.0) {
  messages::CameraInfo info;
  info.header.frame_id = "camera";
  info.width = static_cast<std::uint32_t>(width);
  info.height = static_cast<std::uint32_t>(height);
  info.fx = fx;
  info.fy = fy;
  info.cx = cx;
  info.cy = cy;
  info.skew = skew;
  info.distortion_model = model;
  info.coefficients = std::move(coefficients);
  bool distorted = false;
  for (const double c : info.coefficients) distorted = distorted || c != 0.0;
  info.state = distorted ? messages::CameraState::kRaw : messages::CameraState::kUndistorted;
  messages::validate(info);
  return info;
}

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

}  // namespace holoscan::fiducials::test

#define VERIFY(condition) \
  ::holoscan::fiducials::test::check((condition), #condition, __FILE__, __LINE__)
#define VERIFY_MSG(condition, message) \
  ::holoscan::fiducials::test::check((condition), (message), __FILE__, __LINE__)
