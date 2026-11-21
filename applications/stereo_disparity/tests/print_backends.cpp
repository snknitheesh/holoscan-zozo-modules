// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Prints the stereo backends this platform has, one per line, as the module's
// available_backends() finds them when the test runs: run_and_match.cmake
// reads it for a test that needs a backend, or its absence.

#include <iostream>

#include <holoscan/stereo_disparity/platform.hpp>

int main() {
  namespace sd = holoscan::stereo_disparity;
  for (const sd::Backend backend : sd::available_backends()) {
    std::cout << sd::backend_name(backend) << '\n';
  }
  return 0;
}
