# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_apriltag_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-apriltag 1.0 REQUIRED CONFIG)
message(STATUS "Found holoscan-apriltag ${holoscan-apriltag_VERSION} in ${holoscan-apriltag_DIR}"
  " (vpi backend: ${holoscan-apriltag_WITH_VPI})")

add_executable(apriltag_consumer main.cpp)
target_link_libraries(apriltag_consumer PRIVATE holoscan::apriltag)

add_test(NAME apriltag_installed_consumer COMMAND apriltag_consumer)
set_tests_properties(apriltag_installed_consumer PROPERTIES
  PASS_REGULAR_EXPRESSION "apriltag_consumer: PASS" TIMEOUT 300)
