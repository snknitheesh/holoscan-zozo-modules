# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_stitching_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-stitching 1.0 REQUIRED CONFIG)
message(
  STATUS
  "Found holoscan-stitching ${holoscan-stitching_VERSION} in ${holoscan-stitching_DIR}"
)

add_executable(stitching_consumer main.cpp)
target_link_libraries(stitching_consumer PRIVATE holoscan::stitching)
set_target_properties(stitching_consumer PROPERTIES CXX_EXTENSIONS OFF)

add_test(NAME stitching_installed_consumer COMMAND stitching_consumer)
set_tests_properties(
  stitching_installed_consumer
  PROPERTIES PASS_REGULAR_EXPRESSION "stitching_consumer: PASS" TIMEOUT 300
)
