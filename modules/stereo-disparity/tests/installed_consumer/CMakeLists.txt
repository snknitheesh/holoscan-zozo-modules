# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_stereo_disparity_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-stereo-disparity 1.0 REQUIRED CONFIG)
message(
  STATUS
  "Found holoscan-stereo-disparity ${holoscan-stereo-disparity_VERSION}"
  " in ${holoscan-stereo-disparity_DIR}"
)

add_executable(stereo_disparity_consumer main.cpp)
target_link_libraries(stereo_disparity_consumer PRIVATE holoscan::stereo_disparity)
set_target_properties(stereo_disparity_consumer PROPERTIES CXX_EXTENSIONS OFF)

add_test(NAME stereo_disparity_installed_consumer COMMAND stereo_disparity_consumer)
set_tests_properties(
  stereo_disparity_installed_consumer
  PROPERTIES PASS_REGULAR_EXPRESSION "stereo_disparity_consumer: PASS" TIMEOUT 300
)
