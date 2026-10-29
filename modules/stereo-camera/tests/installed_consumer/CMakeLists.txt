# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_stereo_camera_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-stereo-camera 1.0 REQUIRED CONFIG)
message(
  STATUS
  "Found holoscan-stereo-camera ${holoscan-stereo-camera_VERSION} in ${holoscan-stereo-camera_DIR}"
  " (realsense backend: ${holoscan-stereo-camera_WITH_REALSENSE})"
)

add_executable(stereo_camera_consumer main.cpp)
target_link_libraries(stereo_camera_consumer PRIVATE holoscan::stereo_camera)
# The suite's frame sink and check harness, shared with the module's own tests.
target_include_directories(stereo_camera_consumer PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
set_target_properties(stereo_camera_consumer PROPERTIES CXX_EXTENSIONS OFF)

add_test(
  NAME stereo_camera_installed_consumer
  COMMAND stereo_camera_consumer "${CMAKE_CURRENT_BINARY_DIR}/recording"
)
set_tests_properties(
  stereo_camera_installed_consumer
  PROPERTIES PASS_REGULAR_EXPRESSION "stereo_camera_consumer: PASS" TIMEOUT 300
)
