# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_recorder_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-recorder 1.0 REQUIRED CONFIG)
message(STATUS "Found holoscan-recorder ${holoscan-recorder_VERSION} in ${holoscan-recorder_DIR}")

add_executable(recorder_consumer main.cpp)
target_link_libraries(recorder_consumer PRIVATE holoscan::recorder)
set_target_properties(recorder_consumer PROPERTIES CXX_EXTENSIONS OFF)

add_test(
  NAME recorder_installed_consumer
  COMMAND recorder_consumer "${CMAKE_CURRENT_BINARY_DIR}/recording"
)
set_tests_properties(
  recorder_installed_consumer
  PROPERTIES PASS_REGULAR_EXPRESSION "recorder_consumer: PASS" TIMEOUT 300
)
