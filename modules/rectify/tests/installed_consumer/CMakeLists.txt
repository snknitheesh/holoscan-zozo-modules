# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_rectify_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-rectify 1.0 REQUIRED CONFIG)
message(STATUS "Found holoscan-rectify ${holoscan-rectify_VERSION} in ${holoscan-rectify_DIR}")

add_executable(rectify_consumer main.cpp)
target_link_libraries(rectify_consumer PRIVATE holoscan::rectify)
target_compile_features(rectify_consumer PRIVATE cxx_std_20)
set_target_properties(rectify_consumer PROPERTIES CXX_EXTENSIONS OFF)

add_test(NAME rectify_installed_consumer COMMAND rectify_consumer)
set_tests_properties(
  rectify_installed_consumer
  PROPERTIES PASS_REGULAR_EXPRESSION "rectify_consumer: PASS" TIMEOUT 300
)
