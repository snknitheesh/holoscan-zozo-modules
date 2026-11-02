# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_denoise_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-denoise 1.0 REQUIRED CONFIG)
message(STATUS "Found holoscan-denoise ${holoscan-denoise_VERSION} in ${holoscan-denoise_DIR}")

add_executable(denoise_consumer main.cpp)
target_link_libraries(denoise_consumer PRIVATE holoscan::denoise)
set_target_properties(denoise_consumer PROPERTIES CXX_EXTENSIONS OFF)

add_test(NAME denoise_installed_consumer COMMAND denoise_consumer)
set_tests_properties(
  denoise_installed_consumer
  PROPERTIES PASS_REGULAR_EXPRESSION "denoise_consumer: PASS" TIMEOUT 300
)
