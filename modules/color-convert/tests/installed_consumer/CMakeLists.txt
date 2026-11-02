# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_color_convert_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-color-convert 1.0 REQUIRED CONFIG)
message(
  STATUS
  "Found holoscan-color-convert ${holoscan-color-convert_VERSION} in ${holoscan-color-convert_DIR}"
  " (vic backend: ${holoscan-color-convert_WITH_VPI})"
)

add_executable(color_convert_consumer main.cpp)
target_link_libraries(color_convert_consumer PRIVATE holoscan::color_convert)
set_target_properties(color_convert_consumer PROPERTIES CXX_EXTENSIONS OFF)

add_test(NAME color_convert_installed_consumer COMMAND color_convert_consumer)
set_tests_properties(
  color_convert_installed_consumer
  PROPERTIES PASS_REGULAR_EXPRESSION "color_convert_consumer: PASS" TIMEOUT 300
)
