# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_height_map_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-height-map 1.0 REQUIRED CONFIG)
message(
  STATUS
  "Found holoscan-height-map ${holoscan-height-map_VERSION} in ${holoscan-height-map_DIR}"
)

add_executable(height_map_consumer main.cpp)
target_link_libraries(height_map_consumer PRIVATE holoscan::height_map)
set_target_properties(height_map_consumer PROPERTIES CXX_EXTENSIONS OFF)

add_test(NAME height_map_installed_consumer COMMAND height_map_consumer)
set_tests_properties(
  height_map_installed_consumer
  PROPERTIES PASS_REGULAR_EXPRESSION "height_map_consumer: PASS" TIMEOUT 300
)
