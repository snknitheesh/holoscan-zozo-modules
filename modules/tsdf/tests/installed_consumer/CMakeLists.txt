# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Uses the INSTALLED package, as another project would: put its prefix on
# CMAKE_PREFIX_PATH (the Debian package installs to /usr, which CMake searches).
cmake_minimum_required(VERSION 4.0)
project(holoscan_tsdf_consumer LANGUAGES CXX)
enable_testing()

find_package(holoscan-tsdf 1.0 REQUIRED CONFIG)
message(STATUS "Found holoscan-tsdf ${holoscan-tsdf_VERSION} in ${holoscan-tsdf_DIR}")

add_executable(tsdf_consumer main.cpp)
target_link_libraries(tsdf_consumer PRIVATE holoscan::tsdf)
set_target_properties(tsdf_consumer PROPERTIES CXX_EXTENSIONS OFF)

add_test(NAME tsdf_installed_consumer COMMAND tsdf_consumer)
set_tests_properties(
  tsdf_installed_consumer
  PROPERTIES PASS_REGULAR_EXPRESSION "tsdf_consumer: PASS" TIMEOUT 300
)
